"""अंकIt (Ankit) web app (Gradio).

Run:
    python app.py        then open http://127.0.0.1:7860

Three steps, in the order of the work:
    1 Exam            question paper / marking scheme / model answer -> an editable rubric table
    2 Grade           one student's pages, or a whole class -> live activity -> marks, quotes,
                      feedback, marked pages; class statistics
    3 Check           the flagged answers one by one next to the page; accept or change the marks
    Settings          the grading options a teacher can change
    How it works      slideshow: the approach, pipeline, a sheet's lifecycle, one answer, flags

Environment variables:
    AUTOGRADER_PROFILE              model profile to start with (see config.yaml)
    AUTOGRADER_PUBLIC=1             limits for a public demo (pages, class size, no calculator tool)
    AUTOGRADER_USER / _PASSWORD     ask for a login before the app opens
"""

import contextvars
import os
import queue
import tempfile
import threading
from pathlib import Path

import gradio as gr

from agents.pipeline import build_rubric_stage, grade_batch, grade_sheet, read_sheet
from agents.rubric import validate_rubric
from core import review, trace
from core.analytics import class_analytics
from core.annotate import annotate
from core.batch_io import collect_students
from core.config import ROOT, list_profiles, load_config, resolve_profile_name
from core.llm import LLMError, start_usage
from core.report import save_batch, save_report
from core.settings import CHOICES, RunSettings
from core.utils import clamp, round_half
from ui import render, rubric_table
from ui.theme import CSS, build_theme

PUBLIC = os.getenv("AUTOGRADER_PUBLIC") == "1"
CFG = load_config()
if PUBLIC:
    CFG["ingest"]["max_pages"] = CFG["public"]["max_pages_per_sheet"]
D = RunSettings.from_config(profile=resolve_profile_name())

SAMPLES = ROOT / "samples"


def load_sample_exams() -> dict[str, dict]:
    """Read every samples/*/sample.json (written by the sample builders).

    Returns:
        dict[str, dict]: Exam name -> {"exam": {"question_paper"|"marking_scheme"|"model_answer": [paths]},
            "sheets": {label: [paths]}, "class": path or None}, all paths absolute and existing,
            in the order given by each manifest's "order".
    """
    import json
    exams = []
    for f in sorted(SAMPLES.glob("*/sample.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        base = f.parent

        def ok(paths: list[str], base: Path = base) -> list[str]:
            return [str(base / x) for x in paths if (base / x).exists()]

        exams.append((m.get("order", 99), m["name"], {
            "exam": {k: ok(v) for k, v in m.get("exam", {}).items()},
            "sheets": {label: ok(v) for label, v in m.get("sheets", {}).items() if ok(v)},
            "class": str(base / m["class"]) if m.get("class") and (base / m["class"]).exists() else None,
        }))
    return {name: data for _, name, data in sorted(exams)}


SAMPLE_EXAMS = load_sample_exams()
FIRST_SAMPLE = next(iter(SAMPLE_EXAMS), None)
# Open in light mode even when the computer is set to dark mode (add ?__theme=dark to the URL for dark).
LIGHT_MODE = """<script>
(() => {
  const url = new URL(window.location.href);
  if (!url.searchParams.has("__theme")) {
    url.searchParams.set("__theme", "light");
    window.location.replace(url.href);
  }
})();
</script>"""
FILE_TYPES = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic",
              ".txt", ".md", ".ipynb", ".py"]
# Settings shown in the app, in the order of the controls in the Settings tab.
# Everything else (how passes are combined, seed, the quote/blank/anti-manipulation
# checks being on) comes from config.yaml `defaults:`.
SETTING_KEYS = ["profile", "strictness", "n_samples", "use_examples", "language_hint", "feedback_language",
                "annotate", "grader_input", "rubric_mode", "evidence_check", "guard_detector", "use_reflection",
                "ocr_recheck", "cross_references", "attempt_credit", "ocr_backend", "force_ocr", "use_tools",
                "annotate_locate", "feedback_translator", "class_summary"]

PROFILE_LABELS = {
    "openrouter": "Gemini 2.5 Flash",
    "openrouter_lite": "Gemini 2.5 Flash-Lite (cheaper)",
    "openrouter_gpt": "GPT-4o-mini",
    "ollama_cloud": "Ollama cloud models (via the Ollama app)",
    "ollama_api": "Ollama cloud models (direct)",
    "local": "Small model on this computer",
    "mock": "Demo mode (no AI, made-up marks)",
}


# ================================================================ helpers
def _new_session() -> dict:
    """Return an empty per-browser-tab session.

    Handlers change the session object in place and return that same object.
    Two events can run at once (say a rubric edit while a class is grading),
    and a handler that returned its own copy would overwrite the other's
    change when it finished.

    Returns:
        dict: rubric, rubric_warnings, rubric_mode, entries (graded sheets by name),
        order (names in grading order), current (last graded name), batch (last class run),
        last_run (names graded in the latest run, opened first in Check).
    """
    return {"rubric": None, "rubric_warnings": [], "rubric_mode": "", "entries": {}, "order": [],
            "current": None, "batch": None, "last_run": []}


def _paths(files: str | list | None) -> list[str]:
    """Normalise a Gradio file value to a list of file paths.

    Args:
        files (str | list | None): One path, a list of paths or file objects, or None.

    Returns:
        list[str]: File paths (empty if nothing was uploaded).
    """
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else f.name for f in files]


def _settings(values: tuple) -> RunSettings:
    """Build RunSettings from the Settings-tab control values.

    Args:
        values (tuple): Control values in SETTING_KEYS order.

    Returns:
        RunSettings: Validated settings; hidden options come from config.yaml.

    Raises:
        gr.Error: If a value is not allowed.
    """
    try:
        s = RunSettings.from_config(**dict(zip(SETTING_KEYS, values, strict=True)))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    if PUBLIC:
        s.use_tools = s.use_tools and CFG["public"]["allow_tools"]
    return s


def _stream(fn, *args):
    """Run a pipeline function in a worker thread and stream its activity events.

    Args:
        fn (Callable): The function to run.
        *args: Its arguments.

    Yields:
        tuple[list[dict], dict | None]: (events so far, None) while running, then
        (all events, {"value": result} or {"error": exception}) once at the end.
    """
    events_q: queue.Queue = queue.Queue()
    out: dict = {}

    def target():
        start_usage()
        trace.set_sink(events_q.put)
        try:
            out["value"] = fn(*args)
        except Exception as e:  # noqa: BLE001 - shown to the user below
            out["error"] = e
        finally:
            events_q.put(None)

    threading.Thread(target=contextvars.copy_context().run, args=(target,), daemon=True).start()
    events = []
    while True:
        ev = events_q.get()
        if ev is None:
            break
        events.append(ev)
        yield events, None
    yield events, out


def _fail(events: list, err: Exception) -> str:
    """Render the activity log with an error line at the end.

    Args:
        events (list): Activity events so far.
        err (Exception): The error to show.

    Returns:
        str: HTML for the activity panel.
    """
    import time
    msg = str(err) if isinstance(err, (LLMError, ValueError)) else f"{type(err).__name__}: {err}"
    return render.trace(events + [{"t": time.time(), "stage": "error", "label": "Error", "msg": msg,
                                   "level": "bad"}], False)


def _unique(session: dict, name: str) -> str:
    """Return name, or name (2), name (3)... if a sheet with that name was already graded.

    Args:
        session (dict): The session.
        name (str): Proposed sheet name.

    Returns:
        str: A name not yet used in the session.
    """
    base, n = name, 2
    while name in session["entries"]:
        name = f"{base} ({n})"
        n += 1
    return name


def _student_choices(session: dict) -> list[tuple[str, str]]:
    """Dropdown choices for all graded sheets in the session.

    Args:
        session (dict): The session.

    Returns:
        list[tuple[str, str]]: (label, sheet name). The label is just the name: Gradio keeps showing the
            old label of the selected item when the choices change, so marks or a check status in it
            would go out of date (the sheet's panel shows both).
    """
    return [(name, name) for name in session["order"]]


def _store(session: dict, name: str, report: dict, sheet, settings: RunSettings) -> dict:
    """Save a graded sheet's files and keep it in the session.

    Args:
        session (dict): The session (changed in place).
        name (str): Sheet name.
        report (dict): Report from grade_sheet.
        sheet (Document): The sheet that was graded.
        settings (RunSettings): Settings used (annotate decides if marked pages are drawn).

    Returns:
        dict: The session entry {"report", "sheet", "annotated", "files"}.
    """
    annotated = annotate(report, sheet) if settings.annotate else []
    files = save_report(report, sheet, annotated=annotated)
    session["entries"][name] = {"report": report, "sheet": sheet, "annotated": annotated, "files": files}
    if name not in session["order"]:
        session["order"].append(name)
    return session["entries"][name]


def _gallery(entry: dict, pages: list[int] | None = None) -> list:
    """Page images for a gallery: the marked pages if drawn, else the scans.

    Args:
        entry (dict): A session entry.
        pages (list[int] | None): Only these page numbers (all pages if None or none match).

    Returns:
        list[tuple[Image.Image, str]]: (image, caption) pairs.
    """
    sheet_pages = [p for p in entry["sheet"].pages if p.image is not None]
    imgs = entry["annotated"] or [p.image for p in sheet_pages]
    out = []
    for p, img in zip(sheet_pages, imgs, strict=False):
        if not pages or p.number in pages:
            out.append((img, f"Page {p.number}"))
    return out or [(img, f"Page {p.number}") for p, img in zip(sheet_pages, imgs, strict=False)]


def _downloads(entry: dict) -> list[str]:
    """Files offered for download for one sheet, most useful first.

    Args:
        entry (dict): A session entry.

    Returns:
        list[str]: Paths of the marked PDF, report (Markdown, JSON) and transcript.
    """
    order = ["annotated_pdf", "report_md", "report_json", "transcript_md"]
    return [entry["files"][k] for k in order if k in entry["files"]]


# ================================================================ 1. exam setup
def _sheet_choices(exam: str | None) -> list[str]:
    """Labels of the sample sheets that belong to a sample exam (the first exam if none is chosen)."""
    return list(SAMPLE_EXAMS.get(exam or FIRST_SAMPLE, {}).get("sheets", {}))


def on_sample_exam(exam: str | None):
    """Fill the three upload boxes with a sample exam and point the sheet menu at its students.

    Args:
        exam (str | None): Name of the sample exam.

    Returns:
        tuple: (question paper paths, marking scheme paths, model answer paths, sheet menu update).
    """
    if exam not in SAMPLE_EXAMS:
        return gr.update(), gr.update(), gr.update(), gr.update()
    docs = SAMPLE_EXAMS[exam]["exam"]
    return (docs.get("question_paper") or None, docs.get("marking_scheme") or None,
            docs.get("model_answer") or None, gr.update(choices=_sheet_choices(exam), value=None))


def on_draft_rubric(qp, rub, ma, session: dict, *svals):
    """Draft the rubric from the uploaded documents, streaming the activity log.

    Args:
        qp, rub, ma: Uploaded question paper, marking scheme and model answer files.
        session (dict): The session.
        *svals: Settings-tab values.

    Yields:
        tuple: (activity HTML, rubric table rows, rubric JSON, rubric summary HTML, session,
        "Next: grade" button update, rubric kept in the browser).
    """
    settings = _settings(svals)
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    if not (qp or rub or ma):
        raise gr.Error("Upload at least one of the question paper, marking scheme or model answer "
                       "(or load a sample exam).")
    keep = gr.update()
    for events, out in _stream(build_rubric_stage, _paths(qp), _paths(rub), _paths(ma), settings):
        if out is None:
            yield render.trace(events, True, "Drafting the rubric"), keep, keep, keep, session, keep, keep
            continue
        if "error" in out:
            yield _fail(events, out["error"]), keep, keep, keep, session, keep, keep
            raise gr.Error(str(out["error"]))
        stage = out["value"]
        if session["order"]:  # a new rubric starts a new exam: earlier results would be mixed in
            gr.Info("New rubric drafted, so the sheets graded before were cleared from the app. "
                    "Their files are still in the outputs folder.")
        session.update(entries={}, order=[], current=None, batch=None, last_run=[])
        kept = _keep_rubric(session, stage["rubric"], stage["warnings"], stage["rubric_mode_used"])
        yield (render.trace(events, False), rubric_table.to_rows(stage["rubric"]),
               _json(stage["rubric"]), render.rubric_summary(stage["rubric"], stage["warnings"],
                                                             stage["rubric_mode_used"]), session,
               gr.update(visible=True), kept)


def _keep_rubric(session: dict, rubric: dict, warnings: list[str], mode: str | None = None) -> dict:
    """Store a new rubric in the session and return what the browser keeps across a refresh.

    Sheets graded earlier keep the marks they got with the old rubric; the
    teacher is told so.

    Args:
        session (dict): The session (changed in place).
        rubric (dict): The validated rubric.
        warnings (list[str]): Warnings from validation.
        mode (str | None): How the rubric was drafted (None keeps the current one).

    Returns:
        dict: {"rubric", "rubric_warnings", "rubric_mode"} for the browser's storage.
    """
    session.update(rubric=rubric, rubric_warnings=warnings)
    if mode is not None:
        session["rubric_mode"] = mode
    if session["order"]:
        gr.Info("Rubric updated. Sheets graded before keep their marks; grade them again to use the new rubric.")
    return {"rubric": rubric, "rubric_warnings": warnings, "rubric_mode": session.get("rubric_mode", "")}


def on_new_exam(session: dict) -> tuple:
    """Empty the Grade and Check steps after a new rubric was drafted.

    Args:
        session (dict): The session (already cleared by on_draft_rubric).

    Returns:
        tuple: (activity log, one-sheet result, marked pages, transcript, downloads, class result,
        class downloads, banner, "Check flagged answers" button), then the Check view.
    """
    return (render.trace([], False), render.result(None), [], "", None, render.class_view(None), None, "",
            gr.update(visible=False), *_check_view(session, None))


def _json(rubric: dict) -> str:
    """Pretty-printed rubric JSON for the JSON editor."""
    import json
    return json.dumps(rubric, indent=2, ensure_ascii=False)


def on_table_edit(rows: list, session: dict):
    """Validate and store the rubric after an edit in the table.

    Args:
        rows (list): Table rows.
        session (dict): The session.

    Returns:
        tuple: (summary HTML, rubric JSON or no change, session, rubric kept in the browser).
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    try:
        rubric, warnings = validate_rubric(rubric_table.from_rows(rows, session.get("rubric")))
    except ValueError as e:
        return render.rubric_summary(session.get("rubric"), [f"Not saved: {e}"]), gr.update(), session, gr.update()
    kept = _keep_rubric(session, rubric, warnings)
    return render.rubric_summary(rubric, warnings, session.get("rubric_mode", "")), _json(rubric), session, kept


def on_apply_json(text: str, session: dict):
    """Validate and store a rubric pasted or edited as JSON.

    Args:
        text (str): Rubric JSON.
        session (dict): The session.

    Returns:
        tuple: (table rows, summary HTML, session, rubric kept in the browser).

    Raises:
        gr.Error: If the JSON is not a usable rubric.
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    try:
        rubric, warnings = validate_rubric(text)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    kept = _keep_rubric(session, rubric, warnings)
    return rubric_table.to_rows(rubric), render.rubric_summary(rubric, warnings), session, kept


# ================================================================ 2. grade
def on_mode(mode: str):
    """Show the inputs and results for one student or for a whole class.

    Args:
        mode (str): "one" or "class".

    Returns:
        tuple: Visibility updates for (one-student inputs, class inputs, one-student results, class results).
    """
    one = mode == "one"
    return gr.update(visible=one), gr.update(visible=not one), gr.update(visible=one), gr.update(visible=not one)


def on_sample_sheet(name: str | None, exam: str | None) -> list[str] | None:
    """Put the chosen sample sheet (all its pages) into the upload box.

    Args:
        name (str | None): Sample sheet label.
        exam (str | None): The sample exam it belongs to (the first exam if none is chosen).

    Returns:
        list[str] | None: Page file paths, or None to leave the box empty.
    """
    return SAMPLE_EXAMS.get(exam or FIRST_SAMPLE, {}).get("sheets", {}).get(name)


def on_sample_class(exam: str | None) -> list[str] | None:
    """Put the sample class zip of the chosen sample exam into the upload box.

    Args:
        exam (str | None): Name of the sample exam (the first exam if none is chosen).

    Returns:
        list[str] | None: The zip path, or None if there is no sample class.
    """
    z = SAMPLE_EXAMS.get(exam or FIRST_SAMPLE, {}).get("class")
    if not z:
        gr.Warning("No sample class found. Rebuild it with: python scripts/make_samples.py")
    return [z] if z else None


def _unchecked(session: dict, names: list[str] | None = None) -> list[tuple[str, str | None]]:
    """What nobody has checked yet, in grading order: flagged answers, and whole-sheet notes.

    Args:
        session (dict): The session.
        names (list[str] | None): Only these sheets (all sheets if None).

    Returns:
        list[tuple[str, str | None]]: (sheet name, question id) pairs; the question id is None for a
            whole-sheet note (e.g. text matched to no question).
    """
    out: list[tuple[str, str | None]] = []
    for s in session["order"]:
        if names is not None and s not in names:
            continue
        rep = session["entries"][s]["report"]
        if review.sheet_note_pending(rep):
            out.append((s, None))
        out += [(s, q["id"]) for q in rep["questions"] if q["needs_review"] and not q.get("human_reviewed")]
    return out


def _callout(session: dict, names: list[str]):
    """The banner under the grading panel that leads to the Check step.

    Args:
        session (dict): The session.
        names (list[str]): Sheets that were just graded.

    Returns:
        tuple: (banner HTML, update for the "Check flagged answers" button).
    """
    pending = len(_unchecked(session, names))
    return render.callout(pending, len(names)), gr.update(visible=pending > 0)


def on_grade_one(files, session: dict, *svals):
    """Grade one student's sheet, streaming the activity log.

    The previous sheet's results are cleared as soon as grading starts.

    Args:
        files: The uploaded pages of one sheet.
        session (dict): The session (needs a rubric).
        *svals: Settings-tab values.

    Yields:
        tuple: (activity HTML, result HTML, gallery, transcript, downloads, session, banner HTML,
        check button update).
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    if not session.get("rubric"):
        raise gr.Error("Set up the exam first: draft or paste a rubric in step 1.")
    paths = _paths(files)
    if not paths:
        raise gr.Error("Upload the answer sheet (all pages of one student, in order), or pick a sample sheet.")
    settings = _settings(svals)
    rubric = session["rubric"]

    def job():
        sheet = read_sheet(paths, settings)
        return grade_sheet(rubric, sheet, settings), sheet

    keep = gr.update()
    yield (render.trace([], True, "Grading"), render.empty("Grading…", "Results appear here when the sheet is done."),
           [], "", None, session, "", gr.update(visible=False))
    for events, out in _stream(job):
        if out is None:
            yield (render.trace(events, True, "Grading"), keep, keep, keep, keep, session, keep, keep)
            continue
        if "error" in out:
            yield (_fail(events, out["error"]), render.empty("Grading stopped.", str(out["error"])), keep, keep,
                   keep, session, keep, keep)
            raise gr.Error(str(out["error"]))
        report, sheet = out["value"]
        name = _unique(session, Path(paths[0]).stem if len(paths) == 1 else f"{Path(paths[0]).stem} +{len(paths) - 1}")
        report["meta"]["student"] = name
        entry = _store(session, name, report, sheet, settings)
        session["current"] = name
        session["last_run"] = [name]
        banner, button = _callout(session, [name])
        yield (render.trace(events, False), render.result(report), _gallery(entry), sheet.transcript_markdown(),
               _downloads(entry), session, banner, button)


def on_grade_class(files, session: dict, *svals):
    """Grade every sheet in the upload, streaming the activity log.

    The previous class's results are cleared as soon as grading starts.

    Yields:
        tuple: (activity HTML, class HTML, downloads, session, banner HTML, check button update).
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    if not session.get("rubric"):
        raise gr.Error("Set up the exam first: draft or paste a rubric in step 1.")
    paths = _paths(files)
    if not paths:
        raise gr.Error("Upload one file per student, or a zip with one file or folder per student.")
    settings = _settings(svals)
    rubric = session["rubric"]
    tmp = tempfile.mkdtemp(prefix="autograder_class_")

    def job():
        students = collect_students(paths, tmp)
        if not students:
            raise ValueError("No answer sheets found in the upload.")
        cap = CFG["public"]["max_students_per_batch"]
        if PUBLIC and len(students) > cap:
            raise ValueError(f"The public demo grades at most {cap} sheets at a time.")
        return grade_batch(rubric, students, settings)

    keep = gr.update()
    yield (render.trace([], True, "Grading the class"),
           render.empty("Grading the class…", "Class results appear here when every sheet is done."),
           None, session, "", gr.update(visible=False))
    for events, out in _stream(job):
        if out is None:
            yield render.trace(events, True, "Grading the class"), keep, keep, session, keep, keep
            continue
        if "error" in out:
            yield (_fail(events, out["error"]), render.empty("Grading stopped.", str(out["error"])), keep, session,
                   keep, keep)
            raise gr.Error(str(out["error"]))
        batch = out["value"]
        names = []
        for r in batch["results"]:
            if r["report"]:
                name = _unique(session, r["student"])
                r["report"]["meta"]["student"] = name
                _store(session, name, r["report"], r["sheet"], settings)
                names.append(name)
        session["batch"] = batch
        session["last_run"] = names
        files_out = save_batch(batch)
        banner, button = _callout(session, names)
        yield (render.trace(events, False), render.class_view(batch),
               [files_out["marks_csv"], files_out["analytics_md"], files_out["reports_zip"]], session, banner, button)


# ================================================================ 3. check
CHECK_HEADERS = ["Criterion", "Description", "Max", "Marks", "Student's words", "Check"]


def _status(q: dict) -> str:
    """The Check column of the marks table: "to check" (the grader was unsure), "done" (a teacher
    has looked at it) or "" (not flagged)."""
    if q.get("human_reviewed"):
        return "done"
    return "to check" if q["needs_review"] else ""


def _check_view(session: dict, student: str | None) -> tuple:
    """Everything the Check step shows for one sheet: all its questions at once.

    Args:
        session (dict): The session.
        student (str | None): Sheet name.

    Returns:
        tuple: (progress HTML, sheet menu update, page gallery, flagged-questions HTML,
        marks table rows, feedback table rows, "Accept" button update, tick list update).
    """
    session = session or _new_session()
    total = done = 0
    for s in session["order"]:
        rep = session["entries"][s]["report"]
        flagged = [q for q in rep["questions"] if q["needs_review"] or q.get("human_reviewed")]
        total += len(flagged) + (1 if rep.get("sheet_review_reasons") else 0)
        done += sum(1 for q in flagged if q.get("human_reviewed")) + (1 if rep.get("sheet_checked") else 0)
    progress = render.check_progress(total, done, bool(session["order"]))
    entry = session["entries"].get(student or "")
    if not entry:
        return (progress, gr.update(choices=_student_choices(session), value=None), [],
                render.empty("Nothing open.", "Grade sheets in step 2; sheets with answers that need your eyes "
                                              "are listed here."), gr.update(value=None, visible=False), None,
                gr.update(interactive=False, value="Nothing to accept"), gr.update(choices=[], value=[], visible=False))
    questions = entry["report"]["questions"]
    rows = [[c["id"], c["description"], c["max"], c["awarded"], c.get("evidence", ""), _status(q)]
            for q in questions for c in q["criteria"]]
    feedback = [[q["id"], f"{q['awarded']:g}/{q['max_marks']:g}", q["feedback"]] for q in questions]
    pending = bool(_unchecked(session, [student]))
    rep = entry["report"]
    tick_choices = ([("Whole-sheet note", SHEET_NOTE)] if rep.get("sheet_review_reasons") else []) + [
        (f"{q['id']}  ({q['awarded']:g}/{q['max_marks']:g})", q["id"]) for q in questions
        if q["needs_review"] or q.get("human_reviewed")]
    ticked = ([SHEET_NOTE] if rep.get("sheet_checked") else []) + [q["id"] for q in questions if q.get("human_reviewed")]
    return (progress, gr.update(choices=_student_choices(session), value=student), _gallery(entry),
            render.sheet_checks(rep), gr.update(value=rows, visible=True), feedback,
            gr.update(interactive=pending, value="Accept this sheet as graded" if pending else "Nothing to accept"),
            gr.update(choices=tick_choices, value=ticked, visible=bool(tick_choices)))


def _next_sheet(session: dict, after: str | None = None) -> str | None:
    """The next sheet (after `after`, wrapping round) with something still to check, or None.

    Sheets from the latest grading run come first.
    """
    recent = session.get("last_run") or []
    todo = list(dict.fromkeys(s for s, _ in _unchecked(session, recent))) if recent else []
    todo = todo or list(dict.fromkeys(s for s, _ in _unchecked(session)))
    if not todo:
        return None
    order = session["order"]
    pos = order.index(after) if after in order else -1
    later = [s for s in todo if order.index(s) > pos]
    return (later or todo)[0]


def on_go_check(session: dict):
    """Open the Check step at the first sheet with an answer that still needs a check.

    Returns:
        tuple: (tabs update, then the Check view).
    """
    session = session or _new_session()
    return (gr.Tabs(selected="check"), *_check_view(session, _next_sheet(session) or session.get("current")))


def on_tab_select(student: str | None, session: dict, evt: gr.SelectData):
    """Refresh the Check step when it is opened from the step bar.

    Keeps the sheet that was open; if none was, opens the first one with something to check.

    Returns:
        tuple: The Check view, or no change when another step was opened.
    """
    if evt.value != "Check":
        return tuple(gr.update() for _ in range(8))
    session = session or _new_session()
    if student in session["entries"]:
        return _check_view(session, student)
    return _check_view(session, _next_sheet(session) or session.get("current"))


def on_check_sheet(student: str, session: dict):
    """Show the sheet picked in the menu."""
    return _check_view(session, student)


def on_check_next(student: str, session: dict):
    """Open the next sheet that still has an answer to check."""
    session = session or _new_session()
    nxt = _next_sheet(session, student)
    if nxt is None:
        gr.Info("Every flagged answer is checked.")
        return _check_view(session, student)
    return _check_view(session, nxt)


def _refresh_batch(session: dict) -> None:
    """Recompute class statistics after a teacher changed marks (common mistakes are kept)."""
    batch = session.get("batch")
    if not batch:
        return
    old = {q["id"]: q.get("common_mistakes") for q in batch["analytics"]["questions"]}
    batch["analytics"] = class_analytics(batch["rubric"], [r["report"] for r in batch["results"] if r["report"]])
    for q in batch["analytics"]["questions"]:
        q["common_mistakes"] = old.get(q["id"]) or q.get("common_mistakes")


def _after_check(student: str, session: dict, message: str, advance: bool = True):
    """Redraw the sheet's files and every view after a check, then open the next sheet to check.

    Args:
        student (str): The sheet that was changed.
        session (dict): The session.
        message (str): Short note shown to the teacher.
        advance (bool): Open the next sheet with something to check (False: stay on this sheet).

    Returns:
        tuple: The Check view, then (one-sheet result HTML, one-sheet downloads, class HTML, session).
    """
    entry = session["entries"][student]
    settings = RunSettings.from_config(**{k: v for k, v in entry["report"]["meta"]["settings"].items()
                                          if k != "profile"})
    _store(session, student, entry["report"], entry["sheet"], settings)
    _refresh_batch(session)
    gr.Info(message)
    view = _check_view(session, (_next_sheet(session, student) if advance else None) or student)
    current = session["entries"].get(session.get("current"))
    return (*view, render.result(current["report"]) if current else gr.update(),
            _downloads(current) if current else gr.update(),
            render.class_view(session["batch"]) if session.get("batch") else gr.update(), session)


SHEET_NOTE = "__sheet__"  # value of the whole-sheet note in the tick list


def on_check_tick(student: str, ticked: list[str], session: dict):
    """Tick or untick single answers (and the whole-sheet note) as checked, staying on the sheet.

    Args:
        student (str): Sheet name.
        ticked (list[str]): Question ids ticked in the list, plus SHEET_NOTE if the sheet note is ticked.
        session (dict): The session.

    Raises:
        gr.Error: If no sheet is open.
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    entry = session["entries"].get(student)
    if not entry:
        raise gr.Error("Open a sheet first.")
    rep, ticked = entry["report"], set(ticked or [])
    done, undone = [], []
    for q in rep["questions"]:
        if not (q["needs_review"] or q.get("human_reviewed")):
            continue
        if q["id"] in ticked and not q.get("human_reviewed"):
            review.accept(rep, q["id"])
            done.append(q["id"])
        elif q["id"] not in ticked and q.get("human_reviewed"):
            review.unaccept(rep, q["id"])
            undone.append(q["id"])
    if rep.get("sheet_review_reasons"):
        if SHEET_NOTE in ticked and not rep.get("sheet_checked"):
            review.accept_sheet(rep)
            done.append("the sheet note")
        elif SHEET_NOTE not in ticked and rep.get("sheet_checked"):
            review.unaccept_sheet(rep)
            undone.append("the sheet note")
    msg = "; ".join(x for x in [f"checked {', '.join(done)}" if done else "",
                                f"not checked any more: {', '.join(undone)}" if undone else ""] if x)
    if not _unchecked(session, [student]):
        msg += ". Everything on this sheet is checked; press Next sheet to check to go on"
    return _after_check(student, session, f"{student}: {msg or 'no change'}", advance=False)


def on_check_save(student: str, rows: list, feedback_rows: list, session: dict):
    """Save the teacher's marks and feedback for the whole sheet, then open the next sheet to check.

    Questions whose marks or feedback changed get the new values; every other
    flagged question on the sheet is marked as checked with its marks unchanged.

    Args:
        student (str): Sheet name.
        rows (list): Marks table rows (CHECK_HEADERS).
        feedback_rows (list): Feedback table rows [question, marks, feedback].
        session (dict): The session.

    Raises:
        gr.Error: If no sheet is open or a mark is not a number.
    """
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    entry = session["entries"].get(student)
    if not entry:
        raise gr.Error("Open a sheet first.")
    owner = {c["id"]: q["id"] for q in entry["report"]["questions"] for c in q["criteria"]}
    marks: dict[str, dict[str, float]] = {}
    for r in rows or []:
        if not r or str(r[0]) not in owner:
            continue
        try:
            marks.setdefault(owner[str(r[0])], {})[str(r[0])] = float(r[3])
        except (TypeError, ValueError, IndexError):
            raise gr.Error(f"Marks for {r[0]} must be a number.") from None
    feedback = {str(r[0]): str(r[2] or "") for r in feedback_rows or [] if len(r) > 2}
    changed, accepted = [], 0
    for q in entry["report"]["questions"]:
        new = {cid: clamp(round_half(v), 0, next(c["max"] for c in q["criteria"] if c["id"] == cid))
               for cid, v in marks.get(q["id"], {}).items()}  # as review.override will store them
        moved = any(abs(new.get(c["id"], c["awarded"]) - c["awarded"]) > 1e-9 for c in q["criteria"])
        fb = feedback.get(q["id"], q["feedback"])
        if moved or fb.strip() != (q["feedback"] or "").strip():
            before = q["awarded"]
            review.override(entry["report"], q["id"], new, fb)
            changed.append(f"{q['id']} {before:g} → {q['awarded']:g}")
        elif q["needs_review"] and not q.get("human_reviewed"):
            review.accept(entry["report"], q["id"])
            accepted += 1
    if review.sheet_note_pending(entry["report"]):
        review.accept_sheet(entry["report"])
        accepted += 1
    parts = ([f"changed {', '.join(changed)}"] if changed else []) + ([f"accepted {accepted}"] if accepted else [])
    return _after_check(student, session, f"Saved {student}: " + ("; ".join(parts) or "nothing changed"))


def on_check_accept(student: str, session: dict):
    """Accept every flagged answer on the sheet as graded, then open the next sheet to check."""
    session = session if session is not None else _new_session()  # changed in place, see _new_session
    entry = session["entries"].get(student)
    if not entry:
        raise gr.Error("Open a sheet first.")
    todo = [q["id"] for q in entry["report"]["questions"] if q["needs_review"] and not q.get("human_reviewed")]
    sheet = review.sheet_note_pending(entry["report"])
    if not todo and not sheet:
        raise gr.Error(f"Nothing on {student} is waiting for a check.")
    for qid in todo:
        review.accept(entry["report"], qid)
    review.accept_sheet(entry["report"])
    what = ([f"{len(todo)} answer{'s' if len(todo) != 1 else ''}"] if todo else []) + (["the sheet note"] if sheet else [])
    return _after_check(student, session, f"Accepted {' and '.join(what)} on {student} as graded")


def on_export(session: dict):
    """Write the results again with every check included: the class files, or the last sheet's files.

    Returns:
        dict: Update that shows the download box with the files.

    Raises:
        gr.Error: If nothing was graded yet.
    """
    session = session or _new_session()
    if session.get("batch"):
        _refresh_batch(session)
        out = save_batch(session["batch"])
        return gr.update(value=[out["marks_csv"], out["analytics_md"], out["reports_zip"]], visible=True)
    current = session["entries"].get(session.get("current"))
    if not current:
        raise gr.Error("Grade a sheet or a class first.")
    return gr.update(value=_downloads(current), visible=True)


# ================================================================ settings
def on_settings_change(*svals) -> tuple[str, dict]:
    """Update the settings summary shown in the Grade step, and keep the settings in the browser.

    Args:
        *svals: Settings-tab values.

    Returns:
        tuple[str, dict]: Summary HTML, and {setting name: value} for the browser's storage.
    """
    values = dict(zip(SETTING_KEYS, svals, strict=True))
    try:
        return render.settings_summary(RunSettings.from_config(**values), PROFILE_LABELS), values
    except ValueError as e:
        return render.empty("Settings not valid.", str(e)), gr.update()


def on_load(kept_rubric: dict | None, kept_settings: dict | None, session: dict):
    """Bring back the rubric and settings kept in the browser, after a refresh or a new visit.

    Graded sheets are not kept (their pages are too big for the browser); their
    files are in outputs/, and grading the same files again is instant and free
    because every model reply is cached.

    Args:
        kept_rubric (dict | None): {"rubric", "rubric_warnings", "rubric_mode"} or None.
        kept_settings (dict | None): {setting name: value} or None.
        session (dict): The new, empty session (changed in place).

    Returns:
        tuple: (session, rubric table, rubric JSON, rubric summary, "Next: grade" button), then one
        value per setting control.
    """
    keep = gr.update()
    out = [session, keep, keep, keep, keep]
    try:
        rubric, warnings = validate_rubric((kept_rubric or {}).get("rubric"))
        session.update(rubric=rubric, rubric_warnings=warnings, rubric_mode=kept_rubric.get("rubric_mode", ""))
        out = [session, rubric_table.to_rows(rubric), _json(rubric),
               render.rubric_summary(rubric, warnings, session["rubric_mode"]), gr.update(visible=True)]
    except (ValueError, TypeError, AttributeError):
        pass  # nothing kept, or not a usable rubric: start empty
    values = []
    for key in SETTING_KEYS:
        v = (kept_settings or {}).get(key)
        try:  # a value kept from an older version may no longer be allowed
            ok = v is not None and (v in list_profiles() if key == "profile"
                                    else bool(RunSettings.from_config(**{key: v})))
        except ValueError:
            ok = False
        values.append(v if ok else keep)
    return (*out, *values)


def _lock(locked: bool):
    """Lock or unlock the rubric and the grading buttons while a job runs.

    While a sheet or class is being graded, or a rubric drafted, the rubric
    can't be changed and a second job can't be started; the running job
    already took its copy of the rubric and settings.

    Args:
        locked (bool): True to lock, False to unlock.

    Returns:
        tuple: Updates for (draft button, apply-JSON button, rubric table, grade-one button,
        grade-class button, one/class switch, sample-exam menu).
    """
    on = gr.update(interactive=not locked)
    table = gr.update(interactive=not locked,
                      label="Rubric: locked while a job runs" if locked else
                      "Rubric: edit any cell, add rows for new criteria. Grading uses exactly this.")
    return on, on, table, on, on, on, on


def on_reset_settings() -> list:
    """Default value for every Settings-tab control, from config.yaml.

    Returns:
        list: Values in SETTING_KEYS order.
    """
    defaults = RunSettings.from_config()
    values = []
    for key in SETTING_KEYS:
        values.append(resolve_profile_name() if key == "profile" else getattr(defaults, key))
    return values


# ================================================================ layout
def _settings_tab() -> list:
    """Lay out the Settings tab.

    Returns:
        list: The setting controls, in SETTING_KEYS order.
    """
    gr.HTML(render.settings_intro())
    with gr.Group(elem_classes="ag-card"):
        gr.HTML('<div class="ag-card-title">Main settings</div>')
        with gr.Row():
            s_profile = gr.Dropdown(
                [(PROFILE_LABELS.get(p, p), p) for p in list_profiles()], value=resolve_profile_name(),
                label="AI model", interactive=not PUBLIC,
                info="Which AI service reads the pages and grades. Demo mode uses no AI and gives "
                     "made-up marks, for trying the app.")
            s_strict = gr.Radio(
                [("Lenient", "lenient"), ("Standard", "standard"), ("Strict", "strict")],
                value=D.strictness, label="Strictness",
                info="How much benefit of the doubt an answer gets. Lenient: credit for the key idea "
                     "even if brief. Standard: credit when the answer clearly shows the point. Strict: "
                     "full marks only when every point is stated correctly.")
            s_n = gr.Radio(
                [("1 (fastest)", 1), ("3", 3), ("5 (steadiest)", 5)], value=D.n_samples,
                label="Grading passes",
                info="Each answer is graded this many times, independently, and the marks are combined. "
                     "More passes give steadier marks and a confidence score, but each pass adds cost.")
        with gr.Row():
            s_lang = gr.Dropdown(
                CHOICES["language_hint"], value=D.language_hint, label="Language of the answer sheets",
                info="Choosing it helps the reader with Hindi, Tamil and other scripts. 'auto' works it "
                     "out from the page.")
            s_fblang = gr.Dropdown(
                CHOICES["feedback_language"], value=D.feedback_language, label="Feedback language",
                info="Language of the comments the student gets. 'auto' answers in the language the "
                     "student wrote in.")
        with gr.Row():
            s_examples = gr.Checkbox(
                D.use_examples, label="Use answers I have already marked",
                info="If the rubric holds example answers with your marks, the grader is shown them "
                     "so it follows your standard.")
            s_ann = gr.Checkbox(
                D.annotate, label="Mark the pages in red",
                info="Return the scanned pages as a PDF with ticks, crosses and marks written beside "
                     "each answer.")

    with gr.Accordion("Advanced settings", open=False, elem_classes="ag-advanced"):
        with gr.Group(elem_classes="ag-card"):
            gr.HTML('<div class="ag-card-title">What the grader looks at</div>')
            with gr.Row():
                s_input = gr.Radio(
                    [("Page + reading", "auto"), ("Page image only", "image"), ("Reading only", "text")],
                    value=D.grader_input, label="Grader looks at",
                    info="The scanned page itself, our typed reading of it, or both. Both is the most "
                         "reliable: the page catches reading mistakes, the reading gives exact quotes.")
                s_rmode = gr.Radio(
                    [("Automatic", "auto"), ("Keep my points", "as_is"), ("Split into smaller checks", "expand")],
                    value=D.rubric_mode, label="How the rubric is drafted",
                    info="Keep my points: your marking scheme is used as written. Split: each point is "
                         "broken into smaller checks (useful when you only give a model answer). "
                         "Automatic picks for you.")
        with gr.Group(elem_classes="ag-card"):
            gr.HTML('<div class="ag-card-title">Checks</div>')
            with gr.Row():
                s_ev = gr.Radio(
                    [("Remove the mark", "reject"), ("Keep it, flag for me", "flag")],
                    value=D.evidence_check if D.evidence_check in ("reject", "flag") else "reject",
                    label="A mark without a quote",
                    info="The grader must quote the student's own words for every mark. If the quote is "
                         "not on the sheet, the mark is removed, or kept and flagged for you to look at.")
                s_gdet = gr.Radio(
                    [("Remove it", "strip"), ("Only flag it", "flag")],
                    value=D.guard_detector if D.guard_detector in ("strip", "flag") else "strip",
                    label="Messages to the grader",
                    info="Text such as \"please give full marks\" written on the sheet. Remove: it is "
                         "deleted before grading and the answer is flagged. Flag: it stays, the answer is "
                         "flagged.")
            with gr.Row():
                s_refl = gr.Checkbox(
                    D.use_reflection, label="Second opinion when passes disagree",
                    info="If the grading passes give different marks, a reviewer points out what went "
                         "wrong and the answer is graded once more. Needs 2 or more passes.")
                s_recheck = gr.Checkbox(
                    D.ocr_recheck, label="Double-check the handwriting",
                    info="Where an answer lost marks, the page is read again in case a word or number "
                         "was misread; if so, it is graded again. For messy sheets; one extra step per "
                         "such answer.")
                s_links = gr.Checkbox(
                    D.cross_references, label="Linked questions",
                    info="If an answer says \"see Q3\", or a question builds on an earlier one, the "
                         "grader sees both answers and doesn't take marks off twice for one mistake.")
        with gr.Group(elem_classes="ag-card"):
            gr.HTML('<div class="ag-card-title">Course policy</div>')
            s_attempt = gr.Radio(
                [("Off", 0.0), ("A quarter of the marks", 0.25), ("Half the marks", 0.5)],
                value=D.attempt_credit if D.attempt_credit in (0.0, 0.25, 0.5) else 0.0,
                label="Marks for a genuine attempt",
                info="Give at least this share of the marks to any real attempt on the topic, even if it "
                     "is wrong. Leave off unless your course marks this way.")
        with gr.Group(elem_classes="ag-card"):
            gr.HTML('<div class="ag-card-title">Reading the sheets</div>')
            with gr.Row():
                s_ocr = gr.Radio(
                    [("Automatic", "auto"), ("AI model above", "vlm"), ("Bodhan", "bodhan")],
                    value=D.ocr_backend, label="Handwriting reader",
                    info="Bodhan is built for Indian languages. Automatic uses it for an Indian "
                         "language when a Bodhan key is set, and the AI model otherwise.")
                s_force = gr.Checkbox(
                    D.force_ocr, label="Always read the page images",
                    info="Read the scanned images even when a PDF has typed text inside it. Use it when "
                         "a PDF's hidden text is wrong.")
                s_tools = gr.Checkbox(
                    D.use_tools, label="Check arithmetic with a calculator",
                    info="Before grading, the student's calculations are checked with a calculator. "
                         "Student code is never run.",
                    interactive=not PUBLIC or CFG["public"]["allow_tools"])
        with gr.Group(elem_classes="ag-card"):
            gr.HTML('<div class="ag-card-title">Output</div>')
            with gr.Row():
                s_loc = gr.Checkbox(
                    D.annotate_locate, label="Place marks beside each answer",
                    info="Finds where each answer starts on the page so the red marks sit next to it. "
                         "Off: marks go down the margin in order. One extra step per scanned page.")
                s_fbtr = gr.Radio(
                    [("The grader", "llm"), ("Bodhan translates", "bodhan")],
                    value=D.feedback_translator, label="Who writes feedback in Indian languages",
                    info="The grader writes it directly, or writes it in English and Bodhan translates it.")
                s_class = gr.Checkbox(
                    D.class_summary, label="Common mistakes per question",
                    info="When grading a class, list the mistakes most students made on each question.")
    return [s_profile, s_strict, s_n, s_examples, s_lang, s_fblang, s_ann, s_input, s_rmode, s_ev,
            s_gdet, s_refl, s_recheck, s_links, s_attempt, s_ocr, s_force, s_tools, s_loc, s_fbtr, s_class]


HOW_CAPTIONS = {  # captions for the How it works slides (docs/images/how/NN_<name>.png)
    "approach": "Agents propose, rule checks verify",
    "pipeline": "From exam files to marked pages",
    "sheet-life": "A sheet's lifecycle: every case",
    "question-flow": "Every check on one answer",
    "flags": "What sends an answer to the teacher",
    "journey": "The teacher's three steps",
    "robust": "When things go wrong",
}


def build_app() -> gr.Blocks:
    """Lay out the steps and connect every button to its handler.

    Returns:
        gr.Blocks: The app.
    """
    with gr.Blocks(theme=build_theme(), css=CSS, title=render.APP_TITLE, head=LIGHT_MODE) as demo:
        session = gr.State(_new_session())
        kept_rubric = gr.BrowserState(None, storage_key="ankit_rubric")      # survive a page refresh
        kept_settings = gr.BrowserState(None, storage_key="ankit_settings")
        gr.HTML(render.header())

        with gr.Tabs(elem_classes="ag-steps", selected="exam") as tabs:
            # ---------------------------------------------------------------- 1
            with gr.Tab("Exam", id="exam"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300, elem_classes="ag-panel"):
                        gr.HTML(render.step_title("Set up the exam", render.setup_intro()))
                        qp = gr.File(label="Question paper", file_count="multiple", type="filepath",
                                     file_types=FILE_TYPES, height=110)
                        rub = gr.File(label="Marking scheme", file_count="multiple", type="filepath",
                                      file_types=FILE_TYPES, height=110)
                        ma = gr.File(label="Model answer", file_count="multiple", type="filepath",
                                     file_types=FILE_TYPES, height=110)
                        sample_exam = gr.Dropdown(list(SAMPLE_EXAMS), value=None, label="Or load a sample exam",
                                                  visible=bool(SAMPLE_EXAMS))
                        btn_draft = gr.Button("Draft the rubric", variant="primary", size="lg")
                        setup_trace = gr.HTML(render.trace([], False))
                    with gr.Column(scale=7, min_width=360, elem_classes="ag-panel"):
                        rubric_sum = gr.HTML(render.rubric_summary(None, []))
                        rubric_df = gr.Dataframe(
                            headers=rubric_table.HEADERS, datatype=rubric_table.DATATYPES,
                            value=rubric_table.to_rows(None), type="array", interactive=True,
                            row_count=(1, "dynamic"), col_count=(5, "fixed"), wrap=True,
                            column_widths=["6%", "28%", "43%", "9%", "14%"],
                            label="Rubric: edit any cell, add rows for new criteria. Grading uses exactly this.",
                            elem_classes="ag-df")
                        with gr.Accordion("Model answers and graded examples (JSON)", open=False):
                            gr.Markdown('Add staff-graded answers per question as `"examples": [{"answer": "...", '
                                        '"marks": 2, "comment": "..."}]` to show the grader how strict to be.',
                                        elem_classes="ag-note")
                            rubric_json = gr.Code(language="json", interactive=True, label="Rubric JSON")
                            btn_json = gr.Button("Apply JSON")
                        btn_to_grade = gr.Button("Rubric looks right: grade answer sheets", variant="primary",
                                                 visible=False, elem_classes="ag-next")

            # ---------------------------------------------------------------- 2
            with gr.Tab("Grade", id="grade"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300, elem_classes="ag-panel"):
                        gr.HTML(render.step_title("Grade answer sheets", render.grade_intro()))
                        mode = gr.Radio([("One student", "one"), ("Whole class", "class")], value="one",
                                        show_label=False, container=False, elem_classes="ag-segmented")
                        with gr.Column(visible=True) as one_in:
                            sheet = gr.File(label="One student's answer sheet (all pages, in order)",
                                            file_count="multiple", type="filepath", file_types=FILE_TYPES, height=130)
                            sample_pick = gr.Dropdown(_sheet_choices(None), value=None, label="Or try a sample student",
                                                      info="From the sample exam chosen in step 1.")
                            btn_grade = gr.Button("Grade this sheet", variant="primary", size="lg")
                        with gr.Column(visible=False) as class_in:
                            class_files = gr.File(label="One file per student, or a zip (one file or folder each)",
                                                  file_count="multiple", type="filepath",
                                                  file_types=FILE_TYPES + [".zip"], height=130)
                            btn_sample_class = gr.Button("Use the sample class", size="sm")
                            btn_class = gr.Button("Grade all sheets", variant="primary", size="lg")
                        grade_settings = gr.HTML(render.settings_summary(D, PROFILE_LABELS))
                    with gr.Column(scale=7, min_width=360, elem_classes="ag-panel"):
                        grade_trace = gr.HTML(render.trace([], False))
                        with gr.Row(elem_classes="ag-callout-row"):
                            callout = gr.HTML("")
                            btn_go_check = gr.Button("Check flagged answers", variant="primary", visible=False,
                                                     scale=0, min_width=220)
                with gr.Column(visible=True, elem_classes="ag-results") as one_out:
                    with gr.Tabs():
                        with gr.Tab("Marks and feedback"):
                            result_html = gr.HTML(render.result(None))
                        with gr.Tab("Marked pages"):
                            annotated = gr.Gallery(label="Pages marked in red", columns=1, height=760,
                                                   object_fit="contain", show_label=False)
                        with gr.Tab("Sheet as read"):
                            transcript = gr.Markdown()
                        with gr.Tab("Downloads"):
                            downloads = gr.File(label="Marked PDF, report, full JSON, transcript",
                                                file_count="multiple")
                with gr.Column(visible=False, elem_classes="ag-results") as class_out:
                    class_html = gr.HTML(render.class_view(None))
                    class_downloads = gr.File(label="Marks CSV, class analytics, all reports", file_count="multiple")

            # ---------------------------------------------------------------- 3
            with gr.Tab("Check", id="check"):
                ck_progress = gr.HTML(render.check_progress(0, 0, False))
                with gr.Row(elem_classes="ag-check-bar"):
                    rv_student = gr.Dropdown([], label="Answer sheet", scale=4, allow_custom_value=True)
                    btn_next = gr.Button("Next sheet to check", scale=1, min_width=200)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320, elem_classes="ag-panel"):
                        rv_gallery = gr.Gallery(show_label=False, columns=1, height=760, object_fit="contain")
                    with gr.Column(scale=6, min_width=360, elem_classes="ag-panel"):
                        rv_ticks = gr.CheckboxGroup([], value=[], visible=False, elem_classes="ag-ticks",
                                                    label="Tick each flagged answer once you have checked it",
                                                    info="Ticked answers keep their marks. To change marks, edit "
                                                         "the table below and press Save my changes.")
                        rv_html = gr.HTML(render.empty("Nothing open.", "Grade sheets in step 2; sheets with "
                                                                        "answers that need your eyes are listed here."))
                        rv_df = gr.Dataframe(headers=CHECK_HEADERS, visible=False,
                                             datatype=["str", "str", "number", "number", "str", "str"],
                                             type="array", interactive=True, static_columns=[0, 1, 2, 4, 5],
                                             wrap=True, max_height=560, elem_classes="ag-df",
                                             column_widths=["13%", "31%", "8%", "10%", "25%", "13%"],
                                             label="Every mark on this sheet. Only the Marks column can be changed. "
                                                   "Check: 'to check' = the grader was unsure, 'done' = you have "
                                                   "looked at it.")
                        with gr.Accordion("Feedback to the student", open=False):
                            rv_fb = gr.Dataframe(headers=["Question", "Marks", "Feedback"],
                                                 datatype=["str", "str", "str"], type="array", interactive=True,
                                                 static_columns=[0, 1], wrap=True, elem_classes="ag-df",
                                                 column_widths=["12%", "12%", "76%"])
                        with gr.Row():
                            btn_accept = gr.Button("Accept this sheet as graded", variant="primary")
                            btn_save = gr.Button("Save my changes")
                with gr.Row(elem_classes="ag-export-row"):
                    btn_export = gr.Button("Download updated results", scale=0, min_width=240)
                    ck_downloads = gr.File(label="Results with your checks", file_count="multiple", scale=3,
                                           visible=False)

            # ---------------------------------------------------------------- settings
            with gr.Tab("Settings", id="settings"):
                SETTINGS = _settings_tab()
                with gr.Row():
                    btn_reset = gr.Button("Reset to defaults", size="sm", scale=0)

            # ---------------------------------------------------------------- how it works
            with gr.Tab("How it works", id="help"):
                # slides from the project presentation, shown one at a time (arrows, thumbnails, full screen)
                slides = sorted((ROOT / "docs" / "images" / "how").glob("*.png"))
                if slides:
                    gr.Gallery([(str(s), HOW_CAPTIONS.get(s.stem[3:], "")) for s in slides], show_label=False,
                               preview=True, selected_index=0, columns=len(slides), object_fit="contain",
                               height=820, allow_preview=True, show_download_button=False,
                               show_share_button=False, show_fullscreen_button=True)
                with gr.Accordion("The same in words", open=False):
                    gr.Markdown((ROOT / "ui" / "how_it_works.md").read_text(encoding="utf-8"))

        # ---------------------------------------------------------------- events
        lock = [btn_draft, btn_json, rubric_df, btn_grade, btn_class, mode, sample_exam]

        def locked(event, fn, inputs, outputs):
            """Run fn with the rubric and grading buttons locked, unlocking afterwards even if it fails.

            Returns the event of fn itself, so more steps can follow it."""
            run = event(lambda: _lock(True), None, lock, queue=False, show_progress="hidden").then(fn, inputs, outputs)
            run.then(lambda: _lock(False), None, lock, show_progress="hidden")
            return run

        sample_exam.change(on_sample_exam, sample_exam, [qp, rub, ma, sample_pick])
        check_view = [ck_progress, rv_student, rv_gallery, rv_html, rv_df, rv_fb, btn_accept, rv_ticks]
        locked(btn_draft.click, on_draft_rubric, [qp, rub, ma, session] + SETTINGS,
               [setup_trace, rubric_df, rubric_json, rubric_sum, session, btn_to_grade, kept_rubric]) \
            .success(on_new_exam, session, [grade_trace, result_html, annotated, transcript, downloads, class_html,
                                            class_downloads, callout, btn_go_check] + check_view)
        rubric_df.input(on_table_edit, [rubric_df, session], [rubric_sum, rubric_json, session, kept_rubric])
        btn_json.click(on_apply_json, [rubric_json, session], [rubric_df, rubric_sum, session, kept_rubric])
        btn_to_grade.click(lambda: gr.Tabs(selected="grade"), None, tabs).then(
            None, js="() => { window.scrollTo({top: 0, behavior: 'smooth'}); return []; }")

        mode.change(on_mode, mode, [one_in, class_in, one_out, class_out], show_progress="hidden")
        sample_pick.change(on_sample_sheet, [sample_pick, sample_exam], sheet)
        btn_sample_class.click(on_sample_class, sample_exam, class_files)
        locked(btn_grade.click, on_grade_one, [sheet, session] + SETTINGS,
               [grade_trace, result_html, annotated, transcript, downloads, session, callout, btn_go_check])
        locked(btn_class.click, on_grade_class, [class_files, session] + SETTINGS,
               [grade_trace, class_html, class_downloads, session, callout, btn_go_check])

        to_top = "() => { window.scrollTo({top: 0, behavior: 'smooth'}); return []; }"  # a new sheet starts at the top
        btn_go_check.click(on_go_check, session, [tabs] + check_view).then(None, js=to_top)
        rv_student.input(on_check_sheet, [rv_student, session], check_view).then(None, js=to_top)
        btn_next.click(on_check_next, [rv_student, session], check_view).then(None, js=to_top)
        after = check_view + [result_html, downloads, class_html, session]
        btn_accept.click(on_check_accept, [rv_student, session], after).then(None, js=to_top)
        btn_save.click(on_check_save, [rv_student, rv_df, rv_fb, session], after).then(None, js=to_top)
        rv_ticks.input(on_check_tick, [rv_student, rv_ticks, session], after, show_progress="hidden")
        btn_export.click(on_export, session, ck_downloads)
        tabs.select(on_tab_select, [rv_student, session], check_view, show_progress="hidden")

        gr.on([c.change for c in SETTINGS], on_settings_change, SETTINGS, [grade_settings, kept_settings],
              show_progress="hidden")
        btn_reset.click(on_reset_settings, None, SETTINGS)
        demo.load(on_load, [kept_rubric, kept_settings, session],
                  [session, rubric_df, rubric_json, rubric_sum, btn_to_grade] + SETTINGS)
    return demo




demo = build_app()

if __name__ == "__main__":
    user, password = os.getenv("AUTOGRADER_USER"), os.getenv("AUTOGRADER_PASSWORD")
    demo.queue(default_concurrency_limit=4).launch(
        auth=(user, password) if user and password else None,
        allowed_paths=[str(ROOT / "outputs"), str(ROOT / "samples"), str(ROOT / "docs")])
