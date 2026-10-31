"""HTML for the app's custom panels.

Every piece of text that comes from a document or a model is escaped
(html.escape), so a student's answer can never inject markup into the page."""

from html import escape

from core.report import usage_line

TICK = ('<svg class="ag-tick" viewBox="0 0 26 26" aria-label="full marks"><path d="M4 14 L10 20 L22 5" '
        'fill="none" stroke="var(--ag-red)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')
CROSS = ('<svg class="ag-tick" viewBox="0 0 26 26" aria-label="no marks"><path d="M6 6 L20 20 M20 6 L6 20" '
         'fill="none" stroke="var(--ag-red)" stroke-width="3" stroke-linecap="round"/></svg>')
HALF = ('<svg class="ag-tick" viewBox="0 0 26 26" aria-label="part marks"><path d="M4 14 L10 20 L22 5 M7 19 L17 9" '
        'fill="none" stroke="var(--ag-red)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _e(x) -> str:
    """HTML-escape any value (None becomes an empty string)."""
    return escape(str(x if x is not None else ""))


def _f(x: float) -> str:
    """Format a number without trailing zeros (2.0 -> "2", 1.5 -> "1.5")."""
    return f"{x:g}"


def mark_icon(awarded: float, maximum: float) -> str:
    """Red tick for full marks, cross for none, half tick for part marks.

    Args:
        awarded (float): Marks given.
        maximum (float): Maximum marks.

    Returns:
        str: Inline SVG.
    """
    return TICK if awarded >= maximum else CROSS if awarded <= 0 else HALF


APP_NAME = "Ankit"  # plain-text name (messages, files); अंक (ank) is Hindi for "marks"
APP_TITLE = "अंकIt"  # the name as written: "ank" in Devanagari, "it" in English (browser tab)
APP_NAME_HTML = '<span class="ag-name-hi" lang="hi">अंक</span>It'


def header() -> str:
    """Page header with the app name and a one-line description.

    Returns:
        str: HTML.
    """
    return f"""
<header class="ag-header">
  <h1 title="अंक (ank) is Hindi for marks" aria-label="{APP_NAME}">{APP_NAME_HTML}</h1>
  <p>Grades handwritten and typed answer sheets against your rubric, in English and Indian languages.
     Every mark shows the words that earned it; anything doubtful waits for your check.</p>
  <div class="ag-hand-note" aria-hidden="true">checked ✓</div>
</header>"""


def step_title(title: str, intro_html: str = "") -> str:
    """Heading of a step's main panel, with its short explanation.

    Args:
        title (str): The heading.
        intro_html (str): HTML under the heading.

    Returns:
        str: HTML.
    """
    return f'<div class="ag-step-head"><h2>{_e(title)}</h2>{intro_html}</div>'


def grade_intro() -> str:
    """Short explanation at the top of the Grade step.

    Returns:
        str: HTML.
    """
    return ('<p class="ag-lede">One student, or a whole class at once. Answers the grader is unsure about are set '
            'aside for you in step 3.</p>')


def callout(pending: int, sheets: int) -> str:
    """Banner after grading that says how many answers need a teacher's check.

    Args:
        pending (int): Answers flagged and not yet checked.
        sheets (int): Sheets just graded.

    Returns:
        str: HTML ("" before anything is graded).
    """
    if not sheets:
        return ""
    on = f"{sheets} sheet{'s' if sheets != 1 else ''}"
    if pending:
        what = f"{pending} answer needs" if pending == 1 else f"{pending} answers need"
        return (f'<div class="ag-callout"><span class="ag-stamp">Check</span><div><b>{what} your check</b> on the '
                f'{on} just graded. These are the ones the grader was unsure about; the rest are done.</div></div>')
    return (f'<div class="ag-callout ok"><span class="ag-stamp ok">Done</span><div><b>Nothing needs a check</b> on '
            f'the {on} just graded. The results and downloads are below.</div></div>')


def check_progress(total: int, done: int, graded: bool) -> str:
    """Progress through the flagged answers, shown at the top of the Check step.

    Args:
        total (int): Answers that were flagged (checked or not).
        done (int): Of those, how many a teacher has checked.
        graded (bool): Whether anything has been graded yet.

    Returns:
        str: HTML.
    """
    title = "Check flagged answers"
    if not graded:
        return step_title(title, '<p class="ag-lede">Nothing to check yet. Grade sheets in step 2; answers that '
                                 'need your eyes will be listed here.</p>')
    if total == 0:
        return step_title(title, '<p class="ag-lede">No answer was flagged. You can still open any sheet and '
                                 'question below and change its marks.</p>')
    pct = round(100 * done / total)
    if done == total:
        note = (f'<p class="ag-lede">All {total} flagged answers are checked. Download the results with your '
                'changes at the bottom of this page.</p>')
    else:
        note = ('<p class="ag-lede">Each sheet shows its flagged answers and a table of every mark. Accept the sheet '
                f'or change marks in the table; the next sheet to check opens by itself. {total - done} to go.</p>')
    bar = (f'<div class="ag-progress" role="progressbar" aria-valuemin="0" aria-valuemax="{total}" '
           f'aria-valuenow="{done}"><div class="ag-progress-bar"><i style="width:{pct}%"></i></div>'
           f'<span><b>{done}</b> of {total} checked</span></div>')
    return step_title(title, bar + note)


def setup_intro() -> str:
    """Short explanation at the top of the Exam step.

    Returns:
        str: HTML.
    """
    return ('<p class="ag-lede">Give whatever you have: question paper, marking scheme, model answer, or any mix. '
            'Each can be typed, a PDF, a scan, a phone photo or handwritten. One is enough; a scheme or model '
            'answer makes the grading more reliable.</p>')


def settings_intro() -> str:
    """Short explanation at the top of the Settings tab.

    Returns:
        str: HTML.
    """
    return ('<div class="ag-intro"><p>These settings apply to everything you grade next. The defaults work well '
            'for most exams: <b>Standard</b> strictness, <b>3 grading passes</b>, the grader looking at the '
            '<b>page and our reading of it</b>, and a <b>second opinion</b> when passes disagree. Every mark is '
            'always backed by a quote from the student\'s answer, blank answers get 0, and text written to the '
            'grader never earns marks.</p></div>')


def settings_summary(settings, profile_labels: dict[str, str]) -> str:
    """One line of chips with the settings that matter most, shown on the grading tabs.

    Args:
        settings (RunSettings): Current settings.
        profile_labels (dict[str, str]): Readable names of the model profiles.

    Returns:
        str: HTML.
    """
    looks = {"auto": "page + reading", "image": "page only", "text": "reading only", "both": "page + reading"}
    chips = [
        profile_labels.get(settings.profile or "", settings.profile or ""),
        settings.strictness.capitalize(),
        f"{settings.n_samples} grading pass{'es' if settings.n_samples != 1 else ''}",
        f"looks at {looks.get(settings.grader_input, settings.grader_input)}",
    ]
    if settings.use_examples:
        chips.append("uses your marked examples")
    if settings.use_reflection and settings.n_samples > 1:
        chips.append("second opinion on")
    if settings.ocr_recheck:
        chips.append("double-checks handwriting")
    if settings.attempt_credit:
        chips.append(f"attempt marks {settings.attempt_credit:.0%}")
    chip_html = "".join(f'<span class="ag-chip">{_e(c)}</span>' for c in chips if c)
    return (f'<div class="ag-chips" aria-label="Current settings">{chip_html}'
            f'<span class="ag-chip-note">Change these in the Settings tab</span></div>')


def empty(title: str, body: str) -> str:
    """Placeholder panel shown before there is anything to display.

    Args:
        title (str): Bold first line.
        body (str): Explanation.

    Returns:
        str: HTML.
    """
    return f'<div class="ag-empty"><strong>{_e(title)}</strong><br>{_e(body)}</div>'


# ---------------------------------------------------------------- agent activity
def trace(events: list[dict], running: bool, status: str = "") -> str:
    """Activity log: one line per agent step, newest at the bottom.

    Args:
        events (list[dict]): Events from core.trace ({"t", "stage", "label", "msg", "level"}).
        running (bool): Show the "working" line at the end.
        status (str): Text for the working line.

    Returns:
        str: HTML.
    """
    if not events and not running:
        return empty("Agent activity shows up here.", "Each step the agents take is listed as it happens.")
    t0 = events[0]["t"] if events else 0
    rows = []
    for ev in events[-150:]:
        rows.append(f'<li class="{_e(ev["level"])}"><span class="t">+{ev["t"] - t0:.1f}s</span>'
                    f'<span class="s">{_e(ev["label"])}</span><span class="m">{_e(ev["msg"])}</span></li>')
    tail = f'<div class="ag-running">{_e(status or "Working…")}</div>' if running else ""
    # column-reverse keeps the newest line in view; the tail must come first in the markup
    return f'<div class="ag-trace" role="log" aria-live="polite">{tail}<ol>{"".join(rows)}</ol></div>'


# ---------------------------------------------------------------- rubric
def rubric_summary(rubric: dict | None, warnings: list[str], mode: str = "") -> str:
    """Counts of questions, criteria, marks and examples, plus rubric warnings.

    Args:
        rubric (dict | None): The rubric, or None before one is drafted.
        warnings (list[str]): Warnings from validate_rubric.
        mode (str): Rubric mode used ("as_is" or "expand"), shown as a note.

    Returns:
        str: HTML.
    """
    if not rubric:
        return empty("No rubric yet.", "Upload any of the question paper, marking scheme or model answer, "
                                       "then draft the rubric. You can also load a sample exam.")
    n_q = len(rubric["questions"])
    n_c = sum(len(q["criteria"]) for q in rubric["questions"])
    total = sum(q["max_marks"] for q in rubric["questions"])
    n_ex = sum(len(q.get("examples") or []) for q in rubric["questions"])
    parts = [f"<span><b>{n_q}</b> questions</span>", f"<span><b>{n_c}</b> criteria</span>",
             f"<span><b>{_f(total)}</b> marks in total</span>"]
    if n_ex:
        parts.append(f"<span><b>{n_ex}</b> graded examples</span>")
    if mode:
        parts.append(f'<span class="ag-note">{"kept your marking points as written" if mode == "as_is" else "criteria split into smaller checks"}</span>')
    warn = ""
    if warnings:
        warn = '<ul class="ag-warnings">' + "".join(f"<li>{_e(w)}</li>" for w in warnings) + "</ul>"
    return f'<div class="ag-rubric-sum">{"".join(parts)}</div>{warn}'


# ---------------------------------------------------------------- one question
def _sheet_pending(report: dict) -> bool:
    """Whether the sheet's whole-sheet notes still wait for a teacher."""
    return bool(report.get("sheet_review_reasons")) and not report.get("sheet_checked")


def check_stamp(report: dict) -> str:
    """The stamp next to a sheet's total: what still needs a teacher, or that it is done.

    Args:
        report (dict): A sheet report.

    Returns:
        str: HTML.
    """
    flagged = sum(1 for q in report["questions"] if q["needs_review"] and not q.get("human_reviewed"))
    sheet = _sheet_pending(report)
    if flagged and sheet:
        return f'<span class="ag-stamp">Check {flagged} answer{"s" if flagged != 1 else ""} + sheet</span>'
    if flagged:
        return f'<span class="ag-stamp">Check {flagged} answer{"s" if flagged != 1 else ""}</span>'
    if sheet:
        return '<span class="ag-stamp">Check the sheet</span>'
    teacher = report.get("sheet_checked") or any(q.get("human_reviewed") for q in report["questions"])
    return f'<span class="ag-stamp ok">{"Checked by a teacher" if teacher else "Nothing to check"}</span>'


def sheet_notes(report: dict) -> str:
    """Whole-sheet notes and the text matched to no question, with the pages it is on.

    Args:
        report (dict): A sheet report.

    Returns:
        str: HTML ("" if there is nothing to say about the whole sheet).
    """
    reasons = report.get("sheet_review_reasons") or []
    if not reasons:
        return ""
    if _sheet_pending(report):
        out = '<div class="ag-check"><b>Whole sheet:</b> ' + "; ".join(_e(r) for r in reasons) + "</div>"
    else:
        out = ('<div class="ag-info"><span class="ag-reviewed">Checked by a teacher.</span> '
               + "; ".join(_e(r) for r in reasons) + "</div>")
    text = (report.get("unassigned_text") or "").strip()
    if text:
        pages = report.get("unassigned_pages") or []
        where = f" (page{'s' if len(pages) > 1 else ''} {', '.join(map(str, pages))})" if pages else ""
        out += (f'<details class="ag-more" open><summary>Text not matched to any question{where}</summary>'
                f'<pre>{_e(text)}</pre></details>')
    return out


def sheet_checks(report: dict) -> str:
    """The top of the Check step for one sheet: total, whole-sheet notes, then the flagged answers
    (those still to check first, then those a teacher has checked).

    Args:
        report (dict): A sheet report.

    Returns:
        str: HTML.
    """
    todo = [q for q in report["questions"] if q["needs_review"] and not q.get("human_reviewed")]
    done = [q for q in report["questions"] if q.get("human_reviewed")]
    left = len(todo) + (1 if _sheet_pending(report) else 0)
    status = (f"{left} thing{'s' if left != 1 else ''} to check on this sheet" if left
              else "Everything on this sheet is checked")
    head = (f'<div class="ag-sheet-head"><span class="ag-mark">{_f(report["total"])}/{_f(report["max_total"])}</span>'
            f'<div><b>{_e(report["meta"].get("student") or report["meta"].get("answer_sheet", ""))}</b><br>'
            f'<span class="ag-note">{status}</span></div>{check_stamp(report)}</div>')
    body = sheet_notes(report) + "".join(question(q) for q in todo + done)
    if not body:
        body = '<p class="ag-note">Nothing was flagged on this sheet. You can still change any mark in the table below.</p>'
    return f'<div class="ag-result">{head}{body}</div>'


def question(q: dict, show_answer: bool = True) -> str:
    """One graded question: mark, check reasons, criteria with quotes, feedback.

    Args:
        q (dict): A question result from grade_question.
        show_answer (bool): Add a collapsible section with the answer as read and agent notes.

    Returns:
        str: HTML.
    """
    flagged = q["needs_review"] and not q.get("human_reviewed")
    check = ""
    if flagged:
        check = ('<div class="ag-check"><b>Check this:</b> ' + "; ".join(_e(r) for r in q["review_reasons"]) + "</div>")
    reviewed = '<span class="ag-reviewed">Checked by a teacher</span>' if q.get("human_reviewed") else ""
    tags = [q.get("language", "English")]
    if q.get("grader_input") == "both":
        tags.append("graded from page image + text")
    elif q.get("grader_input") == "image":
        tags.append("graded from page image")
    if len(q.get("sample_totals") or []) > 1:
        tags.append(f"samples {', '.join(_f(x) for x in q['sample_totals'])}")
    tag_html = "".join(f'<span class="ag-tagline">{_e(t)}</span>' for t in tags)
    info = f'<div class="ag-info">{_e("; ".join(q["info"]))}</div>' if q.get("info") else ""

    rows = []
    for c in q["criteria"]:
        ev = f'<span class="ag-quote">{_e(c["evidence"])}</span>' if c.get("evidence") else ""
        reason = f'<span class="ag-reason">{_e(c["reason"])}</span>' if c.get("reason") else ""
        rows.append(f'<tr><td class="v">{mark_icon(c["awarded"], c["max"])}</td><td>{_e(c["description"])}'
                    f'{"<br>" + ev if ev else ""}{reason}</td><td class="m">{_f(c["awarded"])}/{_f(c["max"])}</td></tr>')
    findings = ""
    if q.get("injection_findings"):
        findings = ('<div class="ag-check"><b>Text aimed at the grader:</b> '
                    + "; ".join(f"“{_e(f['text'])}”" for f in q["injection_findings"]) + "</div>")
    more = ""
    if show_answer:
        extra = ""
        if q.get("recheck") and q["recheck"].get("discrepancies"):
            extra += "<p><b>OCR re-check:</b> " + "; ".join(
                f"read “{_e(d.get('transcribed', ''))}”, page says “{_e(d.get('actually_written', ''))}” "
                f"({_e(d.get('confidence', '?'))} confidence)" for d in q["recheck"]["discrepancies"]) + "</p>"
        if q.get("critique"):
            extra += f"<p><b>Critic's notes</b></p><pre>{_e(q['critique'])}</pre>"
        more = (f'<details class="ag-more"><summary>Answer as read, and agent notes</summary>'
                f'<pre>{_e(q["answer_text"] or "(empty)")}</pre>{extra}</details>')
    return f"""
<div class="ag-q">
  <div class="ag-q-mark"><span class="ag-qid">{_e(q["id"])}</span>
    <span class="ag-mark">{_f(q["awarded"])}/{_f(q["max_marks"])}</span>{reviewed}</div>
  <div class="ag-q-body">
    <h4>{_e(q["text"])}</h4>
    <div>{tag_html}</div>{check}{findings}{info}
    <table class="ag-crit">{"".join(rows)}</table>
    <div class="ag-feedback"><b>Feedback:</b> {_e(q["feedback"] or "–")}</div>
    {more}
  </div>
</div>"""


# ---------------------------------------------------------------- whole sheet
def result(report: dict | None) -> str:
    """The whole graded sheet: total, check stamp, facts, then every question.

    Args:
        report (dict | None): Report from grade_sheet, or None before grading.

    Returns:
        str: HTML.
    """
    if not report:
        return empty("No sheet graded yet.", "Upload one student's pages (or pick a sample sheet) and press "
                                             "Grade this sheet.")
    stamp = check_stamp(report)
    u = report["usage"]
    meta = report["meta"]
    facts = [("Answer sheet", meta.get("student") or meta["answer_sheet"]),
             ("Written in", meta.get("language", "English")),
             ("Read by", ", ".join(meta.get("ocr_engines") or ["text layer"])),
             ("Cost of this run", f"₹{u['cost_inr']:.2f} for {u['api_calls']} model calls")]
    fact_html = "".join(f'<div class="ag-fact"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in facts)
    sheet_checks = sheet_notes(report)
    overall = (f'<div class="ag-overall"><h3>Overall feedback</h3>{_e(report["overall_feedback"])}</div>'
               if report.get("overall_feedback") else "")
    qs = "".join(question(q) for q in report["questions"])
    return f"""
<div class="ag-result">
  <div class="ag-result-top">
    <span class="ag-mark big">{_f(report["total"])}/{_f(report["max_total"])}</span>
    {stamp}
    <div class="ag-facts">{fact_html}</div>
  </div>
  {sheet_checks}{overall}
  <div class="ag-section-title">Question by question</div>
  {qs}
  <p class="ag-note">{_e(usage_line(u))}</p>
</div>"""


# ---------------------------------------------------------------- whole class
def class_view(batch: dict | None) -> str:
    """Class results: average, per-question bars, most-missed criteria, common mistakes, students.

    Args:
        batch (dict | None): Result of grade_batch, or None before grading.

    Returns:
        str: HTML.
    """
    if not batch:
        return empty("No class graded yet.", "Upload one file per student, or a zip with one file or folder per "
                                             "student, and press Grade all sheets.")
    a = batch["analytics"]
    failed = [r for r in batch["results"] if r["error"]]
    head = (f'<div class="ag-result-top"><span class="ag-mark big">{round(a["mean_total"], 1):g}/{_f(a["max_total"])}</span>'
            f'<div class="ag-facts"><div class="ag-fact"><span>Sheets graded</span><b>{a["students"]}</b></div>'
            f'<div class="ag-fact"><span>Class average (circled) and median</span><b>{round(a["mean_total"], 2):g} and {_f(a["median_total"])}</b></div>'
            f'<div class="ag-fact"><span>Need a check</span><b>{a["sheets_needing_review"]}</b></div>'
            f'<div class="ag-fact"><span>Cost</span><b>₹{batch["usage"]["cost_inr"]:.2f}</b></div></div></div>')
    fail = ""
    if failed:
        fail = ('<ul class="ag-warnings">' + "".join(f"<li>Could not grade {_e(r['student'])}: {_e(r['error'][:160])}</li>"
                                                    for r in failed) + "</ul>")
    rows = []
    for q in a["questions"]:
        pct = 100 * q["mean"] / q["max_marks"] if q["max_marks"] else 0
        rows.append(f'<tr><td><b>{_e(q["id"])}</b></td><td>{q["mean"]:.2f} / {_f(q["max_marks"])}</td>'
                    f'<td><div class="ag-bar" aria-label="{pct:.0f} percent"><i style="width:{pct:.0f}%"></i></div></td>'
                    f'<td>{q["full_pct"]:g}%</td><td>{q["zero_pct"]:g}%</td><td>{q["needs_review"]}</td></tr>')
    table = ('<table><thead><tr><th>Question</th><th>Average</th><th></th><th>Full marks</th><th>Zero</th>'
             f'<th>To check</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    details = []
    for q in a["questions"]:
        missed = "".join(f"<li>{c['missed_pct']:g}% missed: {_e(c['description'])}</li>"
                         for c in q["criteria"][:4] if c["missed_by"])
        mistakes = f'<div class="ag-mistakes">{_e(q["common_mistakes"])}</div>' if q.get("common_mistakes") else ""
        details.append(f'<div class="ag-section-title">{_e(q["id"])}: {_e(q["text"])}</div>'
                       f'{"<ul class=ag-miss>" + missed + "</ul>" if missed else ""}{mistakes}')
    students = "".join(
        f'<tr><td>{_e(r["student"])}</td><td><span class="ag-mark" style="height:32px;min-width:56px;font-size:19px">'
        f'{_f(r["report"]["total"])}/{_f(r["report"]["max_total"])}</span></td>'
        f'<td>{"<span class=ag-stamp>Check</span>" if r["report"]["needs_review"] else ""}</td></tr>'
        for r in batch["results"] if r["report"])
    return (f'<div class="ag-class ag-result">{head}{fail}<div class="ag-section-title">By question</div>{table}'
            f'{"".join(details)}<div class="ag-section-title">Students</div><table>{students}</table>'
            f'<p class="ag-note">{_e(usage_line(batch["usage"]))}</p></div>')
