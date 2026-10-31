# Code guide

This explains the whole codebase: the idea behind the architecture, what
happens to an answer sheet step by step, what every file does and why, why
things are named the way they are, which paper each idea comes from, and
the questions you are likely to be asked about it.

Read it in this order the first time:
**Part 1 → Part 2 → `agents/pipeline.py` with Part 2 open → Part 3 as a reference.**
Every setting and every ablation is explained in `docs/SETTINGS_AND_ABLATIONS.md`.

---

## Part 1. The idea in one page

### The problem

A TA gives the system a question paper, a marking scheme and/or model answer,
and a student's answer sheet (often a phone photo of handwriting, sometimes
in Hindi or Tamil). The system must give marks per question that a careful
human would agree with, explain them, and admit when it is unsure.

### Why there is no training

We have no large set of graded sheets to train on, and a week, not a month.
Every idea we use from the course papers is applied **at inference time**:
sampling several answers (test-time compute), letting a model act with tools
(ReAct), letting a model critique and retry (Reflexion), splitting a job
between specialised agents (orchestration), and keeping each agent's context
small (ACON, MEM1). Training-based papers (ToolRL, ReTool's RL, AgentGym-RL,
MEM1's RL) are listed as future work in Part 5.

### The one design rule: agents propose, rule checks verify

There are two kinds of component. ("Rule check" means a check written in plain Python, with no model involved. It has nothing to do with running a student's code, which the system never does.)

| | Agents (model calls) | Checks (plain Python) |
|---|---|---|
| What they do | read, interpret, judge | count, compare, verify |
| Strength | understand handwriting, meaning, partial answers | exact, free, can't be talked into anything |
| Weakness | can hallucinate, be inconsistent, be manipulated | can't understand language |
| Examples | OCR, rubric, segmenter, grader, critic, re-check, locator, tool agent | evidence check, grounding check, injection detector, blank check, aggregation, mark limits |

Every agent output passes through a check before it can change a mark. That
single rule answers most "what if the model is wrong?" questions: a mark
needs a quote that really exists in the answer; an answer must really exist
on the sheet; marks can't exceed a criterion's maximum; text asking for marks
is removed by rules a model can't argue with.

### The three stages

1. **Exam setup** (once per exam): turn the marking scheme into a structured
   rubric. A human checks it, then it is frozen.
2. **Grading** (once per sheet): read → match answers to questions → grade
   each question separately and in parallel, with checks → report.
3. **Review** (by a TA): only the flagged questions; accept or change.

---

## Part 2. The journey of one answer sheet

Follow this with `agents/pipeline.py` open. Function names are in brackets.

### Stage 1: exam setup (`build_rubric_stage`)

| Step | Where | What happens | Why |
|---|---|---|---|
| 1.1 Ingest | `core/ingest.py` `load_many` | Each file becomes a `Document` of `Page`s. Typed PDF/Word text is read directly; scans and photos become images marked `needs_ocr`. | A text layer is exact and free; only images need a model. |
| 1.2 OCR | `agents/ocr.py` `run_ocr` | A vision model (or Bodhan for Indic sheets) transcribes image pages. | Same OCR rules as for answer sheets. |
| 1.3 Rubric agent | `agents/rubric.py` `build_rubric` | Question paper + marking scheme + model answer → JSON: questions, criteria, marks. `as_is` copies the TA's marking points; `expand` splits vague ones into checkable steps. | The code needs criteria with ids and marks to check quotes per criterion and combine samples. HITL-Maths: small explicit steps grade best; GLaH: too fine hurts simple questions, so it's a switch. |
| 1.4 Validate | `agents/rubric.py` `validate_rubric` | Code repairs the JSON: renames duplicate ids, drops zero-mark criteria, makes max marks equal the criteria total, warns. | Model JSON is never trusted as-is. |
| 1.5 Human edit | `app.py` tab 1, `ui/rubric_table.py` | TA edits the rubric as a table; then it's frozen for grading. | DR Tulu evolves rubrics during grading; we make them concrete once, before any answer is seen, so early wrong answers can't drag the rubric. |

### Stage 2: grading one sheet (`read_sheet` → `grade_sheet`)

| Step | Where | What happens | Why (paper) |
|---|---|---|---|
| 2.1 Ingest + OCR | `read_sheet` | As in 1.1–1.2. Page images are stored at the size the model sees (1600 px), which keeps 600-dpi scans from filling memory. A page with no ink (measured against its own background: empty, or only show-through from the back of the paper) skips OCR and adds a warning (`ingest.looks_blank`). The OCR prompt forbids fixing mistakes, uses `[illegible]` instead of guessing, marks `[STRUCK: ]`, `[MARGIN: ]`, `[DIAGRAM: ]`, keeps every word in its own script, never translates, and ignores instructions written on the page. Page images are **kept**. | A grader must see the student's real mistakes. Kept images are needed later by the grader, re-check and annotation. |
| 2.2 Segmenter | `agents/segmenter.py` `segment` | One call splits the transcription by question: answers out of order, continued on later pages, labelled "Ans 1", "प्रश्न १", "கேள்வி 3". Returns the page numbers too. Text matching no question goes to `unassigned`. If the reply is unusable, `split_by_labels` splits the sheet at its question labels instead and the sheet is flagged. MCQ answers written as a grid ("1) b  2) c", table cells, numbers over options) are found by `utils.grid_answers`: the segmenter gets a hint for such sheets, and an MCQ it still left empty is filled from the grid. | **Plan-and-Act**: separate deciding *what goes where* from *grading it*. Each question then gets its own small context (**ACON / MEM1** idea). |
| 2.3 Per question, in parallel | `grade_question` | Questions run in a thread pool (`max_parallel` in config). | Independent questions; faster. |

Inside `grade_question`, for each question:

| Step | Where | What happens | Why (paper) |
|---|---|---|---|
| a. Language | `core/lang.py` `detect_language` | Script detection by Unicode block decides the answer's language and the feedback language. | Exact and free; Hindi answers contain English terms, so an Indic script with ≥20% of letters wins. |
| b. Grounding check | `core/utils.py` `coverage` | ≥60% of the segmented answer's lines must be found on the sheet, otherwise flag. | **EngExams**: inventing content for blank regions is "operationally unacceptable". The segmenter could do that; code catches it. |
| c. Injection detector | `core/guard.py` `scan` / `strip` | Rules find sentences aimed at the grader (English, Hinglish, 10 Indian scripts). `strip` removes them before grading; always flagged. | **Inject**: prompt warnings alone left >30% of attacks working; a separate detector did far better. |
| d. Blank check | `core/utils.py` `is_blank` | No letter or digit left after the question's own label (`strip_label`), or only the question copied out (`only_question`) → 0 without a model call and no flag (a single MCQ letter like "D" counts as an answer). Only `[illegible]` → the grader reads the page image; without an image, 0 and flagged. With an image, `[illegible]` in an answer is a note, not a flag. | **EngExams** presence check. Saves calls; a model asked to grade nothing may invent something. |
| d2. Linked questions | `grade_sheet` + `utils.find_references` / `utils.infer_dependencies` | If the answer says "as explained in Q3", or the question builds on an earlier one ("Hence…", "using your answer to Q3", from `depends_on` in the rubric), the other answer is appended for the grader, with the rule that a mistake carried forward is not punished twice. Found by code, not a model. | Standard marking practice (error carried forward). |
| e. Pick page images | `_page_images` | Images of the pages the segmenter named (max 3). | For the grader and the re-check. |
| f. Tool agent (optional) | `agents/tool_agent.py` | ReAct loop: think → `calc` → observe, ≤3 steps, once per question, to check arithmetic. Output passed on as untrusted. Student code is never executed (no safe sandbox); the grader reads and traces code instead. | **ReAct**. Off by default: costs calls, only useful for numeric answers. |
| g. Grader × N | `agents/grader.py` `grade_once` | N independent samples (default 3, temperature 0.7). The grader sees criteria, model answer, optional teacher-graded examples, strictness policy, and the answer as **page image + transcription** for scans. Any correct method or wording counts; examples in the key are only examples; half marks when an answer falls between two levels; code is read, never run. (`grading_style: judgement` swaps in a prompt that uses its own subject knowledge with proportional half marks and a `rubric_note`; tried, not better, kept for ablation I.) Every awarded mark needs an exact quote. Anti-injection rule placed *before* the answer. | **Test-time compute** (Snell et al.): spend compute on several attempts instead of a bigger model. Images: **EDU-CIRCUIT** (38% of transcriptions had errors that silently change grades), **EngExams** and **HITL-Maths** grade from the image. Rule placement: **Inject**. Quotes: make marks checkable. |
| h. Rule checks per sample | `grader.normalize` | "met" = full, "not_met" = 0, partial clamped and rounded to 0.5; **evidence check**: the quote must be found (fuzzy ≥80) in the answer without crossed-out text, else the mark is removed (`reject`) or flagged (`flag`). A quote stitched from several lines or table cells counts if every piece is on the sheet; a one-letter quote must be a whole word. Quotes from injected text never count. | **EngExams**: rigid templates + deterministic validation. This is our main defence against hallucinated marks. |
| i. Aggregate | `agents/aggregator.py` | Per criterion, combine the N marks by a percentile set by strictness (lenient 75th, standard median, strict 25th). Spread of totals → confidence. | **Test-time compute** (aggregating samples). **HITL-Maths** used max-of-5 and it pushed marks above the human range, so we don't use max. |
| j. Critic + re-grade | `agents/critic.py` | Only if samples disagree by >25% of the marks, or a quote failed: the critic writes a short critique; one more grade uses it. | **Reflexion**: verbal feedback, then retry. Conditional because **GLaH** found always-on self-reflection sometimes hurt. |
| k. OCR re-check (off by default) | `agents/recheck.py` | Only if switched on, marks were lost and page images exist: a vision model compares transcription with handwriting, focusing on what the deductions depend on. High-confidence misreading → corrected text → full re-grade. Low-confidence → flag. | **EDU-CIRCUIT**: exactly this routing raised agreement by +4.5 to +8.9 points with only 3–4% sent to humans. Only deducted answers: misreadings mostly hurt students. |
| l. Feedback language | `core/bodhan.py` `translate` (optional) | Feedback in the student's language, written by the grader or translated by Bodhan. | Course asks for Indic support. |
| m. Review flags | end of `grade_question` | Flag if: samples disagreed, a mark was removed, a quote wasn't found, text aimed at the grader (page footers like "Page 2 of 6" are ignored), illegible parts, low grounding, re-check unsure, or (judgement style only) the grader questions the rubric (`rubric_note`). | Every paper keeps a human in charge; flags decide what the human must look at. |

Back in `grade_sheet`:

| Step | Where | What happens | Why |
|---|---|---|---|
| 2.4 Sheet flags | `grade_sheet` | ≥15 words matched to no question → the sheet is flagged (possible unlabelled answer). | So a stray answer is never silently lost. |
| 2.5 Overall feedback | `_overall_feedback` | One call: a paragraph for the student, in their language. | Useful feedback is part of the task. |
| 2.6 Locate | `agents/locator.py` | Where each answer starts on each page: exact from the PDF for typed pages, one vision call per scanned page, asked about every question (the segmenter's page numbers can be wrong). | Only for drawing marks on the page. |
| 2.7 Report | returned dict, `core/report.py` | Marks, evidence, flags, raw samples, usage and cost. Saved as Markdown, JSON, transcript and **annotated PDF** (`core/annotate.py`). | Raw samples are saved so ablations on aggregation/evidence can be recomputed for free (`eval/offline.py`). |

### Stage 3: review and class (`core/review.py`, `grade_batch`)

- **Review:** `review.override` sets marks per criterion (clamped to the max),
  logs who changed what, and the app regenerates the report, PDF and CSV.
  `review.accept` marks a flagged question as checked.
- **Whole class:** `grade_batch` runs `read_sheet` + `grade_sheet` for each
  student (a failing sheet doesn't stop the rest). `core/analytics.py`
  computes averages and most-missed criteria, and one call per question lists
  common mistakes.

---

## Part 3. Every file

### Top level

| File | What and why |
|---|---|
| `app.py` | The web app: three steps (Exam, Grade, Check) plus Settings. One function per button (`on_draft_rubric`, `on_grade_one`, `on_grade_class`, `on_check_save`, ...). `_stream` runs the pipeline in a background thread and yields agent events as they arrive, so the activity log updates live; `_check_view` draws the Check step for one sheet (a tick box per flagged answer, the flagged answers, a table of every mark) and `_next_sheet` finds the next sheet with something to check. `build_app` lays out the steps. Handlers change the session in place (two events can run at once); a running job locks the rubric and grading buttons (`_lock`); the rubric and settings are kept in the browser (`gr.BrowserState`) and restored by `on_load` after a refresh. Session data (rubric, graded sheets) lives in a `gr.State` per browser tab. |
| `cli.py` | Same pipeline in the terminal. `--set key=value` changes any setting, `--batch` grades a class, `--save-rubric` / `--rubric-json` let you edit a rubric between runs. |
| `config.yaml` | Everything you might change without code: model profiles (`mock`, `local`, `ollama_cloud`, `ollama_api`, `openrouter`, `openrouter_lite`), default settings, thresholds, ingest limits, Bodhan endpoints, public-mode limits, prices. |
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` | Dependencies; ruff and pytest settings. |
| `.env.example` | Template for secrets. `.env` is git-ignored. |

### `core/`: plumbing, no grading decisions

| File | What and why |
|---|---|
| `config.py` | Loads config and `.env`. `model_for(role)` answers "which model does this job?": a profile can name a model per role; otherwise image jobs (`ocr`, `recheck`, `grader_vision`, `locator`) use `vision` and the rest use `text`. |
| `settings.py` | `RunSettings`: all switches in one validated dataclass. The app, CLI and ablation runner all build one, so an ablation is "same data, one setting changed". |
| `llm.py` | The only file that calls models. Ollama (local app or the cloud API directly) and OpenRouter all speak the OpenAI API, so one client serves all. Also: retries with back-off (rate limits wait 15–45 s or what the server asks; an empty reply is retried), readable errors, a disk cache keyed on the exact prompt, per-run usage and cost per role (`Usage`, kept in a context variable so parallel users don't mix), a reply cut off at the token limit is asked again with twice the limit (and never cached), and `chat_json` asks once more for complete JSON when a reply can't be read. |
| `mock.py` | A fake model with fixed rules. Lets the app, tests and CI run without keys. Its marks mean nothing. |
| `ingest.py` | Files → `Document` / `Page`. PDF pages with little text or mostly images go to OCR; typed pages keep a preview image and line positions (for exact annotation). Phone photos are auto-rotated; images in .docx are extracted. `looks_blank` / `ink_share` decide if a scanned page is empty. |
| `utils.py` | `parse_json` (finds JSON inside chatty replies; keeps the first object if the model repeats it; mends a complete reply with `json_repair` (e.g. an unescaped quote) but never a cut-off one; repairs LaTeX backslashes and bad escapes), `evidence_supported` (fuzzy quote matching, stitched quotes), `remove_struck`, `is_blank`, `is_page_number`, `coverage`, `normalize_qid` (Q1 = q 1 = प्रश्न १), `find_references` ("see Q3" in an answer), `infer_dependencies` ("Hence…" in a question). |
| `guard.py` | The injection detector: regex patterns per language, sentence splitting (also on `।`), `strip`, and `overlaps` (is a quote taken from flagged text?). |
| `lang.py` | Script and language detection, Indian digits → ASCII, language codes. |
| `bodhan.py` | Bodhan OCR (`indic-ocr` via `/v1/chat/completions`) and translation (`/translate`), per console.bodhan.ai/api-docs. Used when `BODHAN_API_KEY` is set; `placeholder: true` in config.yaml switches it off. |
| `trace.py` | `emit(stage, message)` for the live activity log. The sink is a context variable so it follows the run into worker threads. |
| `annotate.py` | Draws red ticks, crosses, circled marks and notes in a widened margin; handwriting fonts per script. `_place` picks each mark's page and height: the locator's position where it agrees with the segmenter or with where the answer's first line is in the page transcription, else the transcription, else the locator, else the segmenter's page. `_fit` keeps every mark on the page: near its answer, below the one before; on a crowded page the notes are dropped and the marks drawn smaller. |
| `review.py` | Accept or override a question, accept the whole-sheet notes (`accept_sheet`, e.g. text matched to no question), with an audit log. |
| `analytics.py` | Class statistics (no AI). |
| `report.py` | Markdown and JSON reports, annotated PDF, batch CSV and analytics, the usage line. |
| `batch_io.py` | Uploads/zips → students; safe zip extraction (no `../` paths, no links, size limits). |

### `agents/`: one file per agent, plus the orchestrator

| File | Agent | Model role |
|---|---|---|
| `prompts.py` | Every prompt, in one place, so they can be tuned without touching logic. | n/a |
| `ocr.py` | OCR agent | `ocr` (vision) or Bodhan |
| `rubric.py` | Rubric agent + `validate_rubric` | `rubric` |
| `segmenter.py` | Segmenter | `segmenter` |
| `grader.py` | Grader + `normalize` (the rule checks) | `grader` or `grader_vision` |
| `aggregator.py` | Combines samples (code) | none |
| `critic.py` | Critic | `critic` |
| `recheck.py` | OCR re-check | `recheck` (vision) |
| `tool_agent.py` | Tool agent (ReAct) | `tools` |
| `locator.py` | Locator | `locator` (vision) |
| `pipeline.py` | The orchestrator: order, conditions, flags | n/a |

### `ui/`

| File | What and why |
|---|---|
| `theme.py`, `style.css` | The look: white sheets on a graph-paper desk with faint pencil working (drawn at start-up in `theme.pencil_notes`), a red double margin rule, blue-black ink, red pen for marks, an amber stamp for "check"; opens in light mode; dark mode, focus outlines, small screens. |
| `render.py` | HTML for results, activity log, class view. Everything from documents or models is escaped, so student text can't inject HTML. |
| `rubric_table.py` | Rubric ⇄ table rows. |
| `how_it_works.md` | Text for the "How it works" tab. |

### `tools/`, `eval/`, `scripts/`, `tests/`

| File | What and why |
|---|---|
| `tools/math_check.py` | sympy calculator with an allow-list of characters (sympy evaluates text, so input is restricted). |
| `eval/run_ablation.py` | Grades a dataset under several setting variants and compares with human marks. `--questions`, `--sheets` (names or `@split_file.txt`), `--exclude` and `--limit` choose what to grade; `--groups` adds per-group rows (e.g. MCQ vs short). OCR and rubrics are reused across variants. |
| `eval/offline.py` | Recomputes sample count, aggregation and evidence policy from saved raw samples: free ablations. |
| `eval/compare.py` | Each variant vs a baseline with 95% bootstrap confidence intervals (over students), plus per-question MAE. Free. |
| `eval/metrics.py` | MAE, signed bias, QWK, within-1, review rate, flag recall/precision. |
| `eval/variants*.yaml` | The ablation list. |
| `scripts/check_setup.py` | One text and one image call to test a profile. |
| `scripts/list_openrouter_models.py` | Model ids, prices, your credit. |
| `scripts/make_samples.py` | Builds the deep-learning quiz sample (`samples/dl_quiz`). |
| `scripts/make_mendeley_samples.py` | Builds the real-exam sample (`samples/mendeley`) from the imported Mendeley dataset. |
| `scripts/make_redteam.py` | Adds attacks (direct, override, fake rubric, plea, many-shot, subtle, Hindi) to sheets; on scanned PDFs they are written in a handwriting font in the gaps between answers. |
| `scripts/make_degraded.py` | Damaged copies of scanned sheets: blur, low resolution, skew, noise, faded. |
| `scripts/import_mendeley.py` | Converts the Mendeley handwritten dataset to our layout and writes the fixed splits (`split_*.txt`). |
| `scripts/make_calibration.py` | Turns a few teacher-marked sheets into graded examples in `rubric.json`. |
| `scripts/import_saf.py`, `scripts/import_csv.py` | Typed-answer datasets (SAF; any CSV such as JorGPT) for the grader alone. |
| `scripts/list_ollama_models.py` | Model names your Ollama API key can use. |
| `scripts/deploy_space.py` | Publishes to Hugging Face with keys as secrets. |
| `tests/` | 84 tests on the fake model: helpers, detector (many languages), grading logic, rubric table, review, end-to-end runs, zip safety. |

---

## Part 4. Why the names

| Name | Meaning, and why this word |
|---|---|
| **Agent** | A component that calls a model and makes a judgement. Everything else is a "check" or plumbing. |
| **Orchestrator** (`pipeline.py`) | Decides the order and when optional agents run, as in the multi-agent orchestration paper. Ours is a fixed order with conditional steps, so every run is auditable. |
| **Ingest** | Standard data-pipeline term for bringing files in and normalising them. |
| **OCR agent** | Optical character recognition; "agent" because it's a model following rules, not a classic OCR engine. |
| **Rubric agent**, `as_is` / `expand` | It writes the rubric. `as_is` keeps your marking points exactly; `expand` splits them. |
| **Segmenter** | It *segments* the sheet into per-question answers (text segmentation). It is also the planner in Plan-and-Act terms. |
| **Grounding check** | NLP term: an output is *grounded* if it is supported by the source. Here: the segmented answer must be on the sheet. |
| **Presence check / blank check** | From the engineering-exam paper's "presence check": is there an answer at all? |
| **Guard / injection detector** | From "guardrails" and the injection paper's "guardian model"; it guards the grader against prompt injection. |
| **Grader**, **sample** | One grading attempt is a *sample* (as in sampling from the model); N samples = test-time compute. |
| **Evidence check** | Each mark must cite evidence: an exact quote. |
| **Aggregator** | Aggregates samples into one mark. |
| **Strictness percentile** | Strictness picks which percentile of the samples becomes the mark. |
| **Confidence** | 1 − (spread of sample totals ÷ max marks). |
| **Critic** | From Reflexion's self-reflection and actor-critic: it criticises the graders, it doesn't grade. |
| **OCR re-check** | It checks the OCR again, only where marks were lost. |
| **Locator** | It locates answers on the page (a visual grounding task, like locating UI elements in computer-use agents). |
| **Tool agent** | It uses tools (code, calculator), ReAct style. |
| **Profile / role** | A *profile* is a set of models (mock, local, ollama_cloud, openrouter); a *role* is a job inside the pipeline. |
| **Trace** | The event log of a run (as in tracing in software). |
| **Mock** | Standard testing term for a fake dependency. |
| **Flag / needs review** | Marked for a human. |

---

## Part 5. Papers: what we took, where, and what we didn't

### Course papers

| Paper | Taken | Where |
|---|---|---|
| Scaling LLM Test-Time Compute (Snell et al.) | Several independent samples + aggregation; checks acting as verifiers | `pipeline._grade_core`, `aggregator.py`, `grader.normalize` |
| ReAct | Reason → act with a tool → observe loop | `tool_agent.py` |
| Reflexion | Verbal critique, then retry with it | `critic.py` + re-grade in `_grade_core` |
| ReTool | Use tools to verify answers (inference only; we don't train). We kept only the calculator: running student code needs a real sandbox, which we don't have | `tool_agent.py`, `tools/math_check.py` |
| ToolACE | Strict, validated structured calls | JSON schemas in every prompt, `chat_json` repair, `validate_rubric`, `normalize` |
| Tool-call reward model | Judge each call, not only the final answer | in spirit: every agent output is checked by code before use (not a learned reward model) |
| Plan-and-Act | Separate planning (segmenter) from execution (graders) | `segmenter.py`, `pipeline.grade_sheet` |
| Multi-agent evolving orchestration | Specialised agents under an orchestrator; agents called only when needed (critic on disagreement, re-check on deductions) | `pipeline.py` |
| ACON / MEM1 | Keep each agent's context small; pass compact state | one question per grader context; only that question's pages |
| DR Tulu (evolving rubrics) | Turn vague rubric items into concrete checks, but **once and frozen**, approved by a human | `rubric.py` `expand` mode |
| Qwen-CUA / CUA-GYM | Visual grounding on a screen ↔ locating answers on a page | `locator.py` (loose analogy) |
| ToolRL, AgentGym-RL, MEM1's RL, A-Mem, Memp, TMAX, Tongyi DeepResearch | Not used: they need training or long-horizon environments | Future work: RL-tune a grader on TA overrides (ToolRL-style rewards from the review log); procedural memory of how a course grades (Memp) |

### Grading papers (found for this project)

| Paper | Finding | What we did |
|---|---|---|
| EngExams, arXiv 2601.00730 | Presence check, 3 graders, templates + code validation, spread-based review; no reference → +6.4 bias | Blank check, N samples, `normalize`, review flags; warning when there's no reference |
| EDU-CIRCUIT-HW, arXiv 2602.00095 | 38% of transcriptions wrong; symbol errors change grades; re-check only deducted answers | Grader sees the image; `recheck.py` with the same routing |
| HITL-Maths, arXiv 2603.13083 | 5 passes; max biases upward; rubric most sensitive; human verifies all | Percentile not max; editable rubric; review tab |
| GPT4-Hand, arXiv 2411.05231 | No reference over-grades; key only under-grades; key + rubric best | Both inputs supported; ablation A. Our own finding matched "key only under-grades": the grader gave fewer marks than the teacher. A judgement-based prompt with half marks did not fix it (ablation I); more reference, teacher examples and lenient strictness helped |
| GLaH, arXiv 2405.19694 | Rubric refined from graded samples helps; fine rubrics can hurt; one-shot examples help | `as_is`/`expand` switch; `examples` in the rubric |
| Inject, arXiv 2606.03090 | 56.9% score inflation; prompt rules alone not enough; rule before answer better; detector much better | `guard.py`, rule placement, flags |

---

## Part 6. Settings

All settings, their options, defaults, cost and whether the app shows them:
**`docs/SETTINGS_AND_ABLATIONS.md`**, section 1. The short version of why the
main defaults are what they are:

| Setting | Default | Why |
|---|---|---|
| `n_samples` | 3 | Smallest N with a meaningful spread (EngExams used 3). Ablations use 1 to save cost |
| `aggregation` | strictness (median for standard) | Robust to one odd sample; max biases up (HITL-Maths) |
| `grader_input` | auto (image + text for scans) | Seeing the page beat text-only in our ablation C; EDU-CIRCUIT |
| `rubric_mode` | auto (as_is if you gave a rubric) | Expanding was slightly worse in ablation B (GLaH) |
| `evidence_check` | reject | Stops marks for text that is not on the sheet; costs ~0.005 MAE on honest sheets |
| `guard_prompt` / `guard_detector` | on / strip | Both layers (Inject) |
| `use_reflection` | on, conditional | Only fires when passes disagree or a quote failed, so it costs little |
| `ocr_recheck` | off | No measurable gain in our ablation; one extra call per deducted answer. On for messy handwriting |
| `cross_references` | on | Linked questions; no effect when there are none |
| `attempt_credit` | off | A course policy (marks for wrong but on-topic attempts), a teacher's choice |

---

## Part 7. Data structures

- **Document / Page** (`core/ingest.py`): `Page(number, text, image, needs_ocr, source, ocr_engine, lines)`.
- **Rubric** (JSON): `{"questions": [{"id", "text", "max_marks", "model_answer", "criteria": [{"id", "description", "marks", "partial_allowed"}], "notes", "examples": [{"answer", "marks", "comment"}], "depends_on": ["Q2"]}]}`
- **Grader reply** (JSON from the model): `{"criteria": [{"criterion_id", "verdict", "marks_awarded", "evidence", "reason"}], "feedback", "addressed_to_grader", "addressed_text"}` (+ `"rubric_note"` in judgement style)
- **Question result** (`_result` in pipeline.py): marks, criteria with evidence, feedback, confidence, sample totals, review reasons, info, language, injection findings, re-check details, raw samples.
- **Report** (`grade_sheet`): meta (settings, models, OCR engines, language), total, questions, overall feedback, flags, annotation positions, usage, rubric.

---

## Part 8. Questions you will probably be asked

**Why not fine-tune a model?** No graded data at the needed scale and one
week. Every course idea we use works at inference time; the review log
(TA overrides) is exactly the data a later RL or fine-tuning step would need.

**How do you stop the model from inventing marks?** Every mark must quote the
answer; a rule check confirms the quote exists (fuzzy, to tolerate OCR noise) and
removes the mark if not. The segmented answer must also exist on the sheet.

**What if the OCR misreads the handwriting?** Three layers: the grader sees
the page image as well as the text; where marks were lost, a re-check
compares text with handwriting and re-grades if a misreading changed the
result; unreadable text is marked `[illegible]` and flagged, never guessed.

**Why several samples, and why the median?** One sample is noisy; three give
a spread (= confidence) and a robust middle. The max inflates marks
(HITL-Maths); strictness moves the percentile.

**How do you handle "please give me full marks"?** Rules remove it before
grading (in 10+ languages), the prompt says such text earns nothing, the
grader must report it, quotes from it can't earn marks, and the question goes
to a human. Prompt rules alone are not enough (Inject paper).

**What if the answer is in Hindi and the rubric in English?** The OCR keeps
Devanagari; the grader is told to judge content, not language; quotes stay in
Devanagari; feedback comes back in Hindi (or via Bodhan translation).

**Why freeze the rubric?** If it evolved while grading, a run of wrong answers
early on could shift it. A human approves it once instead.

**Why is the critic conditional?** Re-grading every answer costs calls and
always-on self-reflection sometimes made grading worse (GLaH). We only call
it when the samples disagree or a quote failed.

**What does the human still do?** Approve the rubric, check flagged questions,
and own the final marks. Overrides are logged.

**How do you know it works?** `eval/run_ablation.py` against a teacher's
marks on 50 real handwritten sheets (Mendeley), with fixed splits so settings
are chosen on some students and the final system is tested once on others;
`eval/compare.py` gives confidence intervals. Plus injected and damaged copies
of the same sheets for robustness, and a typed dataset (SAF) with questions
we never looked at.

**Why are your marks lower than the teacher's?** We read the reports: the
OCR and segmentation were right; the gap was marking policy. This teacher
gives a mark for most attempts and full marks for half-right answers, while
our grader reads the key more strictly. A prompt that told the grader to
use its own judgement with half marks did not close the gap (ablation I);
teacher examples (calibration) and a lenient policy did. Giving marks to
wrong attempts is left as an explicit course policy (`attempt_credit`), not
hidden tuning.

**Isn't calibration overfitting to one exam?** The code and prompts are the
same for every paper; the examples are data the teacher supplies, like the
rubric. The students used as examples are never used for testing. Without
examples the system still works (that is the `n1` row).

**What does it cost?** Shown on every report. Roughly ₹5–15 per 3-page sheet
with Gemini 2.5 Flash at default settings; re-runs are free from the cache.

**Biggest limitations?** Handwriting recognition on messy pages, diagrams,
the rule-based detector missing new phrasings (covered by the prompt and the
grader's report), and thresholds tuned on little data.
