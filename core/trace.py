"""A live log of what the agents are doing, for the app's "agent activity"
panel and for debugging. The pipeline calls emit(); the caller of the
pipeline can install a sink with set_sink() to receive the events.

The sink lives in a context variable, so it follows the run into worker
threads started with core.llm.submit(), and two users of the web app never
see each other's events."""

import contextvars
import time
from collections.abc import Callable
from typing import Any

_sink: contextvars.ContextVar = contextvars.ContextVar("autograder_trace_sink", default=None)

# stage -> short label used by the UI
STAGES = {
    "ingest": "Reading files", "ocr": "OCR", "rubric": "Rubric agent", "segment": "Segmenter",
    "guard": "Injection guard", "grade": "Grader", "critic": "Critic", "recheck": "OCR re-check",
    "tools": "Tool agent", "locate": "Locator", "summary": "Feedback", "done": "Done", "error": "Error",
}


def set_sink(fn: Callable[[dict], None] | None) -> None:
    """Install the function that receives the events of this run.

    Args:
        fn (Callable[[dict], None] | None): Called with each event dict
            ("t", "stage", "label", "msg", "level" and any extra data); None stops events.
    """
    _sink.set(fn)


def emit(stage: str, message: str, level: str = "info", **data: Any) -> None:
    """Send one event to the run's sink, if there is one.

    Errors raised by the sink are ignored so they cannot stop grading.

    Args:
        stage (str): Pipeline stage, a key of STAGES (e.g. "ocr", "grade").
        message (str): Text shown in the activity panel.
        level (str): "info", "good", "warn" or "bad".
        **data (Any): Extra fields added to the event.
    """
    fn = _sink.get()
    if fn is None:
        return
    try:
        fn({"t": time.time(), "stage": stage, "label": STAGES.get(stage, stage), "msg": message,
            "level": level, **data})
    except Exception:  # a broken sink must never break grading
        pass
