"""RunSettings: every switch that changes how a sheet is graded.

Defaults come from the `defaults:` block of config.yaml. The app, cli.py and
the ablation runner all build one of these, so an ablation is the same data
with one setting changed."""

from dataclasses import asdict, dataclass, fields
from typing import Any

from .config import load_config
from .lang import LANGUAGES

# allowed values of the text settings, checked by RunSettings.validate()
CHOICES = {
    "strictness": ["lenient", "standard", "strict"],
    "aggregation": ["strictness", "mean", "median", "max", "min"],
    "rubric_mode": ["auto", "as_is", "expand"],
    "grader_input": ["auto", "text", "image", "both"],
    "evidence_check": ["reject", "flag", "off"],
    "guard_detector": ["off", "flag", "strip"],
    "language_hint": ["auto"] + LANGUAGES,
    "ocr_backend": ["auto", "vlm", "bodhan"],
    "feedback_language": ["auto"] + LANGUAGES,
    "feedback_translator": ["llm", "bodhan"],
    "grading_style": ["rubric", "judgement"],
}


@dataclass
class RunSettings:
    """Settings for one grading run. The comments in config.yaml list the options.

    Attributes:
        profile (str | None): Config profile; None uses the active one.
        strictness (str): Grading policy: "lenient", "standard" or "strict".
        n_samples (int): Independent grading samples per question.
        aggregation (str): How samples are combined: "strictness" (percentile), "mean",
            "median", "max" or "min".
        temperature (float): Sampling temperature, used when n_samples > 1.
        seed (int): Changes the sample cache keys, to get fresh samples on a repeat run.
        rubric_mode (str): "as_is", "expand", or "auto" (as_is if a rubric is given).
        use_examples (bool): Use staff-graded example answers stored in the rubric.
        grader_input (str): What the grader sees: "auto", "text", "image" or "both".
        evidence_check (str): What happens when a quote is not found in the answer:
            "reject", "flag" or "off".
        guard_prompt (bool): Add anti-injection rules to the prompts.
        guard_detector (str): Rule-based injection detector: "off", "flag" or "strip".
        presence_check (bool): Give blank answers 0 without a model call.
        use_reflection (bool): Run the critic and one re-grade when samples disagree.
        ocr_recheck (bool): Re-read the handwriting where marks were lost.
        use_tools (bool): Let the tool agent check arithmetic with a calculator.
        cross_references (bool): Show the grader the answers a question refers to.
        attempt_credit (float): Share of max marks a genuine attempt earns at least (0-1).
        grading_style (str): "rubric" (follow the criteria) or "judgement" (criteria as a guide).
        force_ocr (bool): OCR the answer sheet even if the PDF has a text layer.
        class_summary (bool): In batch mode, list common mistakes per question.
        language_hint (str): "auto" or a language name; helps OCR and picks the Bodhan language.
        ocr_backend (str): "auto", "vlm" or "bodhan".
        feedback_language (str): "auto" (the student's language) or a language name.
        feedback_translator (str): "llm" (the grader writes it) or "bodhan" (Bodhan translates).
        annotate (bool): Draw red-ink marks on the scanned pages.
        annotate_locate (bool): One vision call per page to place the marks next to each answer.
    """

    profile: str | None = None
    strictness: str = "standard"
    n_samples: int = 3
    aggregation: str = "strictness"
    temperature: float = 0.7
    seed: int = 0
    rubric_mode: str = "auto"
    use_examples: bool = True
    grader_input: str = "auto"
    evidence_check: str = "reject"
    guard_prompt: bool = True
    guard_detector: str = "strip"
    presence_check: bool = True
    use_reflection: bool = True
    ocr_recheck: bool = False
    use_tools: bool = False
    cross_references: bool = True
    attempt_credit: float = 0.0
    grading_style: str = "rubric"
    force_ocr: bool = False
    class_summary: bool = True
    language_hint: str = "auto"
    ocr_backend: str = "auto"
    feedback_language: str = "auto"
    feedback_translator: str = "llm"
    annotate: bool = True
    annotate_locate: bool = True

    @classmethod
    def from_config(cls, **overrides: Any) -> "RunSettings":
        """Build settings from the config.yaml defaults and the given overrides.

        Args:
            **overrides (Any): Setting name -> value; a None value keeps the default.

        Returns:
            RunSettings: Validated settings.

        Raises:
            ValueError: If a setting name is unknown or a value is not allowed.
        """
        values = dict(load_config().get("defaults") or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown setting(s): {unknown}. Valid: {sorted(known)}")
        s = cls(**values)
        s.validate()
        return s

    def validate(self) -> None:
        """Check the text settings against CHOICES and coerce the numbers.

        n_samples becomes at least 1 and attempt_credit is clamped to [0, 1].

        Raises:
            ValueError: If a text setting has a value not in CHOICES.
        """
        for key, options in CHOICES.items():
            if getattr(self, key) not in options:
                raise ValueError(f"{key} must be one of {options}, got {getattr(self, key)!r}")
        self.n_samples = max(1, int(self.n_samples))
        self.seed = int(self.seed)
        self.temperature = float(self.temperature)
        self.attempt_credit = min(1.0, max(0.0, float(self.attempt_credit)))

    def to_dict(self) -> dict:
        """Convert the settings for the JSON report.

        Returns:
            dict: Setting name -> value.
        """
        return asdict(self)
