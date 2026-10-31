"""Language helpers (no AI): which script a text is written in, Indic digits,
and language names and codes used by the prompts and the Bodhan adapter.

Script detection is by Unicode block, which is exact and free. Devanagari is
reported as Hindi unless a language hint is set (the script is also used for
Marathi, Nepali and Sanskrit)."""

# Unicode blocks of the main Indian scripts (Arabic for Urdu)
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
}
LANGUAGE_OF_SCRIPT = {
    "Devanagari": "Hindi", "Bengali": "Bengali", "Gurmukhi": "Punjabi", "Gujarati": "Gujarati",
    "Odia": "Odia", "Tamil": "Tamil", "Telugu": "Telugu", "Kannada": "Kannada",
    "Malayalam": "Malayalam", "Arabic": "Urdu", "Latin": "English",
}
# ISO 639-1 codes (what most Indic APIs, including Bhashini-style ones, expect)
LANGUAGE_CODES = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Nepali": "ne", "Sanskrit": "sa",
    "Bengali": "bn", "Assamese": "as", "Punjabi": "pa", "Gujarati": "gu", "Odia": "or",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Urdu": "ur",
}
LANGUAGES = list(LANGUAGE_CODES)

_INDIC_DIGIT_ZEROS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x06F0, 0x0660]
_DIGIT_MAP = {z + i: str(i) for z in _INDIC_DIGIT_ZEROS for i in range(10)}


def to_ascii_digits(text: str) -> str:
    """Replace Indic and Arabic-Indic digits with ASCII digits.

    'प्रश्न १' -> 'प्रश्न 1'. Works for all the scripts above.

    Args:
        text (str): Any text; None is treated as empty.

    Returns:
        str: The text with ASCII digits.
    """
    return (text or "").translate(_DIGIT_MAP)


def script_counts(text: str) -> dict[str, int]:
    """Count the letters of each script in a text.

    Args:
        text (str): Any text; None is treated as empty.

    Returns:
        dict[str, int]: Script name ("Latin", "Devanagari", ...) -> letter count.
    """
    counts: dict[str, int] = {}
    for ch in text or "":
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["Latin"] = counts.get("Latin", 0) + 1
            continue
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


def detect_language(text: str, min_share: float = 0.2) -> str:
    """Guess the main language of a text from its script.

    An Indic script that makes up at least `min_share` of the letters wins
    over English, because answers written in Hindi or Tamil usually contain
    English technical terms.

    Args:
        text (str): Any text.
        min_share (float): Share of letters an Indic script needs to win.

    Returns:
        str: Language name, e.g. "Hindi"; "English" if no Indic script wins.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return "English"
    indic = {s: c for s, c in counts.items() if s != "Latin"}
    if indic:
        script, n = max(indic.items(), key=lambda kv: kv[1])
        if n / total >= min_share:
            return LANGUAGE_OF_SCRIPT[script]
    return "English"


def is_indic(language: str) -> bool:
    """Check whether a language is one of the known Indian languages.

    Args:
        language (str): Language name.

    Returns:
        bool: True for a known language other than English.
    """
    return language in LANGUAGE_CODES and language != "English"


def code_for(language: str) -> str:
    """Get the ISO 639-1 code of a language.

    Args:
        language (str): Language name.

    Returns:
        str: Code such as "hi"; "en" if the language is unknown.
    """
    return LANGUAGE_CODES.get(language, "en")


def resolve_feedback_language(setting: str, detected: str) -> str:
    """Pick the language for feedback.

    Args:
        setting (str): The feedback_language setting; "auto" means the student's language.
        detected (str): Language detected on the answer sheet.

    Returns:
        str: Language name to write feedback in.
    """
    return detected if setting == "auto" else setting


def has_script(text: str, script: str) -> bool:
    """Check whether a text has any character of a script.

    Args:
        text (str): Any text; None is treated as empty.
        script (str): A key of SCRIPT_RANGES, e.g. "Tamil".

    Returns:
        bool: True if at least one character is in the script's Unicode block.
    """
    lo, hi = SCRIPT_RANGES[script]
    return any(lo <= ord(c) <= hi for c in text or "")
