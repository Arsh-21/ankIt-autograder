"""Rule-based detector for text in an answer that is aimed at the grader.

Catches "give me full marks", "ignore the rubric", "dear sir please be
lenient", fake marking schemes and fake graded examples, in English,
romanised Hindi ("sir pure marks de do") and the main Indian scripts.

Design note: rules are used in addition to a prompt warning. A study of
prompt injection on LLM graders (arXiv 2606.03090) found that prompt warnings
alone left >30% of attacks working on most models, while a separate
detector cut them much further. A detector cannot be talked out
of its job by the text it is checking.

Text is checked sentence by sentence (split on . ! ? । ॥ and new lines).
`flag` only reports a sentence; `strip` also removes it before grading.
Either way the question goes to human review and the report shows what was
found, so a false positive costs a few seconds of checking, never silent marks.
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .utils import normalize_ws

_I = re.I

# ---------------------------------------------------------------- English
_EN = [
    ("asks for marks", r"\b(give|award|grant|assign|allot)\s+(me\s+|us\s+|this\s+(answer\s+)?|him\s+|her\s+)?"
                       r"(full|maximum|max|all|more|extra|bonus|total|good)\s+(marks?|points?|credit|score|grade)"),
    ("asks for marks", r"\bfull\s+(marks?|credit|points?|score)\b"),
    ("override instruction", r"\b(ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,40}\b(rubric|instructions?|"
                             r"criteria|marking|scheme|prompt|rules?|above|previous|reference)\b"),
    ("addresses the grader", r"\b(dear|respected|hello|hi|note\s+to(\s+the)?)\s+(sir|ma'?am|madam|professor|prof|ta|"
                             r"teacher|grader|evaluator|examiner|checker|ai|model)\b"),
    ("addresses the grader", r"(^|\n|\.\s)\s*(grader|evaluator|examiner|checker|ai\s+grader|assistant)\s*[:,]"),
    ("pleading", r"\bplease\s+(be\s+lenient|give|award|grant|pass\s+me|don'?t\s+cut|do\s+not\s+cut|consider\s+my)\b"),
    ("pleading", r"\b(i\s+(studied|worked|prepared)\s+(very\s+|really\s+)?hard|my\s+(scholarship|cgpa|gpa|"
                 r"placement|future)\s+depends)\b"),
    ("fake rubric", r"\b(new|updated|revised|correct|official|actual)\s+(rubric|marking\s+scheme|answer\s+key|"
                    r"model\s+answer|grading\s+(rules|instructions|criteria))\b"),
    ("fake graded example", r"\b(marks?|grade|awarded)\s*[:=]\s*\d+(\.\d+)?\s*/\s*\d+"),
    ("system override", r"\bsystem\s*(prompt|message|override)\b|\bdeveloper\s+mode\b|\bjailbreak\b|\byou\s+are\s+now\b"),
    ("claims correctness", r"\bthis\s+(answer|solution|response)\s+(is|has\s+been)\s+(fully\s+|completely\s+|100%\s+)?"
                           r"(correct|verified|approved|complete)\b[^.\n]{0,30}\b(marks?|credit|grader|rubric|reference)"),
]

# ---------------------------------------------------------------- romanised Hindi (Hinglish)
_HINGLISH = [
    ("asks for marks", r"\b(pure|poore|pura|poora|full|zyada|jyada|achhe|ache)\s+(marks?|number|nambar|ank)\b"),
    ("asks for marks", r"\b(marks?|number|nambar|ank)\s+(de\s*(do|dijiye|dena|dijie)|dedo|dijiye)\b"),
    ("pleading", r"\b(sir|mam|madam)\s+(ji\s+)?(please|plz|pls|plis)\b|\bsir\s+ji\b"),
    ("pleading", r"\b(pass\s+kar\s*(do|dijiye|dena)|maine\s+bahut\s+mehnat)\b"),
    ("override instruction", r"\b(rubric|instructions?|niyam)\s+(ko\s+)?(ignore|nazarandaz|anadekha)\b"),
]

# ---------------------------------------------------------------- Indian scripts
# Phrases for "full marks", "give marks", "please", "sir", "ignore" in each script.
_INDIC = [
    # Hindi / Marathi (Devanagari)
    ("asks for marks", r"(पूरे|पूरा|पूर्ण|अधिकतम|ज़्यादा|ज्यादा|अच्छे)\s*(अंक|नंबर|मार्क्स|गुण)"),
    ("asks for marks", r"(अंक|नंबर|मार्क्स|गुण)\s*(दे\s*दो|दें|दीजिए|दीजिये|दो|द्या|द्यावे)"),
    ("pleading", r"(कृपया|प्लीज़|प्लीज|कृपा\s*करके).{0,30}(अंक|नंबर|मार्क्स|गुण|पास)"),
    ("addresses the grader", r"(सर\s*जी|आदरणीय\s*(सर|महोदय|शिक्षक|परीक्षक)|प्रिय\s*(सर|शिक्षक|परीक्षक)|परीक्षक\s*महोदय|सर\s*कृपया)"),
    ("pleading", r"(मैंने\s*बहुत\s*मेहनत|मेरी\s*छात्रवृत्ति|मी\s*खूप\s*अभ्यास)"),
    ("override instruction", r"(रूब्रिक|निर्देशों|नियमों|उत्तर\s*कुंजी).{0,20}(अनदेखा|नज़रअंदाज़|नजरअंदाज|ध्यान\s*न)"),
    # Tamil
    ("asks for marks", r"(முழு|அதிக)\s*மதிப்பெண்"),
    ("asks for marks", r"மதிப்பெண்(கள்)?\s*(கொடு|வழங்க|தாருங்கள்|போடுங்கள்)"),
    ("pleading", r"தயவு\s*செய்து.{0,30}(மதிப்பெண்|தேர்ச்சி)"),
    ("addresses the grader", r"(ஐயா|மதிப்பிற்குரிய\s*(ஆசிரியர்|தேர்வாளர்))"),
    ("override instruction", r"(விதிகளை|அறிவுறுத்தல்களை|மதிப்பீட்டு\s*முறையை).{0,20}புறக்கணி"),
    # Telugu
    ("asks for marks", r"(పూర్తి|ఎక్కువ)\s*మార్కులు|మార్కులు\s*(ఇవ్వండి|వేయండి)"),
    ("pleading", r"దయచేసి.{0,30}(మార్కులు|పాస్)|సార్\s*దయచేసి"),
    # Bengali
    ("asks for marks", r"(পূর্ণ|পুরো|বেশি)\s*নম্বর|নম্বর\s*(দিন|দেবেন|দাও)"),
    ("pleading", r"(দয়া\s*করে|প্লিজ).{0,30}(নম্বর|পাস)|স্যার\s*(দয়া|প্লিজ)"),
    # Kannada
    ("asks for marks", r"(ಪೂರ್ಣ|ಹೆಚ್ಚು)\s*ಅಂಕ|ಅಂಕ(ಗಳನ್ನು)?\s*(ಕೊಡಿ|ನೀಡಿ)"),
    ("pleading", r"ದಯವಿಟ್ಟು.{0,30}(ಅಂಕ|ಪಾಸ್)"),
    # Malayalam
    ("asks for marks", r"(മുഴുവൻ|കൂടുതൽ)\s*മാർക്ക്|മാർക്ക്\s*(തരണം|തരൂ|നൽകണം)"),
    ("pleading", r"ദയവായി.{0,30}(മാർക്ക്|പാസ്)"),
    # Gujarati
    ("asks for marks", r"(પૂરા|પૂર્ણ|વધુ)\s*(માર્ક્સ|ગુણ)|(માર્ક્સ|ગુણ)\s*(આપો|આપજો)"),
    ("pleading", r"કૃપા\s*કરીને.{0,30}(માર્ક્સ|ગુણ|પાસ)"),
    # Punjabi
    ("asks for marks", r"(ਪੂਰੇ|ਵੱਧ)\s*(ਅੰਕ|ਨੰਬਰ)|(ਅੰਕ|ਨੰਬਰ)\s*ਦੇ\s*ਦਿਓ"),
    ("pleading", r"ਕਿਰਪਾ\s*ਕਰਕੇ.{0,30}(ਅੰਕ|ਨੰਬਰ|ਪਾਸ)|ਸਰ\s*ਜੀ"),
    # Urdu
    ("asks for marks", r"(پورے|زیادہ)\s*(نمبر|مارکس)|(نمبر|مارکس)\s*(دے\s*دیں|دیجیے)"),
    ("pleading", r"براہ\s*کرم.{0,30}(نمبر|پاس)"),
]

PATTERNS = [(kind, re.compile(p, _I)) for kind, p in _EN + _HINGLISH + _INDIC]

_SENTENCE_RE = re.compile(r"[^.!?।॥؟\n]+[.!?।॥؟]?")


@dataclass
class Finding:
    """One flagged sentence.

    Attributes:
        kind (str): Pattern group, e.g. "asks for marks" or "pleading".
        text (str): The sentence, stripped.
        start (int): Start offset of the sentence in the scanned text.
        end (int): End offset of the sentence in the scanned text.
    """

    kind: str
    text: str
    start: int
    end: int


def scan(text: str) -> list[Finding]:
    """Find every sentence that matches a pattern.

    Args:
        text (str): The student's answer.

    Returns:
        list[Finding]: One finding per matching sentence, in order.
    """
    findings = []
    for m in _SENTENCE_RE.finditer(text or ""):
        sentence = m.group(0)
        if not sentence.strip():
            continue
        for kind, pat in PATTERNS:
            # also test with a leading newline so "^ Grader:" patterns work
            if pat.search(sentence) or pat.search("\n" + sentence.lstrip()):
                findings.append(Finding(kind, sentence.strip(), m.start(), m.end()))
                break
    return findings


def strip(text: str, findings: list[Finding]) -> str:
    """Replace flagged sentences with a visible marker.

    Args:
        text (str): The text that was scanned.
        findings (list[Finding]): Output of scan() on that text.

    Returns:
        str: The text with each finding replaced by "[removed: text addressed to the grader]".
    """
    out, last = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        out.append(text[last:f.start])
        out.append(" [removed: text addressed to the grader] ")
        last = f.end
    out.append(text[last:])
    return "".join(out)


def overlaps(evidence: str, findings: list[Finding], threshold: int = 80) -> bool:
    """Check whether an evidence quote comes from flagged text (it cannot earn marks).

    Args:
        evidence (str): Quote given by the grader.
        findings (list[Finding]): Flagged sentences of the answer.
        threshold (int): Minimum rapidfuzz partial_ratio for a match.

    Returns:
        bool: True if the quote matches a flagged sentence; quotes under 3 characters never match.
    """
    ev = normalize_ws(evidence)
    if len(ev) < 3:
        return False
    return any(fuzz.partial_ratio(ev, normalize_ws(f.text)) >= threshold for f in findings)


def as_dicts(findings: list[Finding]) -> list[dict]:
    """Convert findings for the JSON report.

    Args:
        findings (list[Finding]): Flagged sentences.

    Returns:
        list[dict]: [{"kind", "text"}, ...].
    """
    return [{"kind": f.kind, "text": f.text} for f in findings]
