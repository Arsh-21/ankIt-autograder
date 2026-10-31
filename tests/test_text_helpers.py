"""Plain-code helpers: JSON extraction, quote matching, blank answers,
question ids and language detection.

Usage:
    python -m pytest -q tests/test_text_helpers.py
"""

import pytest

from core.lang import detect_language, to_ascii_digits
from core.utils import coverage, evidence_supported, is_blank, normalize_qid, parse_json, remove_struck


@pytest.mark.parametrize("reply, expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! ```json\n{"a": [1, 2,],}\n```', {"a": [1, 2]}),
    ('<think>let me think</think>\n{"ok": true}', {"ok": True}),
    ('Here you go: [1, 2, 3] thanks', [1, 2, 3]),
    # a reply wrapped in ```json whose data itself contains ``` code blocks (seen from a real model)
    ('```json\n{"answer_text": "```python\\ndef f(x):\\n    return x\\n```", "pages": [2]}\n```',
     {"answer_text": "```python\ndef f(x):\n    return x\n```", "pages": [2]}),
    ('{"a": "line1\nline2"}', {"a": "line1\nline2"}),   # raw newline inside a string
])
def test_parse_json_handles_messy_replies(reply, expected):
    assert parse_json(reply) == expected


def test_parse_json_raises_without_json():
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_evidence_must_really_be_in_the_answer():
    answer = "For sigmoid the derivative is at most 0.25 so the gradient shrinks."
    assert evidence_supported("derivative is at most 0.25", answer)
    assert evidence_supported("derivative is atmost 0.25", answer)       # small OCR slip is fine
    assert not evidence_supported("residual connections solve this", answer)
    assert evidence_supported("ग्रेडिएंट छोटा", "इसलिए ग्रेडिएंट छोटा हो जाता है")  # Indic script


def test_crossed_out_text_is_removed_and_blank_detection():
    assert "wrong" not in remove_struck("right [STRUCK: wrong] answer")
    assert is_blank("")
    assert is_blank("[illegible] [STRUCK: something]")
    assert not is_blank("ReLU")
    assert not is_blank("प्रश्न")


def test_coverage_catches_invented_answers():
    sheet = "Q1. Gradients shrink in deep networks.\nQ2. Output is 30 x 30."
    assert coverage("Gradients shrink in deep networks.", sheet) == 1.0
    assert coverage("Batch norm fixes everything for good.", sheet) == 0.0


@pytest.mark.parametrize("raw, expected", [
    ("Q1", "1"), ("q 1", "1"), ("Question 1", "1"), ("Q2(a)", "2a"), ("Ans 4", "4"),
    ("प्रश्न १", "1"), ("கேள்வி 3", "3"), ("ప్రశ్న 2", "2"),
])
def test_question_ids_normalise(raw, expected):
    assert normalize_qid(raw) == expected


def test_language_detection_and_digits():
    assert detect_language("प्रश्न 1. backpropagation के समय gradient छोटा होता है") == "Hindi"
    assert detect_language("கேள்வி 2. output = 30 x 30") == "Tamil"
    assert detect_language("Q1. ReLU fixes vanishing gradients") == "English"
    assert to_ascii_digits("३०") == "30"


def test_mcq_letters_count_as_answers():
    # Mendeley run: bare MCQ letters were treated as blank and their quotes rejected
    assert not is_blank("D")
    assert not is_blank("1. C")
    assert evidence_supported("B", "B")
    assert evidence_supported("b", "Ans: B")
    assert not evidence_supported("B", "A")
    assert not evidence_supported("B", "Bayes rule")  # a letter inside a word is not the answer


def test_page_numbers_are_not_messages():
    from core.utils import is_page_number
    for t in ["1/ total 3", "1 / total 3", "Page 2 of 6", "pg 4", "(3/4)", "Page: 1 of 6", "... 3 of 6",
              "Dano 5 of 6", "Page 6 of r"]:
        assert is_page_number(t), t
    for t in ["please give me full marks", "I studied hard, 2 of 2 marks please", "marks 2 of 2"]:
        assert not is_page_number(t), t


def test_blank_page_detector():
    from PIL import Image, ImageDraw, ImageEnhance

    from core.ingest import looks_blank
    page = Image.new("RGB", (800, 1100), (250, 250, 250))
    ghost = page.copy()
    ImageDraw.Draw(ghost).text((100, 100), "mirrored show-through", fill=(235, 235, 235))
    written = page.copy()
    d = ImageDraw.Draw(written)
    for y in range(100, 600, 30):
        d.line((80, y, 700, y), fill=(20, 20, 60), width=3)
    faded = ImageEnhance.Brightness(ImageEnhance.Contrast(written).enhance(0.35)).enhance(1.25)
    assert looks_blank(page, 0.0005)
    assert looks_blank(ghost, 0.0005)
    assert not looks_blank(written, 0.0005)
    assert not looks_blank(faded, 0.0005)  # faint pencil / weak scan is still writing
    blue = page.copy()
    ImageDraw.Draw(blue).line((80, 100, 600, 100), fill=(90, 110, 220), width=2)
    assert not looks_blank(blue, 0.0005)
    assert not looks_blank(page, 0)


def test_stitched_quotes():
    table = "26. Structured | Semi-structured\n- Are in tabular format | - Not in tabular format\n- format of Excel, CSV | - JSON"
    assert evidence_supported("- Are in tabular format | - Not in tabular format - format of Excel, CSV", table)
    assert not evidence_supported("Are in tabular format, stored in SQL databases with fixed schemas", table)
    lines = "has a decent gap between the\nlinear boundaries.\nClassifier 2: bad.\nThere seems to be 2 different\ntypes of data"
    assert evidence_supported("has a decent gap between the linear boundaries.\nThere seems to be 2 different types", lines)


def test_json_with_mixed_escapes():
    # a correctly escaped backslash next to a bad one (seen from the segmenter on Student_15)
    assert parse_json(r'{"a": "proper \\section and a bad \s here"}')["a"] == r"proper \section and a bad \s here"
    assert parse_json(r'{"a": "\underline{x} é"}')["a"] == "\\underline{x} é"


def test_find_references():
    from core.utils import find_references
    qids = ["Q1", "Q2", "Q3", "Q4(a)", "Q4(b)"]
    assert find_references("As explained in Q3, the gradient vanishes.", qids, "Q1") == ["Q3"]
    assert find_references("Refer answer 2. Also see question 4(b).", qids, "Q1") == ["Q2", "Q4(b)"]
    assert find_references("same as Q.1", qids, "Q1") == []            # never itself
    assert find_references("we see that 3 layers are enough", qids, "Q1") == []
    assert find_references("प्रश्न 2 देखें", qids, "Q1") == ["Q2"]
    assert find_references("already answered in 4", qids, "Q1") == []  # no Q4 in the rubric, only 4a/4b


def test_infer_dependencies():
    from core.utils import infer_dependencies
    qs = [{"id": "Q1", "text": "Define entropy."},
          {"id": "Q2(a)", "text": "Compute the mean of 2, 4, 6."},
          {"id": "Q2(b)", "text": "Hence find the variance."},
          {"id": "Q3", "text": "Using your answer to Question 1, explain information gain."},
          {"id": "Q4", "text": "Hence or otherwise, list two uses."},          # not a sub-part: no guess
          {"id": "Q5", "text": "Explain part 2 of the pipeline in 3 lines."}]  # "part 2" is no question
    assert infer_dependencies(qs) == {"Q2(b)": ["Q2(a)"], "Q3": ["Q1"]}


def test_parse_json_keeps_the_first_of_two_objects():
    """Models sometimes repeat the JSON; the first complete object is used."""
    assert parse_json('{"criteria": [], "feedback": "a"}\n{"criteria": [], "feedback": "b"}')["feedback"] == "a"


def test_cut_off_reply_is_asked_again_with_more_room(monkeypatch):
    """A reply stopped at the token limit is requested again with twice the limit, and not cached."""
    from types import SimpleNamespace

    from core import llm
    calls = []

    class FakeClient:
        class chat:  # noqa: N801 - mirrors the openai client layout
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    calls.append(kw["max_tokens"])
                    done = kw["max_tokens"] >= 200
                    msg = SimpleNamespace(content='{"a": 1}' if done else '{"a": "unfinish')
                    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop" if done else "length")],
                                           usage=None)

    monkeypatch.setattr(llm, "_client", lambda provider: FakeClient)
    monkeypatch.setattr(llm, "model_for", lambda role, profile=None: {"provider": "ollama", "model": "fake"})
    monkeypatch.setattr(llm, "_cache_get", lambda key: None)
    monkeypatch.setattr(llm, "_cache_put", lambda key, text: None)
    assert llm.chat_json("grader", [{"role": "user", "content": "x"}], max_tokens=100) == {"a": 1}
    assert calls == [100, 200]


def test_grid_answers_reads_the_usual_layouts():
    from core.utils import grid_answers, looks_like_grid
    text = ("1) b   2) c   3) a\n| 4 | d |\n5-a 6-b\n 7  8  9\n a  b  c\n"
            "Q21. Option 2) b is wrong because the loss grows.")
    got = grid_answers(text)
    assert got == {"1": "1) b", "2": "2) c", "3": "3) a", "4": "4 | d", "5": "5-a", "6": "6-b",
                   "7": "7: a", "8": "8: b", "9": "9: c"}
    assert looks_like_grid(text) and not looks_like_grid("Q1. (c)\nQ2. The answer is b.")


def test_parse_json_mends_an_unescaped_quote_but_not_a_cut_off_reply():
    from core.utils import parse_json
    got = parse_json('{"questions": [{"id": "Q1", "text": "Which is an "assumption" of LR?", "max_marks": 1}]}')
    assert got["questions"][0]["text"] == 'Which is an "assumption" of LR?'
    with pytest.raises(ValueError):
        parse_json('{"questions": [{"id": "Q1", "text": "Which is')
