"""Critic agent, based on Reflexion.

When grading samples disagree, or a mark was dropped because its quote is
not in the answer, the critic writes a short verbal critique. The critique
goes into one more grading attempt as "problems to fix".
"""

import json

from core.llm import chat

from .prompts import CRITIC_SYSTEM, CRITIC_USER, GUARD_BOUNDARY, GUARD_SYSTEM


def _describe(samples: list[dict]) -> str:
    """Formats the grading samples as plain text for the critic prompt.

    Args:
        samples (list[dict]): Normalised grader outputs, each with "total" and
            per-criterion "criteria".

    Returns:
        str: One block per attempt, listing verdict, marks, evidence and reason
            for every criterion.
    """
    lines = []
    for i, s in enumerate(samples, start=1):
        lines.append(f"Attempt {i}: total {s['total']:g}")
        for cid, c in s["criteria"].items():
            note = ""
            if c["rejected"]:
                note = " (quote NOT found in the answer or came from text aimed at the grader; mark removed)"
            elif c["evidence_flagged"]:
                note = " (quote not found in the transcription)"
            lines.append(f"  {cid}: {c['verdict']} {c['marks']:g} | evidence: {c['evidence'][:150]!r}"
                         f" | reason: {c['reason'][:150]}{note}")
    return "\n".join(lines)


def critique(question: dict, answer_text: str, samples: list[dict], profile: str | None = None,
             guard_prompt: bool = True) -> str:
    """Asks the critic model to point out grading mistakes across the samples.

    Args:
        question (dict): Rubric question with "text" and "criteria".
        answer_text (str): The student's answer as graded.
        samples (list[dict]): Normalised grader outputs to review.
        profile (str | None): Model profile name from config.yaml.
        guard_prompt (bool): Whether to add the anti-injection guard text.

    Returns:
        str: The critique as plain text.

    Raises:
        LLMError: If the model call fails.
    """
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM + (GUARD_SYSTEM if guard_prompt else "")},
        {"role": "user", "content": CRITIC_USER.format(
            question=question["text"],
            criteria_json=json.dumps(question["criteria"], ensure_ascii=False, indent=1),
            answer=answer_text, attempts=_describe(samples),
            boundary=GUARD_BOUNDARY if guard_prompt else "")},
    ]
    return chat("critic", messages, profile=profile, max_tokens=2000)
