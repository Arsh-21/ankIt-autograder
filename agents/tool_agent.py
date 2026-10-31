"""Tool agent (ReAct): think -> act with a tool -> observe, for up to N steps,
before grading. Its observations go to every grader sample, so tools run once
per question, not once per sample.

The only tool is a calculator (sympy, maths characters only), to check a
student's arithmetic. Student code is never executed: without a proper
sandbox (a throw-away container with no network and no file access) running
untrusted code on the grading machine is not safe, so code answers are judged
by the grader reading them (see the code rule in agents/grader.py).

Off by default: extra calls, and only useful for numeric questions. Tool
outputs are passed on as untrusted data and the injection detector runs on them.
"""

from core import guard
from core.config import thresholds
from core.llm import LLMError, chat_json
from tools.math_check import calc

from .prompts import GUARD_BOUNDARY, TOOL_SYSTEM, TOOL_USER

TOOLS = {"calc": calc}


def run_tools(question: dict, answer_text: str, profile: str | None = None,
              guard_prompt: bool = True) -> list[dict]:
    """Runs the ReAct tool loop on one answer.

    Stops at "finish", an unknown action, a non-dict reply, a failed model
    call, or after `tool_max_steps` steps.

    Args:
        question (dict): Rubric question with "text" and "model_answer".
        answer_text (str): The student's answer.
        profile (str | None): Model profile name from config.yaml.
        guard_prompt (bool): Whether to add the anti-injection boundary text.

    Returns:
        list[dict]: One dict per tool call with "thought", "tool", "input",
            "output" and "injection_in_output"; empty for a blank answer.
    """
    if not answer_text.strip():
        return []
    history: list[dict] = []
    for _ in range(thresholds()["tool_max_steps"]):
        hist_txt = "\n".join(
            f"Step {i + 1}: thought={h['thought']!r} action={h['tool']} input={h['input']!r}\n"
            f"Observation: {h['output']}" for i, h in enumerate(history)) or "(none yet)"
        messages = [{"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": TOOL_USER.format(
                        question=question["text"], model_answer=question.get("model_answer") or "(none)",
                        answer=answer_text, history=hist_txt,
                        boundary=GUARD_BOUNDARY if guard_prompt else "")}]
        try:
            step = chat_json("tools", messages, profile=profile, max_tokens=2000)
        except LLMError:
            break
        if not isinstance(step, dict):
            break
        action = str(step.get("action", "finish")).strip()
        if action not in TOOLS:
            break
        output = TOOLS[action](str(step.get("input", "")))
        found = guard.scan(output)
        if found:  # tool output contains text aimed at the grader
            output = guard.strip(output, found)
        history.append({"thought": str(step.get("thought", "")), "tool": action,
                        "input": str(step.get("input", "")), "output": output,
                        "injection_in_output": bool(found)})
    return history


def format_observations(obs: list[dict]) -> str:
    """Formats the tool calls for the grader prompt.

    Args:
        obs (list[dict]): Output of `run_tools`.

    Returns:
        str: One block per tool call with its input and output, or "".
    """
    return "\n\n".join(f"[{o['tool']}] input:\n{o['input']}\noutput:\n{o['output']}" for o in obs)
