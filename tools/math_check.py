"""Calculator for the tool agent: evaluates a maths expression with sympy to
check a student's arithmetic."""

import re

import sympy

# sympy parses text by evaluating it as Python, so only allow maths characters
_ALLOWED = re.compile(r"^[0-9a-zA-Z_+\-*/^().,=\s]*$")
_BLOCKED_WORDS = re.compile(r"__|\b(import|lambda|exec|eval|open|os|sys|getattr|globals|locals)\b")


def calc(expression: str) -> str:
    """Evaluate or simplify an expression, or check an equation.

    Only plain maths characters are accepted, because sympy parses text by
    evaluating it as Python. "^" is read as a power.

    Args:
        expression (str): An expression such as "(32 - 5 + 2*1)/1 + 1", or an
            equation "a = b" to check whether both sides are equal.

    Returns:
        str: The simplified value with a numeric form, "EQUAL" / "NOT equal" for an
            equation, or a message saying why the expression was rejected or failed.
    """
    expr = (expression or "").strip()
    if not expr:
        return "No expression given."
    if len(expr) > 300 or not _ALLOWED.match(expr) or _BLOCKED_WORDS.search(expr):
        return "Expression rejected: only plain maths is allowed."
    expr = expr.replace("^", "**")
    try:
        if "=" in expr:  # "a = b" -> check if both sides are equal
            left, right = expr.split("=", 1)
            diff = sympy.simplify(sympy.sympify(left) - sympy.sympify(right))
            return f"left - right simplifies to {diff} -> {'EQUAL' if diff == 0 else 'NOT equal'}"
        value = sympy.sympify(expr)
        simplified = sympy.simplify(value)
        try:
            numeric = sympy.N(simplified, 10)
            return f"{simplified} (= {numeric})"
        except Exception:
            return str(simplified)
    except Exception as e:  # noqa: BLE001
        return f"Could not evaluate: {e}"
