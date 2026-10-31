"""The rubric as an editable table (one row per criterion) instead of JSON.

A blank Question cell continues the question above, so a criterion is added
by adding a row. Model answers and graded examples are kept from
the stored rubric (edit them in the JSON panel)."""

HEADERS = ["Q", "Question text", "Criterion", "Marks", "Partial"]
DATATYPES = ["str", "str", "str", "number", "bool"]


def to_rows(rubric: dict | None) -> list[list]:
    """Rubric -> table rows, one per criterion; the question text only on its first row.

    Args:
        rubric (dict | None): The rubric, or None for one empty row.

    Returns:
        list[list]: Rows of [question id, question text, criterion, marks, partial ok].
    """
    if not rubric:
        return [["", "", "", 0, True]]
    rows = []
    for q in rubric["questions"]:
        for i, c in enumerate(q["criteria"]):
            rows.append([q["id"], q["text"] if i == 0 else "", c["description"], c["marks"], c["partial_allowed"]])
    return rows


def _as_bool(x) -> bool:
    """Read a table cell as True/False ("yes", "true", "1" count as True)."""
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1", "y")
    return bool(x)


def from_rows(rows: list, previous: dict | None) -> dict:
    """Rebuild the rubric from the table.

    Model answers, examples, notes and links between questions are kept from
    the previous rubric, since the table doesn't show them.

    Args:
        rows (list): Table rows as produced by to_rows (possibly edited).
        previous (dict | None): The rubric before the edit.

    Returns:
        dict: {"questions": [...]} ready for validate_rubric.

    Raises:
        ValueError: With a readable message if a row can't be used.
    """
    prev = {q["id"]: q for q in (previous or {}).get("questions", [])}
    questions, current = [], None
    for n, row in enumerate(rows or [], start=1):
        row = list(row) + [""] * (5 - len(row))
        qid, qtext, desc, marks, partial = (str(row[0] or "").strip(), str(row[1] or "").strip(),
                                            str(row[2] or "").strip(), row[3], row[4])
        if not (qid or qtext or desc):
            continue  # empty row
        if qid and (current is None or qid != current["id"]):
            old = prev.get(qid, {})
            current = {"id": qid, "text": qtext or old.get("text", ""), "max_marks": 0, "criteria": [],
                       "model_answer": old.get("model_answer", ""), "notes": old.get("notes", ""),
                       "examples": old.get("examples", []), "depends_on": old.get("depends_on", [])}
            questions.append(current)
        if current is None:
            raise ValueError(f"Row {n}: the first row needs a question id (e.g. Q1).")
        if qtext and not current["text"]:
            current["text"] = qtext
        try:
            m = float(marks)
        except (TypeError, ValueError):
            raise ValueError(f"Row {n}: marks must be a number, got {marks!r}.") from None
        if not desc:
            raise ValueError(f"Row {n}: the criterion is empty.")
        current["criteria"].append({"id": f"{current['id']}.{len(current['criteria']) + 1}", "description": desc,
                                    "marks": m, "partial_allowed": _as_bool(partial)})
    for q in questions:
        q["max_marks"] = sum(c["marks"] for c in q["criteria"])
    if not questions:
        raise ValueError("The rubric table is empty.")
    return {"questions": questions}
