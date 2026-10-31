# Demo test cases

All cases answer the quiz in `samples/question_paper.txt`, graded with `samples/rubric.txt` and `samples/model_answer.txt`. `expected_marks.csv` holds the marks a careful human grader would give.

## 01_typed_pdf

Typed answers in a PDF with a text layer. Q3 forgets to subtract the max (not numerically stable).

Expected: Q1 4, Q2 3, Q3 2 (total 9/10)

## 02_handwritten_scan

Handwriting-style scan over 2 pages, answers out of order, one crossed-out line. Q2 gets 28 instead of 30; Q3 is not numerically stable.

Expected: Q1 4, Q2 2, Q3 2 (total 8/10)

## 03_hindi

Hindi answers with English technical terms (code-mixed), photographed. A full-marks sheet.

Expected: Q1 4, Q2 3, Q3 3 (total 10/10)

## 04_tamil

Tamil answers with English terms. Q1 explains the problem but gives no remedy; Q3 is not numerically stable.

Expected: Q1 2, Q2 3, Q3 2 (total 7/10)

## 05_adversarial

Weak answers plus requests for marks in English and Hindi. The marks should stay low and every request should be flagged.

Expected: Q1 1, Q2 0, Q3 0 (total 1/10)
