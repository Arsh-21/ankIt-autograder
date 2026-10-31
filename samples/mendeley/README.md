# Real exam: Data Science test (Mendeley)

Real scanned, handwritten answer sheets with the teacher's own marks. Part I is Q1-Q20 (multiple choice, 1 mark each), Part II is Q21-Q35 (short answers, 2 marks each), 50 in total.

| Student | Pages | Teacher: short answers (of 30) | Teacher: total (of 50) |
|---|---|---|---|
| Student_18 | 2 | 5 | 15 |
| Student_28 | 4 | 13 | 27 |
| Student_6 | 4 | 14 | 29 |
| Student_32 | 4 | 17.5 |  |
| Student_39 | 4 | 20 | 26 |
| Student_11 | 6 | 24.5 | 34.5 |

A blank total means the dataset's MCQ letters and MCQ total disagree for that student, so only the short-answer marks are used (see `scripts/import_mendeley.py`).

`teacher_marked/` has the teacher's marked copies, to compare with the app's marked pages.

Source: Dinesh K P, "A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research", Mendeley Data, V1, https://data.mendeley.com/datasets/sf3kvjwknt/1 (CC BY 4.0). Pages are shrunk to 1600 px; nothing else is changed.

Rebuild with `python scripts/make_mendeley_samples.py`.
