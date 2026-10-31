# Deep learning quiz (sample)

A 20-mark quiz with seven questions: two MCQs, two linked numericals (Q4 uses the answer to Q3), a derivation, an explanation and a short program. Built by `scripts/make_samples.py`.

**Exam documents** (`exam/`), each in a different format: the question paper is a scanned printout, the marking scheme a typed PDF, the model answer two phone photos of the teacher's handwritten key. Any of them can be left out; `exam_text/` has the same three as plain text.

**Students** (`sheets/`, all of them in `class.zip`). Expected marks, from applying the marking scheme by hand, are in `expected_marks.csv`.

| Student | Sheet | Expected | Notes |
|---|---|---|---|
| asha | typed PDF, strong | 19/20 | Typed PDF with a text layer, no OCR needed. Q7 explains overflow but not why the result is unchanged. |
| rahul | handwritten scan, 2 pages, out of order | 12/20 | Scanned PDF, 2 pages, Q4 to Q7 out of order, one crossed-out line. Q3 drops the padding (30x30x16); Q4 uses that shape correctly, so it gets full marks (error carried forward). Q5 leaves sigmoid'(z) unexpanded; Q7 is not numerically stable. |
| priya | Hindi, two phone photos | 19/20 | Hindi with English technical terms, photographed on a desk. Q6 leaves out what changes at test time. |
| karthik | Tamil, scanned | 13/20 | Tamil with English terms, 2 scanned pages. Forgets the biases in Q3 and Q4, writes dy/dw for dy/dz and drops the x in Q5, no co-adaptation or ensemble in Q6, no explanation in Q7. |
| meena | weak, mostly blank | 2/20 | Scanned PDF whose second page is empty ruled paper (nothing should be read from it). Q4 and Q7 are not attempted, Q5 says "dont know", Q3 gives no depth and no working. |
| arjun | rushed handwriting, phone photo | 18/20 | Rushed handwriting photographed at an angle, answers labelled "1)", a changed MCQ answer (b crossed out, c written). Q6 has no co-adaptation and no reason why it helps. |
| vikram | asks the grader for marks | 2/20 | Weak answers plus requests for marks in English and Hindi; every request should be flagged and earn nothing. Q4 is worked correctly from his own (wrong) Q3 shape but forgets the biases, so it still earns 2 (error carried forward). |

The handwriting is drawn with handwriting fonts and random jitter, so it is tidier than real handwriting. `samples/mendeley/` has real scanned sheets marked by a teacher.
