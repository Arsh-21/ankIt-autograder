### What happens when you press Grade

1. **Reading.** Typed PDFs and Word files are read directly. Scans and photos go to the OCR agent, which copies the handwriting exactly in its own script (Hindi stays Hindi), marks what it can't read as `[illegible]` and crossed-out text as `[STRUCK]`. For Indian-language sheets it can use Bodhan AI's Indic OCR.
2. **Matching.** The segmenter finds which part of the sheet answers which question, even when answers are out of order or continue on another page. A rule check then confirms that every matched answer really is on the sheet.
3. **Guarding.** A rule-based detector looks for text aimed at the grader ("please give full marks", "ignore the rubric", in English and Indian languages). Those sentences are removed before grading and the question is flagged.
4. **Grading.** Each question is graded several times independently, looking at the page image and the transcription. Every mark must quote the student's own words; a mark whose quote isn't in the answer is removed by a rule check.
5. **Settling.** The independent grades are combined by a percentile that matches the strictness you chose. If they disagree, a critic agent reviews them and one more grade is made.
6. **Re-reading (optional, off by default).** If you turn on *Double-check the handwriting*, then where marks were lost on a scanned page a vision model compares the transcription with the handwriting. If a digit or sign was misread, the answer is re-graded with the correct reading.
7. **Handing over.** Anything uncertain is flagged for you in the Check step, with the reason. The pages come back marked in red, and a whole class gets a marks CSV and a summary of common mistakes.

The design choices are explained, with the research behind each one, in `docs/DESIGN.md`.
