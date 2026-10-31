# Ablations on the Mendeley exam, most important first.
# Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File scripts/run_ablations.ps1
# Everything printed also goes to ablation_log.txt. If one step fails, the next one still runs.
# The held-out test is NOT here: it is run once, afterwards, with the setting chosen from these results.

Start-Transcript -Path ablation_log.txt
$D = "eval/datasets/mendeley"
$S = "--sheets", "@$D/split_ablation.txt", "--sample", "10", "--seed", "0", "--questions", "21-35"
$RT = "Student_12,Student_13,Student_14,Student_15,Student_16"

# 1. references, strictness, baselines (10 students, short answers)
python eval/run_ablation.py --dataset $D --profile openrouter @S --only n1,n1_ref_none,n1_ref_key,n1_ref_scheme,n1_lenient,n1_strict,baseline_plain,baseline_single_call

# 2. prompt injection (5 students): attacked copies, then the same students clean
python scripts/make_redteam.py --dataset $D --out eval/datasets/mendeley_redteam --sheets $RT
python eval/run_ablation.py --dataset eval/datasets/mendeley_redteam --variants eval/variants_redteam.yaml --profile openrouter --sheets $RT --questions 21-35
python eval/run_ablation.py --dataset $D --variants eval/variants_redteam.yaml --profile openrouter --sheets $RT --questions 21-35 --only defence_none,defence_all

# 3. five grading passes (15 students), for the free offline analysis
python eval/run_ablation.py --dataset $D --profile openrouter --sheets "@$D/split_ablation.txt" --sample 15 --seed 0 --questions 21-35 --only n5

# 4. teacher examples, what the grader looks at, finer rubric (same 10 students as step 1)
python eval/run_ablation.py --dataset $D --profile openrouter @S --only n1,n1_calibrated,n1_input_text,n1_input_image,n1_expand

# 5. other models (same 10 students)
python eval/run_ablation.py --dataset $D --profile openrouter_lite @S --only n1
python eval/run_ablation.py --dataset $D --profile openrouter_gpt @S --only n1

Stop-Transcript