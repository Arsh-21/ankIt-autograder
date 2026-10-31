# Second round: confidence intervals, the free offline analysis, the injection test (redone),
# and teacher examples + lenient together. Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File scripts/run_ablations_2.ps1
# Everything printed is also appended to ablation_log2.txt.

$D = "eval/datasets/mendeley"
$LOG = "ablation_log2.txt"

# a. 95% confidence intervals for every round-1 variant against n1 (free)
python eval/compare.py --runs eval/runs/20261031_144845 eval/runs/20261031_160840 --base n1@20261031_144845 2>&1 | Tee-Object -FilePath $LOG -Append

# b. 1, 3 or 5 passes, ways of combining them, quote check on/off, from the n5 run (free)
python eval/offline.py --run eval/runs/20261031_153359 --variant n5 2>&1 | Tee-Object -FilePath $LOG -Append

# c. prompt injection: 4 students x 4 attacks, each defence layer alone and together
python scripts/make_redteam.py --dataset $D --out eval/datasets/mendeley_rt --sheets Student_12,Student_13,Student_14,Student_15 --attacks direct,override,subtle,hindi 2>&1 | Tee-Object -FilePath $LOG -Append
python eval/run_ablation.py --dataset eval/datasets/mendeley_rt --variants eval/variants_redteam.yaml --profile openrouter --questions 21-35 --only defence_none,defence_prompt,defence_detector_strip,defence_all 2>&1 | Tee-Object -FilePath $LOG -Append

# d. teacher examples + lenient together, on the same 10 students as round 1
python eval/run_ablation.py --dataset $D --profile openrouter --sheets "@$D/split_ablation.txt" --sample 10 --seed 0 --questions 21-35 --only n1,n1_calibrated,n1_cal_lenient 2>&1 | Tee-Object -FilePath $LOG -Append