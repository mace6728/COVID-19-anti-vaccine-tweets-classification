# Final Training Log (2026-03-30)

## 1) Goal
- Re-train with an anti-overfitting strategy (no overfit sanity setting).
- Generate a final Kaggle-uploadable CSV.
- Record all actions and outputs in one markdown report.

## 2) Environment and issue handled
- Initial run failed in `.venv_hw1` because:
  - `torch==2.5.1+cu121` with `transformers==5.4.0`
  - `transformers` blocked `torch.load` for torch < 2.6 (CVE-2025-32434 related guard)
- Switched runtime to `.venv` and verified:
  - `torch==2.8.0+cu128`
  - CUDA available (`NVIDIA GeForce RTX 4060 Ti`)
  - `transformers==5.4.0`

## 3) Anti-overfitting training design
To avoid overfitting, this run used strict train/val split with regularization and early stopping logic already in `train.py`:
- Real validation split: `preprocessed/train_preprocessed.csv` and `preprocessed/val_preprocessed.csv`
- Early stopping: `--early-stopping-patience 3`
- Weight decay: `--weight-decay 0.02`
- Dropout: `--dropout 0.25`
- Low LR fine-tuning: `--learning-rate 1.5e-5`
- Gradient accumulation (effective larger batch): `--batch-size 8 --grad-accum-steps 4`
- ASL + pos-weight for class imbalance: `--loss-type asl --asl-use-pos-weight`
- Per-label threshold tuning with fine search grid:
  - start 0.02, end 0.90, step 0.02

## 4) Training command used
```bash
/home/mace/NLP/.venv/bin/python train.py \
  --model-type transformer \
  --pretrained-model digitalepidemiologylab/covid-twitter-bert-v2 \
  --head-type linear \
  --pooling cls \
  --multi-sample-dropout 8 \
  --loss-type asl \
  --asl-gamma-neg 4.0 \
  --asl-gamma-pos 1.0 \
  --asl-clip 0.05 \
  --asl-use-pos-weight \
  --learning-rate 1.5e-5 \
  --weight-decay 0.02 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --epochs 10 \
  --early-stopping-patience 3 \
  --max-length 128 \
  --dropout 0.25 \
  --num-workers 2 \
  --threshold-start 0.02 \
  --threshold-end 0.90 \
  --threshold-step 0.02 \
  --seed 2026 \
  --output-dir artifacts/ctbert_final_no_overfit_v2
```

## 5) Training result summary
From `artifacts/ctbert_final_no_overfit_v2/run_summary.json`:
- `best_val_macro_f1`: 0.6478679581406334
- `default_macro_f1`: 0.6478679581406334
- `tuned_macro_f1`: 0.7156902401364169
- `model_type`: transformer
- `loss_type`: asl

Threshold tuning improved macro-F1 by about +0.0678 over default threshold 0.5.

## 6) Inference and final Kaggle CSV
Inference command:
```bash
/home/mace/NLP/.venv/bin/python predict.py \
  --checkpoint artifacts/ctbert_final_no_overfit_v2/best_model.pt \
  --threshold-file artifacts/ctbert_final_no_overfit_v2/thresholds.json \
  --sample-submission sample_submission.csv \
  --output-file artifacts/ctbert_final_no_overfit_v2/submission_kaggle_final.csv
```

Validation checks:
- Row count matches sample submission:
  - `sample_submission.csv`: 1977 lines
  - `submission_kaggle_final.csv`: 1977 lines
- Column order matches sample format.

## 7) Final deliverables
Primary final submission file for upload:
- `final_submission_kaggle.csv`

Artifact folder outputs:
- `artifacts/ctbert_final_no_overfit_v2/best_model.pt`
- `artifacts/ctbert_final_no_overfit_v2/training_config.json`
- `artifacts/ctbert_final_no_overfit_v2/training_history.json`
- `artifacts/ctbert_final_no_overfit_v2/thresholds.json`
- `artifacts/ctbert_final_no_overfit_v2/run_summary.json`
- `artifacts/ctbert_final_no_overfit_v2/submission_kaggle_final.csv`

## 8) Notes
- This run is not an overfit sanity run. It uses standard train/val evaluation and threshold tuning on validation logits.
- If you want another anti-overfit round, next safe directions are:
  - 5-fold CV with out-of-fold threshold tuning
  - Ensemble of 2-3 seeds with the same regularized config

## 9) Follow-up check and improvement round (v3)

### 9.1 Why improve from v2
After checking v2 artifacts:
- `best_val_macro_f1` was 0.6479 and `tuned_macro_f1` was 0.7157.
- The gap from default to tuned macro-F1 was large (about +0.0678), showing strong dependence on threshold calibration.
- Training used stronger regularization (`dropout=0.25`, `weight_decay=0.02`) and a lower LR (`1.5e-5`).

Improvement hypothesis:
- Slightly relax regularization and increase initial LR to improve representation learning.
- Trigger LR reduction earlier to stabilize later epochs.
- Use finer threshold grid to improve validation calibration.

### 9.2 Modifications applied for v3
- `learning-rate`: `1.5e-5 -> 2e-5`
- `weight-decay`: `0.02 -> 0.01`
- `dropout`: `0.25 -> 0.20`
- `multi-sample-dropout`: `8 -> 6`
- `epochs`: `10 -> 12`
- `scheduler-patience`: `2 -> 1`
- `scheduler-factor`: `0.5 -> 0.7`
- `early-stopping-patience`: `3 -> 4`
- Threshold search: `start=0.01, end=0.95, step=0.01`

### 9.3 v3 training command
```bash
/home/mace/NLP/.venv/bin/python train.py \
  --model-type transformer \
  --pretrained-model digitalepidemiologylab/covid-twitter-bert-v2 \
  --head-type linear \
  --pooling cls \
  --multi-sample-dropout 6 \
  --loss-type asl \
  --asl-gamma-neg 4.0 \
  --asl-gamma-pos 1.0 \
  --asl-clip 0.05 \
  --asl-use-pos-weight \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --epochs 12 \
  --scheduler-factor 0.7 \
  --scheduler-patience 1 \
  --early-stopping-patience 4 \
  --max-length 128 \
  --dropout 0.20 \
  --num-workers 2 \
  --threshold-start 0.01 \
  --threshold-end 0.95 \
  --threshold-step 0.01 \
  --seed 2026 \
  --output-dir artifacts/ctbert_final_no_overfit_v3
```

### 9.4 v2 vs v3 comparison
- v2 (`artifacts/ctbert_final_no_overfit_v2/run_summary.json`)
  - best_val_macro_f1: 0.6478679581
  - tuned_macro_f1: 0.7156902401
  - default_micro_f1: 0.7086230876
  - tuned_micro_f1: 0.7622519239

- v3 (`artifacts/ctbert_final_no_overfit_v3/run_summary.json` and `thresholds.json`)
  - best_val_macro_f1: 0.6624212867
  - tuned_macro_f1: 0.7208608462
  - default_micro_f1: 0.7198275862
  - tuned_micro_f1: 0.7615541922

Delta (v3 - v2):
- best_val_macro_f1: +0.0145533286
- tuned_macro_f1: +0.0051706060
- default_micro_f1: +0.0112044986
- tuned_micro_f1: -0.0006977316

Result:
- Core objective (macro-F1) improved for both default and tuned thresholds.
- Tuned micro-F1 is slightly lower, but macro-F1 is the task priority and is improved.

## 10) Updated inference/output after v3
Inference command:
```bash
/home/mace/NLP/.venv/bin/python predict.py \
  --checkpoint artifacts/ctbert_final_no_overfit_v3/best_model.pt \
  --threshold-file artifacts/ctbert_final_no_overfit_v3/thresholds.json \
  --sample-submission sample_submission.csv \
  --output-file artifacts/ctbert_final_no_overfit_v3/submission_kaggle_final_v3.csv
```

Row-count check:
- `sample_submission.csv`: 1977 lines
- `artifacts/ctbert_final_no_overfit_v3/submission_kaggle_final_v3.csv`: 1977 lines

Promoted latest best submission:
- Copied `artifacts/ctbert_final_no_overfit_v3/submission_kaggle_final_v3.csv`
- to `final_submission_kaggle.csv`

## 11) Current best model and submission
- Best current run directory: `artifacts/ctbert_final_no_overfit_v3`
- Best current submission file: `final_submission_kaggle.csv`

## 12) Default-macro-first implementation (started and executed)

### 12.1 Objective
User priority: maximize **default macro F1** first; other metrics are secondary.

### 12.2 Implemented experiment (v4, multi-seed)
To execute the proposed plan, a robustness run with 3 seeds was performed using stronger regularization (close to v2) and conservative threshold search:

Common v4 settings:
- model: CT-BERT + linear head
- `learning-rate=1.5e-5`
- `weight-decay=0.02`
- `dropout=0.25`
- `multi-sample-dropout=8`
- `epochs=12`
- `scheduler-factor=0.5`, `scheduler-patience=2`
- `early-stopping-patience=4`
- threshold search: `0.05 -> 0.95` with step `0.05`

Runs:
- `artifacts/ctbert_default_macro_v4_seed2026`
- `artifacts/ctbert_default_macro_v4_seed2027`
- `artifacts/ctbert_default_macro_v4_seed2028`

### 12.3 Runtime issue and fix
- During the loop run, seed 2028 stalled with `num_workers=2` (dataloader worker hang symptoms).
- Recovery: terminated the stuck batch and re-ran seed 2028 with `num_workers=0`.
- Re-run completed successfully.

### 12.4 Results (default macro priority)
Collected from each run's `run_summary.json`:

- v2: default macro = 0.6478679581
- v3: default macro = 0.6624212867
- v4 seed2026: default macro = 0.6552377974
- v4 seed2027: default macro = 0.6510394747
- v4 seed2028: default macro = 0.6520793176

v4 robustness summary:
- mean(default macro) = 0.6527855299
- std(default macro) = 0.0017852227

Conclusion under default-macro-first policy:
- **v3 is still the best default-macro run** among tested models.

### 12.5 Default-threshold submission artifact
To align with default-macro-first policy, a strict default-threshold (0.5) submission was generated from v3:

- threshold file: `artifacts/ctbert_final_no_overfit_v3/thresholds_default_0_5.json`
- submission: `artifacts/ctbert_final_no_overfit_v3/submission_kaggle_default05_v3.csv`
- root copy: `final_submission_kaggle_default05_v3.csv`

Row-count validation:
- `sample_submission.csv`: 1977 lines
- `submission_kaggle_default05_v3.csv`: 1977 lines

### 12.6 Next implementation step (already prepared)
If continuing this track, next highest-impact implementation is:
- 5-fold CV (or at least 3-fold) with model selection by mean default macro and low fold variance, then export the best fold/ensemble submission.

## 13) Single-file integration + v2 backbone reruns (2026-04-04)

### 13.1 Objective
- Integrate all project `.py` logic into one runnable file: `HW1_112550043.py`.
- Re-run v2-style experiments from this single file using:
  - CT-BERT
  - BERTweet
  - RoBERTa

### 13.2 Integration status for `HW1_112550043.py`
- Core modules (`config.py`, `dataset.py`, `losses.py`, `threshold_tuner.py`, `evaluate.py`, `models/*.py`, `train.py`, `predict.py`) were already integrated.
- Missing helper logic from `tmp_metrics.py` was added into the unified file:
  - `build_loader_and_model(...)`
  - `evaluate_run(...)`
  - `run_metrics(...)`
- Added a new CLI subcommand:
  - `metrics`

CLI check:
```bash
/home/mace/NLP/.venv/bin/python HW1_112550043.py -h
```
Now shows:
- `preprocess`
- `train`
- `predict`
- `metrics`

### 13.3 v2-style training setup (same policy)
Common settings used for all 3 backbones:
- `--model-type transformer`
- `--head-type linear`
- `--pooling cls`
- `--multi-sample-dropout 8`
- `--loss-type asl --asl-gamma-neg 4.0 --asl-gamma-pos 1.0 --asl-clip 0.05 --asl-use-pos-weight`
- `--learning-rate 1.5e-5 --weight-decay 0.02`
- `--batch-size 8 --grad-accum-steps 4`
- `--epochs 10 --early-stopping-patience 3`
- `--max-length 128 --dropout 0.25`
- `--threshold-start 0.02 --threshold-end 0.90 --threshold-step 0.02`
- `--seed 2026 --num-workers 2`

Backbone-specific runs:
- CT-BERT:
  - `--pretrained-model digitalepidemiologylab/covid-twitter-bert-v2`
  - `--output-dir artifacts/singlefile_v2_ctbert`
- BERTweet:
  - `--pretrained-model vinai/bertweet-base`
  - `--output-dir artifacts/singlefile_v2_bertweet`
- RoBERTa:
  - `--pretrained-model roberta-base`
  - `--output-dir artifacts/singlefile_v2_roberta`

### 13.4 Results (from each `run_summary.json`)
- CT-BERT (`artifacts/singlefile_v2_ctbert/run_summary.json`)
  - `best_val_macro_f1`: 0.6498610898
  - `default_macro_f1`: 0.6498610898
  - `tuned_macro_f1`: 0.7114522523

- BERTweet (`artifacts/singlefile_v2_bertweet/run_summary.json`)
  - `best_val_macro_f1`: 0.5632475977
  - `default_macro_f1`: 0.5632475977
  - `tuned_macro_f1`: 0.6928502978

- RoBERTa (`artifacts/singlefile_v2_roberta/run_summary.json`)
  - `best_val_macro_f1`: 0.6147878909
  - `default_macro_f1`: 0.6147878909
  - `tuned_macro_f1`: 0.7035172071

Default-macro ranking:
1. CT-BERT (0.6499)
2. RoBERTa (0.6148)
3. BERTweet (0.5632)

### 13.5 Submissions generated from single-file pipeline
Predict outputs:
- `artifacts/singlefile_v2_ctbert/submission_kaggle.csv`
- `artifacts/singlefile_v2_bertweet/submission_kaggle.csv`
- `artifacts/singlefile_v2_roberta/submission_kaggle.csv`

Row-count validation:
- `sample_submission.csv`: 1977 lines
- `singlefile_v2_ctbert/submission_kaggle.csv`: 1977 lines
- `singlefile_v2_bertweet/submission_kaggle.csv`: 1977 lines
- `singlefile_v2_roberta/submission_kaggle.csv`: 1977 lines

### 13.6 Metrics subcommand smoke test
Command:
```bash
/home/mace/NLP/.venv/bin/python HW1_112550043.py metrics \
  --name singlefile_v2_ctbert \
  --checkpoint artifacts/singlefile_v2_ctbert/best_model.pt \
  --threshold-file artifacts/singlefile_v2_ctbert/thresholds.json \
  --device cuda --batch-size 128 --num-workers 2
```

Output:
- `default_macro_f1`: 0.649861
- `default_micro_f1`: 0.715041
- `tuned_macro_f1`: 0.711452
- `tuned_micro_f1`: 0.759207
