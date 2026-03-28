# HW1 多標籤分類（LSTM 版本）超詳細實作手冊

本文件是專門給 BiLSTM + Multi-Head Attention（以下簡稱 LSTM 版本）的完整操作與維運指南。

你可以把這份文件當成：

1. 從零開始跑出可提交結果的標準作業流程（SOP）。
2. 跟助教、隊友、未來的你溝通實驗設定的單一真相來源（Single Source of Truth）。
3. 日後做模型改版（例如 Transformer、Label-specific attention）的對照基準文件。

---

## 1. 專案目標與適用範圍

### 1.1 任務定義

- 任務：Twitter 推文的多標籤分類（Multi-label classification）。
- 每筆樣本可同時屬於 0 到多個類別。
- 最終輸出需符合 `sample_submission.csv` 的欄位順序與格式。

### 1.2 本文件只處理 LSTM 路徑

本文件聚焦在 `--model-type bilstm`：

1. 資料前處理（產生 preprocessed CSV 與 metadata）。
2. BiLSTM + Multi-Head Attention 訓練。
3. per-label threshold tuning。
4. 推論與 submission 產生。
5. 常見錯誤排查與最佳實踐。

### 1.3 不在本文件主軸內

- Transformer 設定細節（程式支援，但非本指南核心）。
- 大規模超參搜尋框架（Optuna/Ray Tune）。
- 進階資料增強（EDA、back-translation、LLM augmentation）。

---

## 2. 專案結構（與 LSTM 高度相關）

```text
HW1_dataset/
  HW1_112550043.py                  # 前處理主程式（raw json -> preprocessed CSV）
  train.py                          # 訓練主程式（bilstm / transformer）
  predict.py                        # 推論與 submission 生成
  dataset.py                        # Dataset/DataLoader、vocab、embedding matrix
  evaluate.py                       # macro/micro/per-label F1、loss、predict probs
  threshold_tuner.py                # per-label 閾值搜尋
  losses.py                         # BCE / ASL
  models/
    bilstm_attention.py             # BiLSTM + MHA 模型
  preprocessed/
    train_preprocessed.csv
    val_preprocessed.csv
    test_preprocessed.csv
    metadata.json
  artifacts/
    formal_bilstm/                  # 已驗證可用的一組正式輸出
      best_model.pt
      vocab.json
      thresholds.json
      training_history.json
      training_config.json
      run_summary.json
      submission_formal.csv
```

---

## 3. 標籤與資料契約（Data Contract）

### 3.1 原始資料

- `train.json`：訓練資料，含 `tweet` 與 `labels`。
- `val.json`：驗證資料，含 `tweet` 與 `labels`。
- `test.json`：測試資料，不含真值標籤。
- `sample_submission.csv`：提交模板，定義標籤欄順序。

### 3.2 標籤順序（不可改）

必須嚴格採用以下 12 類順序：

1. ineffective
2. unnecessary
3. pharma
4. rushed
5. side-effect
6. mandatory
7. country
8. ingredients
9. political
10. none
11. conspiracy
12. religious

為什麼不能改？

- 訓練時 logits 第 i 維，必須與 submission 第 i 欄同語義。
- 一旦錯位，模型看似有輸出，實際提交語義全錯。

---

## 4. 前處理 SOP（LSTM 前置條件）

LSTM 訓練依賴 `preprocessed/*.csv` 與 `metadata.json`。
如果你尚未產生，先跑：

```bash
python HW1_112550043.py \
  --data-dir . \
  --output-dir preprocessed \
  --model-type rnn \
  --max-length 128 \
  --strict-label-order \
  --rnn-lowercase \
  --rnn-remove-punct
```

> 備註：LSTM 並非一定要用 rnn-cleaning；但請確保「訓練與推論」共用同一組 preprocessed 輸出。

### 4.1 前處理輸出

1. `train_preprocessed.csv`
2. `val_preprocessed.csv`
3. `test_preprocessed.csv`
4. `metadata.json`

### 4.2 metadata.json 在 LSTM 的關鍵用途

- `label_order`：固定分類維度對齊。
- `recommended_bce_pos_weight`：不平衡類別加權。
- `train_stats` / `val_stats`：分析分布漂移。

---

## 5. 模型設計（BiLSTM + Multi-Head Attention）

模型位於 `models/bilstm_attention.py`。

## 5.1 結構分解

1. Embedding Layer
- 形狀：`[B, T] -> [B, T, D]`
- 支援 random init / GloVe init。
- 可用 `freeze_embedding` 鎖住詞向量。

2. BiLSTM Encoder
- 預設雙向（bidirectional=True）。
- 輸出維度：`hidden_size * 2`（雙向時）。

3. Multi-Head Self-Attention
- 對 LSTM 輸出做 self-attention。
- 透過 `attention_mask == 0` 遮蔽 padding token。

4. Masked Mean Pooling
- 只對有效 token 平均，避免 padding 干擾句向量。

5. LayerNorm + Dropout + Linear
- 提升訓練穩定性與泛化。
- 最終輸出 12 維 logits（不在模型內做 sigmoid）。

## 5.2 重要硬限制

`lstm_out_dim % attention_heads == 0`

- 若 `hidden_size=128` 且雙向，則 `lstm_out_dim=256`。
- `attention_heads` 可用 8、4、16（須整除）。

不整除會直接拋錯。

---

## 6. 訓練流程（train.py）

## 6.1 高層流程

1. 載入 metadata / train / val。
2. 建 vocab（從 train 文本）。
3. 建 DataLoader。
4. 建 BiLSTM 模型。
5. 讀取 `pos_weight` 構建 BCE（或 ASL）。
6. epoch 迭代：train -> val metric。
7. scheduler 監控 val macro-F1。
8. early stopping 保存最佳模型。
9. 針對 val logits 做 per-label threshold tuning。
10. 輸出完整 artifacts。

## 6.2 訓練時使用的關鍵機制

1. Optimizer
- `AdamW`

2. LR Scheduler
- `ReduceLROnPlateau`（監控 `val_macro_f1`）

3. Gradient Control
- `grad_accum_steps`：梯度累積
- `clip_grad_norm_`：梯度裁切

4. Mixed Precision
- 只在 CUDA 啟用，CPU 會自動關閉

5. Early Stopping
- `early_stopping_patience` 個 epoch 未進步即停止

---

## 7. LSTM 常用參數與調整策略

## 7.1 架構參數

- `--embedding-dim`：100 / 200 / 300
- `--hidden-size`：128 / 192 / 256
- `--num-layers`：1 或 2
- `--attention-heads`：須整除輸出維度
- `--dropout`：0.2 ~ 0.4

## 7.2 訓練參數

- `--batch-size`：CPU 常見 32/64
- `--epochs`：20~40（配合 early stopping）
- `--learning-rate`：LSTM 常見 1e-3 起手
- `--weight-decay`：常見 1e-4
- `--grad-clip-norm`：常見 1.0

## 7.3 資料與序列參數

- `--max-length`：常見 128
- `--vocab-max-size`：常見 30000
- `--vocab-min-freq`：常見 2

## 7.4 閾值搜尋參數

- `--threshold-start`
- `--threshold-end`
- `--threshold-step`

預設常用範圍：0.1 ~ 0.9，步長 0.05。

---

## 8. 一鍵可重現指令（LSTM 正式版）

## 8.1 環境安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 8.2 訓練（正式設定）

```bash
python train.py \
  --model-type bilstm \
  --data-dir preprocessed \
  --output-dir artifacts/formal_bilstm \
  --epochs 25 \
  --batch-size 64 \
  --max-length 128 \
  --embedding-dim 200 \
  --hidden-size 128 \
  --num-layers 1 \
  --attention-heads 8 \
  --dropout 0.3 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 2 \
  --early-stopping-patience 5 \
  --threshold-start 0.1 \
  --threshold-end 0.9 \
  --threshold-step 0.05
```

## 8.3 推論與提交檔生成

```bash
python predict.py \
  --data-dir preprocessed \
  --checkpoint artifacts/formal_bilstm/best_model.pt \
  --threshold-file artifacts/formal_bilstm/thresholds.json \
  --sample-submission sample_submission.csv \
  --output-file artifacts/formal_bilstm/submission_formal.csv \
  --batch-size 128
```

## 8.4 提交格式驗證

```bash
wc -l sample_submission.csv artifacts/formal_bilstm/submission_formal.csv
python -c "import pandas as pd; s=pd.read_csv('sample_submission.csv',nrows=0).columns.tolist(); p=pd.read_csv('artifacts/formal_bilstm/submission_formal.csv',nrows=0).columns.tolist(); print('header_match', s==p); print('num_cols', len(p));"
```

---

## 9. 已驗證的正式結果（formal_bilstm）

來源：`artifacts/formal_bilstm/run_summary.json`、`thresholds.json`

- best_val_macro_f1: 0.4406936249688968
- default_macro_f1: 0.4406936249688968
- tuned_macro_f1: 0.45908627383721257
- default_micro_f1: 0.5601012231126107
- tuned_micro_f1: 0.5736738703339882

結論：

1. per-label threshold tuning 明顯有效（macro +0.0184）。
2. 對不平衡多標籤任務，不建議直接固定全部閾值為 0.5。

---

## 10. artifacts 輸出檔案說明

1. `best_model.pt`
- 最佳 macro-F1 權重
- 含 `model_state_dict`、`label_order`、`model_args`
- LSTM 會含 `vocab`

2. `vocab.json`
- token -> index 映射
- 推論必須共用同一份 vocab

3. `training_config.json`
- 完整訓練參數快照

4. `training_history.json`
- 每 epoch 指標

5. `thresholds.json`
- 每類別最佳 threshold
- default vs tuned macro/micro 對照

6. `run_summary.json`
- 精簡摘要（方便彙報/比對）

7. `submission_formal.csv`
- 可直接上傳提交

---

## 11. GloVe 初始化（可選）

若你要接 GloVe：

```bash
python train.py \
  --model-type bilstm \
  --data-dir preprocessed \
  --output-dir artifacts/formal_bilstm_glove \
  --glove-path /path/to/glove.twitter.27B.200d.txt \
  --embedding-dim 200
```

注意：

1. `embedding_dim` 必須和 GloVe 檔案維度一致。
2. 路徑錯誤時，程式會 fallback random init 並印 warning。
3. 請做 random vs glove 對照，不要只看單次結果。

---

## 12. 常見錯誤排查

## 12.1 RuntimeError: transformers is required ...

原因：

- 忘記加 `--model-type bilstm`，走到 transformer 預設路徑。

解法：

1. 補上 `--model-type bilstm`。
2. 或你本來就要 transformer，請安裝 `transformers`。

## 12.2 attention heads 整除錯誤

原因：

- `hidden_size * (2 if bidirectional else 1)` 無法被 `attention_heads` 整除。

解法：

- 調整 `hidden_size` 或 `attention_heads`。

## 12.3 submission row count mismatch

原因：

- `test_file` 或 `data_dir` 指到不同版本資料。

解法：

- 確認訓練與推論使用同一套 preprocessed 檔案。

## 12.4 Exit Code 130

原因：

- 通常是 Ctrl+C 或終端中斷。

解法：

1. 重新執行相同指令。
2. 長訓練建議用 `tmux` 或 `nohup`。

---

## 13. 實驗紀錄最佳實踐

每次訓練至少記錄：

1. 資料版本（preprocessed 來源與日期）
2. 模型參數（hidden, heads, dropout）
3. 訓練參數（lr, batch, max_length, patience）
4. 最佳結果（best macro, tuned macro）
5. 輸出檔位置（best_model, thresholds, submission）

---

## 14. 建議的優化順序

1. 先做 GloVe 對照（random vs glove）。
2. 再做小規模超參搜尋（hidden/dropout/lr）。
3. threshold 搜尋步長由 0.05 改 0.02 做精修。
4. 針對弱類（religious/country/ingredients）做誤差分析。

---

## 15. 提交前 30 秒核對清單

1. 有明確指定 `--model-type bilstm`。
2. `best_model.pt` 與 `thresholds.json` 來自同一 output_dir。
3. submission 欄位順序與 `sample_submission.csv` 一致。
4. submission 列數與 test 行數一致。
5. 保留 `run_summary.json` 供追蹤與報告。

---

## 16. 最短可執行版（TL;DR）

```bash
# 1) 訓練
python train.py --model-type bilstm --data-dir preprocessed --output-dir artifacts/formal_bilstm

# 2) 推論
python predict.py --data-dir preprocessed --checkpoint artifacts/formal_bilstm/best_model.pt --threshold-file artifacts/formal_bilstm/thresholds.json --sample-submission sample_submission.csv --output-file artifacts/formal_bilstm/submission_formal.csv

# 3) 格式驗證
wc -l sample_submission.csv artifacts/formal_bilstm/submission_formal.csv
```

只要這三步成功，基本上就能穩定重現一份可提交的 LSTM 結果。
