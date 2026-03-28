# HW1 Dataset Preprocessing Documentation

## 1. 專案目的與範圍

本文件說明 HW1 多標籤推文分類任務的資料前處理（Data Preprocessing）完整流程，重點包含：

- 標籤轉化與二值化（Label Binarization）
- Twitter 文本清洗（Twitter Text Cleaning）
- 不平衡資料處理支援（Addressing Imbalance）
- 特徵工程與 Tokenization 準備
- 可重現執行方式、輸出格式與驗證清單

本流程已實作於 `preprocess_data.py`，並在目前資料上完成一次可重現執行。

---

## 2. 資料集結構（Raw Data Structure）

目前資料夾結構：

```text
HW1_dataset/
  preprocess_data.py
  sample_submission.csv
  train.json
  val.json
  test.json
  preprocessed/
    metadata.json
    train_preprocessed.csv
    val_preprocessed.csv
    test_preprocessed.csv
```

### 2.1 原始資料檔案角色

| 檔案 | 角色 | 是否含 labels | 備註 |
|---|---|---|---|
| train.json | 訓練資料 | 是 | 每筆為推文與巢狀標籤字典 |
| val.json | 驗證資料 | 是 | 結構同 train |
| test.json | 測試資料 | 否 | 僅有 ID 與 tweet |
| sample_submission.csv | 提交模板 | N/A | 定義 index 與 12 個標籤欄位順序 |

### 2.2 JSON 結構範例

#### train / val 單筆格式

```json
{
  "ID": "1311981051720409089",
  "tweet": "@user ...",
  "labels": {
    "ineffective": [
      {
        "index": 0,
        "start": 6,
        "end": 10,
        "terms": "cant control the Flu"
      }
    ]
  }
}
```

#### test 單筆格式

```json
{
  "ID": 0,
  "tweet": "Confusion is Failure ..."
}
```

關鍵說明：
- `labels` 為巢狀字典，key 是類別名稱。
- `terms/start/end/index` 是 evidence span，分類訓練通常只取 key 是否存在，不直接用 span 做 supervision。

---

## 3. 全域標籤定義（必須固定順序）

標籤順序嚴格遵循 `sample_submission.csv`：

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

程式內建 `EXPECTED_LABEL_ORDER`，並支援 `--strict-label-order` 強制檢查。

---

## 4. 已實作前處理流程（SOP 對應）

## 4.1 階段一：標籤轉化與二值化（Label Binarization）

### 實作方式

- 先讀取 `sample_submission.csv` 表頭，取出 12 類 label。
- 對每筆 train/val：
  - 取 `labels` 字典的 key 集合。
  - 依固定 label order 轉為 multi-hot 向量（有該 key=1，否則=0）。
- 對 test：
  - 因無 gold labels，輸出全 0 佔位（僅為維度一致與推論管線便利）。

### 一致性檢查

- 檢查 train/val/test 每筆標籤維度皆為 12。
- 若計入 index 欄，輸出 CSV 對訓練可視為 13 欄（index + 12 labels，不含其他文字欄位時）。
- 檢查未知標籤 key（non-schema labels）並寫入 metadata。

---

## 4.2 階段二：文本清洗（Twitter Text Cleaning）

### 基礎清洗規則

- URL：
  - `token`（預設）：替換為 `[URL]`
  - `remove`：刪除
  - `keep`：保留原文
- User Handle（@user）：
  - `token`（預設）：替換為 `[USER]`
  - `remove`：刪除
  - `keep`：保留
- Emoji：
  - `demojize`（預設）：轉文字描述（需 `emoji` 套件）
  - `remove`：移除
  - `keep`：保留

### 模型差異化策略

#### Transformer 模式（預設）

- 原則：避免過度清洗，盡量保留語境。
- 動作：URL/User/Emoji 規則 + 空白正規化 + token-based truncation。

#### RNN/LSTM 模式

- 可額外啟用：
  - `--rnn-lowercase`
  - `--rnn-remove-punct`
  - `--rnn-remove-stopwords`
- 適用於傳統詞向量與序列模型管線。

### Truncation

- 以簡易 token split 後截斷至 `max_length`（預設 128）。

---

## 4.3 階段三：不平衡處理（Addressing Imbalance）

### 已實作內容

- 產出每一類別：
  - positive / negative 數量
  - positive_rate
  - 建議 `pos_weight = negative / positive`（供 `BCEWithLogitsLoss` 使用）
- 產出 train vs val 的 label rate drift（比例偏移）

### 注意事項

- 目前資料已提供 train/val 切分，腳本不會重切。
- 若你未來要重新 split，建議使用多標籤分層方法（例如 iterative stratification）維持 rare class 覆蓋。

---

## 4.4 階段四：特徵工程與 Tokenization

### CSV 輸出（預設）

每筆保留：

- 原始 tweet
- 清洗後 tweet_clean
- token 長度欄位
- 12 維標籤欄

### 可選 token ids 匯出（`--export-token-ids`）

- `model-type=transformer`
  - 使用 Hugging Face `AutoTokenizer`
  - 輸出 `input_ids`, `attention_mask`
- `model-type=rnn`
  - 以 train 建 vocab（含 `[PAD]`, `[UNK]`）
  - 輸出固定長度 `input_ids`

---

## 5. 執行方式（Reproducible Commands）

## 5.1 環境需求

- Python 3.10+
- 建議安裝套件：
  - `emoji`（支援 demojize）
  - `transformers`（若需 transformer token ids）

安裝範例：

```bash
pip install emoji
pip install transformers
```

## 5.2 基本執行（Transformer 友善）

```bash
python preprocess_data.py \
  --data-dir . \
  --output-dir preprocessed \
  --model-type transformer \
  --max-length 128 \
  --strict-label-order
```

## 5.3 RNN 模式（較徹底清洗）

```bash
python preprocess_data.py \
  --data-dir . \
  --output-dir preprocessed_rnn \
  --model-type rnn \
  --max-length 128 \
  --rnn-lowercase \
  --rnn-remove-punct \
  --rnn-remove-stopwords \
  --strict-label-order
```

## 5.4 匯出 token ids

```bash
python preprocess_data.py \
  --data-dir . \
  --output-dir preprocessed_tok \
  --model-type transformer \
  --export-token-ids \
  --tokenizer-name bert-base-uncased \
  --max-length 128 \
  --strict-label-order
```

---

## 6. 輸出檔案規格

## 6.1 train_preprocessed.csv / val_preprocessed.csv / test_preprocessed.csv

主要欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| index | int | split 內序號（0-based） |
| ID | str/int | 原始資料 ID |
| tweet | str | 原始推文 |
| tweet_clean | str | 清洗後推文 |
| tweet_raw_len | int | 原始 token 數 |
| tweet_clean_len | int | 清洗後 token 數 |
| split | str | train / val / test |
| 12 labels | int(0/1) | multi-hot 標籤 |

test 資料的 12 labels 為 0 佔位，不可視為真值。

## 6.2 metadata.json

包含：

- `label_order`
- `shape_check`
- `unknown_labels`
- `train_stats`, `val_stats`
- `train_val_rate_drift`
- `recommended_bce_pos_weight`
- `config`（本次執行參數）

## 6.3 readable/（快速可讀化檢視）

為了讓資料能「一目瞭然」，已額外產出 `readable/`：

- `train_preprocessed_readable.md`
- `val_preprocessed_readable.md`
- `test_preprocessed_readable.md`
- `sample_submission_readable.md`
- `train_preprocessed_view.tsv`
- `val_preprocessed_view.tsv`
- `test_preprocessed_view.tsv`

兩種檔案用途：

- `*_readable.md`：摘要報表（總列數、欄位、標籤分布、前 20 筆預覽）
- `*_view.tsv`：扁平化檢視（把 12 個 one-hot 標籤整併成單一 `label` 欄）

建議閱讀順序：

1. 先看 `train_preprocessed_readable.md` 掌握分布與欄位
2. 再看 `train_preprocessed_view.tsv` 逐列掃描內容

---

## 7. 目前資料統計摘要（本次執行結果）

來源：`preprocessed/metadata.json`

### 7.1 基本樣本數

- train: 6956
- val: 987
- test: 1976

### 7.2 標籤維度一致性

- train_dim = 12
- val_dim = 12
- test_dim = 12
- unknown_labels: train={}, val={}

### 7.3 標籤不平衡概況（Train）

| Label | Positive | Positive Rate | Recommended pos_weight |
|---|---:|---:|---:|
| side-effect | 2663 | 38.28% | 1.61 |
| ineffective | 1171 | 16.83% | 4.94 |
| rushed | 1031 | 14.82% | 5.75 |
| pharma | 889 | 12.78% | 6.82 |
| mandatory | 548 | 7.88% | 11.69 |
| unnecessary | 503 | 7.23% | 12.83 |
| none | 440 | 6.33% | 14.81 |
| political | 437 | 6.28% | 14.92 |
| conspiracy | 341 | 4.90% | 19.40 |
| ingredients | 304 | 4.37% | 21.88 |
| country | 140 | 2.01% | 48.69 |
| religious | 45 | 0.65% | 153.58 |

關鍵觀察：
- `religious` 極度稀少，Macro-F1 最容易在此類別受損。
- `side-effect` 佔比最高，若不加權可能導致模型偏向 head class。

### 7.4 Train / Val 比例穩定度

- 各類別 train 與 val 的 positive rate 差異約在 0.0001 到 0.0012。
- 代表現有切分分佈相當一致，驗證集具代表性。

---

## 8. 訓練端整合建議

## 8.1 BCEWithLogitsLoss 權重載入

範例（PyTorch）：

```python
import json
import torch

with open("preprocessed/metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

label_order = meta["label_order"]
pos_weight = [meta["recommended_bce_pos_weight"][k] for k in label_order]
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

最佳實踐：
- 若某類別訓練集中 `positive=0`，`pos_weight` 會是 null，此時需手動處理（例如設為 1.0 或暫停該類別訓練）。
- 訓練時同步監控 macro-F1 與每類別 F1，而不是只看 micro 指標。

## 8.2 Transformer 建議

- 優先採用本腳本預設（最小清洗）。
- max_length 可先用 128，若觀察截斷過多再升至 256。

## 8.3 RNN/LSTM 建議

- 啟用 lowercase / punct removal / stopwords removal 做 baseline。
- 搭配預訓練詞向量（如 GloVe）時，需確認 vocab/token 正規化一致。

---

## 9. 提交格式與作業檢查清單

## 9.1 提交 CSV 檢查

- 欄位順序必須與 `sample_submission.csv` 完全一致：
  - `index` + 12 labels（同順序）
- 不能更動欄位名稱大小寫。

## 9.2 模型與規範檢查

- 確認總參數量小於 1B。
- 禁止使用外部 LLM API（例如 GPT-4 API）直接做推論。
- 報告需量化說明 preprocessing 對 Macro-F1 的影響。

## 9.3 實驗紀錄最佳實踐

每次實驗建議記錄：
- 清洗策略（transformer/rnn 與各 flag）
- max_length
- loss 設定（是否加權）
- 評估指標（macro-F1 / per-label F1）
- 錯誤分析（尤其 rare labels）

---

## 10. 常見問題（FAQ）

### Q1. 為什麼 test 也有 12 個標籤欄且都是 0？
A: 這是為了保持資料格式一致，方便同一套 dataloader 與特徵流程處理；test 的標籤不是 ground truth。

### Q2. labels 裡的 terms 為什麼沒用？
A: 目前任務是 tweet-level multi-label classification，預設 supervision 為 label key 存在與否。

### Q3. 何時需要 `--export-token-ids`？
A: 當你希望前處理階段就固定 tokenization 輸出，減少訓練時重複計算；若要動態 augmentation，可在訓練時再 tokenize。

### Q4. 為什麼不直接做資料增強？
A: 多標籤增強若未人工檢查，容易引入語義漂移與標籤污染。建議先做 class weighting baseline，再針對少數類別做受控增強。

---

## 11. 參考資料（Authoritative References）

- PyTorch 官方文件：`torch.nn.BCEWithLogitsLoss`（pos_weight 定義）
- Hugging Face Transformers 官方文件：`AutoTokenizer`
- Devlin et al., 2019, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> 註：關於「是否過度清洗會傷害 Transformer 表現」在不同語料上可能有差異，建議透過 ablation study（最小清洗 vs 強清洗）在本任務實測確認。

---

## 12. 快速開始（TL;DR）

1. 先跑：

```bash
python preprocess_data.py --data-dir . --output-dir preprocessed --model-type transformer --max-length 128 --strict-label-order
```

2. 使用：
- `preprocessed/train_preprocessed.csv`
- `preprocessed/val_preprocessed.csv`
- `preprocessed/metadata.json` 的 `recommended_bce_pos_weight`

3. 訓練時優先看：
- Macro-F1
- rare class（尤其 religious）F1

4. 送出前確認：
- submission 欄位順序 100% 對齊 `sample_submission.csv`
