# HW1 BERT 版本超詳細實作手冊

本文件是 HW1 多標籤推文分類任務的 BERT 導向完整操作指南。
目標是讓你可以從資料檢查、模型訓練、threshold tuning 到提交檔產生，全流程一次打通，且每個步驟都可重現。

更新日期：2026-03-28

---

## 1. 任務定義與評分重點

### 1.1 任務類型

- 任務：Tweet-level Multi-label Classification
- 每筆推文可同時屬於多個標籤
- 標籤總數：12

標籤固定順序：

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

### 1.2 評分核心

- 主要指標：Macro-F1
- 輔助指標：Micro-F1、Per-label F1

Macro-F1 會平均每個標籤的 F1，因此稀有標籤（如 religious）是否預測成功，對最終分數影響很大。

### 1.3 作業約束

- 模型參數量需小於 1B
- 不使用外部 LLM API 直接推論
- 提交欄位順序必須與 sample_submission.csv 完全一致

---

## 2. 專案架構總覽（目前版本）

目前專案已具備兩條訓練路線：

- bilstm 路線（對照 baseline）
- transformer 路線（本 README 主軸）

主要檔案職責：

- HW1_112550043.py
  - 前處理：JSON 轉 CSV、清洗、標籤二值化、統計與 metadata 產生
- dataset.py
  - DataLoader 與 Dataset
  - 支援 vocab-based（bilstm）與 tokenizer-based（transformer）
- models/transformer_multilabel.py
  - Transformer backbone
  - Multi-sample Dropout
  - Linear head / Label Attention head
- losses.py
  - Asymmetric Loss（ASL）
- evaluate.py
  - loss、macro/micro F1、per-label F1、機率輸出
- threshold_tuner.py
  - per-label threshold 搜尋
- train.py
  - 訓練主流程、驗證、checkpoint、threshold tuning
- predict.py
  - 載入最佳 checkpoint + thresholds
  - 產生 submission.csv

---

## 3. 資料與標籤統計（依 preprocessed/metadata.json）

### 3.1 資料量

- train: 6956
- val: 987
- test: 1976

### 3.2 不平衡特徵

- 最多：side-effect，約 38.28%
- 最少：religious，約 0.65%

結論：

- 單純使用統一 0.5 threshold 常會壓死稀有標籤
- 損失函數與 threshold tuning 對分數提升是必要而非可選

### 3.3 metadata 已可直接利用

preprocessed/metadata.json 已提供：

- label_order
- 各類別 pos_weight（recommended_bce_pos_weight）
- train/val rate drift

這些資訊已被 train.py 直接讀取並套用。

---

## 4. BERT 模型設計細節

Transformer 分支由 AutoModel 載入 backbone，支援 CT-BERT、BERTweet、RoBERTa 類模型。

### 4.1 Backbone

- 輸入：input_ids、attention_mask
- 輸出：last_hidden_state，形狀 [batch, seq_len, hidden]

### 4.2 Pooling

可選：

- cls：取第 1 個 token 表徵
- mean：attention mask 加權平均（忽略 padding）

### 4.3 Head 類型

#### A. linear

- pooled 向量經過多個 dropout 分支
- 每個分支共用同一個 Linear classifier
- 對 logits 做平均

#### B. label_attention

- 為每個標籤建立可學習 label embedding
- 計算 token 與 label embedding 的相似度分數
- 對 token 做 softmax attention，得到每個標籤自己的語意表徵
- 與 pooled 向量融合後輸出每標籤 logit

### 4.4 Multi-sample Dropout

- 透過 multi_sample_dropout 控制分支數（建議 5 到 8）
- 分支越多通常越穩，但訓練時間上升

---

## 5. 損失函數：BCE vs ASL

### 5.1 BCEWithLogitsLoss

- 支援 pos_weight
- 適合作為穩定 baseline
- 參數：--loss-type bce

### 5.2 Asymmetric Loss（ASL）

- 重點處理負樣本過多問題
- 主要參數：
  - --asl-gamma-neg（預設 4.0）
  - --asl-gamma-pos（預設 1.0）
  - --asl-clip（預設 0.05）

ASL 常見優勢：

- 提升 rare labels recall
- 在多標籤且負樣本遠多於正樣本時，通常優於未加權 BCE

注意：

- ASL 也可能造成 precision 下滑，需要配合 threshold tuning 檢查

---

## 6. 訓練流程（train.py）逐步說明

每次執行 train.py，會依序進行：

1. 讀取 CLI 參數
2. 設定 seed（random、numpy、torch）
3. 判斷裝置（CUDA 優先，否則 CPU）
4. 讀 metadata 與 train/val CSV
5. 建立 model + dataloader（依 model-type）
6. 建立 criterion（BCE 或 ASL）
7. 進入 epoch 訓練
8. 每個 epoch 後做 val 評估
9. 以 val Macro-F1 保存 best checkpoint
10. 訓練結束後，對 val logits 做 per-label threshold 搜尋
11. 輸出 run_summary、thresholds、history、checkpoint

### 6.1 重要輸出檔

在 output-dir 下會得到：

- best_model.pt
- training_history.json
- training_config.json
- thresholds.json
- run_summary.json

若是 bilstm，另外有 vocab.json；transformer 不需要。

---

## 7. Threshold Tuning 策略

實作位置：threshold_tuner.py

邏輯：

- 針對每個標籤獨立搜尋 threshold
- 預設搜尋範圍：0.1 到 0.9
- 預設步長：0.05

建議：

- 第一輪可用 0.05 快速掃描
- 第二輪可對稀有標籤改成更細（例如步長 0.01）

---

## 8. 實戰 SOP（CT-BERT 正式訓練）

### 8.1 環境安裝

~~~bash
cd /home/mace/College/kaggle/HW1_dataset
/home/mace/College/kaggle/.venv/bin/pip install -r requirements.txt
~~~

### 8.2 確認資料齊全

~~~bash
ls preprocessed
~~~

至少要有：

- train_preprocessed.csv
- val_preprocessed.csv
- test_preprocessed.csv
- metadata.json

### 8.3 先做 tiny smoke test（可選，但強烈建議）

~~~bash
/home/mace/College/kaggle/.venv/bin/python train.py \
  --model-type transformer \
  --pretrained-model hf-internal-testing/tiny-random-roberta \
  --epochs 1 \
  --batch-size 64 \
  --max-length 64 \
  --learning-rate 2e-4 \
  --output-dir artifacts/smoke_transformer_tiny
~~~

### 8.4 CT-BERT 第一輪正式設定（推薦）

~~~bash
/home/mace/College/kaggle/.venv/bin/python train.py \
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
  --epochs 6 \
  --max-length 128 \
  --dropout 0.2 \
  --num-workers 2 \
  --threshold-start 0.05 \
  --threshold-end 0.95 \
  --threshold-step 0.05 \
  --output-dir artifacts/ctbert_round1_asl
~~~

### 8.5 產生提交檔

~~~bash
/home/mace/College/kaggle/.venv/bin/python predict.py \
  --checkpoint artifacts/ctbert_round1_asl/best_model.pt \
  --threshold-file artifacts/ctbert_round1_asl/thresholds.json \
  --sample-submission sample_submission.csv \
  --output-file artifacts/ctbert_round1_asl/submission.csv
~~~

---

## 9. 參數調校地圖（超詳細）

### 9.1 第一優先調參

- learning-rate（2e-5、3e-5、5e-5）
- batch-size + grad-accum-steps（保持有效 batch）
- loss-type（bce vs asl）
- multi-sample-dropout（5/6/8）

### 9.2 第二優先調參

- head-type（linear vs label_attention）
- pooling（cls vs mean）
- threshold 搜尋範圍與步長

### 9.3 高風險參數

- learning-rate 過高：容易梯度爆炸或過早震盪
- max-length 過大：記憶體壓力上升
- ASL gamma_neg 過大：可能過度懲罰負樣本，precision 波動

### 9.4 建議實驗記錄欄位

每次實驗至少記錄：

- run_id
- backbone
- head_type
- loss_type
- lr / batch / accum
- best val macro_f1
- tuned macro_f1
- per-label F1（特別標註 country、religious）

---

## 10. 資源與硬體建議

### 10.1 GPU 可用時

- 優先用 CUDA
- 建議 batch-size 8 到 16
- grad-accum 調整到有效 batch 32 到 64

### 10.2 僅 CPU 時

- 可以訓練，但速度慢很多
- 建議先做小 epoch、小模型 smoke test
- 正式實驗盡量縮小搜索空間

### 10.3 OOM 處理順序

1. 降 batch-size
2. 提高 grad-accum-steps
3. 降 max-length（例如 256 降到 128）
4. 暫時改用較小 backbone

---

## 11. 常見錯誤與排查

### 11.1 缺 transformers

症狀：

- RuntimeError: transformers is required...

處理：

~~~bash
/home/mace/College/kaggle/.venv/bin/pip install transformers
~~~

### 11.2 標籤順序錯誤

症狀：

- submission 上傳失敗或分數異常低

處理：

- 一律以 sample_submission.csv 的欄位順序為最終準則

### 11.3 訓練中斷（Exit Code 130）

常見原因：

- 手動中止
- shell 中斷

處理：

- 重新執行同一組命令
- 確認 output-dir 有無殘留檔案可續用（目前腳本是重跑訓練，不是自動 resume）

### 11.4 Hugging Face 下載限速

症狀：

- 模型下載慢、warning 提示 unauthenticated

處理：

- 設定 HF_TOKEN 可提高速率（選配）

---

## 12. 提交前最終檢查清單

1. 已完成訓練並有 best_model.pt
2. 已產生 thresholds.json
3. predict.py 已成功產生 submission.csv
4. submission.csv 列數等於 test 筆數
5. submission.csv 欄位名與順序完全對齊 sample_submission.csv
6. 值域為 0/1，無 NaN

---

## 13. 進階衝榜建議（第二輪起）

1. 對稀有標籤做更細 threshold 搜尋（0.01 步長）
2. 比較 linear head 與 label_attention head
3. 進行 3-seed 訓練並做 logit averaging
4. 比較 CT-BERT 與 BERTweet backbone，再做小型 ensemble

---

## 14. 一頁式快速指令（最短路徑）

~~~bash
# 1) train
/home/mace/College/kaggle/.venv/bin/python train.py \
  --model-type transformer \
  --pretrained-model digitalepidemiologylab/covid-twitter-bert-v2 \
  --head-type linear \
  --multi-sample-dropout 6 \
  --loss-type asl \
  --asl-use-pos-weight \
  --learning-rate 2e-5 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --epochs 6 \
  --max-length 128 \
  --output-dir artifacts/ctbert_round1_asl

# 2) predict
/home/mace/College/kaggle/.venv/bin/python predict.py \
  --checkpoint artifacts/ctbert_round1_asl/best_model.pt \
  --threshold-file artifacts/ctbert_round1_asl/thresholds.json \
  --output-file artifacts/ctbert_round1_asl/submission.csv
~~~

---

## 15. 不確定性與風險聲明

- 不同機器的 CUDA、driver、記憶體條件會影響可用 batch-size 與訓練時間。
- 不同 backbone 的 tokenizer 與 vocabulary 分布不同，最佳超參數不一定可直接互換。
- 若 leaderboard 與 val 分數有落差，優先檢查 threshold 過擬合與資料分佈偏差。
