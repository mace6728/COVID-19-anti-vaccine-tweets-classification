#!/usr/bin/env python3
"""Unified HW1 pipeline in a single file.

Subcommands:
- preprocess: build train/val/test preprocessed CSV + metadata
- train: train BiLSTM or Transformer multi-label classifier
- predict: run inference and create submission CSV

For backward compatibility, running this script without a subcommand defaults to
`preprocess`, and legacy preprocess flags still work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None


DEFAULT_LABEL_ORDER = [
    "ineffective",
    "unnecessary",
    "pharma",
    "rushed",
    "side-effect",
    "mandatory",
    "country",
    "ingredients",
    "political",
    "none",
    "conspiracy",
    "religious",
]
EXPECTED_LABEL_ORDER = list(DEFAULT_LABEL_ORDER)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_INDEX = 0
UNK_INDEX = 1

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9\s\[\]#_]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

RNN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

_EMOJI_LIB = None
_EMOJI_IMPORT_ATTEMPTED = False
_EMOJI_WARNED = False


# =====================
# Config dataclasses
# =====================


@dataclass
class DataConfig:
    data_dir: Path = Path("preprocessed")
    train_csv: str = "train_preprocessed.csv"
    val_csv: str = "val_preprocessed.csv"
    test_csv: str = "test_preprocessed.csv"
    metadata_json: str = "metadata.json"
    sample_submission_csv: Path = Path("sample_submission.csv")
    text_column: str = "tweet_clean"
    max_length: int = 128
    vocab_max_size: int = 30000
    vocab_min_freq: int = 2


@dataclass
class ModelConfig:
    embedding_dim: int = 200
    hidden_size: int = 128
    num_layers: int = 1
    attention_heads: int = 8
    dropout: float = 0.3
    layer_norm: bool = True
    bidirectional: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6
    threshold_search_start: float = 0.1
    threshold_search_end: float = 0.9
    threshold_search_step: float = 0.05


@dataclass
class EmbeddingConfig:
    glove_path: Optional[Path] = None
    freeze_embedding: bool = False


@dataclass
class RuntimeConfig:
    output_dir: Path = Path("artifacts")
    device: Optional[str] = None


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class PreprocessConfig:
    data_dir: Path
    output_dir: Path
    model_type: str
    max_length: int
    url_mode: str
    user_mode: str
    emoji_mode: str
    url_token: str
    user_token: str
    rnn_lowercase: bool
    rnn_remove_punct: bool
    rnn_remove_stopwords: bool
    strict_label_order: bool
    export_token_ids: bool
    tokenizer_name: str
    vocab_size: int
    min_freq: int


# =====================
# Models
# =====================


class BiLSTMMultiHeadAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_labels: int,
        num_layers: int = 1,
        attention_heads: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_layer_norm: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != self.embedding.weight.shape:
                raise ValueError(
                    "Pretrained embedding shape mismatch. "
                    f"Expected {self.embedding.weight.shape}, got {pretrained_embeddings.shape}"
                )
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embedding

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        if lstm_out_dim % attention_heads != 0:
            raise ValueError(
                f"LSTM output dim {lstm_out_dim} must be divisible by attention_heads {attention_heads}"
            )

        self.mha = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(lstm_out_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        lstm_outputs, _ = self.lstm(embeddings)

        key_padding_mask = attention_mask == 0
        attn_outputs, _ = self.mha(
            lstm_outputs,
            lstm_outputs,
            lstm_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        mask = attention_mask.unsqueeze(-1)
        masked_attn = attn_outputs * mask
        pooled = masked_attn.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class TransformerMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        dropout: float = 0.2,
        multi_sample_dropout: int = 5,
        head_type: str = "linear",
        pooling: str = "cls",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if AutoModel is None:
            raise RuntimeError(
                "transformers is required for transformer model. "
                "Install it with: pip install transformers"
            )

        if multi_sample_dropout < 1:
            raise ValueError("multi_sample_dropout must be >= 1")
        if head_type not in {"linear", "label_attention"}:
            raise ValueError("head_type must be one of: linear, label_attention")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be one of: cls, mean")

        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.num_labels = num_labels
        self.head_type = head_type
        self.pooling = pooling

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(multi_sample_dropout)])

        if self.head_type == "linear":
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        else:
            self.label_embedding = nn.Parameter(torch.empty(num_labels, self.hidden_size))
            nn.init.xavier_uniform_(self.label_embedding)
            self.label_classifier = nn.Linear(self.hidden_size, 1)

    def _pool_sequence(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        mask = attention_mask.unsqueeze(-1)
        masked = hidden_states * mask
        return masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

    def _forward_linear_head(self, pooled_output: torch.Tensor) -> torch.Tensor:
        logits_list = [self.classifier(dropout(pooled_output)) for dropout in self.dropouts]
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def _forward_label_attention_head(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.einsum("bsh,lh->bsl", sequence_output, self.label_embedding)
        score = score / math.sqrt(self.hidden_size)

        mask = attention_mask.unsqueeze(-1)
        score = score.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(score, dim=1)

        label_repr = torch.einsum("bsh,bsl->blh", sequence_output, attn_weights)
        label_repr = label_repr + pooled_output.unsqueeze(1)

        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(label_repr)
            logits_list.append(self.label_classifier(dropped).squeeze(-1))
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = backbone_outputs.last_hidden_state
        pooled_output = self._pool_sequence(sequence_output, attention_mask)

        if self.head_type == "linear":
            return self._forward_linear_head(pooled_output)

        return self._forward_label_attention_head(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            attention_mask=attention_mask,
        )


# =====================
# Dataset and loaders
# =====================


@dataclass
class VocabBuildConfig:
    max_size: int = 30000
    min_freq: int = 2


class TweetMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        text_column: str,
        label_order: Sequence[str],
        vocab: Dict[str, int],
        max_length: int,
        with_labels: bool,
    ) -> None:
        self.df = data_frame.reset_index(drop=True)
        self.text_column = text_column
        self.label_order = list(label_order)
        self.vocab = vocab
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        token_ids, attention_mask = encode_text_to_ids(text, self.vocab, self.max_length)

        item: Dict[str, torch.Tensor | int | str] = {
            "index": int(row["index"]),
            "ID": str(row["ID"]),
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
        }

        if self.with_labels:
            labels = np.array([int(row[label]) for label in self.label_order], dtype=np.float32)
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item


class TransformerTweetMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        text_column: str,
        label_order: Sequence[str],
        tokenizer_name: str,
        max_length: int,
        with_labels: bool,
    ) -> None:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for transformer datasets. "
                "Install it with: pip install transformers"
            )

        self.df = data_frame.reset_index(drop=True)
        self.text_column = text_column
        self.label_order = list(label_order)
        self.max_length = max_length
        self.with_labels = with_labels

        # Some models (for example BERTweet) may not have a fast tokenizer.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        text = str(row[self.text_column])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor | int | str] = {
            "index": int(row["index"]),
            "ID": str(row["ID"]),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0).float(),
        }

        if self.with_labels:
            labels = np.array([int(row[label]) for label in self.label_order], dtype=np.float32)
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(metadata_path: Path) -> Dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")
    return pd.read_csv(csv_path)


def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()


def build_vocab(texts: Iterable[str], config: VocabBuildConfig) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(str(text)))

    vocab: Dict[str, int] = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    for token, freq in counter.most_common():
        if freq < config.min_freq:
            continue
        if len(vocab) >= config.max_size:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text_to_ids(text: str, vocab: Dict[str, int], max_length: int) -> Tuple[List[int], List[int]]:
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(tok, UNK_INDEX) for tok in tokens][:max_length]
    attention_mask = [1] * len(token_ids)

    pad_len = max_length - len(token_ids)
    if pad_len > 0:
        token_ids.extend([PAD_INDEX] * pad_len)
        attention_mask.extend([0] * pad_len)

    return token_ids, attention_mask


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_order: Sequence[str],
    text_column: str,
    vocab: Dict[str, int],
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TweetMultiLabelDataset(
        data_frame=train_df,
        text_column=text_column,
        label_order=label_order,
        vocab=vocab,
        max_length=max_length,
        with_labels=True,
    )
    val_dataset = TweetMultiLabelDataset(
        data_frame=val_df,
        text_column=text_column,
        label_order=label_order,
        vocab=vocab,
        max_length=max_length,
        with_labels=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def create_test_loader(
    test_df: pd.DataFrame,
    text_column: str,
    label_order: Sequence[str],
    vocab: Dict[str, int],
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    test_dataset = TweetMultiLabelDataset(
        data_frame=test_df,
        text_column=text_column,
        label_order=label_order,
        vocab=vocab,
        max_length=max_length,
        with_labels=False,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def create_transformer_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_order: Sequence[str],
    text_column: str,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TransformerTweetMultiLabelDataset(
        data_frame=train_df,
        text_column=text_column,
        label_order=label_order,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        with_labels=True,
    )
    val_dataset = TransformerTweetMultiLabelDataset(
        data_frame=val_df,
        text_column=text_column,
        label_order=label_order,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        with_labels=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def create_transformer_test_loader(
    test_df: pd.DataFrame,
    text_column: str,
    label_order: Sequence[str],
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    test_dataset = TransformerTweetMultiLabelDataset(
        data_frame=test_df,
        text_column=text_column,
        label_order=label_order,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        with_labels=False,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_embedding_matrix(
    vocab: Dict[str, int],
    embedding_dim: int,
    glove_path: Optional[Path],
) -> np.ndarray:
    matrix = np.random.normal(loc=0.0, scale=0.02, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[PAD_INDEX] = 0.0

    if glove_path is None:
        print("[WARN] No GloVe path provided. Use random embedding initialization.")
        return matrix

    if not glove_path.exists():
        print(f"[WARN] GloVe file does not exist: {glove_path}. Use random initialization.")
        return matrix

    loaded = 0
    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            if token in vocab:
                matrix[vocab[token]] = np.asarray(parts[1:], dtype=np.float32)
                loaded += 1

    coverage = loaded / max(1, len(vocab))
    print(f"[INFO] GloVe coverage: {loaded}/{len(vocab)} ({coverage:.2%})")
    return matrix


# =====================
# Evaluation and tuning
# =====================


@torch.no_grad()
def collect_logits_and_labels(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_batches = []
    label_batches = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_batches.append(logits.detach().cpu().numpy())
        label_batches.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(logits_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)
    return all_logits, all_labels


def evaluate_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str],
    thresholds: Sequence[float] | None = None,
) -> Dict[str, object]:
    probs = 1 / (1 + np.exp(-logits))

    if thresholds is None:
        thresholds = [0.5] * probs.shape[1]

    threshold_array = np.asarray(thresholds).reshape(1, -1)
    preds = (probs >= threshold_array).astype(np.int32)
    y_true = labels.astype(np.int32)

    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)

    per_label_f1: Dict[str, float] = {}
    for i, label in enumerate(label_order):
        per_label_f1[label] = float(f1_score(y_true[:, i], preds[:, i], zero_division=0))

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_label_f1": per_label_f1,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_order: Sequence[str],
    thresholds: Sequence[float] | None = None,
) -> Dict[str, object]:
    model.eval()

    total_loss = 0.0
    total_count = 0
    logits_batches = []
    label_batches = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        logits_batches.append(logits.detach().cpu().numpy())
        label_batches.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(logits_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)

    metrics = evaluate_from_logits(
        logits=all_logits,
        labels=all_labels,
        label_order=label_order,
        thresholds=thresholds,
    )
    metrics["loss"] = float(total_loss / max(1, total_count))
    return metrics


def predict_probabilities(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    prob_batches: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            prob_batches.append(probs.detach().cpu().numpy())

    return np.concatenate(prob_batches, axis=0)


def tune_per_label_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str],
    start: float = 0.1,
    end: float = 0.9,
    step: float = 0.05,
) -> Dict[str, float]:
    thresholds = np.arange(start, end + 1e-8, step)
    best_thresholds: Dict[str, float] = {}

    for i, label in enumerate(label_order):
        y_true = labels[:, i].astype(np.int32)
        y_prob = probabilities[:, i]

        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(np.int32)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        best_thresholds[label] = best_threshold

    return best_thresholds


def thresholds_to_list(best_thresholds: Dict[str, float], label_order: Sequence[str]) -> List[float]:
    return [float(best_thresholds[label]) for label in label_order]


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: Literal["mean", "sum", "none"] = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        prob_pos = probs
        prob_neg = 1.0 - probs

        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(prob_pos.clamp(min=self.eps))
        log_neg = torch.log(prob_neg.clamp(min=self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        pt = targets * prob_pos + (1.0 - targets) * prob_neg
        gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
        one_sided_weight = torch.pow(1.0 - pt, gamma)
        loss = loss * one_sided_weight

        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1.0 - targets)
            loss = loss * class_weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =====================
# Preprocess utilities
# =====================


def get_label_order(metadata_label_order: List[str] | None) -> List[str]:
    if metadata_label_order is None:
        return DEFAULT_LABEL_ORDER
    if len(metadata_label_order) != len(DEFAULT_LABEL_ORDER):
        raise ValueError(
            f"Expected {len(DEFAULT_LABEL_ORDER)} labels, got {len(metadata_label_order)}"
        )
    return metadata_label_order


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def get_label_order_from_sample(sample_csv: Path) -> List[str]:
    with sample_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if not header or header[0] != "index":
        raise ValueError("sample_submission.csv must start with 'index' column")
    labels = header[1:]
    if len(labels) != len(EXPECTED_LABEL_ORDER):
        raise ValueError(f"Expected 12 labels, found {len(labels)} labels")
    return labels


def validate_label_order(order: Sequence[str], strict: bool) -> None:
    if list(order) == EXPECTED_LABEL_ORDER:
        return
    msg = (
        "Label order mismatch.\n"
        f"Found:    {list(order)}\n"
        f"Expected: {EXPECTED_LABEL_ORDER}"
    )
    if strict:
        raise ValueError(msg)
    print(f"[WARN] {msg}")


def apply_url_rule(text: str, mode: str, token: str) -> str:
    if mode == "keep":
        return text
    if mode == "remove":
        return URL_RE.sub(" ", text)
    return URL_RE.sub(f" {token} ", text)


def apply_user_rule(text: str, mode: str, token: str) -> str:
    if mode == "keep":
        return text
    if mode == "remove":
        return USER_RE.sub(" ", text)
    return USER_RE.sub(f" {token} ", text)


def apply_emoji_rule(text: str, mode: str) -> str:
    global _EMOJI_LIB, _EMOJI_IMPORT_ATTEMPTED, _EMOJI_WARNED

    if mode == "keep":
        return text
    if mode == "remove":
        return EMOJI_RE.sub(" ", text)

    if not _EMOJI_IMPORT_ATTEMPTED:
        _EMOJI_IMPORT_ATTEMPTED = True
        try:
            import emoji  # type: ignore

            _EMOJI_LIB = emoji
        except ImportError:
            _EMOJI_LIB = None

    if _EMOJI_LIB is None:
        if not _EMOJI_WARNED:
            print("[WARN] emoji package not installed; fallback to keeping raw emoji.")
            _EMOJI_WARNED = True
        return text

    demojized = _EMOJI_LIB.demojize(text, delimiters=(" ", " "))
    demojized = demojized.replace("_", " ")
    return demojized


def normalize_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize_simple(text: str) -> List[str]:
    return text.split()


def truncate_by_tokens(text: str, max_length: int) -> str:
    tokens = tokenize_simple(text)
    if len(tokens) <= max_length:
        return text
    return " ".join(tokens[:max_length])


def clean_tweet(raw_text: str, cfg: PreprocessConfig) -> str:
    text = raw_text
    text = apply_url_rule(text, cfg.url_mode, cfg.url_token)
    text = apply_user_rule(text, cfg.user_mode, cfg.user_token)
    text = apply_emoji_rule(text, cfg.emoji_mode)

    if cfg.model_type == "rnn":
        if cfg.rnn_lowercase:
            text = text.lower()
        if cfg.rnn_remove_punct:
            text = PUNCT_RE.sub(" ", text)
        text = normalize_spaces(text)
        if cfg.rnn_remove_stopwords:
            tokens = [t for t in text.split() if t not in RNN_STOPWORDS]
            text = " ".join(tokens)
    else:
        text = normalize_spaces(text)

    text = truncate_by_tokens(text, cfg.max_length)
    return normalize_spaces(text)


def binarize_labels(labels_dict: Any, label_order: Sequence[str]) -> Tuple[List[int], List[str]]:
    if not isinstance(labels_dict, dict):
        labels_dict = {}

    keys = set(labels_dict.keys())
    multi_hot = [1 if label in keys else 0 for label in label_order]
    unknown = sorted(k for k in keys if k not in label_order)
    return multi_hot, unknown


def process_split(
    rows: Sequence[Dict[str, Any]],
    split_name: str,
    label_order: Sequence[str],
    cfg: PreprocessConfig,
    has_gold_labels: bool,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Counter]:
    out_rows: List[Dict[str, Any]] = []
    matrix: List[List[int]] = []
    unknown_counter: Counter = Counter()

    for idx, row in enumerate(rows):
        raw_tweet = str(row.get("tweet", ""))
        clean = clean_tweet(raw_tweet, cfg)

        if has_gold_labels:
            vector, unknown = binarize_labels(row.get("labels", {}), label_order)
            for item in unknown:
                unknown_counter[item] += 1
        else:
            vector = [0] * len(label_order)

        matrix.append(vector)
        record: Dict[str, Any] = {
            "index": idx,
            "ID": row.get("ID", idx),
            "tweet": raw_tweet,
            "tweet_clean": clean,
            "tweet_raw_len": len(tokenize_simple(raw_tweet)),
            "tweet_clean_len": len(tokenize_simple(clean)),
            "split": split_name,
        }
        for i, label in enumerate(label_order):
            record[label] = vector[i]
        out_rows.append(record)

    return out_rows, matrix, unknown_counter


def assert_label_dim(matrix: Sequence[Sequence[int]], expected: int, split_name: str) -> None:
    bad_rows = [i for i, row in enumerate(matrix) if len(row) != expected]
    if bad_rows:
        preview = bad_rows[:5]
        raise ValueError(
            f"{split_name} label dimension mismatch at rows {preview}; expected {expected}"
        )


def compute_class_stats(matrix: Sequence[Sequence[int]], label_order: Sequence[str]) -> Dict[str, Any]:
    n = len(matrix)
    m = len(label_order)
    pos_counts = [0] * m
    cardinality = []
    for row in matrix:
        for i, v in enumerate(row):
            pos_counts[i] += int(v)
        cardinality.append(sum(row))

    stats: Dict[str, Any] = {
        "n_samples": n,
        "avg_labels_per_sample": (sum(cardinality) / n) if n else 0.0,
        "max_labels_per_sample": max(cardinality) if cardinality else 0,
        "min_labels_per_sample": min(cardinality) if cardinality else 0,
        "labels": {},
    }

    for i, label in enumerate(label_order):
        pos = pos_counts[i]
        neg = n - pos
        rate = (pos / n) if n else 0.0
        if pos == 0:
            pos_weight = None
        else:
            pos_weight = neg / pos
        stats["labels"][label] = {
            "positive": pos,
            "negative": neg,
            "positive_rate": rate,
            "pos_weight": pos_weight,
        }
    return stats


def compare_split_rates(
    train_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    label_order: Sequence[str],
) -> Dict[str, Any]:
    drift: Dict[str, Any] = {}
    for label in label_order:
        t = train_stats["labels"][label]["positive_rate"]
        v = val_stats["labels"][label]["positive_rate"]
        drift[label] = {
            "train_rate": t,
            "val_rate": v,
            "abs_diff": abs(t - v),
        }
    return drift


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], label_order: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "ID",
        "tweet",
        "tweet_clean",
        "tweet_raw_len",
        "tweet_clean_len",
        "split",
    ] + list(label_order)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rnn_vocab(texts: Iterable[str], vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for txt in texts:
        counter.update(tokenize_simple(txt))

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def ids_from_vocab(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    unk_id = vocab["[UNK]"]
    token_ids = [vocab.get(tok, unk_id) for tok in tokenize_simple(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab["[PAD]"]] * (max_length - len(token_ids)))
    return token_ids


def export_rnn_token_ids(
    output_dir: Path,
    splits: Mapping[str, Sequence[Dict[str, Any]]],
    cfg: PreprocessConfig,
) -> None:
    train_texts = [row["tweet_clean"] for row in splits["train"]]
    vocab = build_rnn_vocab(train_texts, cfg.vocab_size, cfg.min_freq)
    with (output_dir / "rnn_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    for split_name, rows in splits.items():
        out_path = output_dir / f"{split_name}_token_ids.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                payload = {
                    "index": row["index"],
                    "ID": row["ID"],
                    "input_ids": ids_from_vocab(row["tweet_clean"], vocab, cfg.max_length),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def export_transformer_token_ids(
    output_dir: Path,
    splits: Mapping[str, Sequence[Dict[str, Any]]],
    cfg: PreprocessConfig,
) -> None:
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for --export-token-ids with model-type=transformer"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    for split_name, rows in splits.items():
        out_path = output_dir / f"{split_name}_token_ids.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                encoded = tokenizer(
                    row["tweet_clean"],
                    truncation=True,
                    padding="max_length",
                    max_length=cfg.max_length,
                    return_attention_mask=True,
                )
                payload = {
                    "index": row["index"],
                    "ID": row["ID"],
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# =====================
# Train and predict
# =====================


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def metric_to_float(metrics: Dict[str, object], key: str) -> float:
    value = metrics[key]
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    raise TypeError(f"Metric '{key}' is not numeric: {type(value).__name__}")


def build_pos_weight(
    metadata: Dict[str, Any],
    label_order: List[str],
    device: torch.device,
) -> torch.Tensor:
    pos_weight_map = metadata.get("recommended_bce_pos_weight", {})
    weights: List[float] = []
    for label in label_order:
        value = pos_weight_map.get(label, 1.0)
        if value is None:
            value = 1.0
        weights.append(float(value))
    return torch.tensor(weights, dtype=torch.float, device=device)


def build_bilstm_model(
    args: argparse.Namespace,
    label_order: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[nn.Module, DataLoader, DataLoader, Dict[str, int], Dict[str, Any]]:
    vocab = build_vocab(
        texts=train_df[args.text_column].fillna(""),
        config=VocabBuildConfig(max_size=args.vocab_max_size, min_freq=args.vocab_min_freq),
    )
    print(f"[INFO] Vocab size: {len(vocab)}")

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        label_order=label_order,
        text_column=args.text_column,
        vocab=vocab,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    embedding_matrix = build_embedding_matrix(
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        glove_path=args.glove_path,
    )
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)

    model = BiLSTMMultiHeadAttention(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        num_labels=len(label_order),
        pretrained_embeddings=embedding_tensor,
        freeze_embedding=args.freeze_embedding,
    ).to(device)

    model_args = {
        "vocab_size": len(vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "num_labels": len(label_order),
        "freeze_embedding": args.freeze_embedding,
    }
    return model, train_loader, val_loader, vocab, model_args


def build_transformer_model(
    args: argparse.Namespace,
    label_order: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[nn.Module, DataLoader, DataLoader, Dict[str, Any]]:
    train_loader, val_loader = create_transformer_dataloaders(
        train_df=train_df,
        val_df=val_df,
        label_order=label_order,
        text_column=args.text_column,
        tokenizer_name=args.pretrained_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TransformerMultiLabelClassifier(
        pretrained_model_name=args.pretrained_model,
        num_labels=len(label_order),
        dropout=args.dropout,
        multi_sample_dropout=args.multi_sample_dropout,
        head_type=args.head_type,
        pooling=args.pooling,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    model_args = {
        "pretrained_model_name": args.pretrained_model,
        "num_labels": len(label_order),
        "dropout": args.dropout,
        "multi_sample_dropout": args.multi_sample_dropout,
        "head_type": args.head_type,
        "pooling": args.pooling,
        "freeze_backbone": args.freeze_backbone,
    }
    return model, train_loader, val_loader, model_args


def build_criterion(args: argparse.Namespace, pos_weight: torch.Tensor) -> nn.Module:
    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    asl_pos_weight = pos_weight if args.asl_use_pos_weight else None
    return AsymmetricLoss(
        gamma_neg=args.asl_gamma_neg,
        gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
        pos_weight=asl_pos_weight,
    )


def split_validation_for_threshold_calibration(
    val_df: pd.DataFrame,
    calibration_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    if calibration_ratio < 0.0 or calibration_ratio >= 1.0:
        raise ValueError("threshold_calibration_ratio must satisfy 0 <= ratio < 1")

    val_df = val_df.reset_index(drop=True)
    if calibration_ratio == 0.0:
        return val_df, val_df, False

    n_rows = len(val_df)
    if n_rows < 4:
        print(
            "[WARN] Validation split is too small for held-out threshold calibration. "
            "Fallback to tuning on full validation split."
        )
        return val_df, val_df, False

    calibration_size = int(round(n_rows * calibration_ratio))
    calibration_size = min(max(calibration_size, 1), n_rows - 1)

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n_rows)
    calibration_idx = shuffled_idx[:calibration_size]
    selection_idx = shuffled_idx[calibration_size:]

    selection_df = val_df.iloc[selection_idx].reset_index(drop=True)
    calibration_df = val_df.iloc[calibration_idx].reset_index(drop=True)

    if selection_df.empty or calibration_df.empty:
        print(
            "[WARN] Failed to build non-empty held-out threshold calibration split. "
            "Fallback to tuning on full validation split."
        )
        return val_df, val_df, False

    return selection_df, calibration_df, True


def run_train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    device = choose_device(args.device)
    print(f"[INFO] Device: {device}")

    metadata_path = args.data_dir / args.metadata_file
    train_path = args.data_dir / args.train_file
    val_path = args.data_dir / args.val_file

    metadata = load_metadata(metadata_path)
    label_order: List[str] = get_label_order(metadata.get("label_order"))
    if len(label_order) != 12:
        raise ValueError(f"Expected 12 labels, got {len(label_order)}")

    train_df = load_split_csv(train_path)
    val_df = load_split_csv(val_path)

    val_selection_df, val_calibration_df, use_heldout_calibration = split_validation_for_threshold_calibration(
        val_df=val_df,
        calibration_ratio=args.threshold_calibration_ratio,
        seed=args.seed,
    )

    if use_heldout_calibration:
        print(
            "[INFO] Threshold calibration uses held-out validation subset: "
            f"selection={len(val_selection_df)}, calibration={len(val_calibration_df)}"
        )
    else:
        print(f"[INFO] Threshold calibration uses full validation split: n={len(val_selection_df)}")

    if args.model_type == "bilstm":
        model, train_loader, val_loader, vocab, model_args = build_bilstm_model(
            args=args,
            label_order=label_order,
            train_df=train_df,
            val_df=val_selection_df,
            device=device,
        )
    else:
        model, train_loader, val_loader, model_args = build_transformer_model(
            args=args,
            label_order=label_order,
            train_df=train_df,
            val_df=val_selection_df,
            device=device,
        )
        vocab = None

    pos_weight = build_pos_weight(metadata, label_order, device)
    criterion = build_criterion(args=args, pos_weight=pos_weight)

    if args.model_type == "transformer" and args.learning_rate == 1e-3:
        print(
            "[WARN] Transformer backbone typically needs a smaller lr. "
            "Consider --learning-rate 2e-5 to 5e-5."
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    history: List[Dict[str, float]] = []
    best_macro_f1 = -1.0
    patience_counter = 0

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_model_path = args.output_dir / "best_model.pt"
    vocab_path = args.output_dir / "vocab.json"
    training_history_path = args.output_dir / "training_history.json"
    threshold_path = args.output_dir / "thresholds.json"
    config_path = args.output_dir / "training_config.json"

    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json(config_path, serializable_args)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                    scaled_loss = loss / args.grad_accum_steps
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, running_count)
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            label_order=label_order,
        )

        val_macro_f1 = metric_to_float(val_metrics, "macro_f1")
        val_loss = metric_to_float(val_metrics, "loss")

        scheduler.step(val_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_micro_f1": metric_to_float(val_metrics, "micro_f1"),
            "learning_rate": current_lr,
        }
        history.append(epoch_record)

        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f} lr={current_lr:.2e}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            patience_counter = 0
            checkpoint: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "label_order": label_order,
                "model_type": args.model_type,
                "model_args": model_args,
                "max_length": args.max_length,
                "text_column": args.text_column,
                "best_val_macro_f1": best_macro_f1,
            }
            if vocab is not None:
                checkpoint["vocab"] = vocab

            torch.save(checkpoint, best_model_path)
            if vocab is not None:
                save_json(vocab_path, vocab)
            print(f"[INFO] Saved new best checkpoint: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    save_json(training_history_path, {"history": history, "best_val_macro_f1": best_macro_f1})

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    selection_logits, selection_labels = collect_logits_and_labels(model=model, data_loader=val_loader, device=device)

    if use_heldout_calibration:
        if args.model_type == "transformer":
            calibration_dataset = TransformerTweetMultiLabelDataset(
                data_frame=val_calibration_df,
                text_column=args.text_column,
                label_order=label_order,
                tokenizer_name=args.pretrained_model,
                max_length=args.max_length,
                with_labels=True,
            )
        else:
            if vocab is None:
                raise ValueError("BiLSTM model requires vocab for threshold calibration")
            calibration_dataset = TweetMultiLabelDataset(
                data_frame=val_calibration_df,
                text_column=args.text_column,
                label_order=label_order,
                vocab=vocab,
                max_length=args.max_length,
                with_labels=True,
            )

        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        tuning_logits, tuning_labels = collect_logits_and_labels(
            model=model,
            data_loader=calibration_loader,
            device=device,
        )
        tuning_source = "val_calibration_split"
    else:
        tuning_logits, tuning_labels = selection_logits, selection_labels
        tuning_source = "val"

    tuning_probs = 1 / (1 + np.exp(-tuning_logits))

    tuned_thresholds_map = tune_per_label_thresholds(
        probabilities=tuning_probs,
        labels=tuning_labels,
        label_order=label_order,
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )
    tuned_thresholds = thresholds_to_list(tuned_thresholds_map, label_order)

    default_metrics = evaluate_from_logits(
        logits=selection_logits,
        labels=selection_labels,
        label_order=label_order,
        thresholds=[0.5] * len(label_order),
    )
    tuned_metrics = evaluate_from_logits(
        logits=selection_logits,
        labels=selection_labels,
        label_order=label_order,
        thresholds=tuned_thresholds,
    )
    tuning_default_metrics = evaluate_from_logits(
        logits=tuning_logits,
        labels=tuning_labels,
        label_order=label_order,
        thresholds=[0.5] * len(label_order),
    )
    tuning_tuned_metrics = evaluate_from_logits(
        logits=tuning_logits,
        labels=tuning_labels,
        label_order=label_order,
        thresholds=tuned_thresholds,
    )

    threshold_payload = {
        "thresholds": tuned_thresholds_map,
        "default_macro_f1": metric_to_float(default_metrics, "macro_f1"),
        "tuned_macro_f1": metric_to_float(tuned_metrics, "macro_f1"),
        "default_micro_f1": metric_to_float(default_metrics, "micro_f1"),
        "tuned_micro_f1": metric_to_float(tuned_metrics, "micro_f1"),
        "tuning_source": tuning_source,
        "threshold_calibration_ratio": args.threshold_calibration_ratio,
        "selection_val_size": int(len(val_selection_df)),
        "tuning_val_size": int(len(val_calibration_df) if use_heldout_calibration else len(val_selection_df)),
        "tuning_default_macro_f1": metric_to_float(tuning_default_metrics, "macro_f1"),
        "tuning_tuned_macro_f1": metric_to_float(tuning_tuned_metrics, "macro_f1"),
        "tuning_default_micro_f1": metric_to_float(tuning_default_metrics, "micro_f1"),
        "tuning_tuned_micro_f1": metric_to_float(tuning_tuned_metrics, "micro_f1"),
        "search": {
            "start": args.threshold_start,
            "end": args.threshold_end,
            "step": args.threshold_step,
        },
    }
    save_json(threshold_path, threshold_payload)

    run_summary = {
        "label_order": label_order,
        "best_val_macro_f1": best_macro_f1,
        "default_macro_f1": metric_to_float(default_metrics, "macro_f1"),
        "tuned_macro_f1": metric_to_float(tuned_metrics, "macro_f1"),
        "threshold_tuning_source": tuning_source,
        "threshold_calibration_ratio": args.threshold_calibration_ratio,
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "output_dir": os.path.relpath(str(args.output_dir.resolve()), start=str(Path.cwd().resolve())),
    }
    save_json(args.output_dir / "run_summary.json", run_summary)

    print("[DONE] Training completed.")
    print(f"[DONE] Best val macro-F1: {best_macro_f1:.4f}")
    print(
        "[DONE] Threshold tuning source: "
        f"{tuning_source} (selection_n={len(val_selection_df)}, "
        f"tuning_n={len(val_calibration_df) if use_heldout_calibration else len(val_selection_df)})"
    )
    print(
        "[DONE] Threshold tuning macro-F1: "
        f"default={default_metrics['macro_f1']:.4f}, tuned={tuned_metrics['macro_f1']:.4f}"
    )


def run_predict(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    print(f"[INFO] Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint["model_args"]
    label_order = checkpoint["label_order"]
    model_type = checkpoint.get("model_type", "bilstm")
    vocab = checkpoint.get("vocab")
    max_length = int(checkpoint["max_length"])
    text_column = checkpoint.get("text_column", "tweet_clean")

    if model_type == "transformer":
        model = TransformerMultiLabelClassifier(
            pretrained_model_name=model_args["pretrained_model_name"],
            num_labels=model_args["num_labels"],
            dropout=model_args["dropout"],
            multi_sample_dropout=model_args["multi_sample_dropout"],
            head_type=model_args["head_type"],
            pooling=model_args["pooling"],
            freeze_backbone=model_args.get("freeze_backbone", False),
        ).to(device)
    else:
        if vocab is None:
            raise ValueError("BiLSTM checkpoint is missing vocab")

        model = BiLSTMMultiHeadAttention(
            vocab_size=model_args["vocab_size"],
            embedding_dim=model_args["embedding_dim"],
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            attention_heads=model_args["attention_heads"],
            dropout=model_args["dropout"],
            num_labels=model_args["num_labels"],
            pretrained_embeddings=None,
            freeze_embedding=model_args.get("freeze_embedding", False),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    with args.threshold_file.open("r", encoding="utf-8") as f:
        threshold_payload = json.load(f)
    threshold_map = threshold_payload["thresholds"]
    thresholds = [float(threshold_map[label]) for label in label_order]

    test_df = load_split_csv(args.data_dir / args.test_file)
    if model_type == "transformer":
        test_loader = create_transformer_test_loader(
            test_df=test_df,
            text_column=text_column,
            label_order=label_order,
            tokenizer_name=model_args["pretrained_model_name"],
            max_length=max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        if vocab is None:
            raise ValueError("BiLSTM checkpoint is missing vocab")

        test_loader = create_test_loader(
            test_df=test_df,
            text_column=text_column,
            label_order=label_order,
            vocab=vocab,
            max_length=max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    probabilities = predict_probabilities(model=model, data_loader=test_loader, device=device)
    preds = (probabilities >= thresholds).astype(int)

    sample_df = pd.read_csv(args.sample_submission)
    if list(sample_df.columns[1:]) != list(label_order):
        raise ValueError("Label order mismatch between checkpoint and sample_submission.csv")

    submission = sample_df.copy()
    for i, label in enumerate(label_order):
        submission[label] = preds[:, i]

    if len(submission) != len(test_df):
        raise ValueError(
            f"Submission row count mismatch: expected {len(test_df)}, got {len(submission)}"
        )

    submission.to_csv(args.output_file, index=False)
    print(f"[DONE] Submission saved to: {args.output_file.resolve()}")


def build_loader_and_model(
    ckpt_path: Path,
    label_order: List[str],
    val_df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, DataLoader]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_type = checkpoint.get("model_type", "bilstm")
    model_args = checkpoint["model_args"]
    text_column = checkpoint.get("text_column", "tweet_clean")
    max_length = int(checkpoint["max_length"])

    if model_type == "transformer":
        _, val_loader = create_transformer_dataloaders(
            train_df=val_df,
            val_df=val_df,
            label_order=label_order,
            text_column=text_column,
            tokenizer_name=model_args["pretrained_model_name"],
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        model = TransformerMultiLabelClassifier(
            pretrained_model_name=model_args["pretrained_model_name"],
            num_labels=model_args["num_labels"],
            dropout=model_args["dropout"],
            multi_sample_dropout=model_args["multi_sample_dropout"],
            head_type=model_args["head_type"],
            pooling=model_args["pooling"],
            freeze_backbone=model_args.get("freeze_backbone", False),
        ).to(device)
    else:
        vocab = checkpoint.get("vocab")
        if vocab is None:
            raise ValueError("BiLSTM checkpoint is missing vocab")

        _, val_loader = create_dataloaders(
            train_df=val_df,
            val_df=val_df,
            label_order=label_order,
            text_column=text_column,
            vocab=vocab,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        model = BiLSTMMultiHeadAttention(
            vocab_size=model_args["vocab_size"],
            embedding_dim=model_args["embedding_dim"],
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            attention_heads=model_args["attention_heads"],
            dropout=model_args["dropout"],
            num_labels=model_args["num_labels"],
            pretrained_embeddings=None,
            freeze_embedding=model_args.get("freeze_embedding", False),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, val_loader


def evaluate_run(
    name: str,
    ckpt_path: Path,
    threshold_path: Path | None,
    data_dir: Path = Path("preprocessed"),
    val_file: str = "val_preprocessed.csv",
    batch_size: int = 128,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    metadata = load_metadata(data_dir / "metadata.json")
    label_order = get_label_order(metadata.get("label_order"))
    val_df = load_split_csv(data_dir / val_file)

    model, val_loader = build_loader_and_model(
        ckpt_path=ckpt_path,
        label_order=label_order,
        val_df=val_df,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    logits, labels = collect_logits_and_labels(model=model, data_loader=val_loader, device=device)

    default_metrics = evaluate_from_logits(
        logits=logits,
        labels=labels,
        label_order=label_order,
        thresholds=[0.5] * len(label_order),
    )

    result: Dict[str, Any] = {
        "name": name,
        "default_macro_f1": metric_to_float(default_metrics, "macro_f1"),
        "default_micro_f1": metric_to_float(default_metrics, "micro_f1"),
    }
    print(f"=== {name} ===")
    print(f"default_macro_f1 {result['default_macro_f1']:.6f}")
    print(f"default_micro_f1 {result['default_micro_f1']:.6f}")

    if threshold_path is not None:
        with threshold_path.open("r", encoding="utf-8") as f:
            threshold_payload = json.load(f)
        tuned_thresholds = [float(threshold_payload["thresholds"][label]) for label in label_order]
        tuned_metrics = evaluate_from_logits(
            logits=logits,
            labels=labels,
            label_order=label_order,
            thresholds=tuned_thresholds,
        )
        result["tuned_macro_f1"] = metric_to_float(tuned_metrics, "macro_f1")
        result["tuned_micro_f1"] = metric_to_float(tuned_metrics, "micro_f1")
        print(f"tuned_macro_f1 {result['tuned_macro_f1']:.6f}")
        print(f"tuned_micro_f1 {result['tuned_micro_f1']:.6f}")

    print()
    return result


def run_metrics(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    evaluate_run(
        name=args.name,
        ckpt_path=args.checkpoint,
        threshold_path=args.threshold_file,
        data_dir=args.data_dir,
        val_file=args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )


# =====================
# CLI
# =====================


def add_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("preprocess", help="Preprocess raw train/val/test json files")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--model-type", choices=["transformer", "rnn"], default="transformer")
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--url-mode", choices=["token", "remove", "keep"], default="token")
    parser.add_argument("--user-mode", choices=["token", "remove", "keep"], default="token")
    parser.add_argument("--emoji-mode", choices=["keep", "demojize", "remove"], default="demojize")
    parser.add_argument("--url-token", type=str, default="[URL]")
    parser.add_argument("--user-token", type=str, default="[USER]")

    parser.add_argument("--rnn-lowercase", action="store_true")
    parser.add_argument("--rnn-remove-punct", action="store_true")
    parser.add_argument("--rnn-remove-stopwords", action="store_true")

    parser.add_argument("--strict-label-order", action="store_true")
    parser.add_argument("--export-token-ids", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)


def add_train_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train multi-label classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--metadata-file", type=str, default="metadata.json")
    parser.add_argument("--train-file", type=str, default="train_preprocessed.csv")
    parser.add_argument("--val-file", type=str, default="val_preprocessed.csv")
    parser.add_argument("--text-column", type=str, default="tweet_clean")

    parser.add_argument("--model-type", type=str, choices=["bilstm", "transformer"], default="transformer")
    parser.add_argument("--pretrained-model", type=str, default="digitalepidemiologylab/covid-twitter-bert-v2")
    parser.add_argument("--head-type", choices=["linear", "label_attention"], default="linear")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--multi-sample-dropout", type=int, default=5)
    parser.add_argument("--freeze-backbone", action="store_true")

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-max-size", type=int, default=30000)
    parser.add_argument("--vocab-min-freq", type=int, default=2)

    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--loss-type", choices=["bce", "asl"], default="bce")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-use-pos-weight", action="store_true")

    parser.add_argument("--threshold-start", type=float, default=0.1)
    parser.add_argument("--threshold-end", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--threshold-calibration-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of validation split reserved for threshold tuning. "
            "Use >0 to reduce threshold overfitting (e.g., 0.2)."
        ),
    )

    parser.add_argument("--glove-path", type=Path, default=None)
    parser.add_argument("--freeze-embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)


def add_predict_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("predict", help="Run inference and generate submission")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--test-file", type=str, default="test_preprocessed.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)


def add_metrics_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("metrics", help="Evaluate checkpoint metrics on validation split")
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--val-file", type=str, default="val_preprocessed.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW1 unified preprocessing/training/inference pipeline")
    subparsers = parser.add_subparsers(dest="command")

    add_preprocess_subparser(subparsers)
    add_train_subparser(subparsers)
    add_predict_subparser(subparsers)
    add_metrics_subparser(subparsers)

    return parser


def parse_main_args() -> argparse.Namespace:
    parser = build_parser()
    argv = sys.argv[1:]
    known_commands = {"preprocess", "train", "predict", "metrics"}

    if not argv:
        argv = ["preprocess"]
    elif argv[0] in {"-h", "--help"}:
        return parser.parse_args(argv)
    elif argv[0] not in known_commands:
        argv = ["preprocess", *argv]

    return parser.parse_args(argv)


def run_preprocess(args: argparse.Namespace) -> None:
    cfg = PreprocessConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        max_length=args.max_length,
        url_mode=args.url_mode,
        user_mode=args.user_mode,
        emoji_mode=args.emoji_mode,
        url_token=args.url_token,
        user_token=args.user_token,
        rnn_lowercase=args.rnn_lowercase,
        rnn_remove_punct=args.rnn_remove_punct,
        rnn_remove_stopwords=args.rnn_remove_stopwords,
        strict_label_order=args.strict_label_order,
        export_token_ids=args.export_token_ids,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )

    data_dir = cfg.data_dir
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = data_dir / "sample_submission.csv"
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    test_path = data_dir / "test.json"

    label_order = get_label_order_from_sample(sample_path)
    validate_label_order(label_order, strict=cfg.strict_label_order)

    train = load_json(train_path)
    val = load_json(val_path)
    test = load_json(test_path)

    train_rows, train_matrix, train_unknown = process_split(
        train, "train", label_order, cfg, has_gold_labels=True
    )
    val_rows, val_matrix, val_unknown = process_split(
        val, "val", label_order, cfg, has_gold_labels=True
    )
    test_rows, test_matrix, _ = process_split(
        test, "test", label_order, cfg, has_gold_labels=False
    )

    assert_label_dim(train_matrix, 12, "train")
    assert_label_dim(val_matrix, 12, "val")
    assert_label_dim(test_matrix, 12, "test")

    write_csv(out_dir / "train_preprocessed.csv", train_rows, label_order)
    write_csv(out_dir / "val_preprocessed.csv", val_rows, label_order)
    write_csv(out_dir / "test_preprocessed.csv", test_rows, label_order)

    train_stats = compute_class_stats(train_matrix, label_order)
    val_stats = compute_class_stats(val_matrix, label_order)
    drift = compare_split_rates(train_stats, val_stats, label_order)

    pos_weight = {label: train_stats["labels"][label]["pos_weight"] for label in label_order}

    meta = {
        "label_order": label_order,
        "shape_check": {
            "train_dim": len(train_matrix[0]) if train_matrix else 0,
            "val_dim": len(val_matrix[0]) if val_matrix else 0,
            "test_dim": len(test_matrix[0]) if test_matrix else 0,
            "expected_dim": 12,
            "with_index_columns": 13,
        },
        "unknown_labels": {
            "train": dict(train_unknown),
            "val": dict(val_unknown),
        },
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_val_rate_drift": drift,
        "recommended_bce_pos_weight": pos_weight,
        "config": {
            "model_type": cfg.model_type,
            "max_length": cfg.max_length,
            "url_mode": cfg.url_mode,
            "user_mode": cfg.user_mode,
            "emoji_mode": cfg.emoji_mode,
            "rnn_lowercase": cfg.rnn_lowercase,
            "rnn_remove_punct": cfg.rnn_remove_punct,
            "rnn_remove_stopwords": cfg.rnn_remove_stopwords,
            "strict_label_order": cfg.strict_label_order,
            "export_token_ids": cfg.export_token_ids,
            "tokenizer_name": cfg.tokenizer_name,
            "vocab_size": cfg.vocab_size,
            "min_freq": cfg.min_freq,
        },
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if cfg.export_token_ids:
        split_dict = {"train": train_rows, "val": val_rows, "test": test_rows}
        if cfg.model_type == "rnn":
            export_rnn_token_ids(out_dir, split_dict, cfg)
        else:
            export_transformer_token_ids(out_dir, split_dict, cfg)

    print("[OK] Preprocessing finished.")
    print(f"[OK] Output directory: {out_dir.resolve()}")
    print("[OK] Generated files:")
    print("     - train_preprocessed.csv")
    print("     - val_preprocessed.csv")
    print("     - test_preprocessed.csv")
    print("     - metadata.json")
    if cfg.export_token_ids:
        print("     - *_token_ids.jsonl")
        if cfg.model_type == "rnn":
            print("     - rnn_vocab.json")


def main() -> None:
    args = parse_main_args()

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "metrics":
        run_metrics(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
