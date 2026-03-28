from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_INDEX = 0
UNK_INDEX = 1


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
        token_ids, attention_mask = encode_text_to_ids(
            text, self.vocab, self.max_length
        )

        item: Dict[str, torch.Tensor | int | str] = {
            "index": int(row["index"]),
            "ID": str(row["ID"]),
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
        }

        if self.with_labels:
            labels = np.array(
                [int(row[label]) for label in self.label_order], dtype=np.float32
            )
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

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
            labels = np.array(
                [int(row[label]) for label in self.label_order], dtype=np.float32
            )
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


def encode_text_to_ids(
    text: str, vocab: Dict[str, int], max_length: int
) -> Tuple[List[int], List[int]]:
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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
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
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
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
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def build_embedding_matrix(
    vocab: Dict[str, int],
    embedding_dim: int,
    glove_path: Optional[Path],
) -> np.ndarray:
    matrix = np.random.normal(
        loc=0.0, scale=0.02, size=(len(vocab), embedding_dim)
    ).astype(np.float32)
    matrix[PAD_INDEX] = 0.0

    if glove_path is None:
        print("[WARN] No GloVe path provided. Use random embedding initialization.")
        return matrix

    if not glove_path.exists():
        print(
            f"[WARN] GloVe file does not exist: {glove_path}. Use random initialization."
        )
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
