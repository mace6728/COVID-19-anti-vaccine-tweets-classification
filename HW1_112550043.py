#!/usr/bin/env python3
"""Preprocess HW1 vaccine tweet dataset for multi-label classification.

Stages implemented:
1. Label binarization with strict label order validation.
2. Twitter text cleaning with model-aware strategy (transformer or rnn).
3. Imbalance analysis + BCEWithLogitsLoss-compatible pos_weight export.
4. Truncation and optional token-id export (RNN vocab or HF tokenizer).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

EXPECTED_LABEL_ORDER = [
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


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="HW1 multi-label data preprocessing")
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

    args = parser.parse_args()
    return PreprocessConfig(
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
        # Transformer setting: avoid over-cleaning to preserve context signals.
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
        # BCEWithLogitsLoss recommendation: pos_weight = neg/pos.
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
    train_stats: Dict[str, Any], val_stats: Dict[str, Any], label_order: Sequence[str]
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


def build_rnn_vocab(
    texts: Iterable[str],
    vocab_size: int,
    min_freq: int,
) -> Dict[str, int]:
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
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for --export-token-ids with model-type=transformer"
        ) from exc

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


def main() -> None:
    cfg = parse_args()
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

    pos_weight = {
        label: train_stats["labels"][label]["pos_weight"] for label in label_order
    }

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


if __name__ == "__main__":
    main()
