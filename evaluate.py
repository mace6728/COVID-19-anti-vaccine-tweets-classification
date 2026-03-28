from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits_and_labels(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
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
        per_label_f1[label] = float(
            f1_score(y_true[:, i], preds[:, i], zero_division=0)
        )

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
