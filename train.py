from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import DEFAULT_LABEL_ORDER
from dataset import (
    VocabBuildConfig,
    build_embedding_matrix,
    build_vocab,
    create_dataloaders,
    create_transformer_dataloaders,
    load_metadata,
    load_split_csv,
    set_seed,
)
from evaluate import collect_logits_and_labels, evaluate_from_logits, evaluate_model
from losses import AsymmetricLoss
from models import BiLSTMMultiHeadAttention, TransformerMultiLabelClassifier
from threshold_tuner import thresholds_to_list, tune_per_label_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-label tweet classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/transformer")
    )
    parser.add_argument("--metadata-file", type=str, default="metadata.json")
    parser.add_argument("--train-file", type=str, default="train_preprocessed.csv")
    parser.add_argument("--val-file", type=str, default="val_preprocessed.csv")
    parser.add_argument("--text-column", type=str, default="tweet_clean")

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bilstm", "transformer"],
        default="transformer",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="digitalepidemiologylab/covid-twitter-bert-v2",
    )
    parser.add_argument(
        "--head-type", choices=["linear", "label_attention"], default="linear"
    )
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

    parser.add_argument("--glove-path", type=Path, default=None)
    parser.add_argument("--freeze-embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def metric_to_float(metrics: Dict[str, object], key: str) -> float:
    value = metrics[key]
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    raise TypeError(f"Metric '{key}' is not numeric: {type(value).__name__}")


def build_pos_weight(
    metadata: Dict,
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
    train_df,
    val_df,
    device: torch.device,
):
    vocab = build_vocab(
        texts=train_df[args.text_column].fillna(""),
        config=VocabBuildConfig(
            max_size=args.vocab_max_size, min_freq=args.vocab_min_freq
        ),
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
    train_df,
    val_df,
    device: torch.device,
):
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


def build_criterion(
    args: argparse.Namespace,
    pos_weight: torch.Tensor,
) -> nn.Module:
    if args.loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    asl_pos_weight = pos_weight if args.asl_use_pos_weight else None
    return AsymmetricLoss(
        gamma_neg=args.asl_gamma_neg,
        gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
        pos_weight=asl_pos_weight,
    )


def main() -> None:
    args = parse_args()
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
    label_order: List[str] = metadata.get("label_order", DEFAULT_LABEL_ORDER)
    if len(label_order) != 12:
        raise ValueError(f"Expected 12 labels, got {len(label_order)}")

    train_df = load_split_csv(train_path)
    val_df = load_split_csv(val_path)

    if args.model_type == "bilstm":
        model, train_loader, val_loader, vocab, model_args = build_bilstm_model(
            args=args,
            label_order=label_order,
            train_df=train_df,
            val_df=val_df,
            device=device,
        )
    else:
        model, train_loader, val_loader, model_args = build_transformer_model(
            args=args,
            label_order=label_order,
            train_df=train_df,
            val_df=val_df,
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

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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

    serializable_args = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
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

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = (step % args.grad_accum_steps == 0) or (
                step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm
                )
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
            checkpoint = {
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

    save_json(
        training_history_path, {"history": history, "best_val_macro_f1": best_macro_f1}
    )

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_logits, val_labels = collect_logits_and_labels(
        model=model, data_loader=val_loader, device=device
    )
    val_probs = 1 / (1 + np.exp(-val_logits))

    tuned_thresholds_map = tune_per_label_thresholds(
        probabilities=val_probs,
        labels=val_labels,
        label_order=label_order,
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )
    tuned_thresholds = thresholds_to_list(tuned_thresholds_map, label_order)

    default_metrics = evaluate_from_logits(
        logits=val_logits,
        labels=val_labels,
        label_order=label_order,
        thresholds=[0.5] * len(label_order),
    )
    tuned_metrics = evaluate_from_logits(
        logits=val_logits,
        labels=val_labels,
        label_order=label_order,
        thresholds=tuned_thresholds,
    )

    threshold_payload = {
        "thresholds": tuned_thresholds_map,
        "default_macro_f1": metric_to_float(default_metrics, "macro_f1"),
        "tuned_macro_f1": metric_to_float(tuned_metrics, "macro_f1"),
        "default_micro_f1": metric_to_float(default_metrics, "micro_f1"),
        "tuned_micro_f1": metric_to_float(tuned_metrics, "micro_f1"),
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
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "output_dir": str(args.output_dir.resolve()),
    }
    save_json(args.output_dir / "run_summary.json", run_summary)

    print("[DONE] Training completed.")
    print(f"[DONE] Best val macro-F1: {best_macro_f1:.4f}")
    print(
        "[DONE] Threshold tuning macro-F1: "
        f"default={default_metrics['macro_f1']:.4f}, tuned={tuned_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
