from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from dataset import create_test_loader, create_transformer_test_loader, load_split_csv
from evaluate import predict_probabilities
from models import BiLSTMMultiHeadAttention, TransformerMultiLabelClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and generate submission file"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--test-file", type=str, default="test_preprocessed.csv")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ensemble-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list of checkpoints for probability averaging ensemble.",
    )
    parser.add_argument("--threshold-file", type=Path, required=True)
    parser.add_argument(
        "--sample-submission", type=Path, default=Path("sample_submission.csv")
    )
    parser.add_argument("--output-file", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.ensemble_checkpoints:
        checkpoint_paths = [Path(p) for p in args.ensemble_checkpoints]
        if args.checkpoint is not None:
            checkpoint_paths.append(Path(args.checkpoint))
        return checkpoint_paths

    if args.checkpoint is None:
        raise ValueError("Provide --checkpoint or --ensemble-checkpoints")
    return [Path(args.checkpoint)]


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    model_args = checkpoint["model_args"]
    model_type = checkpoint.get("model_type", "bilstm")

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
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, model_type

    vocab = checkpoint.get("vocab")
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
    return model, model_type


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"[INFO] Device: {device}")

    checkpoint_paths = resolve_checkpoint_paths(args)
    print(f"[INFO] Checkpoints: {len(checkpoint_paths)}")

    checkpoint = torch.load(checkpoint_paths[0], map_location=device)
    model_args = checkpoint["model_args"]
    label_order = checkpoint["label_order"]
    model_type = checkpoint.get("model_type", "bilstm")
    vocab = checkpoint.get("vocab")
    max_length = int(checkpoint["max_length"])
    text_column = checkpoint.get("text_column", "tweet_clean")

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

    probabilities_accum = []
    for checkpoint_path in checkpoint_paths:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if ckpt["label_order"] != label_order:
            raise ValueError(
                f"Label order mismatch in ensemble checkpoint: {checkpoint_path}"
            )

        ckpt_model_args = ckpt["model_args"]
        ckpt_model_type = ckpt.get("model_type", "bilstm")
        if ckpt_model_type != model_type:
            raise ValueError(
                f"Model type mismatch in ensemble checkpoint: {checkpoint_path}"
            )
        if int(ckpt["max_length"]) != max_length:
            raise ValueError(
                f"Max length mismatch in ensemble checkpoint: {checkpoint_path}"
            )
        if ckpt.get("text_column", "tweet_clean") != text_column:
            raise ValueError(
                f"Text column mismatch in ensemble checkpoint: {checkpoint_path}"
            )

        if model_type == "transformer":
            if (
                ckpt_model_args.get("pretrained_model_name")
                != model_args.get("pretrained_model_name")
            ):
                raise ValueError(
                    "Ensemble transformer checkpoints must use the same tokenizer/model "
                    f"as first checkpoint: {checkpoint_path}"
                )
        else:
            if ckpt.get("vocab") != vocab:
                raise ValueError(
                    f"BiLSTM vocab mismatch in ensemble checkpoint: {checkpoint_path}"
                )

        model, _ = build_model_from_checkpoint(ckpt, device)
        probabilities_accum.append(
            predict_probabilities(model=model, data_loader=test_loader, device=device)
        )

    probabilities = probabilities_accum[0]
    if len(probabilities_accum) > 1:
        probabilities = sum(probabilities_accum) / len(probabilities_accum)
        print(
            "[INFO] Ensemble probability averaging completed "
            f"({len(probabilities_accum)} checkpoints)."
        )

    preds = (probabilities >= thresholds).astype(int)

    sample_df = pd.read_csv(args.sample_submission)
    if list(sample_df.columns[1:]) != list(label_order):
        raise ValueError(
            "Label order mismatch between checkpoint and sample_submission.csv"
        )

    submission = sample_df.copy()
    for i, label in enumerate(label_order):
        submission[label] = preds[:, i]

    if len(submission) != len(test_df):
        raise ValueError(
            f"Submission row count mismatch: expected {len(test_df)}, got {len(submission)}"
        )

    submission.to_csv(args.output_file, index=False)
    print(f"[DONE] Submission saved to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
