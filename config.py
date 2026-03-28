from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


def get_label_order(metadata_label_order: List[str] | None) -> List[str]:
    if metadata_label_order is None:
        return DEFAULT_LABEL_ORDER
    if len(metadata_label_order) != len(DEFAULT_LABEL_ORDER):
        raise ValueError(
            f"Expected {len(DEFAULT_LABEL_ORDER)} labels, got {len(metadata_label_order)}"
        )
    return metadata_label_order
