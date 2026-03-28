from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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
        self.layer_norm = (
            nn.LayerNorm(lstm_out_dim) if use_layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        lstm_outputs, _ = self.lstm(embeddings)

        # key_padding_mask: True indicates masked positions for MHA.
        key_padding_mask = attention_mask == 0
        attn_outputs, _ = self.mha(
            lstm_outputs,
            lstm_outputs,
            lstm_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Masked mean pooling avoids biased representation from padded tokens.
        mask = attention_mask.unsqueeze(-1)
        masked_attn = attn_outputs * mask
        pooled = masked_attn.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
