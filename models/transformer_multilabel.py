from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from transformers import AutoModel  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None


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

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(multi_sample_dropout)]
        )

        if self.head_type == "linear":
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        else:
            # Learnable label embeddings interact with token states.
            self.label_embedding = nn.Parameter(
                torch.empty(num_labels, self.hidden_size)
            )
            nn.init.xavier_uniform_(self.label_embedding)
            self.label_classifier = nn.Linear(self.hidden_size, 1)

    def _pool_sequence(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        mask = attention_mask.unsqueeze(-1)
        masked = hidden_states * mask
        return masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

    def _forward_linear_head(
        self, pooled_output: torch.Tensor
    ) -> torch.Tensor:
        logits_list = [self.classifier(dropout(pooled_output)) for dropout in self.dropouts]
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def _forward_label_attention_head(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # score shape: [batch, seq_len, num_labels]
        score = torch.einsum("bsh,lh->bsl", sequence_output, self.label_embedding)
        score = score / math.sqrt(self.hidden_size)

        mask = attention_mask.unsqueeze(-1)
        score = score.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(score, dim=1)

        # label_repr shape: [batch, num_labels, hidden]
        label_repr = torch.einsum("bsh,bsl->blh", sequence_output, attn_weights)
        label_repr = label_repr + pooled_output.unsqueeze(1)

        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(label_repr)
            logits_list.append(self.label_classifier(dropped).squeeze(-1))
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
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
