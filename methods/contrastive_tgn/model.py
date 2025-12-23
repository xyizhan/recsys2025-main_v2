from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


@dataclass
class TGNOutput:
    user_embeddings: torch.Tensor
    pos_item_embeddings: torch.Tensor
    pos_item_ids: torch.Tensor
    lengths: torch.Tensor


class TGNEncoder(nn.Module):
    """
    Minimal TGN-style encoder with GRU updates and Δt encoding.
    """

    def __init__(
        self,
        type_vocab_size: int,
        item_bucket_count: int,
        embed_dim: int,
        delta_hidden_dim: int = 64,
        dropout: float = 0.1,
        delta_scale: float = 3600.0,
    ) -> None:
        super().__init__()
        self.item_bucket_count = int(item_bucket_count)
        self.embed_dim = embed_dim
        self.delta_scale = float(delta_scale)

        self.type_embedding = nn.Embedding(type_vocab_size, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_bucket_count, embed_dim, padding_idx=0)
        self.delta_encoder = nn.Sequential(
            nn.Linear(1, delta_hidden_dim),
            nn.SiLU(),
            nn.Linear(delta_hidden_dim, embed_dim),
        )
        self.input_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def _hash_items(self, sku_ids: torch.Tensor) -> torch.Tensor:
        return torch.remainder(sku_ids, self.item_bucket_count)

    def forward(self, batch) -> TGNOutput:
        type_ids: torch.Tensor = batch["type_ids"]
        sku_ids: torch.Tensor = batch["sku_ids"]
        timestamps: torch.Tensor = batch["timestamps"]
        mask: torch.Tensor = batch["mask"]

        if not torch.is_floating_point(timestamps):
            timestamps = timestamps.float()

        hashed_items = self._hash_items(sku_ids)

        type_emb = self.type_embedding(type_ids.clamp(min=0))
        item_emb = self.item_embedding(hashed_items)

        delta = torch.zeros_like(timestamps)
        delta[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        delta = torch.clamp(delta, min=0.0)
        delta = torch.log1p(delta / max(1e-6, self.delta_scale))
        delta_emb = self.delta_encoder(delta.unsqueeze(-1))

        inputs = self.input_ln(type_emb + item_emb + delta_emb)
        inputs = inputs * mask.unsqueeze(-1)
        inputs = self.dropout(inputs)

        lengths = mask.long().sum(dim=1)
        safe_lengths = lengths.clone()
        safe_lengths[safe_lengths <= 0] = 1

        packed = pack_padded_sequence(
            inputs,
            lengths=safe_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        user_emb = hidden[-1]
        user_emb = F.normalize(user_emb, dim=-1)

        batch_indices = torch.arange(type_ids.size(0), device=type_ids.device)
        last_indices = torch.clamp(lengths - 1, min=0)
        pos_item_ids = hashed_items[batch_indices, last_indices]
        pos_item_embeddings = self.item_embedding(pos_item_ids)
        pos_item_embeddings = F.normalize(pos_item_embeddings, dim=-1)

        user_emb = torch.where(
            lengths.unsqueeze(-1) > 0,
            user_emb,
            torch.zeros_like(user_emb),
        )
        pos_item_embeddings = torch.where(
            lengths.unsqueeze(-1) > 0,
            pos_item_embeddings,
            torch.zeros_like(pos_item_embeddings),
        )

        return TGNOutput(
            user_embeddings=user_emb,
            pos_item_embeddings=pos_item_embeddings,
            pos_item_ids=pos_item_ids,
            lengths=lengths,
        )

    def generate_embeddings(self, batch):
        with torch.no_grad():
            output = self.forward(batch)
            return output.user_embeddings


def sampled_ranking_loss(
    user_embeddings: torch.Tensor,
    pos_item_embeddings: torch.Tensor,
    lengths: torch.Tensor,
    item_embedding: nn.Embedding,
    item_bucket_count: int,
    num_negatives: int = 20,
) -> torch.Tensor:
    valid_mask = lengths > 0
    if not torch.any(valid_mask):
        return user_embeddings.new_tensor(0.0)
    user_valid = user_embeddings[valid_mask]
    pos_valid = pos_item_embeddings[valid_mask]

    pos_scores = torch.sum(user_valid * pos_valid, dim=-1)
    neg_ids = torch.randint(
        low=0,
        high=item_bucket_count,
        size=(user_valid.size(0), num_negatives),
        device=user_embeddings.device,
    )
    neg_emb = item_embedding(neg_ids)
    neg_scores = torch.einsum("bnd,bd->bn", neg_emb, user_valid)

    diff = pos_scores.unsqueeze(1) - neg_scores
    loss = -F.logsigmoid(diff).mean()
    return loss
