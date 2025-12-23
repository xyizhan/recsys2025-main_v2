from __future__ import annotations

from dataclasses import dataclass
import math

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
        neighbor_agg: str = "none",
        neighbor_k: int = 5,
    ) -> None:
        super().__init__()
        self.item_bucket_count = int(item_bucket_count)
        self.embed_dim = embed_dim
        self.delta_scale = float(delta_scale)
        self.neighbor_agg = neighbor_agg
        self.neighbor_k = int(neighbor_k)

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
        if self.neighbor_agg not in {"none", "mean", "tgat"}:
            raise ValueError(f"Unsupported neighbor aggregator '{self.neighbor_agg}'")
        if self.neighbor_agg == "tgat":
            self.tgat_query = nn.Linear(embed_dim, embed_dim, bias=False)
            self.tgat_key = nn.Linear(embed_dim, embed_dim, bias=False)
            self.tgat_value = nn.Linear(embed_dim, embed_dim, bias=False)
            self.tgat_time_mlp = nn.Sequential(
                nn.Linear(1, delta_hidden_dim),
                nn.SiLU(),
                nn.Linear(delta_hidden_dim, embed_dim),
            )

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

        neighbor_context = self._compute_neighbor_context(
            item_emb=item_emb,
            timestamps=timestamps,
            mask=mask,
        )

        inputs = self.input_ln(type_emb + item_emb + delta_emb + neighbor_context)
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

    def _compute_neighbor_context(
        self,
        item_emb: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.neighbor_agg == "none" or self.neighbor_k <= 0:
            return torch.zeros_like(item_emb)
        if self.neighbor_agg == "mean":
            return self._mean_neighbor_context(item_emb, mask)
        if self.neighbor_agg == "tgat":
            return self._tgat_neighbor_context(item_emb, timestamps, mask)
        raise RuntimeError(f"Unknown neighbor aggregator '{self.neighbor_agg}'")

    def _mean_neighbor_context(
        self,
        item_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_item = item_emb * mask.unsqueeze(-1)
        context = torch.zeros_like(item_emb)
        counts = torch.zeros_like(mask, dtype=item_emb.dtype)
        for offset in range(1, self.neighbor_k + 1):
            shifted_emb = torch.roll(masked_item, shifts=offset, dims=1)
            shifted_emb[:, :offset, :] = 0
            context = context + shifted_emb
            shifted_mask = torch.roll(mask.float(), shifts=offset, dims=1)
            shifted_mask[:, :offset] = 0
            counts = counts + shifted_mask
        counts_unsqueezed = counts.unsqueeze(-1)
        counts_safe = counts_unsqueezed.clamp(min=1e-6)
        avg = context / counts_safe
        zero_mask = (counts_unsqueezed > 0)
        avg = torch.where(zero_mask, avg, torch.zeros_like(avg))
        return avg

    def _tgat_neighbor_context(
        self,
        item_emb: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_item = item_emb * mask.unsqueeze(-1)
        query = self.tgat_query(item_emb)
        key_base = self.tgat_key(masked_item)
        value_base = self.tgat_value(masked_item)
        scores_list = []
        value_list = []
        mask_list = []
        for offset in range(1, self.neighbor_k + 1):
            shifted_key = torch.roll(key_base, shifts=offset, dims=1)
            shifted_value = torch.roll(value_base, shifts=offset, dims=1)
            shifted_mask = torch.roll(mask.float(), shifts=offset, dims=1)
            shifted_time = torch.roll(timestamps, shifts=offset, dims=1)
            shifted_key[:, :offset, :] = 0
            shifted_value[:, :offset, :] = 0
            shifted_mask[:, :offset] = 0
            shifted_time[:, :offset] = 0
            delta = torch.clamp(timestamps - shifted_time, min=0.0)
            delta = torch.log1p(delta / max(self.delta_scale, 1e-6))
            time_emb = self.tgat_time_mlp(delta.unsqueeze(-1))
            attn_key = shifted_key + time_emb
            scores = torch.sum(query * attn_key, dim=-1) / math.sqrt(self.embed_dim)
            scores_list.append(scores)
            value_list.append(shifted_value)
            mask_list.append(shifted_mask)
        if not scores_list:
            return torch.zeros_like(item_emb)
        scores = torch.stack(scores_list, dim=-1)
        values = torch.stack(value_list, dim=2)
        neighbor_mask = torch.stack(mask_list, dim=-1)
        inf_mask = torch.full_like(scores, -1e9)
        scores = torch.where(neighbor_mask > 0, scores, inf_mask)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * neighbor_mask
        normalizer = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        weights = weights / normalizer
        context = torch.sum(weights.unsqueeze(-1) * values, dim=2)
        context = torch.where(mask.unsqueeze(-1), context, torch.zeros_like(context))
        return context


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
