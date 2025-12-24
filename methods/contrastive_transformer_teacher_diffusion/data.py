from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from methods.contrastive_transformer.data import (
    TYPE_TO_ID,
    build_client_sequences,
    load_events_df,
)

FieldBatch = Dict[str, torch.Tensor]


def _compute_delta_seconds(timestamps) -> np.ndarray:
    if timestamps.size == 0:
        return np.zeros(0, dtype=np.float32)
    # timestamps is numpy array of datetime64[ns]
    ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    diffs = np.diff(ns, prepend=ns[0])
    seconds = diffs.astype(np.float64) / 1e9
    seconds = np.clip(seconds, 0.0, 7 * 24 * 3600)
    return seconds.astype(np.float32)


def collate_sequences(batch_seqs: List, max_len: int, stats_feat: List[np.ndarray] | None = None, stats_dim: int = 0) -> FieldBatch:
    B = len(batch_seqs)
    device = torch.device("cpu")
    type_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    sku_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    cat_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    price_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    url_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    query_vec = torch.zeros((B, max_len, 16), dtype=torch.float32, device=device)
    delta_times = torch.zeros((B, max_len), dtype=torch.float32, device=device)

    sku_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    cat_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    price_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    url_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    query_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    delta_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    stats_vec = torch.zeros((B, stats_dim), dtype=torch.float32, device=device) if stats_dim > 0 else None

    for i, df in enumerate(batch_seqs):
        if df is None or df.shape[0] == 0:
            continue
        L = min(df.shape[0], max_len)
        sub = df.iloc[-L:]
        et_list = sub["event_type"].tolist()
        type_ids[i, :L] = torch.tensor([TYPE_TO_ID.get(t, 2) for t in et_list], dtype=torch.long)
        delta_vals = _compute_delta_seconds(sub["timestamp"].to_numpy())
        delta_times[i, :L] = torch.tensor(delta_vals, dtype=torch.float32)
        delta_mask[i, :L] = 1.0

        for name, target, mask in [
            ("sku", sku_ids, sku_mask),
            ("category", cat_ids, cat_mask),
            ("price", price_ids, price_mask),
            ("url", url_ids, url_mask),
        ]:
            vals = sub[name].fillna(0).astype(np.int64).to_numpy()
            mask_vals = (~sub[name].isna()).astype(np.float32).to_numpy()
            target[i, :L] = torch.tensor(vals, dtype=torch.long)
            mask[i, :L] = torch.tensor(mask_vals, dtype=torch.float32)

        q_rows: List[np.ndarray] = []
        q_mask_vals: List[float] = []
        for t, q in zip(et_list, sub["query"].tolist() if "query" in sub.columns else [None] * L):
            if t == "search_query" and q is not None and str(q).lower() != "nan":
                from methods.contrastive_transformer.data import _parse_query

                q_rows.append(_parse_query(str(q)))
                q_mask_vals.append(1.0)
            else:
                q_rows.append(np.zeros(16, dtype=np.float32))
                q_mask_vals.append(0.0)
        qmat = np.stack(q_rows, axis=0) if len(q_rows) > 0 else np.zeros((L, 16), dtype=np.float32)
        query_vec[i, :L, :] = torch.tensor(qmat, dtype=torch.float32)
        query_mask[i, :L] = torch.tensor(q_mask_vals, dtype=torch.float32)
        if stats_vec is not None and stats_feat is not None:
            feat = stats_feat[i]
            if feat is None:
                stats_vec[i, :] = torch.zeros(stats_dim, dtype=torch.float32)
            else:
                vec = np.asarray(feat, dtype=np.float32)
                if vec.shape[0] != stats_dim:
                    vec = np.resize(vec, stats_dim)
                stats_vec[i, :] = torch.tensor(vec, dtype=torch.float32)

    batch = {
        "type_ids": type_ids,
        "sku_ids": sku_ids,
        "cat_ids": cat_ids,
        "price_ids": price_ids,
        "url_ids": url_ids,
        "query_vec": query_vec,
        "delta_times": delta_times,
        "sku_mask": sku_mask,
        "cat_mask": cat_mask,
        "price_mask": price_mask,
        "url_mask": url_mask,
        "query_mask": query_mask,
        "delta_mask": delta_mask,
    }
    if stats_vec is not None:
        batch["stats_vec"] = stats_vec
    return batch


def augment_views(
    batch: FieldBatch,
    mask_prob: float = 0.15,
    drop_prob: float = 0.1,
) -> Tuple[FieldBatch, FieldBatch]:
    def _make_view() -> FieldBatch:
        device = batch["type_ids"].device
        B, L = batch["type_ids"].shape
        keep = (torch.rand((B, L), device=device) > drop_prob)
        keep_long = keep.long()
        keep_float = keep.float()
        keep_float3 = keep_float.unsqueeze(-1)
        view: FieldBatch = {}
        for key, tensor in batch.items():
            if tensor.dim() == 3:
                view[key] = tensor * keep_float3
            else:
                dtype = tensor.dtype
                if dtype == torch.long:
                    view[key] = tensor * keep_long
                else:
                    view[key] = tensor * keep_float
        if "stats_vec" in batch:
            view["stats_vec"] = batch["stats_vec"]
        mask_token = torch.tensor(TYPE_TO_ID["MASK"], device=device)
        mask_flag = torch.rand((B, L), device=device) < mask_prob
        view["type_ids"] = torch.where(mask_flag, mask_token, view["type_ids"])
        return view

    return _make_view(), _make_view()


__all__ = [
    "load_events_df",
    "build_client_sequences",
    "collate_sequences",
    "augment_views",
]
