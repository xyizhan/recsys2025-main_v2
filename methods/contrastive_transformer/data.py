import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

from data_utils.data_dir import DataDir
from data_utils.utils import load_with_properties

EVENT_TYPES = [
    "product_buy",
    "add_to_cart",
    "remove_from_cart",
    "page_visit",
    "search_query",
]

# map event_type to integer id (reserve 0 for MASK, 1 for CLS, start real events at 2)
TYPE_TO_ID = {
    "MASK": 0,
    "CLS": 1,
    "product_buy": 2,
    "add_to_cart": 3,
    "remove_from_cart": 4,
    "page_visit": 5,
    "search_query": 6,
}


def _parse_query(q: str) -> np.ndarray:
    q = q.strip().replace("[", "").replace("]", "")
    if q == "":
        return np.zeros(16, dtype=np.float32)
    arr = np.array([int(s) for s in q.split(" ") if s != ""], dtype=np.float32)
    if arr.size != 16:
        # pad or crop to 16
        if arr.size < 16:
            arr = np.pad(arr, (0, 16 - arr.size))
        else:
            arr = arr[:16]
    # scale integers (0..255) to 0..1
    return arr / 255.0


def load_events_df(data_dir: DataDir) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for et in EVENT_TYPES:
        df = load_with_properties(data_dir=data_dir, event_type=et)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs[et] = df
    return dfs


def build_client_sequences(dfs: Dict[str, pd.DataFrame], relevant_client_ids: np.ndarray) -> Dict[int, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for et, df in dfs.items():
        # keep only relevant clients
        df = df[df["client_id"].isin(relevant_client_ids)].copy()
        df["event_type"] = et
        # ensure missing columns exist
        for col in ["sku", "category", "price", "url", "query"]:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[["client_id", "timestamp", "event_type", "sku", "category", "price", "url", "query"]])
    all_events = pd.concat(frames, axis=0, ignore_index=True)
    all_events.sort_values(["client_id", "timestamp"], inplace=True)
    groups: Dict[int, pd.DataFrame] = {cid: grp for cid, grp in all_events.groupby("client_id")}
    return groups


def collate_sequences(batch_seqs: List[pd.DataFrame], max_len: int) -> Dict[str, torch.Tensor]:
    B = len(batch_seqs)
    type_ids = torch.zeros((B, max_len), dtype=torch.long)
    sku_ids = torch.zeros((B, max_len), dtype=torch.long)
    cat_ids = torch.zeros((B, max_len), dtype=torch.long)
    price_ids = torch.zeros((B, max_len), dtype=torch.long)
    url_ids = torch.zeros((B, max_len), dtype=torch.long)
    query_vec = torch.zeros((B, max_len, 16), dtype=torch.float32)

    for i, df in enumerate(batch_seqs):
        if df is None or df.shape[0] == 0:
            continue
        L = min(df.shape[0], max_len)
        # take most recent max_len
        sub = df.iloc[-L:]
        type_ids[i, :L] = torch.tensor([TYPE_TO_ID.get(t, 2) for t in sub["event_type"].tolist()], dtype=torch.long)
        # NaN -> 0 index (will hash to some bucket)
        for name, target in [("sku", sku_ids), ("category", cat_ids), ("price", price_ids), ("url", url_ids)]:
            vals = sub[name].fillna(0).astype(np.int64).to_numpy()
            target[i, :L] = torch.tensor(vals, dtype=torch.long)
        qmat = np.stack([_parse_query(q) for q in sub["query"].astype(str).tolist()], axis=0) if "query" in sub.columns else np.zeros((L, 16), dtype=np.float32)
        query_vec[i, :L, :] = torch.tensor(qmat, dtype=torch.float32)
    return {
        "type_ids": type_ids,
        "sku_ids": sku_ids,
        "cat_ids": cat_ids,
        "price_ids": price_ids,
        "url_ids": url_ids,
        "query_vec": query_vec,
    }


def augment_views(batch: Dict[str, torch.Tensor], mask_prob: float = 0.1, drop_prob: float = 0.1) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # simple subsampling and masking over time dimension
    def _aug_one(x: torch.Tensor) -> torch.Tensor:
        B, L = x.size(0), x.size(1)
        keep = torch.rand((B, L), device=x.device) > drop_prob
        x = x * keep.long()
        return x

    def _mask_types(t: torch.Tensor) -> torch.Tensor:
        B, L = t.size(0), t.size(1)
        m = torch.rand((B, L), device=t.device) < mask_prob
        t = torch.where(m, torch.tensor(TYPE_TO_ID["MASK"], device=t.device), t)
        return t

    v1 = {k: _aug_one(v) if v.dim() == 2 else v for k, v in batch.items()}
    v2 = {k: _aug_one(v) if v.dim() == 2 else v for k, v in batch.items()}
    v1["type_ids"] = _mask_types(v1["type_ids"])
    v2["type_ids"] = _mask_types(v2["type_ids"])
    return v1, v2