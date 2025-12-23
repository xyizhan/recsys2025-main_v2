from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from methods.contrastive_transformer.data import TYPE_TO_ID


def _timestamps_to_seconds(timestamps: pd.Series) -> np.ndarray:
    """
    Convert pandas datetime Series to float seconds since epoch.
    """
    values = timestamps.values
    if not np.issubdtype(values.dtype, np.datetime64):
        values = pd.to_datetime(timestamps).values
    ns = values.astype("datetime64[ns]").astype(np.int64)
    return (ns.astype(np.float64) / 1e9).astype(np.float32)


def collate_tgn_sequences(
    batch_seqs: List[Optional[pd.DataFrame]],
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Turn a list of per-client DataFrames into padded tensors for the TGN encoder.
    """
    batch_size = len(batch_seqs)
    type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    sku_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    timestamps = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for idx, df in enumerate(batch_seqs):
        if df is None or df.empty:
            continue
        sub = df.tail(max_len)
        length = sub.shape[0]
        etypes = [TYPE_TO_ID.get(t, 2) for t in sub["event_type"].tolist()]
        type_ids[idx, :length] = torch.tensor(etypes, dtype=torch.long)
        sku_vals = sub["sku"].fillna(0).astype(np.int64).to_numpy()
        sku_ids[idx, :length] = torch.from_numpy(sku_vals)
        ts = _timestamps_to_seconds(sub["timestamp"])
        timestamps[idx, :length] = torch.from_numpy(ts)
        mask[idx, :length] = True

    return {
        "type_ids": type_ids,
        "sku_ids": sku_ids,
        "timestamps": timestamps,
        "mask": mask,
    }
