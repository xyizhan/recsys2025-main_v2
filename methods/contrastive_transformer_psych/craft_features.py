# python -m methods.contrastive_transformer_psych.craft_features
import argparse
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pickle

from data_utils.data_dir import DataDir
from methods.contrastive_transformer_psych.data import (
    build_client_sequences,
    load_events_df,
    compute_user_statistics,
)

warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-compute handcrafted user statistics.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to split_data directory used by DataDir",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="methods/contrastive_transformer_psych/user_stats_cache.pkl",
        help="Where to save the serialized feature dictionary",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    data_dir = DataDir(Path(args.data_dir))
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    print(f"[craft_features] Loading input events from {data_dir.data_dir}")
    dfs = load_events_df(data_dir=data_dir)
    print("[craft_features] Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)

    cache: dict[int, np.ndarray] = {}
    all_stats = []
    client_ids = []

    for client_id, df in tqdm(client_groups.items(), desc="calculate features"):
        stats = compute_user_statistics(df)
        cache[client_id] = stats
        all_stats.append(stats)
        client_ids.append(client_id)
    all_stats = np.stack(all_stats, axis=0)

    means = np.zeros(all_stats.shape[1], dtype=np.float64)
    stds = np.zeros_like(means)
    for i in range(all_stats.shape[1]):
        feature = all_stats[:, i]
        median = np.median(feature)
        mad = np.median(np.abs(feature - median))
        if mad == 0:
            mad = np.std(feature) + 1e-8
        means[i] = median
        stds[i] = mad * 1.4826

    standardized_cache: dict[int, np.ndarray] = {}
    for client_id in tqdm(client_ids, desc="standardize features"):
        raw_stats = cache[client_id]
        standardized = (raw_stats - means) / (stds + 1e-8)
        standardized_cache[client_id] = standardized.astype(np.float32)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(standardized_cache, f)
    print(f"[craft_features] Saved {len(standardized_cache)} user stats to {output_path}")


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
