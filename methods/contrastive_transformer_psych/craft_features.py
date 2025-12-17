# python -m methods.contrastive_transformer_psych.craft_features
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from scipy import stats
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
import gc

from data_utils.data_dir import DataDir
from methods.contrastive_transformer_psych.data import (
    EVENT_TYPES,
    TYPE_TO_ID,
    augment_views,
    build_client_sequences,
    collate_sequences,
    load_events_df,
    compute_user_statistics,
    STATS_DIM
)

warnings.filterwarnings('ignore')

reference_time = datetime(2022, 11, 9)  
start_date = datetime(2022, 6, 23)  


def main():
    data_dir = DataDir(Path("/root/shared-nvme/data"))
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    print("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    print("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)

    cache_dir = "/root/shared-nvme/data/feature"
    cache_path = Path(cache_dir) / 'user_stats_cache.pkl'
    cache = {}
    all_stats = []
    client_ids = []

    for client_id, df in tqdm(client_groups.items(), desc="calculate features"):
        stats = compute_user_statistics(df)
        cache[client_id] = stats
        all_stats.append(stats)
        client_ids.append(client_id)
    all_stats = np.stack(all_stats, axis=0)

    means = np.zeros(all_stats.shape[1])
    stds = np.zeros(all_stats.shape[1])
    
    for i in range(all_stats.shape[1]):
        feature = all_stats[:, i]
        median = np.median(feature)
        mad = np.median(np.abs(feature - median))
        if mad == 0:
            mad = np.std(feature) + 1e-8
        means[i] = median
        stds[i] = mad * 1.4826
    
    standardized_cache = {}
    for client_id in tqdm(client_ids, desc="standardize features"):
        raw_stats = cache[client_id]
        standardized = (raw_stats - means) / (stds + 1e-8)
        standardized_cache[client_id] = standardized.astype(np.float32)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(standardized_cache, f)
    print(f"预计算完成，缓存已保存到 {cache_path}")

if __name__ == "__main__":
    main()
    