import argparse
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsFeatureBaseline:
    def __init__(self, stats_feature_path: str):
        self.stats_feature_path = stats_feature_path
        
        # 加载手工特征
        with open(stats_feature_path, 'rb') as f:
            self.stats_features = pickle.load(f)
        
        logger.info(f"Loaded stats features for {len(self.stats_features)} clients")
    
    def _stack_features(self, features_dict: Dict[int, np.ndarray]) -> np.ndarray:
        all_features = []
        client_ids = []
        
        for client_id, feat in features_dict.items():
            if feat is not None and len(feat) > 0:
                all_features.append(feat)
                client_ids.append(client_id)
        
        if not all_features:
            raise ValueError("No valid features found!")
        
        all_features = np.stack(all_features)  # [n_clients, feat_dim]
        
        return np.array(client_ids), all_features
    
    def generate_embeddings(self, relevant_client_ids: np.ndarray) -> tuple:
        relevant_features = {}
        missing_clients = 0
        
        for client_id in relevant_client_ids:
            client_id_int = int(client_id)
            if client_id_int in self.stats_features:
                relevant_features[client_id_int] = self.stats_features[client_id_int]
            else:
                if len(self.stats_features) > 0:
                    sample_feat = next(iter(self.stats_features.values()))
                    relevant_features[client_id_int] = np.zeros_like(sample_feat)
                else:
                    relevant_features[client_id_int] = np.zeros(46)  # STATS_DIM
                missing_clients += 1
        
        if missing_clients > 0:
            logger.warning(f"{missing_clients} clients not found in stats features, using zero vectors")

        client_ids, embeddings = self._stack_features(relevant_features)
        
        logger.info(f"Generated embeddings for {len(client_ids)} clients: shape={embeddings.shape}")
        
        return client_ids, embeddings.astype(np.float16)


def main():
    embeddings_dir = '/root/shared-nvme/data/embedding/stats_features'
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相关用户ID
    relevant_client_ids = np.load('/root/shared-nvme/data/input/relevant_clients.npy')
    logger.info(f"Loaded {len(relevant_client_ids)} relevant clients")
    
    baseline = StatsFeatureBaseline(
        stats_feature_path='/root/shared-nvme/data/feature/user_stats_cache.pkl'
    )
    
    client_ids, embeddings = baseline.generate_embeddings(relevant_client_ids)

    output_client_path = embeddings_dir / "client_ids.npy"
    output_embedding_path = embeddings_dir / "embeddings.npy"
    
    np.save(output_client_path, client_ids)
    np.save(output_embedding_path, embeddings)
    
    logger.info(f"Saved client_ids to {output_client_path}")
    logger.info(f"Saved embeddings to {output_embedding_path}")
    
    # 打印一些统计信息
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")
    logger.info(f"Embeddings mean: {embeddings.mean():.6f}")
    logger.info(f"Embeddings std: {embeddings.std():.6f}")
    logger.info(f"Embeddings min/max: [{embeddings.min():.6f}, {embeddings.max():.6f}]")


if __name__ == "__main__":
    main()