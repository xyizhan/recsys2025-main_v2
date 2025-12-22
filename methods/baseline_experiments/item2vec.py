# pip install torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple
# python -m methods.baseline.item2vec
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from collections import defaultdict
import logging
from tqdm import tqdm
from data_utils.data_dir import DataDir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Item2VecBaseline:
    def __init__(self, embedding_dim=512, window=5, min_count=5):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.item_vectors = None  # sku -> vector
        self.item_to_idx = None
        
    def build_sequences(self, dfs, event_types=['product_buy', 'add_to_cart']):
        user_sequences = defaultdict(list)
        sku_counter = defaultdict(int)
        
        for event_type in event_types:
            df = dfs[event_type]
            for _, row in tqdm(df.iterrows(), desc=f"Processing {event_type}"):
                user_sequences[row['client_id']].append(row['sku'])
                sku_counter[row['sku']] += 1
        
        # 过滤低频商品
        filtered_sequences = {}
        for client_id, skus in user_sequences.items():
            filtered = [s for s in skus if sku_counter[s] >= self.min_count]
            if len(filtered) >= 2:
                filtered_sequences[client_id] = filtered
        
        return filtered_sequences, sku_counter
    
    def train_item_vectors(self, sequences):
        sequences_list = list(sequences.values())
        
        model = Word2Vec(
            sentences=sequences_list,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            epochs=10,
            seed=42
        )
        
        self.item_vectors = {item: model.wv[item] for item in model.wv.index_to_key}
        self.item_to_idx = {item: idx for idx, item in enumerate(model.wv.index_to_key)}
        
        logger.info(f"Trained on {len(sequences)} users, {len(self.item_vectors)} unique items")
        return model
    
    def get_user_embedding(self, sku_list):
        if not self.item_vectors:
            raise ValueError("Model not trained yet")
        
        valid_vecs = []
        for sku in sku_list:
            if sku in self.item_vectors:
                valid_vecs.append(self.item_vectors[sku])
        
        if not valid_vecs:
            return np.zeros(self.embedding_dim)
        
        user_vec = np.mean(valid_vecs, axis=0)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm
        return user_vec
    
    def generate_all_embeddings(self, sequences, relevant_client_ids):
        embeddings = []
        valid_client_ids = []
        
        for client_id in tqdm(relevant_client_ids, desc="Generating embeddings"):
            client_id_int = int(client_id)
            if client_id_int in sequences:
                vec = self.get_user_embedding(sequences[client_id_int])
                embeddings.append(vec)
                valid_client_ids.append(client_id_int)
            else:
                embeddings.append(np.zeros(self.embedding_dim))
                valid_client_ids.append(client_id_int)
        
        return np.array(valid_client_ids), np.array(embeddings, dtype=np.float16)

def main():
    # 加载数据
    data_dir = DataDir(Path('/root/shared-nvme/data'))
    relevant_client_ids = np.load('/root/shared-nvme/data/input/relevant_clients.npy')
    logger.info(f"Loaded {len(relevant_client_ids)} relevant clients")
    
    from methods.contrastive_transformer_psych.data import load_events_df
    dfs = load_events_df(data_dir)
    
    # 训练Item2Vec
    baseline = Item2VecBaseline(
        embedding_dim=512,
        window=7,
        min_count=5
    )
    
    sequences, _ = baseline.build_sequences(dfs)
    baseline.train_item_vectors(sequences)
    
    # 生成嵌入
    client_ids, embeddings = baseline.generate_all_embeddings(sequences, relevant_client_ids)
    
    # 保存
    embeddings_dir = Path('/root/shared-nvme/data/embedding/item2vec')
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    
    logger.info(f"Saved embeddings for {len(client_ids)} clients")
    logger.info(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()