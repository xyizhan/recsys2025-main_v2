import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from data_utils.data_dir import DataDir
from methods.contrastive_transformer_psych.data import (
    EVENT_TYPES,
    TYPE_TO_ID,
    build_client_sequences,
    collate_sequences,
    load_events_df,
    STATS_DIM
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="Directory with split_data output")
    p.add_argument("--embeddings-dir", type=str, required=True, help="Output directory for embeddings.npy and client_ids.npy")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--gnn-layers", type=int, default=3, help="Number of GNN layers")
    p.add_argument("--batch-size", type=int, default=128, help="Smaller batch for graph data")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature for contrastive loss")
    p.add_argument("--contrastive-weight", type=float, default=1.0, help="Weight for contrastive loss")
    p.add_argument("--prediction-weight", type=float, default=0.3, help="Weight for event prediction loss")
    return p

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    logits = (z1 @ z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, targets)
    loss2 = F.cross_entropy(logits.t(), targets)
    return (loss1 + loss2) * 0.5

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (B, L, D)
            pos_enc = self.pe[positions]  # (B, L, D)
            return x + pos_enc
        else:
            # (N, D)
            pos_enc = self.pe[positions]  # (N, D)
            return x + pos_enc

class EventEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        sku_buckets = 65536
        cat_buckets = 4096
        url_buckets = 65536
        price_buckets = 128
        type_buckets = 8
        
        # 事件编码
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_single_event(self, event_dict):
        x = self.emb_type(self._hash(event_dict['type_id'], self.emb_type.num_embeddings))
        if event_dict.get('sku_id') is not None:
            x = x + self.emb_sku(self._hash(event_dict['sku_id'], self.emb_sku.num_embeddings))
        if event_dict.get('cat_id') is not None:
            x = x + self.emb_cat(self._hash(event_dict['cat_id'], self.emb_cat.num_embeddings))
        if event_dict.get('price_id') is not None:
            x = x + self.emb_price(self._hash(event_dict['price_id'], self.emb_price.num_embeddings))
        if event_dict.get('url_id') is not None:
            x = x + self.emb_url(self._hash(event_dict['url_id'], self.emb_url.num_embeddings))
        if event_dict.get('query_vec') is not None:
            x = x + self.query_proj(event_dict['query_vec'])
        
        return self.ln(self.gelu(x))

class LightGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class TemporalLightGNN(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = EventEncoder(d_model=d_model, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=512)
        
        self.gnn_layers = nn.ModuleList([
            LightGNNLayer(d_model, d_model) for _ in range(gnn_layers)
        ])
        
        self.user_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim)
        )
        
        # 事件预测头
        self.event_pred_head = nn.Linear(d_model, len(EVENT_TYPES))
        self.edge_dropout = nn.Dropout(dropout)
        
    def build_temporal_graph(self, batch, event_embeddings):
        B, L, D = event_embeddings.shape
        user_nodes = torch.zeros(B, D, device=event_embeddings.device)
        
        # 所有节点：[用户节点, 事件节点]
        all_nodes = torch.cat([user_nodes, event_embeddings.reshape(B*L, D)], dim=0)
        
        # 构建边
        edge_list = []
        for b in range(B):
            user_idx = b
            for l in range(L):
                event_idx = B + b * L + l
                edge_list.append([user_idx, event_idx])
                edge_list.append([event_idx, user_idx])  # 双向
        
        # 事件->事件时序边 (t -> t+1)
        for b in range(B):
            for l in range(L-1):
                event_idx1 = B + b * L + l
                event_idx2 = B + b * L + (l + 1)
                edge_list.append([event_idx1, event_idx2])
                edge_list.append([event_idx2, event_idx1])  # 双向
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=event_embeddings.device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=event_embeddings.device)
        
        # 位置编码
        positions = torch.arange(L, device=event_embeddings.device).unsqueeze(0).expand(B, L).reshape(-1)
        event_indices = torch.arange(B*L, device=event_embeddings.device) + B
        all_nodes[event_indices] = self.pos_encoder(all_nodes[event_indices].reshape(-1, D), positions)
        
        return all_nodes, edge_index, B, L
    
    def forward(self, batch, return_predictions=False):
        B, L = batch["type_ids"].shape
        event_embeddings_list = []
        type_emb = self.event_encoder.emb_type(
            self.event_encoder._hash(batch["type_ids"], self.event_encoder.emb_type.num_embeddings)
        )
        
        event_embeddings = type_emb
        if "sku_ids" in batch:
            sku_emb = self.event_encoder.emb_sku(
                self.event_encoder._hash(batch["sku_ids"], self.event_encoder.emb_sku.num_embeddings)
            )
            event_embeddings = event_embeddings + sku_emb
            
        if "cat_ids" in batch:
            cat_emb = self.event_encoder.emb_cat(
                self.event_encoder._hash(batch["cat_ids"], self.event_encoder.emb_cat.num_embeddings)
            )
            event_embeddings = event_embeddings + cat_emb
            
        if "price_ids" in batch:
            price_emb = self.event_encoder.emb_price(
                self.event_encoder._hash(batch["price_ids"], self.event_encoder.emb_price.num_embeddings)
            )
            event_embeddings = event_embeddings + price_emb
            
        if "url_ids" in batch:
            url_emb = self.event_encoder.emb_url(
                self.event_encoder._hash(batch["url_ids"], self.event_encoder.emb_url.num_embeddings)
            )
            event_embeddings = event_embeddings + url_emb
            
        if "query_vec" in batch and batch["query_vec"] is not None:
            query_emb = self.event_encoder.query_proj(batch["query_vec"])
            event_embeddings = event_embeddings + query_emb
        
        event_embeddings = self.event_encoder.ln(self.event_encoder.gelu(event_embeddings))
        
        nodes, edge_index, num_users, seq_len = self.build_temporal_graph(batch, event_embeddings)
        x = nodes
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
        
        # 提取用户节点
        user_embeddings = x[:num_users]  # (B, D)
        user_embed = self.user_proj(user_embeddings)
        user_embed = F.normalize(user_embed, dim=-1)
        
        if self.training or return_predictions:
            event_nodes = x[num_users:].reshape(B, seq_len, -1)
            event_pred_logits = self.event_pred_head(event_nodes)  # (B, L, num_types)
            
            return {
                'user_embed': user_embed,
                'event_pred_logits': event_pred_logits,
                'nodes': nodes,
                'edge_index': edge_index
            }
        else:
            return {
                'user_embed': user_embed
            }


def compute_loss(model, batch, temperature=0.2, contrastive_weight=1.0, prediction_weight=0.3):
    # 创建两个增强视图
    with torch.no_grad():
        view1_outputs = model(batch, return_predictions=True)
        batch2 = {k: v.clone() for k, v in batch.items()}
        if "type_ids" in batch2:
            mask_prob = 0.1
            mask = torch.rand(batch2["type_ids"].shape, device=batch2["type_ids"].device) < mask_prob
            batch2["type_ids"][mask] = TYPE_TO_ID.get("MASK", 0)
    
    view2_outputs = model(batch2, return_predictions=True)
    
    # 对比学习损失
    contrastive_loss = info_nce_loss(
        view1_outputs['user_embed'], 
        view2_outputs['user_embed'],
        temperature=temperature
    )
    
    # 事件预测损失
    next_event_labels = prepare_sequential_labels(batch)  # (B, L)
    event_pred_logits = view1_outputs['event_pred_logits']  # (B, L, num_types)
    event_pred_logits_flat = event_pred_logits.reshape(-1, event_pred_logits.size(-1))
    next_event_labels_flat = next_event_labels.reshape(-1)
    
    # 只计算有效标签的损失
    valid_mask = next_event_labels_flat != -100
    if valid_mask.any():
        prediction_loss = F.cross_entropy(
            event_pred_logits_flat[valid_mask],
            next_event_labels_flat[valid_mask]
        )
    else:
        prediction_loss = torch.tensor(0.0, device=batch["type_ids"].device)
    
    # 总损失
    total_loss = contrastive_weight * contrastive_loss + prediction_weight * prediction_loss
    
    return total_loss, contrastive_loss.item(), prediction_loss.item() if valid_mask.any() else 0.0

def prepare_sequential_labels(batch):
    type_ids = batch["type_ids"]
    B, L = type_ids.shape
    
    if L <= 1:
        return torch.full((B, L), -100, dtype=torch.long, device=type_ids.device)
    
    next_event_labels = torch.full((B, L), -100, dtype=torch.long, device=type_ids.device)
    next_event_labels[:, :-1] = type_ids[:, 1:]  # 每个位置预测下一个位置
    
    valid_mask = (next_event_labels >= 2) & (next_event_labels <= 6)
    next_event_labels[valid_mask] = next_event_labels[valid_mask] - 2  # 映射到0-4
    next_event_labels[~valid_mask] = -100  # 无效标签
    
    return next_event_labels

def train_lightgnn(
    model: TemporalLightGNN,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    log_interval: int,
    use_amp: bool,
    amp_device_type: str,
    temperature: float,
    contrastive_weight: float,
    prediction_weight: float,
):
    ids = list(client_groups.keys())
    if len(ids) == 0:
        logger.warning("No clients available for training")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = max(1, math.ceil(len(ids) / batch_size))
    
    logger.info(f"Training on {len(ids)} clients ({steps_per_epoch} steps/epoch)")

    for ep in range(epochs):
        np.random.shuffle(ids)
        epoch_start = time.time()
        total_loss = 0
        total_contrastive_loss = 0
        total_prediction_loss = 0
        
        for step_idx in range(steps_per_epoch):
            # 准备批次数据
            start = step_idx * batch_size
            batch_ids = ids[start: start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                # 计算损失
                loss, contrastive_loss, prediction_loss = compute_loss(
                    model, batch, temperature, contrastive_weight, prediction_weight
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss
            total_prediction_loss += prediction_loss
            
            # 日志记录
            if (step_idx + 1) % log_interval == 0:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - step_idx - 1)
                
                logger.info(
                    f"Epoch {ep+1}/{epochs} | Step {step_idx+1}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | Contrastive: {contrastive_loss:.4f} | "
                    f"Prediction: {prediction_loss:.4f} | ETA: {eta:.1f}s"
                )
        
        avg_loss = total_loss / steps_per_epoch
        avg_contrastive_loss = total_contrastive_loss / steps_per_epoch
        avg_prediction_loss = total_prediction_loss / steps_per_epoch
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {ep+1} finished in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | Contrastive: {avg_contrastive_loss:.4f} | "
            f"Prediction: {avg_prediction_loss:.4f}"
        )

def generate_embeddings(
    model: TemporalLightGNN,
    device: torch.device,
    client_groups,
    relevant_client_ids: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    log_interval: int,
):
    model.eval()
    client_ids = relevant_client_ids.astype(np.int64)
    embeddings = np.zeros((client_ids.shape[0], embedding_dim), dtype=np.float16)
    
    total = client_ids.shape[0]
    steps = max(1, math.ceil(total / batch_size))
    
    start_time = time.time()
    
    with torch.no_grad():
        for step_idx, i in enumerate(range(0, total, batch_size)):
            batch_ids = client_ids[i: i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]   
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch, return_predictions=False)
            batch_embeddings = outputs["user_embed"].detach().cpu().numpy()
            
            end_idx = min(i + batch_size, total)
            embeddings[i:end_idx] = batch_embeddings.astype(np.float16)
            
            if (step_idx + 1) % log_interval == 0:
                processed = end_idx
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                logger.info(
                    f"Embedding gen {step_idx+1}/{steps} | "
                    f"{processed}/{total} samples | {rate:.1f} samples/s"
                )
    
    return client_ids, embeddings

def main(params):
    # 初始化
    data_dir = DataDir(Path(params.data_dir))
    embeddings_dir = Path(params.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    
    # 加载数据
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    logger.info("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    logger.info("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)
    
    # 设备设置
    if params.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(params.device)
    
    use_amp = device.type == "cuda" and not params.no_amp
    
    # 创建模型
    model = TemporalLightGNN(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        gnn_layers=params.gnn_layers,
        dropout=params.dropout,
    )
    model.to(device)
    
    logger.info(f"Using device: {device} (AMP: {use_amp})")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    logger.info("Training Temporal LightGNN...")
    train_lightgnn(
        model=model,
        device=device,
        client_groups=client_groups,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        epochs=params.epochs,
        lr=params.lr,
        log_interval=params.log_interval,
        use_amp=use_amp,
        amp_device_type="cuda" if device.type == "cuda" else "cpu",
        temperature=params.temperature,
        contrastive_weight=params.contrastive_weight,
        prediction_weight=params.prediction_weight,
    )
    
    # 生成嵌入
    logger.info("Generating embeddings...")
    client_ids, embeddings = generate_embeddings(
        model=model,
        device=device,
        client_groups=client_groups,
        relevant_client_ids=relevant_client_ids,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        embedding_dim=params.embedding_dim,
        log_interval=params.log_interval,
    )
    
    # 保存结果
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    logger.info(f"Saved embeddings for {len(client_ids)} clients to {embeddings_dir}")
    logger.info(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())