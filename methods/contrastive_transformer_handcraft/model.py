import torch
from torch import nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        length = x.size(1)
        return x + self.pe[:length].unsqueeze(0)


class FeatureGateFusion(nn.Module):
    """手工特征门控融合模块"""
    def __init__(self, d_model: int, user_feat_dim: int, product_feat_dim: int):
        super().__init__()
        self.user_gate = nn.Sequential(
            nn.Linear(user_feat_dim + d_model, d_model),
            nn.Sigmoid()
        )
        self.user_proj = nn.Linear(user_feat_dim, d_model)
        
        self.product_gate = nn.Sequential(
            nn.Linear(product_feat_dim + d_model, d_model),
            nn.Sigmoid()
        )
        self.product_proj = nn.Linear(product_feat_dim, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, event_embed: torch.Tensor, user_features: Optional[torch.Tensor] = None, 
                product_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合事件嵌入和手工特征
        
        Args:
            event_embed: (B, L, D) 事件嵌入
            user_features: (B, user_dim) 用户特征
            product_features: (B, L, product_dim) 商品特征
        
        Returns:
            融合后的特征表示
        """
        x = event_embed
        
        # 用户特征门控融合
        if user_features is not None:
            user_proj = self.user_proj(user_features).unsqueeze(1)  # (B, 1, D)
            # 广播用户特征到所有时间步
            user_proj_expanded = user_proj.expand(-1, x.size(1), -1)  # (B, L, D)
            # 计算门控权重
            gate_input = torch.cat([x, user_proj_expanded], dim=-1)
            user_gate_weights = self.user_gate(gate_input)  # (B, L, D)
            # 门控融合
            x = x + user_gate_weights * user_proj_expanded
        
        # 商品特征门控融合
        if product_features is not None:
            product_proj = self.product_proj(product_features)  # (B, L, D)
            # 计算门控权重
            gate_input = torch.cat([x, product_proj], dim=-1)
            product_gate_weights = self.product_gate(gate_input)  # (B, L, D)
            # 门控融合
            x = x + product_gate_weights * product_proj
        
        return self.layer_norm(x)


class EventEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        user_feat_dim: int = 64,  # 根据实际特征维度调整
        product_feat_dim: int = 32,  # 根据实际特征维度调整
        use_handcrafted_features: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_handcrafted_features = use_handcrafted_features
        
        # token embeddings (hashed buckets) - 和图片中一致
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        
        # query vector (length 16 integers) projection
        self.query_proj = nn.Linear(16, d_model)
        
        # special tokens
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)  # 这里使用我们定义的位置编码
        
        # 手工特征融合模块
        if use_handcrafted_features:
            self.feature_fusion = FeatureGateFusion(d_model, user_feat_dim, product_feat_dim)
        
        # transformer encoder
        dim_ff = max(d_model, int(d_model * ffn_mult))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        # projection head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(inplace=True), 
            nn.Linear(d_model, embed_dim)
        )
        
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        """哈希函数，确保ID在bucket范围内"""
        return (ids % buckets).clamp(min=0)
    
    def event_embed(
        self,
        type_ids: torch.Tensor,
        sku_ids: torch.Tensor | None,
        cat_ids: torch.Tensor | None,
        price_ids: torch.Tensor | None,
        url_ids: torch.Tensor | None,
        query_vec: torch.Tensor | None,
        user_features: torch.Tensor | None = None,
        product_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """生成事件嵌入"""
        # 基础事件嵌入
        x = self.emb_type(self._hash(type_ids, self.emb_type.num_embeddings))
        if sku_ids is not None:
            x = x + self.emb_sku(self._hash(sku_ids, self.emb_sku.num_embeddings))
        if cat_ids is not None:
            x = x + self.emb_cat(self._hash(cat_ids, self.emb_cat.num_embeddings))
        if price_ids is not None:
            x = x + self.emb_price(self._hash(price_ids, self.emb_price.num_embeddings))
        if url_ids is not None:
            x = x + self.emb_url(self._hash(url_ids, self.emb_url.num_embeddings))
        if query_vec is not None:
            x = x + self.query_proj(query_vec)
        
        x = self.ln(self.gelu(x))
        
        # 手工特征融合
        if self.use_handcrafted_features and (user_features is not None or product_features is not None):
            x = self.feature_fusion(x, user_features, product_features)
        
        return x
    
    def forward(self, batch, user_features: torch.Tensor | None = None, 
                product_features: torch.Tensor | None = None):
        """前向传播"""
        x = self.event_embed(
            type_ids=batch["type_ids"],
            sku_ids=batch.get("sku_ids"),
            cat_ids=batch.get("cat_ids"),
            price_ids=batch.get("price_ids"),
            url_ids=batch.get("url_ids"),
            query_vec=batch.get("query_vec"),
            user_features=user_features,
            product_features=product_features,
        )
        
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))  # 添加位置编码
        x = self.encoder(x)
        
        # 取CLS token作为序列表示
        h = x[:, 0, :]
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        return z


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE对比学习损失函数"""
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    
    logits = (z1 @ z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    
    loss1 = nn.functional.cross_entropy(logits, targets)
    loss2 = nn.functional.cross_entropy(logits.t(), targets)
    return (loss1 + loss2) * 0.5