from __future__ import annotations
from .handcrafted_features import HandcraftedFeatureLoader

import torch
from torch import nn

from methods.contrastive_transformer.model import PositionalEncoding


class MaslowPsychologicalVAE(nn.Module):
    """Hierarchical VAE that mimics dependency between psychological needs."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 160,
        hidden_dim: int = 256,
        need_levels: int = 5,
    ) -> None:
        super().__init__()
        if need_levels < 1:
            raise ValueError("need_levels must be >= 1")
        self.latent_dim = latent_dim
        self.need_levels = need_levels
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dependency_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, max(1, need_levels - 1)),
            nn.Sigmoid(),
        )
        self.chunk_sizes = self._build_chunk_sizes(latent_dim, need_levels)
        self.need_means = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.chunk_sizes])
        self.need_logvars = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.chunk_sizes])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    @staticmethod
    def _build_chunk_sizes(latent_dim: int, need_levels: int) -> list[int]:
        base = latent_dim // need_levels
        rem = latent_dim % need_levels
        sizes = []
        for i in range(need_levels):
            extra = 1 if i < rem else 0
            sizes.append(max(1, base + extra))
        return sizes

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor, sample_latent: bool) -> torch.Tensor:
        if not sample_latent:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, sequence_repr: torch.Tensor, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        hidden = self.encoder(sequence_repr)
        dep_weights = self.dependency_gate(hidden)
        chunk_z = []
        chunk_mu = []
        chunk_logvar = []
        for idx in range(self.need_levels):
            mean = self.need_means[idx](hidden)
            logvar = self.need_logvars[idx](hidden)
            if dep_weights.size(1) > 0 and idx > 0:
                gate = dep_weights[:, idx - 1 : idx]
                mean = mean * gate
                logvar = logvar * gate
            z = self.reparameterize(mean, logvar, sample_latent)
            chunk_z.append(z)
            chunk_mu.append(mean)
            chunk_logvar.append(logvar)
        latent = torch.cat(chunk_z, dim=-1)
        mu = torch.cat(chunk_mu, dim=-1)
        logvar = torch.cat(chunk_logvar, dim=-1)
        recon = self.decoder(latent)
        hierarchy_loss = torch.tensor(0.0, device=sequence_repr.device)
        if dep_weights.size(1) > 1:
            diffs = dep_weights[:, 1:] - dep_weights[:, :-1]
            hierarchy_loss = torch.relu(diffs).mean()
        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "recon": recon,
            "dependency_weights": dep_weights,
            "hierarchy_loss": hierarchy_loss,
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class DualPsychVAETransformer(nn.Module):
    """Transformer encoder with dual VAE heads (behavioral + psychological) and handcrafted features."""

    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        behavioral_latent_dim: int = 128,
        psych_latent_dim: int = 160,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        fusion_behavior: float = 0.5,
        fusion_psych: float = 0.5,
        recon_dim: int = 5,
        use_handcrafted: bool = True,
        handcrafted_dir: str = "./handcrafted_features",
        handcrafted_fusion_method: str = "concat",  # "concat", "add", or "weighted"
    ) -> None:
        super().__init__()
        self.mps_compatible = True
        self.d_model = d_model
        self.behavioral_latent_dim = behavioral_latent_dim
        self.psych_latent_dim = psych_latent_dim
        self.fusion_behavior = fusion_behavior
        self.fusion_psych = fusion_psych
        self.use_handcrafted = use_handcrafted
        self.handcrafted_fusion_method = handcrafted_fusion_method
        
        # 初始化手工特征加载器
        if self.use_handcrafted:
            self.handcrafted_loader = HandcraftedFeatureLoader(handcrafted_dir)
            self.user_feat_dim, self.product_feat_dim = self.handcrafted_loader.get_feature_dimensions()
            print(f"手工特征加载完成: 用户维度={self.user_feat_dim}, 商品维度={self.product_feat_dim}")
        else:
            self.user_feat_dim, self.product_feat_dim = 0, 0
        
        # 手工特征融合层
        if self.use_handcrafted and self.user_feat_dim > 0:
            if handcrafted_fusion_method == "concat":
                self.user_feature_fusion = nn.Sequential(
                    nn.Linear(d_model + self.user_feat_dim, d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            elif handcrafted_fusion_method == "add":
                # 确保维度匹配
                if self.user_feat_dim != d_model:
                    self.user_feature_proj = nn.Linear(self.user_feat_dim, d_model)
                else:
                    self.user_feature_proj = nn.Identity()
            elif handcrafted_fusion_method == "weighted":
                self.user_feature_attention = nn.Sequential(
                    nn.Linear(d_model + self.user_feat_dim, d_model // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model // 2, 1),
                    nn.Sigmoid()
                )
        
        # 原有的嵌入层
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        dim_ff = max(d_model, int(d_model * ffn_mult))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, embed_dim))
        
        # Behavioral VAE head
        self.behavior_mu = nn.Linear(d_model, behavioral_latent_dim)
        self.behavior_logvar = nn.Linear(d_model, behavioral_latent_dim)
        hid = max(behavioral_latent_dim * 2, d_model)
        self.behavior_decoder = nn.Sequential(
            nn.Linear(behavioral_latent_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, recon_dim),
        )
        self.behavior_to_embed = nn.Linear(behavioral_latent_dim, embed_dim)
        
        # Psychological VAE head
        self.psych_vae = MaslowPsychologicalVAE(
            input_dim=d_model,
            latent_dim=psych_latent_dim,
            hidden_dim=max(256, d_model),
        )
        self.psych_to_embed = nn.Linear(psych_latent_dim, embed_dim)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def _fuse_handcrafted_features(self, sequence_repr, client_ids):
        """融合手工特征到序列表示中"""
        if not self.use_handcrafted or self.user_feat_dim == 0:
            return sequence_repr
        
        # 获取批量手工特征
        user_handcrafted = []
        for client_id in client_ids:
            feature = self.handcrafted_loader.get_user_feature(client_id.item())
            user_handcrafted.append(feature)
        
        user_handcrafted = torch.stack(user_handcrafted).to(sequence_repr.device)
        
        # 根据融合方法处理特征
        if self.handcrafted_fusion_method == "concat":
            combined = torch.cat([sequence_repr, user_handcrafted], dim=1)
            return self.user_feature_fusion(combined)
        
        elif self.handcrafted_fusion_method == "add":
            proj_features = self.user_feature_proj(user_handcrafted)
            return sequence_repr + proj_features
        
        elif self.handcrafted_fusion_method == "weighted":
            combined = torch.cat([sequence_repr, user_handcrafted], dim=1)
            attention_weights = self.user_feature_attention(combined)
            return sequence_repr * attention_weights + user_handcrafted * (1 - attention_weights)
        
        else:
            return sequence_repr

    def encode_events(self, batch) -> torch.Tensor:
        x = self.emb_type(self._hash(batch["type_ids"], self.emb_type.num_embeddings))
        if "sku_ids" in batch:
            x = x + self.emb_sku(self._hash(batch["sku_ids"], self.emb_sku.num_embeddings))
        if "cat_ids" in batch:
            x = x + self.emb_cat(self._hash(batch["cat_ids"], self.emb_cat.num_embeddings))
        if "price_ids" in batch:
            x = x + self.emb_price(self._hash(batch["price_ids"], self.emb_price.num_embeddings))
        if "url_ids" in batch:
            x = x + self.emb_url(self._hash(batch["url_ids"], self.emb_url.num_embeddings))
        if "query_vec" in batch:
            x = x + self.query_proj(batch["query_vec"])
        x = self.ln(self.gelu(x))
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        return x[:, 0, :]

    def forward(self, batch, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        # 获取客户端ID用于手工特征融合
        client_ids = batch.get("client_id", torch.zeros(batch["type_ids"].size(0), dtype=torch.long))
        
        # 编码事件序列
        h = self.encode_events(batch)
        
        # 融合手工特征
        h = self._fuse_handcrafted_features(h, client_ids)
        
        proj = self.proj(h)
        behavior_mu = self.behavior_mu(h)
        behavior_logvar = self.behavior_logvar(h)
        if sample_latent:
            behavior_std = torch.exp(0.5 * behavior_logvar)
            behavior_eps = torch.randn_like(behavior_std)
            behavior_latent = behavior_mu + behavior_eps * behavior_std
        else:
            behavior_latent = behavior_mu
        behavior_recon = self.behavior_decoder(behavior_latent)
        psych_out = self.psych_vae(h, sample_latent=sample_latent)
        fused = proj
        if self.fusion_behavior > 0:
            fused = fused + self.fusion_behavior * self.behavior_to_embed(behavior_mu)
        if self.fusion_psych > 0:
            fused = fused + self.fusion_psych * self.psych_to_embed(psych_out["latent"])
        embed = nn.functional.normalize(fused, dim=-1)
        return {
            "embed": embed,
            "sequence_hidden": h,
            "behavior_mu": behavior_mu,
            "behavior_logvar": behavior_logvar,
            "behavior_recon": behavior_recon,
            "psych_latent": psych_out["latent"],
            "psych_mu": psych_out["mu"],
            "psych_logvar": psych_out["logvar"],
            "psych_recon": psych_out["recon"],
            "hierarchy_loss": psych_out["hierarchy_loss"],
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))