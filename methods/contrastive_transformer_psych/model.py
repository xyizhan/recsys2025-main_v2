from __future__ import annotations

import torch
from torch import nn 

from methods.contrastive_transformer.model import PositionalEncoding

# version 1: gate + concat
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

# version 2: gate + hierachical + concat
class HierarchicalPsychVAE(nn.Module):
    def __init__(self, input_dim: int,
                 latent_dim: int = 160,
                 hidden_dim: int = 256,
                 need_levels: int = 5,):
        super().__init__()

        if need_levels < 1:
            raise ValueError("need_levels must be >= 1")
        self.latent_dim = latent_dim
        self.need_levels = need_levels

        # 需求层次定义
        self.need_levels_name = [
            'physiological',     # 生理需求
            'safety',            # 安全需求
            'love_belonging',    # 社交需求
            'esteem',            # 尊重需求
            'self_actualization' # 自我实现
        ]

        self.chunk_sizes = self._build_chunk_sizes(latent_dim, need_levels)
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

        # 编码器
        self.need_encoders = nn.ModuleList()
        last = hidden_dim
        for i in range(self.need_levels):
            if i > 0:
                layer_input_dim = last + self.chunk_sizes[i-1]
            else:
                layer_input_dim = last
            
            last = layer_input_dim 
            
            encoder = nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2 * self.chunk_sizes[i])
            )
            self.need_encoders.append(encoder)
            
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
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

    # 层次约束损失：确保低层需求强度不低于高层需求
    def hierarchical_constraint_loss(self, need_mus):
        loss = 0.0    
        for i in range(len(need_mus) - 1):
            lower_level = need_mus[i]      # 低层需求
            higher_level = need_mus[i + 1]  # 高层需求 
            # 违背程度
            violation = torch.relu(higher_level - lower_level)  
            # 平均违背程度
            loss += violation.mean()
        
        return loss / (len(need_mus) - 1)
    
    def forward(self, sequence_repr: torch.Tensor, sample_latent: bool = True) -> dict[str, torch.Tensor]:       
        hidden = self.encoder(sequence_repr)
        dep_weights = self.dependency_gate(hidden)

        need_mus = []
        need_logvars = []
        need_z = []
        cumulative_input = hidden
        
        # 逐层处理需求层次
        for i in range(self.need_levels):
            # 编码
            encoder_output = self.need_encoders[i](cumulative_input)
            mu = encoder_output[:, :self.chunk_sizes[i]]
            logvar = encoder_output[:, self.chunk_sizes[i]:]

            if dep_weights.size(1) > 0 and i > 0:
                gate = dep_weights[:, i-1 : i]
                mu = mu * gate
                logvar = logvar * gate
            
            # 重参数化采样
            z = self.reparameterize(mu, logvar, sample_latent)
            
            # 存储结果
            need_mus.append(mu)
            need_logvars.append(logvar)
            need_z.append(z)
            
            cumulative_input = torch.cat([hidden] + need_z, dim=-1)
        
        # 合并所有需求层次
        latent = torch.cat(need_z, dim=-1)
        mu = torch.cat(need_mus, dim=-1)
        logvar = torch.cat(need_logvars, dim=-1)
        recon = self.decoder(latent)
        hierarchy_loss = self.hierarchical_constraint_loss(need_mus)
        
        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "recon": recon,
            "hierarchy_loss": hierarchy_loss,
        }
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class DualPsychVAETransformer(nn.Module):
    """Transformer encoder with dual VAE heads (behavioral + psychological)."""

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
        stats_dim: int = 15,
        stats_version: str = 'concat'  # concat gate none
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.behavioral_latent_dim = behavioral_latent_dim
        self.psych_latent_dim = psych_latent_dim
        self.fusion_behavior = fusion_behavior
        self.fusion_psych = fusion_psych
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.stats_proj = nn.Sequential(nn.BatchNorm1d(stats_dim),
                                        nn.Linear(stats_dim, d_model))

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
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), 
                                  nn.ReLU(inplace=True), 
                                  nn.Linear(d_model, embed_dim))
        
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
        self.psych_vae = HierarchicalPsychVAE(
            input_dim=d_model,
            latent_dim=psych_latent_dim,
            hidden_dim=max(256, d_model),
        )
        self.psych_to_embed = nn.Linear(psych_latent_dim, embed_dim)

        # 心理学代理特征
        self.stats_version = stats_version
        if stats_version == 'concat':
            self.h_fusion = nn.Sequential(nn.LayerNorm(d_model * 2),
                                            nn.Linear(d_model * 2, d_model),
                                            nn.GELU())
        elif stats_version == 'gate': 
            self.stats_to_film = nn.Linear(d_model, d_model * 2)   
        
        # Optional self-supervision heads
        self.mask_predictor = nn.Linear(d_model, type_buckets)
        order_hidden = max(64, d_model // 2)
        self.order_classifier = nn.Sequential(
            nn.Linear(d_model, order_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(order_hidden, 2),
        )

        # 自适应 fusion_weight
        self.gated_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, 1), 
            nn.Sigmoid() 
        ) 

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_events(self, batch, return_sequence: bool = False) -> torch.Tensor:
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
        if return_sequence:
            return x

        stats_x = self.stats_proj(batch["stats_vec"])
        return x[:, 0, :], stats_x

    def forward(self, batch, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        h, stats_emb = self.encode_events(batch)

        if self.stats_version == 'concat':    # 拼接映射到d_model
            h_aug = torch.cat([h, stats_emb], dim=-1)
            h = self.h_fusion(h_aug)

        elif self.stats_version == 'gate':    # FiLM
            gamma_beta = self.stats_to_film(stats_emb)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            
            gamma = torch.sigmoid(gamma)   # [0, 1]
            beta = torch.tanh(beta) * 0.5  # [-0.5, 0.5]
            h = h * gamma + beta

        # behaviour
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

        # psych
        psych_out = self.psych_vae(h, sample_latent=sample_latent)

        # 自适应的融合权重
        behavior_embed_raw = self.behavior_to_embed(behavior_mu)
        psych_embed_raw = self.psych_to_embed(psych_out["latent"])
        concat_embed = torch.cat([behavior_embed_raw, psych_embed_raw], dim=-1)
        
        gate = self.gated_fusion(concat_embed)
        fused = gate * behavior_embed_raw + (1 - gate) * psych_embed_raw       
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

    def predict_mask_logits(self, batch) -> torch.Tensor:
        seq = self.encode_events(batch, return_sequence=True)
        tokens = seq[:, 1:, :]
        return self.mask_predictor(tokens)

    def classify_order(self, batch) -> torch.Tensor:
        h = self.encode_events(batch, return_sequence=False)
        return self.order_classifier(h)
