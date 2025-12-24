from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn

from methods.contrastive_transformer.model import PositionalEncoding


class TypeConditionedSequenceEncoder(nn.Module):
    field_names = ("sku", "cat", "url", "price", "query", "delta")

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, d_model),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.field_gate = nn.Embedding(type_buckets, len(self.field_names))
        nn.init.xavier_uniform_(self.field_gate.weight)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        dim_ff = max(d_model, int(d_model * ffn_mult))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        type_ids = batch["type_ids"]
        x = self.emb_type(self._hash(type_ids, self.emb_type.num_embeddings))
        gates = torch.sigmoid(self.field_gate(type_ids))
        contribs: Dict[str, torch.Tensor] = {}
        contribs["sku"] = self.emb_sku(self._hash(batch["sku_ids"], self.emb_sku.num_embeddings))
        contribs["cat"] = self.emb_cat(self._hash(batch["cat_ids"], self.emb_cat.num_embeddings))
        contribs["url"] = self.emb_url(self._hash(batch["url_ids"], self.emb_url.num_embeddings))
        contribs["price"] = self.emb_price(self._hash(batch["price_ids"], self.emb_price.num_embeddings))
        contribs["query"] = self.query_proj(batch["query_vec"])
        delta = batch["delta_times"].unsqueeze(-1)
        contribs["delta"] = self.delta_proj(delta)

        for idx, fname in enumerate(self.field_names):
            mask_name = f"{fname}_mask" if f"{fname}_mask" in batch else None
            mask = batch[mask_name] if mask_name else None
            field = contribs[fname]
            gate = gates[..., idx].unsqueeze(-1)
            if mask is not None:
                gate = gate * mask.unsqueeze(-1)
            x = x + gate * field

        x = self.ln(self.act(x))
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        return x[:, 0, :]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


class ConditionalDiffusionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, noisy_latent: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        temb = self._sinusoidal_embedding(timesteps, self.time_embed_dim)
        time_features = self.time_embed(temb)
        cond_features = self.cond_proj(cond)
        latent_features = self.latent_proj(noisy_latent)
        x = torch.cat([latent_features, cond_features, time_features], dim=-1)
        return self.net(x)

    @staticmethod
    def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        device = timesteps.device
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class UnifiedDiffusionModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        latent_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        diffusion_timesteps: int = 50,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        diffusion_hidden_dim: int = 512,
        diffusion_time_embed_dim: int = 64,
        cfg_drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_kwargs = dict(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            ffn_mult=ffn_mult,
            sku_buckets=sku_buckets,
            cat_buckets=cat_buckets,
            url_buckets=url_buckets,
            price_buckets=price_buckets,
            type_buckets=type_buckets,
        )
        self.student_encoder = TypeConditionedSequenceEncoder(**encoder_kwargs)
        self.teacher_encoder = TypeConditionedSequenceEncoder(**encoder_kwargs)
        self.student_head = ProjectionHead(d_model, latent_dim, embed_dim)
        self.teacher_head = ProjectionHead(d_model, latent_dim, embed_dim)
        self.diffusion_head = ConditionalDiffusionHead(
            latent_dim=embed_dim,
            cond_dim=d_model,
            time_embed_dim=diffusion_time_embed_dim,
            hidden_dim=diffusion_hidden_dim,
        )
        self.cfg_drop_prob = cfg_drop_prob
        self.null_condition = nn.Parameter(torch.zeros(d_model))
        self._init_diffusion_buffers(
            timesteps=diffusion_timesteps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )
        self._sync_teacher()

    def _sync_teacher(self) -> None:
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_encoder.requires_grad_(False)
        self.teacher_head.requires_grad_(False)
        self.teacher_encoder.eval()
        self.teacher_head.eval()

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for t_param, s_param in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
        for t_param, s_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
        self.teacher_encoder.eval()
        self.teacher_head.eval()

    def student_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden = self.student_encoder(batch)
        embed = self.student_head(hidden)
        return {"hidden": hidden, "embed": embed}

    @torch.no_grad()
    def teacher_embed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.teacher_encoder(batch)
        embed = self.teacher_head(hidden)
        return embed.detach()

    def diffusion_loss(self, condition_hidden: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
        B = teacher_target.size(0)
        device = teacher_target.device
        noise = torch.randn_like(teacher_target)
        t = torch.randint(0, self.diffusion_timesteps, (B,), device=device, dtype=torch.long)
        alphas_bar = self.alphas_cumprod.index_select(0, t).unsqueeze(-1)
        sqrt_ab = torch.sqrt(alphas_bar)
        sqrt_one_minus = torch.sqrt(1 - alphas_bar)
        noisy_latent = sqrt_ab * teacher_target + sqrt_one_minus * noise
        cond = self._apply_cfg_dropout(condition_hidden)
        eps_pred = self.diffusion_head(noisy_latent, cond, t)
        return nn.functional.mse_loss(eps_pred, noise)

    def _apply_cfg_dropout(self, condition_hidden: torch.Tensor) -> torch.Tensor:
        B = condition_hidden.size(0)
        drop_mask = torch.rand((B,), device=condition_hidden.device) < self.cfg_drop_prob
        if drop_mask.any():
            cond = condition_hidden.clone()
            null_vec = self.null_condition.to(condition_hidden.device)
            count = int(drop_mask.sum().item())
            cond[drop_mask] = null_vec.unsqueeze(0).expand(count, -1)
            return cond
        return condition_hidden

    def predict_noise(
        self,
        noisy_latent: torch.Tensor,
        condition_hidden: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_weight: float = 0.0,
    ) -> torch.Tensor:
        if guidance_weight <= 0:
            cond = condition_hidden
            return self.diffusion_head(noisy_latent, cond, timesteps)
        cond_pred = self.diffusion_head(noisy_latent, condition_hidden, timesteps)
        uncond = self.null_condition.to(condition_hidden.device).unsqueeze(0).expand_as(condition_hidden)
        uncond_pred = self.diffusion_head(noisy_latent, uncond, timesteps)
        return (1.0 + guidance_weight) * cond_pred - guidance_weight * uncond_pred

    def sample_latent(
        self,
        condition_hidden: torch.Tensor,
        steps: int | None = None,
        method: str = "ddim",
        guidance_weight: float = 0.0,
    ) -> torch.Tensor:
        steps = steps if steps is not None else self.diffusion_timesteps
        steps = min(max(1, steps), self.diffusion_timesteps)
        if method.lower() == "ddpm":
            return self._sample_latent_ddpm(condition_hidden, guidance_weight)
        if method.lower() != "ddim":
            raise ValueError(f"Unsupported sampling method: {method}")
        return self._sample_latent_ddim(condition_hidden, steps, guidance_weight)

    def _sample_latent_ddpm(self, condition_hidden: torch.Tensor, guidance_weight: float) -> torch.Tensor:
        B = condition_hidden.size(0)
        device = condition_hidden.device
        latent = torch.randn((B, self.embed_dim), device=device)
        for i in reversed(range(self.diffusion_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_theta = self.predict_noise(latent, condition_hidden, t, guidance_weight)
            beta_t = self.betas[i]
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]
            sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[i]
            latent = sqrt_recip_alpha * (latent - beta_t / sqrt_one_minus_ab * eps_theta)
            if i > 0:
                noise = torch.randn_like(latent)
                var = self.posterior_variance[i]
                latent = latent + torch.sqrt(var) * noise
        return latent

    def _build_schedule(self, steps: int) -> Tuple[int, ...]:
        if steps >= self.diffusion_timesteps:
            return tuple(range(self.diffusion_timesteps - 1, -1, -1))
        values = torch.linspace(self.diffusion_timesteps - 1, 0, steps)
        idxs = torch.round(values).to(torch.long).tolist()
        schedule = []
        for idx in idxs:
            if not schedule or schedule[-1] != idx:
                schedule.append(idx)
        if schedule[-1] != 0:
            schedule.append(0)
        return tuple(schedule)

    def _sample_latent_ddim(
        self,
        condition_hidden: torch.Tensor,
        steps: int,
        guidance_weight: float,
    ) -> torch.Tensor:
        B = condition_hidden.size(0)
        device = condition_hidden.device
        latent = torch.randn((B, self.embed_dim), device=device)
        schedule = self._build_schedule(steps)
        for i, t_idx in enumerate(schedule):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_theta = self.predict_noise(latent, condition_hidden, t, guidance_weight)
            alpha_t = self.alphas_cumprod[t_idx].to(device=device)
            alpha_prev = torch.tensor(1.0, device=device)
            if i + 1 < len(schedule):
                alpha_prev = self.alphas_cumprod[schedule[i + 1]].to(device=device)
            sqrt_alpha_t = torch.sqrt(alpha_t).clamp(min=1e-6)
            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=0.0))
            x0_pred = (latent - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t
            sqrt_alpha_prev = torch.sqrt(alpha_prev).clamp(min=1e-6)
            sqrt_one_minus_alpha_prev = torch.sqrt(torch.clamp(1 - alpha_prev, min=0.0))
            latent = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * eps_theta
        return latent

    def _init_diffusion_buffers(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.diffusion_timesteps = timesteps
        self.embed_dim = self.student_head.net[-1].out_features

    @torch.no_grad()
    def sample_and_aggregate(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int,
        sampling_method: str,
        sampling_steps: int,
        guidance_weight: float,
    ) -> torch.Tensor:
        hidden = self.student_encoder(batch)
        acc = torch.zeros((hidden.size(0), self.embed_dim), device=hidden.device)
        for _ in range(max(1, num_samples)):
            latent = self.sample_latent(
                hidden,
                steps=sampling_steps,
                method=sampling_method,
                guidance_weight=guidance_weight,
            )
            acc = acc + nn.functional.normalize(latent, dim=-1)
        return acc / float(max(1, num_samples))
