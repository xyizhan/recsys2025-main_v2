from __future__ import annotations

import math
import torch
from torch import nn

from methods.contrastive_transformer.model import PositionalEncoding


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    timesteps = timesteps.float()
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
    )
    args = timesteps.unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FieldWiseFiLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(self, base: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(self.net(cond), 2, dim=-1)
        return base * (1.0 + scale) + shift


class ConditionalDiffusionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, noisy_latent: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        temb = self.time_embed(temb)
        cond_proj = self.cond_proj(cond)
        x = torch.cat([noisy_latent, cond_proj, temb], dim=-1)
        return self.net(x)


class ContrastiveDiffusionEncoder(nn.Module):
    """Contrastive transformer encoder with conditional diffusion head."""

    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        latent_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        diffusion_timesteps: int = 50,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        diffusion_hidden_dim: int = 512,
        diffusion_time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.film_sku = FieldWiseFiLM(d_model)
        self.film_cat = FieldWiseFiLM(d_model)
        self.film_url = FieldWiseFiLM(d_model)
        self.film_price = FieldWiseFiLM(d_model)
        self.film_query = FieldWiseFiLM(d_model)
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
        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, embed_dim),
        )
        self.diffusion_head = ConditionalDiffusionHead(
            latent_dim=latent_dim,
            cond_dim=d_model,
            time_embed_dim=diffusion_time_embed_dim,
            hidden_dim=diffusion_hidden_dim,
        )
        self._init_diffusion_buffers(
            timesteps=diffusion_timesteps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )

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

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_events(self, batch, return_sequence: bool = False) -> torch.Tensor:
        if "type_ids" not in batch:
            raise ValueError("batch must contain type_ids for base embedding")
        x = self.emb_type(self._hash(batch["type_ids"], self.emb_type.num_embeddings))
        if "sku_ids" in batch:
            sku_emb = self.emb_sku(self._hash(batch["sku_ids"], self.emb_sku.num_embeddings))
            x = self.film_sku(x, sku_emb)
        if "cat_ids" in batch:
            cat_emb = self.emb_cat(self._hash(batch["cat_ids"], self.emb_cat.num_embeddings))
            x = self.film_cat(x, cat_emb)
        if "price_ids" in batch:
            price_emb = self.emb_price(self._hash(batch["price_ids"], self.emb_price.num_embeddings))
            x = self.film_price(x, price_emb)
        if "url_ids" in batch:
            url_emb = self.emb_url(self._hash(batch["url_ids"], self.emb_url.num_embeddings))
            x = self.film_url(x, url_emb)
        if "query_vec" in batch:
            query_emb = self.query_proj(batch["query_vec"])
            x = self.film_query(x, query_emb)
        x = self.ln(self.gelu(x))
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        if return_sequence:
            return x
        return x[:, 0, :]

    def forward(
        self,
        batch,
        sampling_method: str = "ddim",
        sampling_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_events(batch)
        base_latent = self.latent_proj(hidden)
        latent = self.sample_latent(hidden, steps=sampling_steps, method=sampling_method)
        embed = nn.functional.normalize(self.embed_proj(latent), dim=-1)
        return {
            "embed": embed,
            "base_latent": base_latent,
            "hidden": hidden,
        }

    def predict_noise(self, noisy_latent: torch.Tensor, hidden: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.diffusion_head(noisy_latent, hidden, timesteps)

    def diffusion_loss(
        self,
        base_latent: torch.Tensor,
        hidden: torch.Tensor,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = base_latent.size(0)
        device = base_latent.device
        if noise is None:
            noise = torch.randn_like(base_latent)
        if timesteps is None:
            timesteps = torch.randint(0, self.diffusion_timesteps, (B,), device=device)
        alphas_bar = self.alphas_cumprod.index_select(0, timesteps).unsqueeze(-1)
        sqrt_ab = torch.sqrt(alphas_bar)
        sqrt_one_minus = torch.sqrt(1 - alphas_bar)
        noisy_latent = sqrt_ab * base_latent + sqrt_one_minus * noise
        eps_pred = self.predict_noise(noisy_latent, hidden, timesteps)
        loss = nn.functional.mse_loss(eps_pred, noise)
        return loss

    def sample_latent(self, hidden: torch.Tensor, steps: int | None = None, method: str = "ddim") -> torch.Tensor:
        steps = steps if steps is not None else self.diffusion_timesteps
        steps = min(max(1, steps), self.diffusion_timesteps)
        if method.lower() == "ddpm":
            if steps != self.diffusion_timesteps:
                raise ValueError("DDPM sampling requires steps equal to diffusion_timesteps")
            return self._sample_latent_ddpm(hidden)
        if method.lower() != "ddim":
            raise ValueError(f"Unsupported sampling method: {method}")
        return self._sample_latent_ddim(hidden, steps)

    def _sample_latent_ddpm(self, hidden: torch.Tensor) -> torch.Tensor:
        B = hidden.size(0)
        device = hidden.device
        latent = torch.randn((B, self.latent_dim), device=device)
        for i in reversed(range(self.diffusion_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_theta = self.predict_noise(latent, hidden, t)
            beta_t = self.betas[i]
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]
            sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[i]
            latent = sqrt_recip_alpha * (latent - beta_t / sqrt_one_minus_ab * eps_theta)
            if i > 0:
                noise = torch.randn_like(latent)
                var = self.posterior_variance[i]
                latent = latent + torch.sqrt(var) * noise
        return latent

    def _build_sampling_schedule(self, steps: int) -> list[int]:
        if steps >= self.diffusion_timesteps:
            return list(range(self.diffusion_timesteps - 1, -1, -1))
        values = torch.linspace(self.diffusion_timesteps - 1, 0, steps)
        indices = torch.round(values).to(torch.long).cpu().tolist()
        schedule: list[int] = []
        for idx in indices:
            if not schedule or schedule[-1] != idx:
                schedule.append(idx)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _sample_latent_ddim(self, hidden: torch.Tensor, steps: int) -> torch.Tensor:
        B = hidden.size(0)
        device = hidden.device
        latent = torch.randn((B, self.latent_dim), device=device)
        schedule = self._build_sampling_schedule(steps)
        for i, t_idx in enumerate(schedule):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_theta = self.predict_noise(latent, hidden, t)
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
