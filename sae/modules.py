# sae/modules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ==========================================
# Config & Utils
# ==========================================

@dataclass
class SAEConfig:
    d_in: int
    d_latent: int       # d_in * expansion_factor
    k: int
    act_name: str = "relu"
    act_eps: float = 1e-6
    alpha_aux: float = 1/32
    k_aux_factor: int = 2
    lr: float = 1e-4
    device: Optional[str] = None

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def compute_zscore_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes mean and std for normalization."""
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std

def zscore_transform(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean) / std

# ==========================================
# SAE Model (Top-K Hard Sparsity)
# ==========================================

class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.d_in = cfg.d_in
        self.d_latent = cfg.d_latent
        self.k = cfg.k
        self.k_aux_factor = cfg.k_aux_factor
        self.act_eps = cfg.act_eps

        self.encoder = nn.Linear(cfg.d_in, cfg.d_latent, bias=True)
        self.decoder = nn.Linear(cfg.d_latent, cfg.d_in, bias=True)
        self.act = nn.ReLU()

        # Buffer to track "dead" neurons
        self.register_buffer("epoch_on_count", torch.zeros(cfg.d_latent, dtype=torch.long))

    def start_epoch(self):
        self.epoch_on_count.zero_()

    def mark_activity(self, h: torch.Tensor):
        on_mask = (h > self.act_eps).any(dim=0)
        self.epoch_on_count.add_(on_mask.to(self.epoch_on_count.dtype))

    def dead_latent_indices(self) -> torch.Tensor:
        return (self.epoch_on_count == 0).nonzero(as_tuple=False).flatten()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        self.mark_activity(h)

        # Top-K selection
        val, idx = torch.topk(h, k=self.k, dim=1)
        mask_topk = torch.zeros_like(h, dtype=torch.bool)
        mask_topk.scatter_(1, idx, True)

        h_topk = torch.where(mask_topk, h, torch.zeros_like(h))
        x_hat = self.decode(h_topk)

        return x_hat, h, h_topk

    def aux_reconstruct_error(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Auxiliary task: reconstruct error using only dead neurons."""
        dead_idx = self.dead_latent_indices()
        if dead_idx.numel() == 0:
            return torch.zeros_like(e), None

        k_aux = min(self.k * self.k_aux_factor, self.d_latent)

        # Encode error
        h_e = self.encode(e)

        # Select dead neurons
        h_dead = h_e[:, dead_idx]

        # Top-K on dead neurons
        if h_dead.shape[1] > k_aux:
            _, local_topk = torch.topk(h_dead, k=k_aux, dim=1)
            mask_dead = torch.zeros_like(h_dead, dtype=torch.bool)
            mask_dead.scatter_(1, local_topk, True)
            h_dead = torch.where(mask_dead, h_dead, torch.zeros_like(h_dead))

        # Partial reconstruction
        z = torch.zeros_like(h_e)
        z[:, dead_idx] = h_dead
        e_hat = self.decoder(z)

        return e_hat, z

# ==========================================
# Loss Function
# ==========================================

@dataclass
class LossOut:
    total: torch.Tensor
    rec: torch.Tensor
    aux: torch.Tensor

def sae_loss_func(model: SAE, x: torch.Tensor, alpha_aux: float) -> LossOut:
    x_hat, _, _ = model(x)
    diff = x - x_hat

    # Reconstruction Loss (Normalized MSE)
    L_rec = (diff.pow(2).sum(dim=1) / x.shape[1]).mean()

    # Auxiliary Loss
    e_hat, _ = model.aux_reconstruct_error(diff)
    L_aux = (diff - e_hat).pow(2).mean()

    total = L_rec + alpha_aux * L_aux
    return LossOut(total, L_rec, L_aux)
