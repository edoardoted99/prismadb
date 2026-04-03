# sae/modules.py
"""
Sparse Autoencoder with Top-K hard sparsity.
Aligned with O'Neill et al. (2024) "Disentangling Dense Embeddings with SAEs".
"""
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
    alpha_aux: float = 1/32       # Paper: α usually set to 1/32
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

def _geometric_median(X: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Compute the geometric median of a set of points (Weiszfeld's algorithm)."""
    y = X.mean(dim=0)
    for _ in range(max_iter):
        diffs = X - y.unsqueeze(0)
        norms = diffs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        weights = 1.0 / norms
        y_new = (X * weights).sum(dim=0) / weights.sum()
        if (y_new - y).norm() < tol:
            break
        y = y_new
    return y

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

    def initialize_from_data(self, X: torch.Tensor):
        """
        Paper-aligned initialization (Appendix A.1):
        1. b_pre (encoder bias) = -geometric_median(X)
        2. Decoder columns normalized to unit length
        3. Encoder rows set parallel to decoder columns
        4. Encoder magnitudes matched to input magnitudes
        """
        with torch.no_grad():
            # 1. Encoder bias from geometric median (Bricken et al. 2023)
            sample = X[:min(4096, len(X))]
            geo_med = _geometric_median(sample)
            self.encoder.bias.data = -geo_med.repeat(self.d_latent // self.d_in + 1)[:self.d_latent]

            # 2. Normalize decoder columns to unit length
            self.normalize_decoder()

            # 3. Set encoder directions parallel to decoder directions
            # Decoder weight: [d_in, d_latent] -> columns are feature directions
            # Encoder weight: [d_latent, d_in] -> rows are feature directions
            self.encoder.weight.data = self.decoder.weight.data.T.clone()

            # 4. Scale encoder magnitudes to match input magnitudes (Gao et al. 2024)
            avg_norm = X[:min(4096, len(X))].norm(dim=1).mean()
            encoder_norms = self.encoder.weight.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.encoder.weight.data *= avg_norm / encoder_norms

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weight columns to unit length (Paper Appendix A.1)."""
        W = self.decoder.weight.data  # [d_in, d_latent]
        norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data = W / norms

    @torch.no_grad()
    def gradient_projection(self):
        """
        Remove gradient component parallel to decoder weight directions.
        Decouples Adam optimizer from decoder normalization (Bricken et al. 2023).
        """
        if self.decoder.weight.grad is None:
            return
        W = self.decoder.weight.data  # [d_in, d_latent]
        G = self.decoder.weight.grad  # [d_in, d_latent]
        # Project out the component of grad along each column direction
        dots = (G * W).sum(dim=0, keepdim=True)
        norms_sq = (W * W).sum(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.grad = G - W * (dots / norms_sq)

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

def sae_loss_func(model: SAE, x: torch.Tensor, alpha_aux: float,
                  global_norm_factor: float = None) -> LossOut:
    """
    Paper-aligned loss:
    L = (1/d)||x - x_hat||^2 + alpha * L_aux

    global_norm_factor: precomputed 1/d for primary MSE (global, fixed at training start)
    AuxK uses per-batch normalization.
    """
    x_hat, _, _ = model(x)
    diff = x - x_hat

    # Reconstruction Loss: (1/d) * ||x - x_hat||^2, averaged over batch
    if global_norm_factor is None:
        global_norm_factor = 1.0 / x.shape[1]
    L_rec = (diff.pow(2).sum(dim=1) * global_norm_factor).mean()

    # Auxiliary Loss: per-batch normalized ||e - e_hat||^2
    e_hat, _ = model.aux_reconstruct_error(diff.detach())
    residual = diff - e_hat
    # Per-batch normalization: normalize by batch's error magnitude
    batch_norm = diff.detach().pow(2).sum(dim=1).mean().clamp_min(1e-8)
    L_aux = residual.pow(2).sum(dim=1).mean() / batch_norm

    total = L_rec + alpha_aux * L_aux
    return LossOut(total, L_rec, L_aux)
