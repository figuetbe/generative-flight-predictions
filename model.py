#!/usr/bin/env python3
"""
Flow Matching Model for generative flight trajectory prediction.

This module contains the neural network architectures used for conditional
flow matching of aircraft trajectories. The model consists of:
- SinusoidalPositionalEncoding: For sequence positional information
- TimeEmbedding: For flow matching time step encoding
- HistoryEncoder: Transformer encoder for historical flight data
- FutureDenoiser: Transformer decoder for trajectory generation
- FlowMatchingModel: Main model combining all components

Also includes utility functions for model loading and sampling.
"""

import math
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TimeEmbedding(nn.Module):
    """Time step embedding for flow matching."""

    def __init__(self, d_model: int, hidden: int = 256, emb_dim: int = 128):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.linspace(0, math.log(10_000), emb_dim // 2, dtype=torch.float32)
            ),
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )

    def forward(self, t):
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        ang = t.unsqueeze(-1) * self.freqs  # (B, F)
        temb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, emb_dim)
        return self.proj(temb)  # (B, d_model)


class HistoryEncoder(nn.Module):
    """Transformer encoder for processing flight history sequences."""

    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout, context_dim=3):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.context_proj = nn.Linear(context_dim, d_model) if context_dim > 0 else None
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        z = self.input(x)
        z = self.pos(z)
        if self.context_proj is not None and context is not None:
            c = self.context_proj(context).unsqueeze(1)  # (B,1,D)
            z = torch.cat([c, z], dim=1)
        return self.norm(self.enc(z))


class FutureDenoiser(nn.Module):
    """Transformer decoder for denoising future trajectory predictions."""

    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout):
        super().__init__()
        self.input = nn.Linear(in_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=512)
        self.t_proj_tokens = nn.Linear(d_model, d_model)
        self.t_proj_memory = nn.Linear(d_model, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, in_dim)

    def forward(self, xt, mem, t_emb):
        z = self.pos(self.input(xt))
        z = z + self.t_proj_tokens(t_emb).unsqueeze(1)
        mem = mem + self.t_proj_memory(t_emb).unsqueeze(1)
        return self.output(self.norm(self.dec(tgt=z, memory=mem)))


class FlowMatchingModel(nn.Module):
    """Flow Matching Model for conditional generative trajectory prediction.

    This model combines historical flight data with contextual information
    to generate probabilistic predictions of future aircraft trajectories.

    Args:
        d_model: Model dimension for transformer layers
        nhead: Number of attention heads
        enc_layers: Number of encoder layers
        dec_layers: Number of decoder layers
        ff: Feed-forward network dimension
        dropout: Dropout probability
        in_dim: Input feature dimension
        context_dim: Context feature dimension
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=8,
        ff=4 * 512,
        dropout=0.1,
        in_dim=7,
        context_dim=8,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.encoder = HistoryEncoder(
            in_dim, d_model, nhead, enc_layers, ff, dropout, context_dim=context_dim
        )
        self.time_emb = TimeEmbedding(d_model=d_model)
        self.denoiser = FutureDenoiser(in_dim, d_model, nhead, dec_layers, ff, dropout)

    def forward(self, x_hist, x_t, t_scalar, context):
        mem = self.encoder(x_hist, context)
        t_emb = self.time_emb(t_scalar)
        return self.denoiser(x_t, mem, t_emb)


def sample_xt_and_target(y, t):
    """Sample intermediate states and targets for flow matching training."""
    eps = torch.randn_like(y)
    t_ = t.view(-1, 1, 1)
    x_t = (1.0 - t_) * eps + t_ * y
    v_star = y - eps
    return x_t, v_star, eps


def load_model_checkpoint(checkpoint_path: str, device=None) -> FlowMatchingModel:
    """Load model checkpoint and return configured model.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (auto-detects if None)

    Returns:
        Loaded FlowMatchingModel instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle potential module prefix issues
    state = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )

    # Get model configuration from checkpoint or use defaults
    cfg = ckpt.get(
        "model_cfg",
        dict(
            d_model=512,
            nhead=8,
            enc_layers=6,
            dec_layers=8,
            ff=4 * 512,
            dropout=0.1,
            in_dim=7,
            context_dim=8,
        ),
    )

    # Initialize and load model
    model = FlowMatchingModel(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def get_model_config() -> Dict[str, Any]:
    """Get default model configuration."""
    return dict(
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=8,
        ff=4 * 512,
        dropout=0.1,
        in_dim=7,
        context_dim=8,
    )
