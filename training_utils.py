#!/usr/bin/env python3
"""
Training utilities for Flow Matching Model.

This module contains training-related utilities including:
- Learning rate schedulers (WarmupCosine)
- Exponential Moving Average (EMA)
- Training functions optimized for the Flow Matching Model
"""

import math
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import FlowMatchingModel, sample_xt_and_target
from utils import make_loader


class WarmupCosine:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.last_step = -1
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.last_step += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.last_step < self.warmup:
                lr = base * (self.last_step + 1) / self.warmup
            else:
                t = (self.last_step - self.warmup) / max(
                    1, self.max_steps - self.warmup
                )
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (
                    1 + math.cos(math.pi * t)
                )
            g["lr"] = lr


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                # Keep non-float buffers in sync
                self.shadow[k] = v

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def swap_into(self, model):
        """Swap EMA weights into model and return restore function."""
        msd = model.state_dict()

        def _swap():
            for k, v in msd.items():
                if k not in self.shadow:
                    continue
                if v.dtype.is_floating_point:
                    tmp = v.detach().clone()
                    v.data.copy_(self.shadow[k].to(device=v.device, dtype=v.dtype))
                    self.shadow[k] = tmp
                else:
                    pass  # Keep current buffer values

        _swap()
        return _swap


def train_cfm(
    train_ds,
    val_ds,
    epochs=200,
    batch_size=2048,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_steps=2000,
    ema_decay=0.9995,
    patience=50,
    model_cfg=None,
    ckpt_path="best_cfm_geo.pt",
    aux_w=0.05,
    accum_steps=1,
    compile_mode="reduce-overhead",
    device=None,
):
    """
    Train the Flow Matching Model with optimized settings.

    Uses bfloat16 autocast, fused AdamW, torch.compile, and EMA.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        grad_clip: Gradient clipping threshold
        warmup_steps: Number of warmup steps for learning rate
        ema_decay: EMA decay rate
        patience: Early stopping patience
        model_cfg: Model configuration dictionary
        ckpt_path: Path to save checkpoints
        aux_w: Weight for auxiliary loss
        accum_steps: Gradient accumulation steps
        compile_mode: Torch compile mode
        device: Device to train on (auto-detects if None)

    Returns:
        Trained FlowMatchingModel instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    # Initialize model
    model = FlowMatchingModel(**(model_cfg or {})).to(device)

    # Resume from checkpoint if available
    resume = False
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = OrderedDict(
            (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[resume] Loaded {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
        resume = True

    # Compile model for better performance
    try:
        model = torch.compile(model, mode=compile_mode)
    except Exception as e:
        print(f"[compile] Fallback: {e}")

    # Initialize optimizer
    try:
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
    except TypeError:
        opt = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )

    # Setup scheduler and EMA
    max_steps = epochs * max(1, len(train_loader))
    sched = WarmupCosine(
        opt, warmup_steps=warmup_steps, max_steps=max_steps, min_lr=lr * 0.05
    )
    ema = EMA(model, decay=ema_decay)

    # Loss weights
    pos_w, vel_w = 1.0, 0.1

    # Early stopping variables
    best_val, bad = float("inf"), 0

    # Mixed precision setup
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    def run_epoch(loader, train=True, log_components=False):
        """Run one training or validation epoch."""
        if train:
            model.train()
        else:
            model.eval()

        tot = n = 0
        pos_tot = vel_tot = aux_tot = 0.0

        if train:
            opt.zero_grad(set_to_none=True)

        for step, (xb, yb, cb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            # Sample random time steps
            t = torch.rand(xb.size(0), 1, device=device)

            with (
                torch.set_grad_enabled(train),
                torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
                ),
            ):
                x_t, _, eps = sample_xt_and_target(yb, t)
                v_pred = model(xb, x_t, t, cb)  # Predict velocity field
                y_pred = eps + v_pred  # Predict target trajectory

                # Compute losses
                pos_loss = F.mse_loss(y_pred[..., :3], yb[..., :3])  # Position (x,y,z)
                vel_loss = F.mse_loss(
                    y_pred[..., 3:6], yb[..., 3:6]
                )  # Velocity (vx,vy,vz)
                aux_loss = (
                    F.mse_loss(y_pred[..., 6:7], yb[..., 6:7]) if aux_w > 0 else 0.0
                )

                loss = pos_w * pos_loss + vel_w * vel_loss + aux_w * aux_loss

                # Gradient accumulation
                if train and accum_steps > 1:
                    loss = loss / accum_steps

            if train:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model)

            # Accumulate statistics
            tot += float(loss) * (accum_steps if train and accum_steps > 1 else 1.0)
            n += 1
            pos_tot += float(pos_loss)
            vel_tot += float(vel_loss)
            aux_tot += (
                float(aux_loss) if isinstance(aux_loss, torch.Tensor) else aux_loss
            )

        avg_loss = tot / max(1, n)
        if log_components:
            aux_scaled = aux_w * (aux_tot / max(1, n))
            print(
                f"    Loss breakdown: total={avg_loss:.6f} | "
                f"pos={pos_tot / max(1, n):.6f} | vel={vel_tot / max(1, n):.6f} | "
                f"aux*={aux_scaled:.6f}"
            )
        return avg_loss

    # Evaluate baseline if resuming
    if resume:
        restore = ema.swap_into(model)
        best_val = run_epoch(val_loader, train=False, log_components=True)
        restore()
        print(f"[resume] Baseline validation loss = {best_val:.6f}")

    print("Starting training...")
    for ep in range(1, epochs + 1):
        t0 = time.time()

        # Training epoch
        tr = run_epoch(train_loader, True, log_components=True)

        # Validation with EMA weights
        restore = ema.swap_into(model)
        va = run_epoch(val_loader, False, log_components=True)
        restore()

        dt = time.time() - t0
        print(f"Epoch {ep:03d} | Train {tr:.6f} | Val {va:.6f} | {dt:.1f}s")

        # Checkpointing and early stopping
        if va < best_val - 1e-5:
            best_val, bad = va, 0
            ema.copy_to(model)
            torch.save(
                {"model_state": model.state_dict(), "model_cfg": model_cfg}, ckpt_path
            )
            print("  ✓ Saved best model")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    ckpt = torch.load(ckpt_path, map_location=device)
    clean = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )
    best_model = FlowMatchingModel(**ckpt["model_cfg"]).to(device)
    best_model.load_state_dict(clean, strict=True)

    return best_model
