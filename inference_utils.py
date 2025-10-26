from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def sample_future_heun(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    context: torch.Tensor,
    T_out: int,
    n_steps: int = 64,
    G: float = 1.0,
    use_autocast: bool = True,
) -> torch.Tensor:
    """
    Heun integrator for Conditional Flow Matching, returning normalized futures.

    Args:
        model: torch.nn.Module with forward (x_hist, x_t, t_scalar, context)
        x_hist: (B, L, D) normalized, aircraft-centric history
        context: (B, C) normalized context
        T_out: number of future steps
        n_steps: time discretization of the flow (default 64)
        G: guidance/scaling factor for the velocity field
        use_autocast: enable CUDA autocast if available

    Returns:
        x: (B, T_out, D) normalized future sequence
    """
    model.eval()
    B, _, D = x_hist.shape

    # Initialize the future trajectory with random noise from standard normal distribution
    x = torch.randn(B, T_out, D, device=x_hist.device, dtype=x_hist.dtype)

    # Calculate time step size for discretizing the continuous flow
    dt = 1.0 / n_steps

    amp_enabled = (
        (x_hist.device.type == "cuda") or (x_hist.device.type == "mps")
    ) and use_autocast
    amp_dtype = torch.bfloat16

    # Perform Heun integration over n_steps to solve the flow ODE
    for k in range(n_steps):
        # Calculate current and next time points for this integration step
        # Clamp to avoid numerical issues at t=1.0
        t0v = min(k * dt, 1.0 - 1e-6)
        t1v = min((k + 1) * dt, 1.0 - 1e-6)

        # Create time tensors for the entire batch
        t0 = torch.full((B, 1), t0v, device=x_hist.device, dtype=x_hist.dtype)
        t1 = torch.full((B, 1), t1v, device=x_hist.device, dtype=x_hist.dtype)

        with torch.amp.autocast(
            device_type=x_hist.device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            # Compute velocity field at current time t0 and position x
            v1 = model(x_hist, x, t0, context)

            # Take Euler step to estimate position at next time (predictor step)
            x_pred = x + (G * v1) * dt

            # Compute velocity field at predicted position and next time t1
            v2 = model(x_hist, x_pred, t1, context)

            # Take weighted average of velocities and update position (corrector step)
            x = x + 0.5 * (G * v1 + G * v2) * dt
    return x


@torch.no_grad()
def sample_many(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    ctx: torch.Tensor,
    T_out: int,
    n_steps: int = 64,
    G: float = 1.0,
    n_samples: int = 200,
    chunk: int = 128,
) -> torch.Tensor:
    """
    Draw many normalized futures by repeating inputs in chunks to fit memory.

    Returns:
        (n_samples, B, T_out, D) after caller reshapes
    """
    # Initialize list to collect generated samples
    outs = []

    # Process samples in chunks to avoid GPU memory overflow
    # This allows generating large numbers of samples even with limited memory
    for s0 in range(0, n_samples, chunk):
        # Calculate number of samples for this chunk (handle remainder)
        n = min(chunk, n_samples - s0)

        # Repeat the history and context tensors for this chunk
        # x_hist.repeat(n, 1, 1) creates n copies along batch dimension
        # ctx.repeat(n, 1) creates n copies of context vectors
        repeated_hist = x_hist.repeat(n, 1, 1)
        repeated_ctx = ctx.repeat(n, 1)

        # Generate samples for this chunk using the Heun integrator
        chunk_samples = sample_future_heun(
            model,
            repeated_hist,
            repeated_ctx,
            T_out=T_out,
            n_steps=n_steps,
            G=G,
        )

        # Store the generated chunk
        outs.append(chunk_samples)

    # Concatenate all chunks along the batch dimension to create final output
    # Result has shape (n_samples, B, T_out, D) where B is original batch size
    return torch.cat(outs, dim=0)


@torch.no_grad()
def denorm_seq_to_global(
    seq_norm: torch.Tensor,
    ctx_norm: torch.Tensor,
    feat_mean: torch.Tensor | np.ndarray,
    feat_std: torch.Tensor | np.ndarray,
    ctx_mean: torch.Tensor | np.ndarray,
    ctx_std: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Convert normalized, aircraft-centric sequence back to global frame.

    Mirrors the implementation in the notebook.

    Args:
        seq_norm: (B, T, D) normalized sequence
        ctx_norm: (B, C) normalized context
        feat_mean/std: per-feature stats of length D
        ctx_mean/std: per-context stats of length >= 5 (x0,y0,z0,cos,sin,...)

    Returns:
        (B, T, D) in global coordinates/units
    """
    # Extract dimensions for tensor operations
    B, T, D = seq_norm.shape

    # Convert feature statistics to tensors with proper shape for broadcasting
    # Reshape to (1, 1, D) so they can be broadcasted across batch and time dimensions
    fm = torch.as_tensor(feat_mean, dtype=seq_norm.dtype, device=seq_norm.device).view(
        1, 1, -1
    )
    fs = torch.as_tensor(feat_std, dtype=seq_norm.dtype, device=seq_norm.device).view(
        1, 1, -1
    )

    # Denormalize the sequence using feature statistics: seq = seq_norm * std + mean
    seq = seq_norm * fs + fm

    # Get context dimension and convert context statistics to tensors
    C = ctx_norm.size(-1)
    cm = torch.as_tensor(
        ctx_mean[:C], dtype=ctx_norm.dtype, device=ctx_norm.device
    ).view(1, C)
    cs = torch.as_tensor(
        ctx_std[:C], dtype=ctx_norm.dtype, device=ctx_norm.device
    ).view(1, C)

    # Denormalize the context: ctx_raw = ctx_norm * std + mean
    ctx_raw = ctx_norm * cs + cm

    # Extract cosine and sine values for rotation matrix (aircraft heading)
    # These represent the aircraft's orientation in the global frame
    c = ctx_raw[:, 3:4]  # cos(heading)
    s = ctx_raw[:, 4:5]  # sin(heading)

    # Rotate position coordinates from aircraft-centric to global frame
    # Apply 2D rotation matrix: [cos θ, -sin θ; sin θ, cos θ]
    x_local = seq[..., 0]  # aircraft-centric x (forward direction)
    y_local = seq[..., 1]  # aircraft-centric y (right direction)
    seq[..., 0] = c * x_local - s * y_local  # global x
    seq[..., 1] = s * x_local + c * y_local  # global y

    # Rotate velocity coordinates using the same rotation matrix
    # Velocities transform the same way as positions under rotation
    vx_local = seq[..., 3]  # aircraft-centric velocity x
    vy_local = seq[..., 4]  # aircraft-centric velocity y
    seq[..., 3] = c * vx_local - s * vy_local  # global velocity x
    seq[..., 4] = s * vx_local + c * vy_local  # global velocity y

    # Add reference position to convert from relative to absolute coordinates
    # ctx_raw[:, :3] contains the aircraft's reference position (x0, y0, z0)
    ref_xyz = ctx_raw[:, :3].view(B, 1, 3)
    seq[..., :3] = seq[..., :3] + ref_xyz

    # Return the sequence in global coordinates with proper units
    return seq
