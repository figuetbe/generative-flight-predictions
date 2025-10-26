#!/usr/bin/env python3
"""
Generate future trajectory predictions for a single flight clip.

Main entry: predict_trajectories(flight_df, ...)
Returns lat/lon/alt samples with the correct timing and orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# ----- If you already have these elsewhere, you can delete the imports below and use your own -----
# We keep them here to make the script self-contained and robust.
from utils import aircraft_centric_transform

# =========================
# Geometry / transforms
# =========================


def denorm_seq_to_global(
    seq_norm: torch.Tensor,
    ctx_norm: torch.Tensor,
    feat_mean,
    feat_std,
    ctx_mean,
    ctx_std,
) -> torch.Tensor:
    """
    Inverse of training transform with the CORRECT rotation (R^T):
      local -> global uses:
        [xg]   [ c  s] [xl]
        [yg] = [-s  c] [yl]
    Also applies de-normalization and reference translation.
    seq_norm: (N, T, D)
    ctx_norm: (N, C)
    Returns:  (N, T, D) in global LV95 (meters, m/s, rad/s)
    """
    N, T, D = seq_norm.shape
    device = seq_norm.device
    dtype = seq_norm.dtype

    fm = torch.as_tensor(feat_mean, dtype=dtype, device=device).view(1, 1, -1)
    fs = torch.as_tensor(feat_std, dtype=dtype, device=device).view(1, 1, -1)
    seq = seq_norm * fs + fm  # (N,T,D)

    C = ctx_norm.size(-1)
    cm = torch.as_tensor(ctx_mean[:C], dtype=dtype, device=device).view(1, C)
    cs = torch.as_tensor(ctx_std[:C], dtype=dtype, device=device).view(1, C)
    ctx_raw = ctx_norm * cs + cm  # (N,C)

    # rotation params
    c = ctx_raw[:, 3:4]  # cosθ  (N,1)
    s = ctx_raw[:, 4:5]  # sinθ  (N,1)

    # R^T for local->global
    x_local = seq[..., 0]
    y_local = seq[..., 1]
    vx_local = seq[..., 3]
    vy_local = seq[..., 4]

    x_global = c * x_local + s * y_local
    y_global = -s * x_local + c * y_local
    vx_global = c * vx_local + s * vy_local
    vy_global = -s * vx_local + c * vy_local

    seq[..., 0] = x_global
    seq[..., 1] = y_global
    seq[..., 3] = vx_global
    seq[..., 4] = vy_global

    # translate by absolute reference (x,y,z)
    ref_xyz = ctx_raw[:, :3].view(N, 1, 3)
    seq[..., :3] = seq[..., :3] + ref_xyz
    return seq


# =========================
# Preprocess flight data
# =========================


def preprocess_flight_data(flight_df: pd.DataFrame) -> np.ndarray:
    """
    Build the exact input feature array the model expects for ONE flight:
      - project lon/lat -> LV95 x,y
      - compute vx,vy from track & groundspeed; z,vz from altitude, vertical_rate
      - compute psi_rate exactly as in training
      - validates input has exactly 60 consecutive samples at 1Hz intervals
      - returns shape (60,7)
    """
    df = flight_df.copy()

    # --- projection: WGS84 -> LV95 (always_xy to avoid axis confusion)
    import pyproj

    crs_lv95 = pyproj.CRS.from_epsg(2056)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    to_lv95 = pyproj.Transformer.from_crs(crs_wgs84, crs_lv95, always_xy=True)

    # required cols
    needed = [
        "longitude",
        "latitude",
        "track",
        "groundspeed",
        "altitude",
        "vertical_rate",
        "timestamp",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"preprocess: missing columns: {missing}")

    # timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # project
    x_coords, y_coords = to_lv95.transform(
        df["longitude"].to_numpy(), df["latitude"].to_numpy()
    )
    df["x"] = x_coords
    df["y"] = y_coords

    # kinematics
    KNOTS2MPS = 0.5144444444444444
    FTPM2MPS = 0.3048 / 60.0
    track_rad = np.deg2rad(df["track"].to_numpy())
    spd_mps = df["groundspeed"].to_numpy() * KNOTS2MPS
    df["vx"] = spd_mps * np.sin(track_rad)
    df["vy"] = spd_mps * np.cos(track_rad)
    df["z"] = df["altitude"] * 0.3048
    df["vz"] = df["vertical_rate"] * FTPM2MPS

    # Assume input is already properly sorted and resampled to exactly 60 samples at 1Hz
    if len(df) != 60:
        raise ValueError(f"Expected exactly 60 samples, got {len(df)}")

    # Check that timestamps are consecutive with exactly 1s intervals
    # timestamps = df["timestamp"].sort_values().reset_index(drop=True)
    # expected_timestamps = timestamps.iloc[0] + pd.to_timedelta(range(60), unit="s")
    # if not timestamps.equals(expected_timestamps):
    #     raise ValueError("Timestamps must be consecutive with exactly 1s intervals")

    # compute psi_rate exactly like training (group shift over one series)
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    vxm = np.roll(vx, 1)
    vxm[0] = vx[0]
    vym = np.roll(vy, 1)
    vym[0] = vy[0]
    cross = vxm * vy - vym * vx
    dot = vxm * vx + vym * vy
    psi_rate = -np.arctan2(cross, dot)
    psi_rate = np.nan_to_num(psi_rate, nan=0.0, posinf=0.0, neginf=0.0)
    psi_rate = np.clip(psi_rate, -0.25, 0.25)
    df["psi_rate"] = psi_rate

    features = ["x", "y", "z", "vx", "vy", "vz", "psi_rate"]
    X_raw = df[features].to_numpy().astype(np.float32)  # (60,7)
    return X_raw


# =========================
# Sampler wrapper (handles shape)
# =========================


def sample_many_safe(
    model, x_hist, ctx, T_out, n_steps, G, n_samples, chunk=128
) -> torch.Tensor:
    """
    Wraps your sample_many version to ensure output is (n_samples, T_out, D).
    """
    from inference_utils import sample_many as _sample_many  # use your implementation

    y = _sample_many(
        model,
        x_hist,
        ctx,
        T_out=T_out,
        n_steps=n_steps,
        G=G,
        n_samples=n_samples,
        chunk=min(chunk, n_samples),
    )
    # If sample_many returns (n_samples*B, T, D), reshape when B=1:
    if x_hist.size(0) == 1 and y.dim() == 3 and y.size(0) == n_samples:
        return y
    if x_hist.size(0) == 1 and y.size(0) == n_samples * 1:
        return y.view(n_samples, y.size(1), y.size(2))
    # fallback (already (n_samples, T, D))
    return y


# =========================
# Resampling (5s baseline -> arbitrary)
# =========================


def resample_predictions(
    predictions: torch.Tensor, sampling_rate: float
) -> torch.Tensor:
    """
    predictions: (n_samples, 12, D) at +5,+10,…,+60 seconds
    sampling_rate: new step in seconds. We resample onto [5, 5+sampling_rate, 5+2*sampling_rate, …, 60].
    """
    assert predictions.dim() == 3, "predictions must be (S, T, D)"
    S, T, D = predictions.shape
    # Original prediction times are +5s … +60s (12 points)
    original_times = np.arange(5, 61, 5, dtype=float)  # [5,10,...,60]
    new_times = np.arange(
        5, 60 + 1e-6, sampling_rate, dtype=float
    )  # [5, 5+sampling_rate, 5+2*sampling_rate, ..., ~60]

    # Interpolate along time for each feature independently
    out = torch.empty(
        (S, len(new_times), D), dtype=predictions.dtype, device=predictions.device
    )
    pred_np = predictions.detach().cpu().numpy()  # (S,T,D)
    for f in range(D):
        vals = pred_np[:, :, f]  # (S,T)
        res = np.vstack(
            [np.interp(new_times, original_times, vals_i) for vals_i in vals]
        )  # (S, len(new_times))
        out[:, :, f] = torch.from_numpy(res).to(out.device, out.dtype)
    return out


# =========================
# Lat/Lon conversion
# =========================


def predictions_to_latlonalt(predictions_np: np.ndarray) -> np.ndarray:
    """
    predictions_np: (n_samples, n_timesteps, >=3) with x,y,z in LV95 meters.
    Returns: (n_samples, n_timesteps, 3) as [lat, lon, alt_ft]
    """
    import pyproj

    crs_lv95 = pyproj.CRS.from_epsg(2056)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    to_wgs84 = pyproj.Transformer.from_crs(crs_lv95, crs_wgs84, always_xy=True)

    S, T, D = predictions_np.shape
    out = np.zeros((S, T, 3), dtype=float)
    for s in range(S):
        lon, lat = to_wgs84.transform(predictions_np[s, :, 0], predictions_np[s, :, 1])
        alt_ft = predictions_np[s, :, 2] / 0.3048
        out[s, :, 0] = lat
        out[s, :, 1] = lon
        out[s, :, 2] = alt_ft
    return out


def predict_trajectories(
    flight_df: pd.DataFrame,
    model: torch.nn.Module,
    feat_mean,
    feat_std,
    ctx_mean,
    ctx_std,
    device: torch.device,
    n_samples: int = 10,
    sampling_rate: int = 5,  # seconds
    n_steps: int = 64,
    guidance_scale: float = 1.0,
) -> np.ndarray:
    """
    End-to-end: raw df -> (n_samples, n_times, 3=[lat,lon,alt_ft])
    """
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    # 1) Build features (assumes flight_df is already 1Hz resampled to 60 samples)
    X_raw = preprocess_flight_data(flight_df)  # (60,7)

    # 2) Aircraft-centric transform (parity with training)
    X_raw_b = X_raw[None, :, :]
    Y_dummy = np.zeros((1, 1, X_raw.shape[1]), dtype=X_raw.dtype)
    X_t_b, _Y_t_b, C_raw_b = aircraft_centric_transform(X_raw_b, Y_dummy)
    X_t, C_raw = X_t_b[0], C_raw_b[0]

    # 3) Normalize
    fm = np.asarray(feat_mean, np.float32)
    fs = np.asarray(feat_std, np.float32) + 1e-8
    cm = np.asarray(ctx_mean, np.float32)
    cs = np.asarray(ctx_std, np.float32) + 1e-8
    X_norm = ((X_t - fm) / fs).astype(np.float32)
    C_norm = ((C_raw - cm[: len(C_raw)]) / cs[: len(C_raw)]).astype(np.float32)

    x_hist = torch.from_numpy(X_norm).unsqueeze(0).to(device)
    ctx = torch.from_numpy(C_norm).unsqueeze(0).to(device)

    # 4) Sample futures at the model's native stride (5s → 12 steps up to +60s)
    T_out = 12
    futures_norm = sample_many_safe(
        model=model,
        x_hist=x_hist,
        ctx=ctx,
        T_out=T_out,
        n_steps=int(n_steps),
        G=float(guidance_scale),
        n_samples=int(n_samples),
        chunk=128,
    )  # (n_samples, 12, D)

    # 5) Denormalize & map back to global LV95
    ctx_rep = ctx.repeat(n_samples, 1)
    futures_global = denorm_seq_to_global(
        futures_norm, ctx_rep, feat_mean, feat_std, ctx_mean, ctx_std
    )
    # futures_global is torch.Tensor (n_samples, 12, D)

    # 6) Optional resampling from +5..+60s to desired sampling grid
    if abs(sampling_rate - 5.0) > 1e-6:
        futures_global = resample_predictions(futures_global, sampling_rate)

    # 7) Convert to lat/lon/alt (feet)
    predictions_latlonalt = predictions_to_latlonalt(
        futures_global.detach().cpu().numpy()
    )
    return predictions_latlonalt


def latlonalt_to_cartesian(latlonalt: np.ndarray) -> np.ndarray:
    """
    Convert lat/lon/alt to cartesian coordinates (x, y, z) in meters.

    Parameters:
    latlonalt: array with [lat, lon, alt_ft] in last dimension
               Can be: (n_points, 3) or (n_samples, n_timesteps, 3) or (n_timesteps, 3)

    Returns:
    cartesian: array with [x, y, z] in meters, same shape as input
    """
    import pyproj

    # Convert feet to meters
    alt_m = latlonalt[..., 2] * 0.3048

    # LV95 projection for x, y
    crs_lv95 = pyproj.CRS.from_epsg(2056)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs_wgs84, crs_lv95, always_xy=True)

    lat = latlonalt[..., 0]
    lon = latlonalt[..., 1]

    # Handle different shapes
    if lat.ndim == 1:  # (n_points,) - single set of points
        x, y = transformer.transform(lon, lat)
        return np.stack([x, y, alt_m], axis=-1)
    elif lat.ndim == 2:  # (n_samples, n_timesteps) - multiple trajectories
        cartesian = np.zeros((*lat.shape, 3), dtype=float)
        for i in range(lat.shape[0]):
            x, y = transformer.transform(lon[i], lat[i])
            cartesian[i, :, 0] = x
            cartesian[i, :, 1] = y
            cartesian[i, :, 2] = alt_m[i]
        return cartesian
    else:  # (n_samples, n_timesteps, n_features) - 3D case
        cartesian = np.zeros_like(latlonalt)
        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                x, y = transformer.transform(lon[i, j], lat[i, j])
                cartesian[i, j, 0] = x
                cartesian[i, j, 1] = y
                cartesian[i, j, 2] = alt_m[i, j]
        return cartesian


def calculate_trajectory_spacings(
    f1_latlonalt: np.ndarray, f2_latlonalt: np.ndarray
) -> pd.DataFrame:
    """
    Calculate closest point of approach (CPA) spacings for each f1, f2 sample combination.

    Parameters:
    f1_latlonalt: (n_samples_f1, n_timesteps, 3) array with [lat, lon, alt_ft]
    f2_latlonalt: (n_samples_f2, n_timesteps, 3) array with [lat, lon, alt_ft]

    Returns:
    DataFrame with columns:
    - sample_1: index of f1 sample
    - sample_2: index of f2 sample
    - CPA_3D_meter: 3D distance at closest point of approach in meters
    - index: time index where CPA occurs
    - hori_spacing_NM: horizontal spacing in nautical miles
    - vert_spacing_ft: vertical spacing in feet
    """
    n_samples_f1, n_timesteps, _ = f1_latlonalt.shape
    n_samples_f2 = f2_latlonalt.shape[0]

    # Convert to cartesian coordinates
    f1_cart = latlonalt_to_cartesian(f1_latlonalt)
    f2_cart = latlonalt_to_cartesian(f2_latlonalt)

    results = []

    for i in range(n_samples_f1):
        for j in range(n_samples_f2):
            # Get trajectories for this sample pair
            traj1 = f1_cart[i]  # (n_timesteps, 3)
            traj2 = f2_cart[j]  # (n_timesteps, 3)

            # Calculate pairwise distances between all time points
            # traj1: (n_timesteps, 3), traj2: (n_timesteps, 3)
            # diff: (n_timesteps, n_timesteps, 3)
            diff = traj1[:, np.newaxis, :] - traj2[np.newaxis, :, :]
            distances_3d = np.linalg.norm(diff, axis=2)  # (n_timesteps, n_timesteps)

            # Find minimum distance and its indices
            min_idx = np.unravel_index(np.argmin(distances_3d), distances_3d.shape)
            cpa_3d_meter = distances_3d[min_idx]
            time_idx = min_idx[0]  # Use the time index from trajectory 1

            # Calculate horizontal and vertical spacing at CPA
            pos1 = traj1[min_idx[0]]
            pos2 = traj2[min_idx[1]]

            # Horizontal distance (x, y plane)
            hori_dist_m = np.linalg.norm(pos1[:2] - pos2[:2])

            # Vertical distance
            vert_dist_m = abs(pos1[2] - pos2[2])

            # Convert to requested units
            hori_spacing_NM = hori_dist_m / 1852.0  # meters to nautical miles
            vert_spacing_ft = vert_dist_m / 0.3048  # meters to feet

            results.append(
                {
                    "sample_1": i,
                    "sample_2": j,
                    "CPA_3D_meter": cpa_3d_meter,
                    "index": time_idx,
                    "hori_spacing_NM": hori_spacing_NM,
                    "vert_spacing_ft": vert_spacing_ft,
                }
            )

    return pd.DataFrame(results)


def calculate_conflicting_pair_spacings(conflicting_pair) -> pd.DataFrame:
    """
    Calculate horizontal and vertical spacing between two conflicting flights at each timestamp.

    Parameters:
    conflicting_pair: Traffic object containing two flight trajectories

    Returns:
    DataFrame with columns:
    - timestamp: timestamp of the observation
    - hori_spacing_NM: horizontal spacing in nautical miles
    - vert_spacing_ft: vertical spacing in feet
    - flight_1_id: ID of first flight
    - flight_2_id: ID of second flight
    """
    # Extract data for both flights
    f1_data = conflicting_pair[0].data.copy()
    f2_data = conflicting_pair[1].data.copy()

    # Ensure timestamps are datetime
    f1_data["timestamp"] = pd.to_datetime(f1_data["timestamp"], utc=True)
    f2_data["timestamp"] = pd.to_datetime(f2_data["timestamp"], utc=True)

    # Merge on timestamp to get paired observations
    merged = pd.merge(
        f1_data[["timestamp", "latitude", "longitude", "altitude", "flight_id"]],
        f2_data[["timestamp", "latitude", "longitude", "altitude", "flight_id"]],
        on="timestamp",
        suffixes=("_f1", "_f2"),
        how="inner",  # Only timestamps where both flights have data
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "hori_spacing_NM",
                "vert_spacing_ft",
                "flight_1_id",
                "flight_2_id",
            ]
        )

    # Convert to cartesian coordinates for distance calculation
    # Create arrays for latlonalt conversion
    f1_coords = merged[["latitude_f1", "longitude_f1", "altitude_f1"]].values
    f2_coords = merged[["latitude_f2", "longitude_f2", "altitude_f2"]].values

    f1_cart = latlonalt_to_cartesian(f1_coords)  # (n_timestamps, 3)
    f2_cart = latlonalt_to_cartesian(f2_coords)  # (n_timestamps, 3)

    # Calculate spacings at each timestamp
    results = []
    for i, (idx, row) in enumerate(merged.iterrows()):
        pos1 = f1_cart[i]
        pos2 = f2_cart[i]

        # Horizontal distance (x, y plane)
        hori_dist_m = np.linalg.norm(pos1[:2] - pos2[:2])

        # Vertical distance
        vert_dist_m = abs(pos1[2] - pos2[2])

        # Convert to requested units
        hori_spacing_NM = hori_dist_m / 1852.0  # meters to nautical miles
        vert_spacing_ft = vert_dist_m / 0.3048  # meters to feet

        results.append(
            {
                "timestamp": row["timestamp"],
                "hori_spacing_NM": hori_spacing_NM,
                "vert_spacing_ft": vert_spacing_ft,
                "flight_1_id": row["flight_id_f1"],
                "flight_2_id": row["flight_id_f2"],
            }
        )

    return pd.DataFrame(results)
