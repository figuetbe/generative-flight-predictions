from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import DataLoader, Dataset
from traffic.core import Traffic

# ---------------------- Constants ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

lvl95 = pyproj.Proj("EPSG:2056")

KNOTS2MPS = 0.5144444444444444
FTPM2MPS = 0.3048 / 60.0

CACHE_DIR = pathlib.Path("./dataset_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "cfm-v4"

# ---------------------- Utilities ----------------------


@contextmanager
def stage(name: str):
    t0 = pd.Timestamp.now().timestamp()
    logging.info(f"[start] {name}")
    try:
        yield
    finally:
        dt = pd.Timestamp.now().timestamp() - t0
        logging.info(f"[done]  {name} in {dt:.1f}s")


def df_fingerprint(df: pd.DataFrame, cols: Tuple[str, ...]) -> Dict[str, Any]:
    sub = df.loc[:, ["flight_id", "timestamp", *cols]].copy()
    sub["flight_id"] = sub["flight_id"].astype("string[python]")
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
    h = pd.util.hash_pandas_object(sub, index=False).values
    digest = hashlib.sha256(h.tobytes()).hexdigest()
    return {
        "n_rows": int(len(sub)),
        "n_flights": int(sub["flight_id"].nunique()),
        "ts_min": str(sub["timestamp"].min()),
        "ts_max": str(sub["timestamp"].max()),
        "sha256": digest,
    }


def stable_json_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def make_stats_key(fp: Dict[str, Any], prep: Dict[str, Any]) -> str:
    blob = {"ver": CACHE_VERSION, "role": "norm_stats", "df": fp, "prep": prep}
    return stable_json_hash(blob)[:16]


def make_dataset_key(
    fp: Dict[str, Any], prep: Dict[str, Any], samp: Dict[str, Any]
) -> str:
    blob = {
        "ver": CACHE_VERSION,
        "role": "dataset",
        "df": fp,
        "prep": prep,
        "samp": samp,
    }
    return stable_json_hash(blob)[:16]


def cache_paths(basekey: str, statskey: str) -> Dict[str, pathlib.Path]:
    base = CACHE_DIR / basekey
    return {
        "x_tr": base.with_suffix(".X_train.npy"),
        "y_tr": base.with_suffix(".Y_train.npy"),
        "c_tr": base.with_suffix(".C_train.npy"),
        "x_va": base.with_suffix(".X_val.npy"),
        "y_va": base.with_suffix(".Y_val.npy"),
        "c_va": base.with_suffix(".C_val.npy"),
        "x_te": base.with_suffix(".X_test.npy"),
        "y_te": base.with_suffix(".Y_test.npy"),
        "c_te": base.with_suffix(".C_test.npy"),
        "meta_tr": base.with_suffix(".meta_train.parquet"),
        "meta_va": base.with_suffix(".meta_val.parquet"),
        "meta_te": base.with_suffix(".meta_test.parquet"),
        "stats": (CACHE_DIR / statskey).with_suffix(".norm_stats.json"),
        "manifest": base.with_suffix(".split_manifest.json"),
        "key": base.with_suffix(".key.json"),
        "summary": base.with_suffix(".summary.json"),
    }


def cache_exists(paths: Dict[str, pathlib.Path]) -> bool:
    need = [
        "x_tr",
        "y_tr",
        "c_tr",
        "x_va",
        "y_va",
        "c_va",
        "x_te",
        "y_te",
        "c_te",
        "meta_tr",
        "meta_va",
        "meta_te",
        "stats",
        "manifest",
        "key",
        "summary",
    ]
    return all(paths[k].exists() for k in need)


# ---------------------- Feature engineering ----------------------


def load_and_engineer(input_parquet: str) -> pd.DataFrame:
    trajs = Traffic.from_file(input_parquet)

    import pyarrow as pa
    import pyarrow.compute as pc

    table = pa.Table.from_pandas(trajs.data, preserve_index=False)
    new_cols = []
    for name, col in zip(table.column_names, table.columns):
        if pa.types.is_string(col.type):
            col = pc.cast(col, pa.large_string())
        new_cols.append(col)
    table = pa.table(new_cols, names=table.column_names)
    trajs.data = table.to_pandas(types_mapper=pd.ArrowDtype)

    trajs.data = trajs.data.loc[
        ~trajs.data.callsign.str.contains("ASR|OOAS|FCK", na=False)
    ]

    track_rad = np.deg2rad(trajs.data["track"].to_numpy())
    spd_mps = trajs.data["groundspeed"].to_numpy() * KNOTS2MPS
    trajs.data["vx"] = spd_mps * np.sin(track_rad)
    trajs.data["vy"] = spd_mps * np.cos(track_rad)
    trajs.data["z"] = trajs.data["altitude"] * 0.3048
    trajs.data["vz"] = trajs.data["vertical_rate"] * FTPM2MPS

    trajs = trajs.compute_xy(lvl95)

    df = trajs.data[["timestamp", "flight_id", "x", "y", "z", "vx", "vy", "vz"]].copy()
    df["flight_id"] = df["flight_id"].astype("string[python]")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["flight_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    g = df.groupby("flight_id", sort=False)
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    vxm = g["vx"].shift(1).to_numpy()
    vym = g["vy"].shift(1).to_numpy()

    cross = vxm * vy - vym * vx
    dot = vxm * vx + vym * vy
    psi_rate = -np.arctan2(cross, dot)

    df["psi_rate"] = pd.Series(psi_rate, index=df.index)
    df["psi_rate"] = np.nan_to_num(df["psi_rate"], nan=0.0, posinf=0.0, neginf=0.0)
    df["psi_rate"] = np.clip(df["psi_rate"], -0.25, 0.25)

    return df


# ---------------------- Windowing & sampling ----------------------


@dataclass
class WindowParams:
    input_len: int = 60
    output_horizon: int = 60
    output_stride: int = 5
    overlap: bool = False

    def to_prep_dict(self) -> Dict[str, Any]:
        return {
            "input_len": int(self.input_len),
            "output_horizon": int(self.output_horizon),
            "output_stride": int(self.output_stride),
            "overlap": bool(self.overlap),
            "features": ["x", "y", "z", "vx", "vy", "vz", "psi_rate"],
        }


@dataclass
class TurnSampling:
    min_turn_frac: float = 0.30
    turn_thr: float = 0.01
    consec: int = 3
    consider_hist: bool = True
    consider_future: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_turn_frac": float(self.min_turn_frac),
            "turn_thr": float(self.turn_thr),
            "consec": int(self.consec),
            "consider_hist": bool(self.consider_hist),
            "consider_future": bool(self.consider_future),
        }


def _enumerate_windows(
    df: pd.DataFrame, features: Tuple[str, ...], params: WindowParams
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
    needed = ["timestamp", "flight_id", *features]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    work = (
        df.loc[:, needed]
        .sort_values(["flight_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    work["flight_id"] = work["flight_id"].astype("string[python]").astype("category")
    codes = work["flight_id"].cat.codes.to_numpy()

    T = params.input_len + params.output_horizon
    K = int(codes.max()) + 1 if len(codes) else 0
    lengths = np.bincount(codes, minlength=K)
    valid_mask = lengths >= T
    valid_codes = np.nonzero(valid_mask)[0]
    valid_counts = lengths[valid_mask] - (T - 1)

    first = np.empty_like(codes, dtype=bool)
    first[0] = True
    if len(codes) > 1:
        first[1:] = codes[1:] != codes[:-1]
    flight_starts = np.flatnonzero(first)
    code_order = codes[first]
    startpos_map = dict(zip(code_order.tolist(), flight_starts.tolist()))

    per_flight_starts = [
        startpos_map[int(fc)] + np.arange(int(cnt), dtype=np.int64)
        for fc, cnt in zip(valid_codes, valid_counts)
    ]
    if len(per_flight_starts) == 0:
        raise ValueError("No valid windows.")
    all_starts = np.concatenate(per_flight_starts, axis=0)

    flight_spans = {
        int(fc): (startpos_map[int(fc)], int(lengths[int(fc)])) for fc in valid_codes
    }
    return all_starts, codes, flight_spans


def _triad_flags(
    df: pd.DataFrame,
    features: Tuple[str, ...],
    params: WindowParams,
    turn: TurnSampling,
    codes: np.ndarray,
    flight_spans: Dict[int, Tuple[int, int]],
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    b_all = df["psi_rate"].to_numpy() > float(turn.turn_thr)
    tri_true = np.zeros_like(b_all, dtype=np.uint8)

    K = int(codes.max()) + 1 if len(codes) else 0
    first = np.empty_like(codes, dtype=bool)
    first[0] = True
    if len(codes) > 1:
        first[1:] = codes[1:] != codes[:-1]
    flight_starts = np.flatnonzero(first)

    lengths = np.bincount(codes, minlength=K)

    for fcode in range(K):
        L = int(lengths[fcode])
        if L == 0:
            continue
        start = int(flight_starts[np.where((codes[flight_starts] == fcode))[0][0]])
        if L < turn.consec:
            continue
        bf = b_all[start : start + L].astype(np.uint8)
        wsum = np.convolve(bf, np.ones(turn.consec, dtype=np.uint8), mode="valid")
        tri_f = (wsum == turn.consec).astype(np.uint8)
        tri_true[start : start + L - turn.consec + 1] = tri_f

    tri_ps = np.zeros(len(tri_true) + 1, dtype=np.int64)
    tri_ps[1:] = np.cumsum(tri_true)
    return tri_true, tri_ps


def _any_triad(tri_ps: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    b = np.clip(b, 0, len(tri_ps) - 1)
    a = np.clip(a, 0, len(tri_ps) - 1)
    return (tri_ps[b] - tri_ps[a]) > 0


def cadence_clean_window_mask(
    df: pd.DataFrame,
    codes: np.ndarray,
    all_starts: np.ndarray,
    window_len: int,
    tol: float = 0.0,
) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    dt = ts.groupby(df["flight_id"], sort=False).diff().dt.total_seconds().to_numpy()

    if tol > 0.0:
        is_bad = ~np.isclose(dt, 1.0, atol=tol)
    else:
        is_bad = dt != 1.0

    first = np.empty_like(codes, dtype=bool)
    first[0] = True
    if len(codes) > 1:
        first[1:] = codes[1:] != codes[:-1]
    is_bad[first] = False

    bad_int = is_bad.astype(np.int64)
    csum = np.cumsum(bad_int)

    first_idx = np.where(first, np.arange(len(codes)), 0)
    flight_start_index_for_row = np.maximum.accumulate(first_idx)

    csum_f = csum - csum[flight_start_index_for_row]

    ends = all_starts + (window_len - 1)
    count_bad = csum_f[ends] - csum_f[all_starts]
    return count_bad == 0


def sample_windows(
    df: pd.DataFrame,
    n_samps: int,
    params: WindowParams,
    turn: TurnSampling,
    rng: np.random.Generator,
    uniform_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    features = ("x", "y", "z", "vx", "vy", "vz", "psi_rate")
    all_starts, codes, flight_spans = _enumerate_windows(df, features, params)

    T = params.input_len + params.output_horizon
    mask = cadence_clean_window_mask(df, codes, all_starts, T, tol=0.0)
    all_starts = all_starts[mask]
    if all_starts.size == 0:
        raise ValueError(
            "No valid windows after enforcing 1 Hz cadence. Resample your data."
        )

    tri_true, tri_ps = _triad_flags(df, features, params, turn, codes, flight_spans)

    T = params.input_len + params.output_horizon
    hist_has = (
        _any_triad(tri_ps, all_starts, all_starts + params.input_len - turn.consec + 1)
        if turn.consider_hist
        else np.zeros_like(all_starts, dtype=bool)
    )
    fut_starts = all_starts + params.input_len
    fut_has = (
        _any_triad(
            tri_ps, fut_starts, fut_starts + params.output_horizon - turn.consec + 1
        )
        if turn.consider_future
        else np.zeros_like(all_starts, dtype=bool)
    )
    has_turn = hist_has | fut_has

    total_windows = all_starts.shape[0]
    n_pick = min(int(n_samps), total_windows)

    if uniform_only or (turn.min_turn_frac <= 0.0):
        choices = rng.choice(total_windows, size=n_pick, replace=False)
        abs_starts = all_starts[choices]
    else:
        desired_pos = int(np.ceil(min(turn.min_turn_frac, 1.0) * n_pick))
        pos_pool = all_starts[has_turn]
        neg_pool = all_starts[~has_turn]
        n_pos_avail = int(pos_pool.shape[0])
        n_neg_avail = int(neg_pool.shape[0])

        if n_pos_avail == 0:
            choices = rng.choice(total_windows, size=n_pick, replace=False)
            abs_starts = all_starts[choices]
        else:
            n_pos = min(desired_pos, n_pos_avail, n_pick)
            n_neg = n_pick - n_pos
            if n_neg > n_neg_avail:
                n_neg = n_neg_avail
                n_pos = min(n_pick - n_neg, n_pos_avail)
            pos_sel = rng.choice(n_pos_avail, size=n_pos, replace=False)
            neg_sel = rng.choice(n_neg_avail, size=n_neg, replace=False)
            abs_starts = np.concatenate([pos_pool[pos_sel], neg_pool[neg_sel]], axis=0)
            rng.shuffle(abs_starts)

    feat_arr = df[list(features)].to_numpy()
    idx = abs_starts[:, None] + np.arange(T, dtype=np.int64)[None, :]
    XY = feat_arr[idx]
    X_raw = XY[:, : params.input_len, :]
    Y_full = XY[:, params.input_len :, :]

    fut_sel = (
        np.arange(params.output_stride, params.output_horizon + 1, params.output_stride)
        - 1
    ).astype(np.int64)
    if params.overlap:
        last_x = X_raw[:, -1:, :]
        Y_raw = np.concatenate([last_x, Y_full[:, fut_sel, :]], axis=1)
    else:
        Y_raw = Y_full[:, fut_sel, :]

    ts_arr = df["timestamp"].to_numpy()
    codes_all_starts = codes[abs_starts]
    flights_cats = df["flight_id"].astype("category").cat
    meta = pd.DataFrame(
        {
            "flight_id": flights_cats.categories[codes_all_starts].astype(str).tolist(),
            "start_row": abs_starts,
            "last_hist_row": abs_starts + (params.input_len - 1),
            "last_hist_timestamp": ts_arr[abs_starts + (params.input_len - 1)],
            "has_turn_hist": hist_has[np.searchsorted(all_starts, abs_starts)].astype(
                bool
            ),
            "has_turn_fut": fut_has[np.searchsorted(all_starts, abs_starts)].astype(
                bool
            ),
        }
    )
    return X_raw, Y_raw, has_turn, meta


# ---------------------- Geometry & normalization ----------------------


def rotate_xy_inplace(arr: np.ndarray, c: np.ndarray, s: np.ndarray) -> None:
    x = arr[..., 0].astype(np.float64)
    y = arr[..., 1].astype(np.float64)
    arr[..., 0] = c[:, None] * x - s[:, None] * y
    arr[..., 1] = s[:, None] * x + c[:, None] * y
    vx = arr[..., 3].astype(np.float64)
    vy = arr[..., 4].astype(np.float64)
    arr[..., 3] = c[:, None] * vx - s[:, None] * vy
    arr[..., 4] = s[:, None] * vx + c[:, None] * vy


def aircraft_centric_transform(
    X_raw: np.ndarray, Y_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = X_raw[:, -1, :3].copy()
    X_t = X_raw.copy()
    Y_t = Y_raw.copy()
    X_t[..., :3] -= refs[:, None, :]
    Y_t[..., :3] -= refs[:, None, :]

    vx_last = X_raw[:, -1, 3].astype(np.float64)
    vy_last = X_raw[:, -1, 4].astype(np.float64)
    vz_last = X_raw[:, -1, 5].astype(np.float64)
    gs_last = np.hypot(vx_last, vy_last).astype(np.float64)
    eps = 1e-8
    c = np.where(gs_last > eps, vy_last / (gs_last + eps), 1.0)
    s = np.where(gs_last > eps, vx_last / (gs_last + eps), 0.0)

    rotate_xy_inplace(X_t, c, s)
    rotate_xy_inplace(Y_t, c, s)

    psi_rate_last = X_raw[:, -1, 6].astype(np.float32)
    C_raw = np.stack(
        [
            refs[:, 0].astype(np.float32),
            refs[:, 1].astype(np.float32),
            refs[:, 2].astype(np.float32),
            c.astype(np.float32),
            s.astype(np.float32),
            gs_last.astype(np.float32),
            vz_last.astype(np.float32),
            psi_rate_last,
        ],
        axis=1,
    )
    return X_t, Y_t, C_raw


@dataclass
class StatsConfig:
    stats_seed: int = 42
    stats_sample_size: int = 1_000_000

    def to_prep_dict(self) -> Dict[str, Any]:
        return {
            "stats_seed": int(self.stats_seed),
            "stats_sample_size": int(self.stats_sample_size),
        }


def compute_or_load_norm_stats(
    df_train: pd.DataFrame,
    wparams: WindowParams,
    turn: TurnSampling,
    stats_cfg: StatsConfig,
    fp: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, pathlib.Path]:
    prep = {
        **wparams.to_prep_dict(),
        **{"transform": "aircraft_centric"},
        **stats_cfg.to_prep_dict(),
    }
    stats_key = make_stats_key(fp, prep)
    stats_path = (CACHE_DIR / stats_key).with_suffix(".norm_stats.json")
    if stats_path.exists():
        norm_stats = json.loads(stats_path.read_text())
        return norm_stats, stats_key, stats_path

    rng = np.random.default_rng(stats_cfg.stats_seed)

    X_raw, Y_raw, _has_turn, _meta = sample_windows(
        df_train,
        n_samps=stats_cfg.stats_sample_size,
        params=wparams,
        turn=turn,
        rng=rng,
        uniform_only=True,
    )

    X_t, Y_t, C_raw = aircraft_centric_transform(X_raw, Y_raw)

    XY = np.concatenate([X_t, Y_t], axis=1).astype(np.float64)
    feat_mean = XY.reshape(-1, XY.shape[-1]).mean(axis=0)
    feat_std = XY.reshape(-1, XY.shape[-1]).std(axis=0, dtype=np.float64) + 1e-8

    ctx_mean = C_raw.mean(axis=0)
    ctx_std = C_raw.std(axis=0) + 1e-8

    norm_stats = {
        "feat_mean": feat_mean.astype(np.float32).tolist(),
        "feat_std": feat_std.astype(np.float32).tolist(),
        "ctx_mean": ctx_mean.astype(np.float32).tolist(),
        "ctx_std": ctx_std.astype(np.float32).tolist(),
        "stats_key": stats_key,
        "prep": prep,
    }
    stats_path.write_text(json.dumps(norm_stats, indent=2))
    return norm_stats, stats_key, stats_path


# ---------------------- Split & dataset build ----------------------


@dataclass
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    split_seed: int = 42

    def to_prep_dict(self) -> Dict[str, Any]:
        return {
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "split_seed": self.split_seed,
        }


@dataclass
class SamplingConfig:
    n_train: int = 2_000_000
    n_val: int = 500_000
    n_test: int = 400_000
    train_turn: TurnSampling = field(
        default_factory=lambda: TurnSampling(min_turn_frac=0.30)
    )
    val_turn: TurnSampling = field(
        default_factory=lambda: TurnSampling(min_turn_frac=0.30)
    )
    test_turn: TurnSampling = field(
        default_factory=lambda: TurnSampling(min_turn_frac=0.0)
    )

    def to_samp_dict(self) -> Dict[str, Any]:
        return {
            "n_train": int(self.n_train),
            "n_val": int(self.n_val),
            "n_test": int(self.n_test),
            "train_turn": self.train_turn.to_dict(),
            "val_turn": self.val_turn.to_dict(),
            "test_turn": self.test_turn.to_dict(),
        }


def split_flights(
    df: pd.DataFrame, scfg: SplitConfig
) -> Tuple[List[str], List[str], List[str]]:
    flights = df["flight_id"].astype(str).unique()
    rng = np.random.RandomState(scfg.split_seed)
    rng.shuffle(flights)
    n = len(flights)
    n_tr = int(round(scfg.train_frac * n))
    n_va = int(round(scfg.val_frac * n))
    n_te = n - n_tr - n_va
    train_f, val_f, test_f = (
        flights[:n_tr],
        flights[n_tr : n_tr + n_va],
        flights[n_tr + n_va :],
    )
    assert len(set(train_f) & set(val_f)) == 0
    assert len(set(train_f) & set(test_f)) == 0
    assert len(set(val_f) & set(test_f)) == 0
    return list(map(str, train_f)), list(map(str, val_f)), list(map(str, test_f))


class CFMDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, C: np.ndarray):
        self.X, self.Y, self.C = X, Y, C

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.Y[i], dtype=torch.float32)
        c = torch.tensor(self.C[i], dtype=torch.float32)
        return x, y, c


def make_loader(ds: Dataset, bs: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=min(os.cpu_count() or 1, 8),
        pin_memory=True,
        drop_last=True,
    )


def build_or_load_dataset(
    df: pd.DataFrame,
    wparams: WindowParams,
    scfg: SplitConfig,
    samp: SamplingConfig,
    stats_cfg: StatsConfig,
):
    features = ("x", "y", "z", "vx", "vy", "vz", "psi_rate")

    fp = df_fingerprint(df, features)
    prep = {
        **wparams.to_prep_dict(),
        **scfg.to_prep_dict(),
        "transform": "aircraft_centric",
        "features": list(features),
    }
    stats_key = make_stats_key(
        fp,
        {
            **wparams.to_prep_dict(),
            **scfg.to_prep_dict(),
            **stats_cfg.to_prep_dict(),
            "transform": "aircraft_centric",
        },
    )
    dset_key = make_dataset_key(fp, prep, samp.to_samp_dict())
    paths = cache_paths(dset_key, stats_key)

    if cache_exists(paths):
        print(f"[cache] hit: {dset_key}")
        X_train = np.load(paths["x_tr"], mmap_mode="r")
        Y_train = np.load(paths["y_tr"], mmap_mode="r")
        C_train = np.load(paths["c_tr"], mmap_mode="r")
        X_val = np.load(paths["x_va"], mmap_mode="r")
        Y_val = np.load(paths["y_va"], mmap_mode="r")
        C_val = np.load(paths["c_va"], mmap_mode="r")
        X_test = np.load(paths["x_te"], mmap_mode="r")
        Y_test = np.load(paths["y_te"], mmap_mode="r")
        C_test = np.load(paths["c_te"], mmap_mode="r")
        norm_stats = json.loads(paths["stats"].read_text())
        meta_train = pd.read_parquet(paths["meta_tr"])
        meta_val = pd.read_parquet(paths["meta_va"])
        meta_test = pd.read_parquet(paths["meta_te"])
        manifest = json.loads(paths["manifest"].read_text())
        summary = json.loads(paths["summary"].read_text())
        return (
            X_train,
            Y_train,
            C_train,
            X_val,
            Y_val,
            C_val,
            X_test,
            Y_test,
            C_test,
            norm_stats,
            meta_train,
            meta_val,
            meta_test,
            manifest,
            summary,
        )

    print(f"[cache] miss: {dset_key} -> building…")

    train_flights, val_flights, test_flights = split_flights(df, scfg)

    df_train = df[df["flight_id"].astype(str).isin(train_flights)].copy()
    df_val = df[df["flight_id"].astype(str).isin(val_flights)].copy()
    df_test = df[df["flight_id"].astype(str).isin(test_flights)].copy()

    norm_stats, stats_key_, stats_path = compute_or_load_norm_stats(
        df_train, wparams, samp.train_turn, stats_cfg, fp
    )

    feat_mean = np.array(norm_stats["feat_mean"], dtype=np.float32)
    feat_std = np.array(norm_stats["feat_std"], dtype=np.float32)
    ctx_mean = np.array(norm_stats["ctx_mean"], dtype=np.float32)
    ctx_std = np.array(norm_stats["ctx_std"], dtype=np.float32)

    rng_tr = np.random.default_rng(42)
    rng_va = np.random.default_rng(43)
    rng_te = np.random.default_rng(44)

    Xtr_raw, Ytr_raw, _turn_tr, meta_train = sample_windows(
        df_train,
        n_samps=samp.n_train,
        params=wparams,
        turn=samp.train_turn,
        rng=rng_tr,
        uniform_only=False,
    )
    Xva_raw, Yva_raw, _turn_va, meta_val = sample_windows(
        df_val,
        n_samps=samp.n_val,
        params=wparams,
        turn=samp.val_turn,
        rng=rng_va,
        uniform_only=False,
    )
    Xte_raw, Yte_raw, _turn_te, meta_test = sample_windows(
        df_test,
        n_samps=samp.n_test,
        params=wparams,
        turn=samp.test_turn,
        rng=rng_te,
        uniform_only=True,
    )

    Xtr_t, Ytr_t, Ctr_raw = aircraft_centric_transform(Xtr_raw, Ytr_raw)
    Xva_t, Yva_t, Cva_raw = aircraft_centric_transform(Xva_raw, Yva_raw)
    Xte_t, Yte_t, Cte_raw = aircraft_centric_transform(Xte_raw, Yte_raw)

    X_train = ((Xtr_t - feat_mean) / feat_std).astype(np.float32)
    Y_train = ((Ytr_t - feat_mean) / feat_std).astype(np.float32)
    X_val = ((Xva_t - feat_mean) / feat_std).astype(np.float32)
    Y_val = ((Yva_t - feat_mean) / feat_std).astype(np.float32)
    X_test = ((Xte_t - feat_mean) / feat_std).astype(np.float32)
    Y_test = ((Yte_t - feat_mean) / feat_std).astype(np.float32)

    C_train = ((Ctr_raw - ctx_mean) / ctx_std).astype(np.float32)
    C_val = ((Cva_raw - ctx_mean) / ctx_std).astype(np.float32)
    C_test = ((Cte_raw - ctx_mean) / ctx_std).astype(np.float32)

    def _turn_frac(meta: pd.DataFrame) -> Dict[str, float]:
        return {
            "turn_hist_frac": float(np.mean(meta["has_turn_hist"].to_numpy()))
            if len(meta)
            else float("nan"),
            "turn_fut_frac": float(np.mean(meta["has_turn_fut"].to_numpy()))
            if len(meta)
            else float("nan"),
        }

    summary = {
        "sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "turn_fractions": {
            "train": _turn_frac(meta_train),
            "val": _turn_frac(meta_val),
            "test": _turn_frac(meta_test),
        },
        "windowing": wparams.to_prep_dict(),
        "normalization_stats_key": stats_key_,
    }

    np.save(paths["x_tr"], X_train, allow_pickle=False)
    np.save(paths["y_tr"], Y_train, allow_pickle=False)
    np.save(paths["c_tr"], C_train, allow_pickle=False)
    np.save(paths["x_va"], X_val, allow_pickle=False)
    np.save(paths["y_va"], Y_val, allow_pickle=False)
    np.save(paths["c_va"], C_val, allow_pickle=False)
    np.save(paths["x_te"], X_test, allow_pickle=False)
    np.save(paths["y_te"], Y_test, allow_pickle=False)
    np.save(paths["c_te"], C_test, allow_pickle=False)

    meta_train.to_parquet(paths["meta_tr"], index=False)
    meta_val.to_parquet(paths["meta_va"], index=False)
    meta_test.to_parquet(paths["meta_te"], index=False)

    manifest = {
        "train_flights": train_flights,
        "val_flights": val_flights,
        "test_flights": test_flights,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2))

    paths["summary"].write_text(json.dumps(summary, indent=2))

    key_info = {
        "cache_version": CACHE_VERSION,
        "dataset_key": dset_key,
        "stats_key": stats_key_,
        "fingerprint": fp,
        "prep": prep,
        "sampling": samp.to_samp_dict(),
    }
    paths["key"].write_text(json.dumps(key_info, indent=2))

    if not paths["stats"].exists():
        paths["stats"].write_text(json.dumps(norm_stats, indent=2))

    return (
        X_train,
        Y_train,
        C_train,
        X_val,
        Y_val,
        C_val,
        X_test,
        Y_test,
        C_test,
        norm_stats,
        meta_train,
        meta_val,
        meta_test,
        manifest,
        summary,
    )
