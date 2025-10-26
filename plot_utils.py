# plot_utils.py
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import torch


def plot_latlon_spaghetti(
    x_hist_glob: np.ndarray,
    y_glob_all: np.ndarray,
    y_true_glob: Optional[np.ndarray],
    N_SAMPLES: int,
) -> go.Figure:
    import numpy as np
    import plotly.graph_objects as go
    from pyproj import Transformer

    # Transformer from LV95 (EPSG:2056) back to WGS84 (EPSG:4326)
    to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def xy_to_lonlat(arr_xy):
        """Convert XY coordinates to longitude/latitude."""
        flat = arr_xy.reshape(-1, 2)
        lon, lat = to_wgs84.transform(flat[:, 0], flat[:, 1])  # returns lon, lat
        lon = np.asarray(lon).reshape(arr_xy.shape[:-1])
        lat = np.asarray(lat).reshape(arr_xy.shape[:-1])
        return lon, lat

    # History -> lon/lat (B, L, 3) -> use x,y
    hist_xy = x_hist_glob[:, :, :2]  # (B, L, 2)
    hist_lon, hist_lat = xy_to_lonlat(hist_xy)
    B = x_hist_glob.shape[0]

    # Ground truth future (optional)
    if y_true_glob is not None:
        true_xy = y_true_glob[:, :, :2]  # (B, T, 2)
        true_lon, true_lat = xy_to_lonlat(true_xy)
    else:
        true_lon = true_lat = None

    # Predictions: (S, B, T, D) -> keep first two dims as-is
    pred_xy = y_glob_all[:, :, :, :2]  # (S, B, T, 2)
    S, B, T, _ = pred_xy.shape
    pred_lon, pred_lat = xy_to_lonlat(pred_xy.reshape(S * B, T, 2))
    pred_lon = pred_lon.reshape(S, B, T)
    pred_lat = pred_lat.reshape(S, B, T)

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig_map = go.Figure()

    # Histories (thin gray) + t=0 markers
    for b in range(B):
        fig_map.add_trace(
            go.Scattermap(
                lon=hist_lon[b],
                lat=hist_lat[b],
                mode="markers",
                line=dict(width=1, color="rgba(0,0,0,1)"),
                marker=dict(size=4, color="black"),
                showlegend=(b == 0),
                name="Last Minute",
            )
        )
        fig_map.add_trace(
            go.Scattermap(
                lon=[hist_lon[b, -1]],
                lat=[hist_lat[b, -1]],
                mode="markers",
                marker=dict(size=6, color="purple"),
                showlegend=(b == 0),
                name="t=0",
            )
        )

    # Predictions (spaghetti)
    for b in range(B):
        color = palette[b % len(palette)]
        for s in range(S):
            fig_map.add_trace(
                go.Scattermap(
                    lon=pred_lon[s, b],
                    lat=pred_lat[s, b],
                    mode="lines",
                    line=dict(width=2, color=color),
                    opacity=0.40,
                    showlegend=(s == 0),
                    name=f"{N_SAMPLES} Predictions",
                )
            )
    # Ground truth (if available)
    if true_lon is not None:
        for b in range(B):
            fig_map.add_trace(
                go.Scattermap(
                    lon=true_lon[b],
                    lat=true_lat[b],
                    mode="markers",
                    line=dict(width=2, color="red"),
                    showlegend=(b == 0),
                    name="Ground Truth",
                )
            )
    # Auto-center the map roughly over the data
    # we center on the last history
    lon_center = hist_lon[0, -1]
    lat_center = hist_lat[0, -1]

    fig_map.update_layout(
        # title=f"CFM: {B} case(s) — {S} samples each (Latitude/Longitude)",
        map_style="carto-positron",
        # center on traj
        map=dict(
            style="carto-positron",
            # style="basic",
            center=dict(lon=lon_center, lat=lat_center),
            zoom=10.1,
        ),
        # legend horizontal on top
        legend=dict(
            itemsizing="trace",
            orientation="h",
            yanchor="bottom",
            y=1,
            x=1,
            xanchor="right",
        ),
        # legend=dict(itemsizing="trace"),
        height=500,
        width=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig_map.show()
    return fig_map
