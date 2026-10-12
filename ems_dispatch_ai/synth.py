"""Synthetic Riyadh EMS demand. All assumptions are illustrative, not real statistics.

Rate(cell, day, hour) = city_daily_total * spatial_weight(cell) * hour_profile(hour | calendar) * day_factor.
Incidents are Poisson draws; locations uniform within the cell.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import calendar as cal
from .geo import Grid

# (lat, lon, weight, spread_deg, kind) demand hotspots. kind: C=commercial, R=residential, I=industrial
HOTSPOTS = [
    (24.690, 46.685, 1.0, 0.030, "C"),   # Olaya / King Fahd Rd
    (24.633, 46.715, 0.9, 0.025, "C"),   # Batha / downtown
    (24.665, 46.740, 0.6, 0.025, "R"),   # Malaz
    (24.725, 46.815, 0.7, 0.035, "R"),   # Naseem / east
    (24.590, 46.665, 0.7, 0.030, "R"),   # Suwaidi / west
    (24.755, 46.630, 0.5, 0.030, "R"),   # Nakheel / north-west
    (24.570, 46.780, 0.6, 0.030, "I"),   # southern industrial area
    (24.800, 46.700, 0.4, 0.040, "R"),   # northern ring road / growth areas
]


def kind_multiplier(kind: str, hour: int, d: date) -> float:
    """Time-of-day shift of demand between commercial, residential and industrial areas."""
    ramadan, off = cal.is_ramadan(d), cal.is_weekend(d) or cal.is_eid(d)
    if kind == "C":
        if ramadan:
            return 1.6 if (hour >= 21 or hour <= 2) else 0.8
        return 1.5 if 9 <= hour <= 17 else (1.1 if 18 <= hour <= 23 else 0.5)
    if kind == "I":
        if off or ramadan and not 8 <= hour <= 15:
            return 0.3
        return 1.9 if 6 <= hour <= 16 else 0.35
    return 1.4 if (hour >= 18 or hour <= 6) else 0.75  # residential

BASE_HOURLY = np.array([0.55, 0.45, 0.38, 0.33, 0.32, 0.38, 0.60, 0.95, 1.05, 1.00, 1.00, 1.05,
                        1.10, 1.10, 1.05, 1.05, 1.15, 1.25, 1.30, 1.30, 1.30, 1.20, 1.00, 0.75])
RAMADAN_HOURLY = np.array([1.10, 1.00, 0.80, 0.55, 0.40, 0.30, 0.30, 0.45, 0.55, 0.60, 0.65, 0.70,
                           0.75, 0.80, 0.90, 1.10, 1.55, 1.85, 0.70, 0.90, 1.20, 1.35, 1.40, 1.30])
WEEKEND_HOURLY = np.array([0.80, 0.70, 0.55, 0.45, 0.38, 0.35, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00,
                           1.05, 1.05, 1.05, 1.10, 1.20, 1.30, 1.35, 1.40, 1.40, 1.35, 1.20, 1.00])


def _kernels(grid: Grid) -> np.ndarray:
    cen = grid.centroids()
    return np.array([wt * np.exp(-(((cen[:, 0] - lat) ** 2 + (cen[:, 1] - lon) ** 2) / (2 * s ** 2)))
                     for lat, lon, wt, s, _ in HOTSPOTS])


def spatial_weights(grid: Grid, hour: int = 12, d: date = date(2025, 1, 6)) -> np.ndarray:
    """Share of city demand per cell for a given hour and day (sums to 1)."""
    k = _kernels(grid)
    m = np.array([kind_multiplier(h[4], hour, d) for h in HOTSPOTS])
    w = 0.05 + (m[:, None] * k).sum(0)
    return w / w.sum()


def hour_profile(d: date) -> np.ndarray:
    if cal.is_ramadan(d):
        p = RAMADAN_HOURLY
    elif cal.is_weekend(d) or cal.is_eid(d):
        p = WEEKEND_HOURLY
    else:
        p = BASE_HOURLY
    return p / p.sum()


def day_factor(d: date) -> float:
    f = 1.0
    if cal.is_hajj(d):
        f *= 0.88   # many residents travel for Hajj / Eid al-Adha holiday
    if cal.is_eid(d):
        f *= 1.08   # festivities, road traffic
    if cal.is_ramadan(d):
        f *= 1.05
    if d.month in (6, 7, 8):
        f *= 1.06   # summer heat illness
    return f


def expected_rates(start: date, days: int, grid: Grid, daily_total: float = 150.0) -> np.ndarray:
    """Array (days, 24, n_cells) of expected incidents per hour per cell."""
    kern = _kernels(grid)
    out = np.empty((days, 24, grid.n_cells))
    for k in range(days):
        d = start + timedelta(days=k)
        m = np.array([[kind_multiplier(h[4], hr, d) for h in HOTSPOTS] for hr in range(24)])  # (24, n_hot)
        w = 0.05 + m @ kern
        w /= w.sum(axis=1, keepdims=True)
        out[k] = daily_total * day_factor(d) * hour_profile(d)[:, None] * w
    return out


def hourly_counts(start: date, days: int, grid: Grid, seed: int = 0, daily_total: float = 150.0) -> pd.DataFrame:
    """Long table: date, hour, cell, count (+ the true expected rate, for analysis only)."""
    rng = np.random.default_rng(seed)
    lam = expected_rates(start, days, grid, daily_total)
    counts = rng.poisson(lam)
    d_idx, h_idx, c_idx = np.indices(lam.shape)
    dates = pd.to_datetime(start) + pd.to_timedelta(d_idx.ravel(), unit="D")
    return pd.DataFrame({"date": dates, "hour": h_idx.ravel(), "cell": c_idx.ravel(),
                         "count": counts.ravel(), "rate_true": lam.ravel()})


def incidents(start: date, days: int, grid: Grid, seed: int = 0, daily_total: float = 150.0) -> pd.DataFrame:
    """Individual incidents with timestamps (minutes from start), lat/lon and priority (1 = highest)."""
    rng = np.random.default_rng(seed + 1)
    hc = hourly_counts(start, days, grid, seed, daily_total)
    hc = hc[hc["count"] > 0]
    rep = hc.loc[hc.index.repeat(hc["count"])].reset_index(drop=True)
    cen = grid.centroids()
    n = len(rep)
    lat = cen[rep["cell"], 0] + (rng.random(n) - 0.5) * grid.dlat
    lon = cen[rep["cell"], 1] + (rng.random(n) - 0.5) * grid.dlon
    day_off = (rep["date"] - pd.Timestamp(start)).dt.days.to_numpy()
    t = day_off * 1440 + rep["hour"].to_numpy() * 60 + rng.random(n) * 60
    pri = rng.choice([1, 2, 3], size=n, p=[0.2, 0.5, 0.3])
    df = pd.DataFrame({"t_min": t, "lat": lat, "lon": lon, "cell": rep["cell"], "priority": pri,
                       "date": rep["date"], "hour": rep["hour"]})
    return df.sort_values("t_min").reset_index(drop=True)
