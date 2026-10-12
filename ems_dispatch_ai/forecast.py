"""Demand forecasting: gradient boosting (Poisson loss) vs. a historical-average baseline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance

from . import calendar as cal
from .geo import Grid

FEATURES = ["cell", "row", "col", "hour", "dow", "weekend", "ramadan", "eid", "hajj", "month",
            "hour_sin", "hour_cos", "cell_rate", "cell_hour_rate"]


def calendar_frame(dates: pd.Series) -> pd.DataFrame:
    days = pd.Series(pd.to_datetime(dates).dt.date.unique())
    f = pd.DataFrame({
        "day": days,
        "ramadan": days.map(cal.is_ramadan).astype(int),
        "eid": days.map(cal.is_eid).astype(int),
        "hajj": days.map(cal.is_hajj).astype(int),
        "weekend": days.map(cal.is_weekend).astype(int),
    })
    return f


def make_features(df: pd.DataFrame, grid: Grid, cell_rate: np.ndarray, cell_hour_rate: np.ndarray) -> pd.DataFrame:
    """df needs columns date, hour, cell."""
    out = df[["date", "hour", "cell"]].copy()
    out["day"] = pd.to_datetime(out["date"]).dt.date
    out = out.merge(calendar_frame(out["date"]), on="day", how="left")
    out["row"], out["col"] = np.divmod(out["cell"].to_numpy(), grid.n_cols)
    out["dow"] = pd.to_datetime(out["date"]).dt.dayofweek
    out["month"] = pd.to_datetime(out["date"]).dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["cell_rate"] = cell_rate[out["cell"]]
    out["cell_hour_rate"] = cell_hour_rate[out["cell"], out["hour"]]
    return out[FEATURES]


@dataclass
class DemandForecaster:
    grid: Grid
    max_iter: int = 300
    learning_rate: float = 0.06
    random_state: int = 0
    model: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def fit(self, train: pd.DataFrame) -> "DemandForecaster":
        g = train.groupby("cell")["count"].mean()
        self.cell_rate = np.zeros(self.grid.n_cells)
        self.cell_rate[g.index] = g.to_numpy()
        gh = train.groupby(["cell", "hour"])["count"].mean()
        self.cell_hour_rate = np.zeros((self.grid.n_cells, 24))
        self.cell_hour_rate[gh.index.get_level_values(0), gh.index.get_level_values(1)] = gh.to_numpy()
        X = make_features(train, self.grid, self.cell_rate, self.cell_hour_rate)
        self.model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            categorical_features=[FEATURES.index("cell")], random_state=self.random_state)
        self.model.fit(X, train["count"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = make_features(df, self.grid, self.cell_rate, self.cell_hour_rate)
        return np.clip(self.model.predict(X), 1e-6, None)

    def predict_block(self, day, hours, n_cells=None) -> np.ndarray:
        """Expected incidents per cell summed over `hours` on `day` -> (n_cells,)."""
        n = n_cells or self.grid.n_cells
        df = pd.DataFrame([(pd.Timestamp(day), h, c) for h in hours for c in range(n)],
                          columns=["date", "hour", "cell"])
        return self.predict(df).reshape(len(hours), n).sum(axis=0)


class HistoricalAverage:
    """Baseline: mean count per (cell, hour, weekend flag) in training data."""

    def fit(self, train: pd.DataFrame) -> "HistoricalAverage":
        t = train.assign(weekend=pd.to_datetime(train["date"]).dt.dayofweek.isin([4, 5]).astype(int))
        self.table = t.groupby(["cell", "hour", "weekend"])["count"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        wk = pd.to_datetime(df["date"]).dt.dayofweek.isin([4, 5]).astype(int)
        idx = pd.MultiIndex.from_arrays([df["cell"], df["hour"], wk])
        return np.clip(self.table.reindex(idx).fillna(0).to_numpy(), 1e-6, None)


def evaluate(y: np.ndarray, pred: np.ndarray, rate_true: np.ndarray | None = None) -> dict:
    """Count-level metrics; with a synthetic ground-truth rate also the error vs the true intensity
    (removes irreducible Poisson noise and shows how well the *demand pattern* is learned)."""
    out = {"mae": float(np.mean(np.abs(y - pred))),
            "poisson_deviance": float(mean_poisson_deviance(y, pred)),
            "total_bias_pct": float((pred.sum() - y.sum()) / max(y.sum(), 1) * 100)}
    if rate_true is not None:
        out["rate_mae"] = float(np.mean(np.abs(rate_true - pred)))
        out["rate_mape_pct"] = float(np.sum(np.abs(rate_true - pred)) / np.sum(rate_true) * 100)
    return out
