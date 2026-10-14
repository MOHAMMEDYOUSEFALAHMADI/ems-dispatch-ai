"""Riyadh grid, distances and travel-time estimation (haversine fallback, optional OSRM)."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

# Bounding box roughly covering central Riyadh.
LAT_MIN, LAT_MAX = 24.52, 24.86
LON_MIN, LON_MAX = 46.56, 46.90
EARTH_R_KM = 6371.0088

# Approximate public locations of major hospitals (used as synthetic transport destinations).
HOSPITALS = {
    "King Faisal Specialist Hospital": (24.672, 46.676),
    "King Fahad Medical City": (24.689, 46.705),
    "King Khalid University Hospital": (24.716, 46.620),
    "Prince Sultan Military Medical City": (24.646, 46.714),
    "King Abdulaziz Medical City (NGHA)": (24.746, 46.852),
    "Security Forces Hospital": (24.735, 46.693),
}


@dataclass(frozen=True)
class Grid:
    n_rows: int = 10
    n_cols: int = 10

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def dlat(self) -> float:
        return (LAT_MAX - LAT_MIN) / self.n_rows

    @property
    def dlon(self) -> float:
        return (LON_MAX - LON_MIN) / self.n_cols

    def centroids(self) -> np.ndarray:
        """(n_cells, 2) array of (lat, lon); cell id = row * n_cols + col, row 0 = south."""
        r, c = np.divmod(np.arange(self.n_cells), self.n_cols)
        return np.column_stack([LAT_MIN + (r + 0.5) * self.dlat, LON_MIN + (c + 0.5) * self.dlon])

    def cell_of(self, lat, lon):
        r = np.clip(((np.asarray(lat) - LAT_MIN) / self.dlat).astype(int), 0, self.n_rows - 1)
        c = np.clip(((np.asarray(lon) - LON_MIN) / self.dlon).astype(int), 0, self.n_cols - 1)
        return r * self.n_cols + c


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def speed_kmh(hour: int, ramadan: bool = False) -> float:
    """Emergency-vehicle average road speed by hour of day (synthetic Riyadh traffic profile)."""
    if ramadan:
        if 16 <= hour <= 18:  # pre-iftar rush
            return 26.0
        if hour >= 21 or hour <= 1:  # busy Ramadan nights
            return 34.0
    if hour in (6, 7, 8, 12, 13, 16, 17, 18):  # commute / school / afternoon peaks
        return 28.0
    if 0 <= hour <= 5:
        return 55.0
    return 42.0


CIRCUITY = 1.3  # road distance / great-circle distance (typical urban grid)


class HaversineRouter:
    """Travel time in minutes = haversine * circuity / speed(hour)."""

    name = "haversine"

    def matrix(self, origins: np.ndarray, dests: np.ndarray, hour: int = 12, ramadan: bool = False) -> np.ndarray:
        o, d = np.asarray(origins, float), np.asarray(dests, float)
        km = haversine_km(o[:, None, 0], o[:, None, 1], d[None, :, 0], d[None, :, 1])
        return km * CIRCUITY / speed_kmh(hour, ramadan) * 60.0

    def minutes(self, a, b, hour: int = 12, ramadan: bool = False) -> float:
        return float(self.matrix(np.array([a]), np.array([b]), hour, ramadan)[0, 0])


class OSRMRouter(HaversineRouter):
    """Uses an OSRM server's /table API when OSRM_URL is set; falls back to haversine on any error.

    OSRM durations are free-flow car times, so they are scaled by the same hour-of-day congestion factor.
    """

    name = "osrm"

    def __init__(self, base_url: str | None = None, timeout: float = 10.0):
        self.base_url = (base_url or os.environ.get("OSRM_URL", "")).rstrip("/")
        self.timeout = timeout
        self.fallbacks = 0

    def matrix(self, origins, dests, hour=12, ramadan=False):
        if not self.base_url:
            self.fallbacks += 1
            return super().matrix(origins, dests, hour, ramadan)
        try:
            pts = list(origins) + list(dests)
            coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in pts)
            src = ";".join(str(i) for i in range(len(origins)))
            dst = ";".join(str(len(origins) + i) for i in range(len(dests)))
            url = f"{self.base_url}/table/v1/driving/{coords}?sources={src}&destinations={dst}&annotations=duration"
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                data = json.loads(r.read())
            secs = np.array(data["durations"], dtype=float)
            return secs / 60.0 * (42.0 / speed_kmh(hour, ramadan))
        except Exception:
            self.fallbacks += 1
            return super().matrix(origins, dests, hour, ramadan)


def get_router():
    return OSRMRouter() if os.environ.get("OSRM_URL") else HaversineRouter()
