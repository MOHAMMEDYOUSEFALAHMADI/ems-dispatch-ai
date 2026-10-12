from datetime import date

import numpy as np
import pytest

from ems_dispatch_ai import calendar as cal
from ems_dispatch_ai import synth
from ems_dispatch_ai.geo import Grid, HaversineRouter, OSRMRouter, haversine_km, speed_kmh


def test_haversine_riyadh_jeddah():
    assert haversine_km(24.7136, 46.6753, 21.4858, 39.1925) == pytest.approx(846, abs=10)


def test_grid_roundtrip():
    g = Grid()
    cen = g.centroids()
    assert cen.shape == (100, 2)
    assert (g.cell_of(cen[:, 0], cen[:, 1]) == np.arange(100)).all()


def test_travel_time_rush_hour_slower():
    r = HaversineRouter()
    a, b = (24.70, 46.68), (24.65, 46.72)
    assert r.minutes(a, b, hour=7) > r.minutes(a, b, hour=11) > r.minutes(a, b, hour=3)
    assert speed_kmh(17, ramadan=True) < speed_kmh(17)


def test_osrm_falls_back_without_server(monkeypatch):
    monkeypatch.delenv("OSRM_URL", raising=False)
    r = OSRMRouter(base_url="http://127.0.0.1:9", timeout=0.5)
    m = r.matrix(np.array([[24.7, 46.7]]), np.array([[24.6, 46.6]]))
    assert r.fallbacks == 1 and m[0, 0] > 0


def test_calendar():
    assert cal.is_ramadan(date(2025, 3, 15)) and not cal.is_ramadan(date(2025, 4, 15))
    assert cal.is_hajj(date(2025, 6, 6))
    assert cal.is_weekend(date(2025, 4, 18))  # Friday


def test_ramadan_shifts_demand_to_night_and_pre_iftar():
    g = Grid()
    normal = synth.expected_rates(date(2025, 4, 14), 1, g).sum(axis=2)[0]   # Monday
    ram = synth.expected_rates(date(2025, 3, 10), 1, g).sum(axis=2)[0]      # Monday in Ramadan
    assert ram[17] > normal[17] and ram[23] > normal[23] and ram[11] < normal[11]


def test_spatial_shift_commercial_by_day_residential_by_night():
    g = Grid()
    olaya = g.cell_of(24.69, 46.685)
    naseem = g.cell_of(24.725, 46.815)
    day, night = synth.spatial_weights(g, 11), synth.spatial_weights(g, 22)
    assert day[olaya] > night[olaya] and night[naseem] > day[naseem]
    assert day.sum() == pytest.approx(1)


def test_incidents_deterministic_and_in_bbox():
    g = Grid()
    a = synth.incidents(date(2025, 4, 1), 2, g, seed=3)
    b = synth.incidents(date(2025, 4, 1), 2, g, seed=3)
    assert a.equals(b) and 200 < len(a) < 400
    assert (g.cell_of(a["lat"], a["lon"]) == a["cell"]).all()
    assert a["t_min"].is_monotonic_increasing
