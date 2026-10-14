from datetime import date

import numpy as np
import pandas as pd

from ems_dispatch_ai import synth
from ems_dispatch_ai.geo import Grid, HaversineRouter
from ems_dispatch_ai.simulate import (
    Policy,
    SimConfig,
    legacy_posts,
    nearest_available,
    simulate,
    static_policy,
    summarize,
)


def test_nearest_available_skips_busy():
    pos = np.array([[24.70, 46.70], [24.60, 46.60], [24.71, 46.71]])
    a, t = nearest_available(pos, np.array([True, True, False]), (24.712, 46.712), HaversineRouter(), 12, False)
    assert a == 0 and t > 0


def test_simulation_invariants_and_determinism():
    g, cfg = Grid(), SimConfig(n_ambulances=12)
    inc = synth.incidents(date(2025, 4, 15), 2, g, seed=3)
    pol = static_policy("legacy", legacy_posts(g, cfg.n_ambulances))
    r1 = simulate(inc, date(2025, 4, 15), pol, g, cfg)
    r2 = simulate(inc, date(2025, 4, 15), pol, g, cfg)
    assert len(r1) == len(inc) and r1.equals(r2)
    assert (r1["response_min"] >= cfg.call_processing_min + cfg.chute_min).all()
    assert (r1["wait_min"] >= 0).all()
    s = summarize(r1, 2)
    assert 0 <= s["pct_within_8"] <= s["pct_within_10"] <= 100


def test_queueing_when_fleet_tiny():
    g = Grid()
    inc = synth.incidents(date(2025, 4, 15), 1, g, seed=3)
    r = simulate(inc, date(2025, 4, 15), static_policy("one", legacy_posts(g, 1)), g, SimConfig(n_ambulances=1))
    assert (r["wait_min"] > 0).mean() > 0.5


def test_dynamic_policy_relocates():
    g, cfg = Grid(), SimConfig(n_ambulances=4, block_hours=4)
    cen = g.centroids()
    plans = {0: cen[[0, 9, 90, 99]], 1: cen[[44, 45, 54, 55]]}
    pol = Policy("toggle", lambda day, hour: plans[(hour // 4) % 2], dynamic=True)
    inc = pd.DataFrame(
        {
            "t_min": [10.0, 300.0, 500.0],
            "lat": cen[[44, 45, 54], 0],
            "lon": cen[[44, 45, 54], 1],
            "cell": [44, 45, 54],
            "priority": [1, 2, 3],
        }
    )
    r = simulate(inc, date(2025, 4, 15), pol, g, cfg)
    assert r.attrs["relocations"] >= 4
    assert r["travel_min"].iloc[1] < r["travel_min"].iloc[0]  # after relocation units are near the centre
