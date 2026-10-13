"""Discrete-event simulation of ambulance dispatch under different posting policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import calendar as cal
from .coverage import kmeans_sites, mclp
from .geo import HOSPITALS, Grid, HaversineRouter


@dataclass
class SimConfig:
    n_ambulances: int = 24
    call_processing_min: float = 1.5     # call-taking + triage
    chute_min: float = 1.0               # crew turnout
    on_scene_min: float = 18.0
    transport_prob: float = 0.7
    handover_min: float = 20.0
    block_hours: int = 4                 # re-optimisation interval for dynamic policy
    threshold_min: float = 8.0           # response-time target used by the optimiser
    backup_weight: float = 0.3
    tail_threshold_min: float = 15.0     # secondary target protecting the 90th percentile
    tail_weight: float = 0.5
    seed: int = 0


def nearest_available(amb_pos: np.ndarray, available: np.ndarray, incident: tuple[float, float],
                      router: HaversineRouter, hour: int, ramadan: bool) -> tuple[int, float]:
    """Index of the available ambulance with the smallest estimated travel time, and that time."""
    idx = np.flatnonzero(available)
    tt = router.matrix(amb_pos[idx], np.array([incident]), hour, ramadan)[:, 0]
    k = int(np.argmin(tt))
    return int(idx[k]), float(tt[k])


@dataclass
class Policy:
    """Returns ambulance post coordinates (n_ambulances, 2) for the block starting at (day, hour)."""
    name: str
    posts_fn: callable = field(repr=False)
    dynamic: bool = False


def static_policy(name: str, posts: np.ndarray) -> Policy:
    return Policy(name, lambda day, hour: posts, dynamic=False)


def legacy_posts(grid: Grid, n: int) -> np.ndarray:
    return kmeans_sites(grid.centroids(), n, seed=0)


def mclp_posts(grid: Grid, demand: np.ndarray, n: int, cfg: SimConfig, router: HaversineRouter,
               hour: int, ramadan: bool) -> np.ndarray:
    cen = grid.centroids()
    tt = router.matrix(cen, cen, hour, ramadan)
    travel_budget = cfg.threshold_min - cfg.call_processing_min - cfg.chute_min
    overhead = cfg.call_processing_min + cfg.chute_min
    sites = mclp(demand, tt, n, travel_budget, cfg.backup_weight,
                 tail_threshold_min=cfg.tail_threshold_min - overhead, tail_weight=cfg.tail_weight)
    return cen[sites]


def simulate(incidents: pd.DataFrame, start: date, policy: Policy, grid: Grid, cfg: SimConfig,
             router: HaversineRouter | None = None) -> pd.DataFrame:
    router = router or HaversineRouter()
    t0 = datetime.combine(start, time())
    rng = np.random.default_rng(cfg.seed)
    hosp = np.array(list(HOSPITALS.values()))
    n = cfg.n_ambulances
    posts = np.array(policy.posts_fn(start, 0), dtype=float)
    pos_free = posts.copy()            # where each ambulance is when it becomes free
    free_at = np.zeros(n)              # minute when ambulance becomes free
    at_post_at = np.zeros(n)           # minute when it is back at its post
    relocations, block_end = 0, cfg.block_hours * 60
    rows = []
    for inc in incidents.itertuples(index=False):
        t = inc.t_min
        while policy.dynamic and t >= block_end:   # re-plan at block boundaries
            day = (t0 + timedelta(minutes=block_end)).date()
            new_posts = np.array(policy.posts_fn(day, int((block_end % 1440) // 60)), dtype=float)
            cur = np.where((at_post_at <= block_end)[:, None], posts, pos_free)
            cost = router.matrix(cur, new_posts, int((block_end % 1440) // 60))
            r, c = linear_sum_assignment(cost)
            order = np.empty(n, dtype=int)
            order[r] = c
            for a in range(n):
                tgt = new_posts[order[a]]
                if np.allclose(tgt, posts[a]):
                    continue
                if free_at[a] <= block_end:          # idle -> drive to new post now
                    idle_pos = cur[a]
                    if cost[a, order[a]] > 0.5:
                        relocations += 1
                    pos_free[a] = idle_pos
                    free_at[a] = min(free_at[a], block_end)
                    at_post_at[a] = block_end + cost[a, order[a]]
                posts[a] = tgt                        # busy units return to the new post later
            block_end += cfg.block_hours * 60
        d = t0 + timedelta(minutes=t)
        hour, ram = d.hour, cal.is_ramadan(d.date())
        avail = free_at <= t
        cur_pos = np.where((at_post_at <= t)[:, None], posts, pos_free)
        if avail.any():
            a, travel = nearest_available(cur_pos, avail, (inc.lat, inc.lon), router, hour, ram)
            wait, depart = 0.0, t
        else:                                         # queue: first unit to free up takes the call
            a = int(np.argmin(free_at))
            wait, depart = free_at[a] - t, free_at[a]
            travel = router.minutes(tuple(pos_free[a]), (inc.lat, inc.lon), hour, ram)
        response = cfg.call_processing_min + cfg.chute_min + wait + travel
        arrive = t + response
        if rng.random() < cfg.transport_prob:
            h_tt = router.matrix(np.array([[inc.lat, inc.lon]]), hosp, hour, ram)[0]
            h = int(np.argmin(h_tt))
            done, end_pos = arrive + cfg.on_scene_min + h_tt[h] + cfg.handover_min, hosp[h]
        else:
            done, end_pos = arrive + cfg.on_scene_min, np.array([inc.lat, inc.lon])
        free_at[a] = done
        pos_free[a] = end_pos
        at_post_at[a] = done + router.minutes(tuple(end_pos), tuple(posts[a]), hour, ram)
        rows.append((t, inc.priority, a, wait, travel, response))
    out = pd.DataFrame(rows, columns=["t_min", "priority", "ambulance", "wait_min", "travel_min", "response_min"])
    out.attrs["relocations"] = relocations
    return out


def summarize(res: pd.DataFrame, days: int) -> dict:
    r = res["response_min"]
    return {
        "incidents": int(len(res)),
        "median_min": round(float(r.median()), 2),
        "p90_min": round(float(r.quantile(0.9)), 2),
        "pct_within_8": round(float((r <= 8).mean() * 100), 1),
        "pct_within_10": round(float((r <= 10).mean() * 100), 1),
        "pct_queued": round(float((res["wait_min"] > 0).mean() * 100), 2),
        "p1_median_min": round(float(res.loc[res["priority"] == 1, "response_min"].median()), 2),
        "relocations_per_day": round(res.attrs.get("relocations", 0) / days, 1),
    }
