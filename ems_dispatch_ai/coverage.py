"""Ambulance posting optimisation: Maximal Covering Location Problem (Church & ReVelle 1974)
with an optional backup-coverage term (Hogan & ReVelle 1986), solved exactly as a MILP (SciPy/HiGHS).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack


def coverage_sets(tt: np.ndarray, threshold_min: float) -> np.ndarray:
    """tt: (n_sites, n_demand) travel minutes -> boolean cover matrix."""
    return tt <= threshold_min


def mclp(
    demand: np.ndarray,
    tt: np.ndarray,
    p: int,
    threshold_min: float,
    backup_weight: float = 0.3,
    max_per_site: int = 1,
    time_limit: float = 20.0,
    tail_threshold_min: float | None = None,
    tail_weight: float = 0.0,
) -> list[int]:
    """Choose p ambulance posts (site indices, repeats allowed if max_per_site > 1).

    maximise   sum_i d_i * (y1_i + backup_weight * y2_i + tail_weight * y3_i)
    s.t.       y1_i + y2_i <= sum_{j: t_ji <= T} x_j      (primary + backup coverage within T)
               y2_i <= y1_i
               y3_i <= sum_{j: t_ji <= T_tail} x_j         (optional 'tail' coverage, protects p90)
               sum_j x_j = p ;  x_j in {0..max_per_site} ;  y binary
    Solved exactly with SciPy's MILP interface to HiGHS.
    """
    a = coverage_sets(tt, threshold_min).astype(float)
    n_s, n_d = a.shape
    d = np.asarray(demand, float)
    use_tail = tail_threshold_min is not None and tail_weight > 0
    ny = 3 if use_tail else 2
    c = np.concatenate([np.zeros(n_s), -d, -backup_weight * d] + ([-tail_weight * d] if use_tail else []))
    I, Z = eye(n_d, format="csr"), csr_matrix((n_d, n_d))
    blocks_cov = [-csr_matrix(a.T), I, I] + ([Z] if use_tail else [])
    blocks_ord = [csr_matrix((n_d, n_s)), -I, I] + ([Z] if use_tail else [])
    cons = [
        LinearConstraint(hstack(blocks_cov), -np.inf, 0),
        LinearConstraint(hstack(blocks_ord), -np.inf, 0),
        LinearConstraint(hstack([csr_matrix(np.ones((1, n_s))), csr_matrix((1, ny * n_d))]), p, p),
    ]
    if use_tail:
        at = coverage_sets(tt, tail_threshold_min).astype(float)
        cons.append(LinearConstraint(hstack([-csr_matrix(at.T), Z, Z, I]), -np.inf, 0))
    ub = np.concatenate([np.full(n_s, max_per_site), np.ones(ny * n_d)])
    res = milp(
        c,
        constraints=cons,
        integrality=np.ones(n_s + ny * n_d),
        bounds=Bounds(0, ub),
        options={"time_limit": time_limit, "disp": False},
    )
    if res.x is None:
        return greedy_mclp(demand, tt, p, threshold_min, backup_weight)
    x = np.round(res.x[:n_s]).astype(int)
    sites = [j for j in range(n_s) for _ in range(x[j])]
    return sites if len(sites) == p else greedy_mclp(demand, tt, p, threshold_min, backup_weight)


def greedy_mclp(demand, tt, p, threshold_min, backup_weight=0.3) -> list[int]:
    a = coverage_sets(tt, threshold_min)
    times = np.zeros(a.shape[1], dtype=int)
    chosen: list[int] = []
    for _ in range(p):
        gain = []
        for j in range(a.shape[0]):
            if j in chosen:
                gain.append(-1)
                continue
            g = demand * a[j] * np.where(times == 0, 1.0, np.where(times == 1, backup_weight, 0.0))
            gain.append(g.sum())
        j = int(np.argmax(gain))
        chosen.append(j)
        times += a[j]
    return chosen


def covered_fraction(demand: np.ndarray, tt: np.ndarray, sites: list[int], threshold_min: float) -> float:
    if not sites:
        return 0.0
    covered = (tt[sites] <= threshold_min).any(axis=0)
    return float((demand * covered).sum() / demand.sum())


def kmeans_sites(points: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """Unweighted k-means used to build a geographically even 'legacy' station layout."""
    rng = np.random.default_rng(seed)
    c = points[rng.choice(len(points), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((points[:, None] - c[None]) ** 2).sum(-1), axis=1)
        c = np.array([points[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
    return c
