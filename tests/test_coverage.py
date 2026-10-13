import itertools

import numpy as np

from ems_dispatch_ai.coverage import covered_fraction, greedy_mclp, kmeans_sites, mclp


def brute_force(d, tt, p, T, bw):
    best, arg = -1, None
    a = tt <= T
    for comb in itertools.combinations(range(tt.shape[0]), p):
        k = a[list(comb)].sum(0)
        v = (d * (k >= 1)).sum() + bw * (d * (k >= 2)).sum()
        if v > best + 1e-9:
            best, arg = v, comb
    return best


def value(d, tt, sites, T, bw):
    k = (tt[sites] <= T).sum(0)
    return (d * (k >= 1)).sum() + bw * (d * (k >= 2)).sum()


def test_mclp_is_optimal_on_small_instances():
    rng = np.random.default_rng(0)
    for _ in range(5):
        tt = rng.uniform(0, 15, (8, 12))
        d = rng.uniform(0, 5, 12)
        sites = mclp(d, tt, 3, 6.0, backup_weight=0.3)
        assert len(sites) == 3
        assert value(d, tt, sites, 6.0, 0.3) >= brute_force(d, tt, 3, 6.0, 0.3) - 1e-6


def test_mclp_at_least_as_good_as_greedy():
    rng = np.random.default_rng(1)
    tt = rng.uniform(0, 20, (30, 40))
    d = rng.gamma(1.0, 2.0, 40)
    s1, s2 = mclp(d, tt, 5, 7.0), greedy_mclp(d, tt, 5, 7.0)
    assert value(d, tt, s1, 7.0, 0.3) >= value(d, tt, s2, 7.0, 0.3) - 1e-6


def test_tail_term_improves_tail_coverage():
    rng = np.random.default_rng(2)
    tt = rng.uniform(0, 25, (25, 30))
    d = rng.gamma(1.0, 2.0, 30)
    plain = mclp(d, tt, 4, 6.0, backup_weight=0)
    tail = mclp(d, tt, 4, 6.0, backup_weight=0, tail_threshold_min=12.0, tail_weight=0.5)
    assert covered_fraction(d, tt, tail, 12.0) >= covered_fraction(d, tt, plain, 12.0)


def test_kmeans_sites_deterministic():
    pts = np.random.default_rng(0).uniform(size=(50, 2))
    assert np.allclose(kmeans_sites(pts, 4), kmeans_sites(pts, 4))
