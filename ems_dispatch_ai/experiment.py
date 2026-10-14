"""End-to-end experiment: forecast -> optimise posts -> simulate -> metrics + charts.

python -m ems_dispatch_ai.experiment [--quick]
Writes results/*.json and docs/*.png (used by README and the Streamlit dashboard).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as cal
from . import synth
from .forecast import DemandForecaster, HistoricalAverage, evaluate
from .geo import HOSPITALS, Grid, HaversineRouter
from .simulate import Policy, SimConfig, legacy_posts, mclp_posts, simulate, static_policy, summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS, DOCS = ROOT / "results", ROOT / "docs"
TRAIN_START, TRAIN_END = date(2024, 1, 1), date(2025, 2, 14)
TEST_START, TEST_END = date(2025, 2, 15), date(2025, 4, 14)
SCENARIOS = {"ramadan": date(2025, 3, 8), "normal": date(2025, 4, 15)}


def build_policies(grid, fc, cfg, router, start, days):
    cen = grid.centroids()
    n = cfg.n_ambulances
    legacy = legacy_posts(grid, n)
    # static MCLP: one plan from forecast average demand over the horizon
    avg = sum(fc.predict_block(start + timedelta(days=k), list(range(24))) for k in range(days)) / days
    static = mclp_posts(grid, avg, n, cfg, router, hour=12, ramadan=cal.is_ramadan(start))
    cache: dict = {}

    def dynamic(day, hour):
        key = (pd.Timestamp(day).date(), hour)
        if key not in cache:
            hrs = list(range(hour, hour + cfg.block_hours))
            dem = fc.predict_block(key[0], hrs)
            cache[key] = mclp_posts(grid, dem, n, cfg, router, hour + cfg.block_hours // 2,
                                    cal.is_ramadan(key[0]))
        return cache[key]

    return [static_policy("baseline_static", legacy),
            static_policy("ai_static_mclp", static),
            Policy("ai_dynamic_mclp", dynamic, dynamic=True)], cache


def run(quick: bool = False, out_results=RESULTS, out_docs=DOCS):
    t0 = time.time()
    out_results.mkdir(exist_ok=True)
    out_docs.mkdir(exist_ok=True)
    grid, router = Grid(), HaversineRouter()
    train = synth.hourly_counts(TRAIN_START, (TRAIN_END - TRAIN_START).days + 1, grid, seed=1)
    test = synth.hourly_counts(TEST_START, (TEST_END - TEST_START).days + 1, grid, seed=2)
    fc = DemandForecaster(grid, max_iter=120 if quick else 300).fit(train)
    ha = HistoricalAverage().fit(train)
    p_gb, p_ha = fc.predict(test), ha.predict(test)
    ram = test["date"].dt.date.map(cal.is_ramadan).to_numpy()
    y, rt = test["count"].to_numpy(), test["rate_true"].to_numpy()
    fmetrics = {
        "train_rows": len(train), "test_rows": len(test),
        "gradient_boosting": evaluate(y, p_gb, rt),
        "historical_average": evaluate(y, p_ha, rt),
        "true_rate_oracle": evaluate(y, rt, rt),
        "ramadan_only": {
            "gradient_boosting": evaluate(y[ram], p_gb[ram], rt[ram]),
            "historical_average": evaluate(y[ram], p_ha[ram], rt[ram]),
        },
    }
    test = test.assign(pred_gb=p_gb, pred_ha=p_ha)

    cfg = SimConfig()
    days = 4 if quick else 14
    sims, summaries, posts_out = {}, {}, {}
    for scen, start in SCENARIOS.items():
        inc = synth.incidents(start, days, grid, seed=3 if scen == "ramadan" else 4)
        policies, cache = build_policies(grid, fc, cfg, router, start, days)
        summaries[scen] = {}
        for pol in policies:
            res = simulate(inc, start, pol, grid, cfg, router)
            summaries[scen][pol.name] = summarize(res, days)
            sims[(scen, pol.name)] = res
        posts_out[scen] = {
            "baseline_static": policies[0].posts_fn(start, 0).tolist(),
            "ai_static_mclp": policies[1].posts_fn(start, 0).tolist(),
            "ai_dynamic_mclp": {f"{d} {h:02d}:00": p.tolist() for (d, h), p in sorted(cache.items())},
            "demand_forecast_by_block": {f"{d} {h:02d}:00": fc.predict_block(d, list(range(h, h + cfg.block_hours))).tolist()
                                         for (d, h) in sorted(cache)},
        }

    sens = {}
    if not quick:  # fleet-size sensitivity on the normal scenario
        start = SCENARIOS["normal"]
        inc = synth.incidents(start, 7, grid, seed=5)
        for n in (16, 20, 24, 28, 32):
            c = SimConfig(n_ambulances=n)
            pols, _ = build_policies(grid, fc, c, router, start, 7)
            sens[n] = {p.name: summarize(simulate(inc, start, p, grid, c, router), 7) for p in pols}

    meta = {"grid": [grid.n_rows, grid.n_cols], "sim_days": days, "config": cfg.__dict__,
            "runtime_s": round(time.time() - t0, 1), "router": router.name}
    (out_results / "forecast_metrics.json").write_text(json.dumps(fmetrics, indent=2))
    (out_results / "simulation_metrics.json").write_text(json.dumps({"meta": meta, "scenarios": summaries,
                                                                     "fleet_sensitivity": sens}, indent=2))
    (out_results / "posts.json").write_text(json.dumps({"centroids": grid.centroids().tolist(),
                                                        "hospitals": HOSPITALS, "scenarios": posts_out}))
    charts(test, sims, summaries, sens, grid, posts_out, out_docs)
    return fmetrics, summaries, sens


def charts(test, sims, summaries, sens, grid, posts_out, out_docs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline_static": "#9e9e9e", "ai_static_mclp": "#1f77b4", "ai_dynamic_mclp": "#d62728"}
    # 1. forecast vs actual, city total per hour around the start of Ramadan 2025
    city = test.groupby(["date", "hour"])[["count", "pred_gb", "pred_ha", "rate_true"]].sum().reset_index()
    city["ts"] = city["date"] + pd.to_timedelta(city["hour"], unit="h")
    w = city[(city["ts"] >= "2025-02-25") & (city["ts"] < "2025-03-07")]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(w["ts"], w["count"], color="#bbb", lw=1, label="actual (synthetic)")
    ax.plot(w["ts"], w["pred_ha"], color="#555", lw=1.2, ls="--", label="historical average")
    ax.plot(w["ts"], w["pred_gb"], color="#d62728", lw=1.6, label="gradient boosting")
    ax.axvline(pd.Timestamp("2025-03-01"), color="green", ls=":", lw=1.5)
    ax.text(pd.Timestamp("2025-03-01 02:00"), ax.get_ylim()[1] * 0.92, "Ramadan begins", color="green")
    ax.set_ylabel("incidents / hour (city)")
    ax.set_title("Hourly demand forecast vs actual - transition into Ramadan 2025")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_docs / "forecast_ramadan.png", dpi=130)
    plt.close(fig)

    # 2. response-time CDFs
    fig, axes = plt.subplots(1, len(summaries), figsize=(11, 3.8), sharey=True)
    for ax, scen in zip(np.atleast_1d(axes), summaries):
        for pol in summaries[scen]:
            r = np.sort(sims[(scen, pol)]["response_min"].to_numpy())
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=colors[pol], label=pol, lw=1.8)
        for x in (8, 10):
            ax.axvline(x, color="k", ls=":", lw=0.8)
        ax.set_xlim(0, 25)
        ax.set_title(f"{scen} scenario")
        ax.set_xlabel("response time (min)")
    np.atleast_1d(axes)[0].set_ylabel("share of calls")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Response-time distribution by posting policy (simulation)")
    fig.tight_layout()
    fig.savefig(out_docs / "response_cdf.png", dpi=130)
    plt.close(fig)

    # 3. bar chart of key metrics
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, label) in zip(axes, [("median_min", "median (min)"), ("p90_min", "90th percentile (min)"),
                                       ("pct_within_8", "% within 8 min")]):
        x = np.arange(len(summaries))
        pols = list(next(iter(summaries.values())))
        for i, pol in enumerate(pols):
            ax.bar(x + (i - 1) * 0.27, [summaries[s][pol][key] for s in summaries], 0.27, color=colors[pol], label=pol)
        ax.set_xticks(x, list(summaries))
        ax.set_title(label)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_docs / "policy_metrics.png", dpi=130)
    plt.close(fig)

    # 4. map: forecast demand + posts for one block
    p = posts_out["normal"]
    blk = sorted(p["ai_dynamic_mclp"])[4]
    dem = np.array(p["demand_forecast_by_block"][blk])
    cen = grid.centroids()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    sc = ax.scatter(cen[:, 1], cen[:, 0], c=dem, s=420, marker="s", cmap="YlOrRd")
    fig.colorbar(sc, ax=ax, label="forecast incidents in block")
    b = np.array(p["baseline_static"])
    d = np.array(p["ai_dynamic_mclp"][blk])
    ax.scatter(b[:, 1], b[:, 0], marker="o", facecolors="none", edgecolors="k", s=90, label="legacy posts")
    ax.scatter(d[:, 1], d[:, 0], marker="^", color="#1f4e9e", s=70, label="AI posts (this block)")
    hs = np.array(list(HOSPITALS.values()))
    ax.scatter(hs[:, 1], hs[:, 0], marker="P", color="green", s=80, label="hospitals")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Riyadh synthetic grid - block {blk}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_docs / "posts_map.png", dpi=130)
    plt.close(fig)

    if sens:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ns = sorted(sens)
        for pol in colors:
            ax.plot(ns, [sens[n][pol]["p90_min"] for n in ns], marker="o", color=colors[pol], label=pol)
        ax.set_xlabel("ambulances on duty")
        ax.set_ylabel("p90 response (min)")
        ax.set_title("Fleet-size sensitivity (normal week)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_docs / "fleet_sensitivity.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    f, s, sens = run(a.quick)
    print(json.dumps({"forecast": {k: f[k] for k in ("gradient_boosting", "historical_average", "ramadan_only")},
                      "simulation": s}, indent=2))


if __name__ == "__main__":
    main()
