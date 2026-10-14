"""Streamlit dashboard:  streamlit run app.py   (run `python -m ems_dispatch_ai.experiment` first)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ems_dispatch_ai.triage import TriageAssistant

RES = Path(__file__).resolve().parent / "results"
st.set_page_config(page_title="EMS Dispatch AI - Riyadh (synthetic)", layout="wide")
st.title("EMS Dispatch AI - Riyadh (synthetic data)")
st.caption("Research prototype on synthetic data. Not an operational dispatch system and not clinical advice.")

if not (RES / "simulation_metrics.json").exists():
    st.warning("No results yet. Run `python -m ems_dispatch_ai.experiment` first.")
    st.stop()

sim = json.loads((RES / "simulation_metrics.json").read_text())
fc = json.loads((RES / "forecast_metrics.json").read_text())
posts = json.loads((RES / "posts.json").read_text())

scen = st.sidebar.selectbox("Scenario", list(sim["scenarios"]))
policy = st.sidebar.radio("Posting policy", ["baseline_static", "ai_static_mclp", "ai_dynamic_mclp"], index=2)
blocks = sorted(posts["scenarios"][scen]["ai_dynamic_mclp"])
block = st.sidebar.select_slider("Time block", blocks, value=blocks[min(4, len(blocks) - 1)])

tab_map, tab_metrics, tab_forecast, tab_triage = st.tabs(["Map", "Policy comparison", "Forecast", "Triage assistant"])

with tab_map:
    cen = np.array(posts["centroids"])
    dem = np.array(posts["scenarios"][scen]["demand_forecast_by_block"][block])
    p = posts["scenarios"][scen][policy]
    pts = np.array(p[block] if isinstance(p, dict) else p)
    demand_df = pd.DataFrame({"lat": cen[:, 0], "lon": cen[:, 1], "demand": dem})
    posts_df = pd.DataFrame({"lat": pts[:, 0], "lon": pts[:, 1]})
    hosp_df = pd.DataFrame([{"name": k, "lat": v[0], "lon": v[1]} for k, v in posts["hospitals"].items()])
    try:
        import pydeck as pdk

        layers = [
            pdk.Layer("HeatmapLayer", demand_df, get_position="[lon, lat]", get_weight="demand", radius_pixels=60),
            pdk.Layer(
                "ScatterplotLayer",
                posts_df,
                get_position="[lon, lat]",
                get_fill_color=[30, 80, 200],
                get_radius=600,
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                hosp_df,
                get_position="[lon, lat]",
                get_fill_color=[0, 150, 60],
                get_radius=500,
                pickable=True,
            ),
        ]
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(latitude=24.69, longitude=46.72, zoom=10),
                tooltip={"text": "{name}"},
                map_style=None,
            )
        )
    except ImportError:
        st.map(posts_df)
    st.caption(f"Heat = forecast incidents in block {block}; blue = ambulance posts ({policy}); green = hospitals.")

with tab_metrics:
    df = pd.DataFrame(sim["scenarios"][scen]).T
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.bar_chart(df[["median_min", "p90_min"]])
    c2.bar_chart(df[["pct_within_8", "pct_within_10"]])
    if sim.get("fleet_sensitivity"):
        st.subheader("Fleet-size sensitivity (p90, minutes)")
        st.line_chart(
            pd.DataFrame(
                {int(n): {k: v["p90_min"] for k, v in d.items()} for n, d in sim["fleet_sensitivity"].items()}
            ).T
        )

with tab_forecast:
    st.json({k: fc[k] for k in ("gradient_boosting", "historical_average", "ramadan_only")})
    img = Path(__file__).resolve().parent / "docs" / "forecast_ramadan.png"
    if img.exists():
        st.image(str(img))

with tab_triage:
    st.write(
        "Paste a call note in Arabic or English. Default backend is a transparent rule-based mock; "
        "set TRIAGE_LLM_BASE_URL / TRIAGE_LLM_MODEL to use an OpenAI-compatible model."
    )
    note = st.text_area("Call note", "رجل عمره ٦٠ سنة عنده ألم في الصدر وتعرق")
    if st.button("Triage"):
        st.json(TriageAssistant().triage(note).to_dict())
