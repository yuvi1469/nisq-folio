"""NISQ-Folio results dashboard (read-only).

Usage::

    uv run streamlit run dashboard/app.py -- --config configs/main.yaml

(or set the environment variable NISQ_FOLIO_CONFIG instead of passing ``--config``).

Shows the grid's progress, the noiseless results (ADR-026), the noisy analysis (ADR-028) for
every complete cell, and a per-instance explorer. It only reads result files, so it is safe to
open while an experiment is running; use the "Reload results" button to pick up new tasks.
Numbers come from the same code as the reports, with fewer bootstrap replicates (500) for
speed, so confidence intervals can differ slightly from the reports of record.
"""

from __future__ import annotations

import argparse
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

from nisq_folio.analysis.dashboard_data import DashboardData, load_dashboard
from nisq_folio.analysis.summary import cell_table, mixer_table
from nisq_folio.config import load_config

CONFIG_ENV = "NISQ_FOLIO_CONFIG"  # alternative to --config (used by the tests)
LEVELS = [0, 0.25, 0.5, 1]
MIXERS = {"adapt": "ADAPT", "xy_ring": "XY ring", "x": "X"}
COLOURS = alt.Scale(domain=list(MIXERS.values()), range=["#2a78d6", "#eb6834", "#1baf7a"])
SHAPES = alt.Scale(domain=list(MIXERS.values()), range=["circle", "square", "triangle-up"])


def _config_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get(CONFIG_ENV, "configs/main.yaml"))
    return parser.parse_known_args(sys.argv[1:])[0].config


@st.cache_data(show_spinner="Reading results...", ttl=300)
def _load(path: str, stamp: int) -> DashboardData:
    """Cached load; ``stamp`` changes when the user asks for a reload."""
    del stamp
    return load_dashboard(load_config(path))


def _named(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Mixer=df["mixer"].map(MIXERS))


def _reference(y: float = 1.0) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"y": [y]}))
        .mark_rule(strokeDash=[4, 4], color="#8a8985")
        .encode(y="y:Q")
    )


def progress_tab(data: DashboardData) -> None:
    """Grid progress by N, mixer and noise level."""
    p = data.progress
    q = p[p["kind"] == "qaoa"]
    done, total = int(p["done"].sum()), int(p["total"].sum())
    noisy = q[q["noise_level"] > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks finished", f"{done:,} / {total:,}", f"{100 * done / total:.0f}%")
    c2.metric("Noisy QAOA tasks", f"{int(noisy['done'].sum()):,} / {int(noisy['total'].sum()):,}")
    c3.metric("Complete noisy sizes", ", ".join(_complete_sizes(data)) or "none yet")
    table = noisy.pivot_table(
        index=["n", "mixer"], columns="noise_level", values="share", aggfunc="first"
    )
    st.dataframe(table.style.format("{:.0%}"), use_container_width=True)


def _complete_sizes(data: DashboardData) -> list[str]:
    n1 = data.noisy.get("N1", pd.DataFrame())
    return [f"N={n}" for n in sorted(n1["n"].unique())] if not n1.empty else []


def noiseless_tab(data: DashboardData) -> None:
    """ADR-026 noiseless amplification and mixer comparisons."""
    den = st.radio("Covariance", ["none", "rmt_clip"], horizontal=True, key="den0")
    q = data.qaoa[(data.qaoa["noise_level"] == 0) & (data.qaoa["denoising"] == den)]
    cells = _named(cell_table(q))
    chart = alt.Chart(cells).encode(
        x=alt.X("N:O", title="Number of assets N"),
        color=alt.Color("Mixer:N", scale=COLOURS),
        tooltip=[
            "Mixer",
            "N",
            alt.Tooltip("A_median:Q", format=".2f"),
            alt.Tooltip("share_A_gt_1:Q", format=".0%"),
        ],
    )
    band = chart.mark_rule(strokeWidth=2).encode(
        y=alt.Y("A_q25:Q", scale=alt.Scale(type="log"), title="Amplification A (median, IQR)"),
        y2="A_q75:Q",
    )
    points = chart.mark_point(size=80, filled=True).encode(
        y="A_median:Q", shape=alt.Shape("Mixer:N", scale=SHAPES)
    )
    st.altair_chart(band + chart.mark_line().encode(y="A_median:Q") + points + _reference(),
                    use_container_width=True)  # fmt: skip
    st.caption("Dashed line: random feasible guessing (A = 1). Hover a point for its values.")
    st.dataframe(cells.drop(columns=["mixer"]), use_container_width=True, hide_index=True)
    st.subheader("Paired mixer comparisons")
    st.dataframe(mixer_table(q.assign(denoising="none")), use_container_width=True,
                 hide_index=True)  # fmt: skip


def noise_tab(data: DashboardData) -> None:
    """ADR-028 noisy analysis for every complete cell."""
    if not data.noisy or data.noisy["N1"].empty:
        st.info("No cell is complete at every noise level yet.")
        return
    den = st.radio("Covariance", ["none", "rmt_clip"], horizontal=True, key="den1")
    n1 = _named(data.noisy["N1"])
    n1 = n1[n1["denoising"] == den]
    metric = st.radio("Show", ["Amplification A", "Feasible probability"], horizontal=True)
    for n in sorted(n1["n"].unique()):
        c = n1[n1["n"] == n]
        base = alt.Chart(c).encode(
            x=alt.X(
                "noise_level:Q",
                title="Noise scale m (1 = FakeKingston medians)",
                axis=alt.Axis(values=LEVELS, format="g"),
            ),
            color=alt.Color("Mixer:N", scale=COLOURS),
            tooltip=[
                "Mixer",
                "noise_level",
                alt.Tooltip("A_median:Q", format=".2f"),
                alt.Tooltip("A_ci_low:Q", format=".2f"),
                alt.Tooltip("A_ci_high:Q", format=".2f"),
                alt.Tooltip("p_feasible_median:Q", format=".2f"),
            ],
        )
        if metric == "Amplification A":
            log = alt.Scale(type="log")
            y = alt.Y("A_median:Q", scale=log, title="A (median, 95% CI)")
            band = base.mark_area(opacity=0.15).encode(
                y=alt.Y("A_ci_low:Q", scale=log), y2="A_ci_high:Q"
            )
            layers = [band, base.mark_line().encode(y=y), _reference()]
        else:
            y = alt.Y("p_feasible_median:Q", title="Feasible probability (median)")
            layers = [base.mark_line().encode(y=y)]
        points = base.mark_point(size=70, filled=True).encode(
            y=y, shape=alt.Shape("Mixer:N", scale=SHAPES)
        )
        st.markdown(f"**N = {n}**")
        st.altair_chart(alt.layer(*layers, points), use_container_width=True)
    st.subheader("Crossover m* (median A falls to 1)")
    n2 = data.noisy["N2"]
    st.dataframe(_named(n2[n2["denoising"] == den]), use_container_width=True, hide_index=True)
    st.subheader("Verdicts (median A > 1 and A > 1 on at least two thirds of instances)")
    v = data.noisy["V"]
    st.dataframe(v[v["denoising"] == den], use_container_width=True, hide_index=True)
    st.subheader("Crossover comparisons: bootstrap P(first > second)")
    st.dataframe(data.noisy["N3"], use_container_width=True, hide_index=True)


def instance_tab(data: DashboardData) -> None:
    """One instance's amplification across noise levels and mixers."""
    q = data.qaoa
    c1, c2, c3 = st.columns(3)
    n = c1.selectbox("N", sorted(q["n"].unique()))
    den = c2.selectbox("Covariance", ["none", "rmt_clip"])
    instance = c3.selectbox("Instance", sorted(q[q["n"] == n]["instance"].unique()))
    rows = _named(q[(q["n"] == n) & (q["denoising"] == den) & (q["instance"] == instance)])
    cols = ["Mixer", "noise_level", "A", "p_optimal", "p_feasible", "expected_gap",
            "mode_optimal", "total_shots"]  # fmt: skip
    st.dataframe(rows[cols].sort_values(["Mixer", "noise_level"]), use_container_width=True,
                 hide_index=True)  # fmt: skip
    c = data.classical
    st.dataframe(c[(c["n"] == n) & (c["denoising"] == den) & (c["instance"] == instance)],
                 use_container_width=True, hide_index=True)  # fmt: skip


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="NISQ-Folio results", layout="wide")
    path = _config_path()
    st.title("NISQ-Folio results")
    st.caption(f"Config: `{path}`. Read-only; pre-registered analyses ADR-026 and ADR-028.")
    if "stamp" not in st.session_state:
        st.session_state.stamp = 0
    if st.button("Reload results"):
        st.session_state.stamp += 1
    data = _load(path, st.session_state.stamp)
    if data.qaoa.empty:
        st.warning("No finished QAOA tasks yet.")
        return
    tabs = st.tabs(["Progress", "Noiseless", "Noise", "Instance explorer"])
    with tabs[0]:
        progress_tab(data)
    with tabs[1]:
        noiseless_tab(data)
    with tabs[2]:
        noise_tab(data)
    with tabs[3]:
        instance_tab(data)


main()
