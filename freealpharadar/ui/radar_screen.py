"""The "Radar Screen" landing view.

A Plotly scatter of final Score vs. Market Cap, coloured by sector, with hover
tooltips summarising each company. Dynamic filters for sector, market-cap range
and score threshold sit above the chart. A ranked table sits below.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from freealpharadar.scoring import ScoreResult


def _results_to_df(results: List[ScoreResult]) -> pd.DataFrame:
    """Flatten score results into a DataFrame for plotting/filtering."""
    rows = []
    for r in results:
        moat = next(
            (c for c in r.contributions if c.group == "Disruption & Moat"), None
        )
        moat_desc = f"{moat.label}: z={moat.zscore:+.2f}" if moat else "n/a"
        rows.append(
            {
                "ticker": r.ticker,
                "name": r.name,
                "sector": r.sector or "Unknown",
                "score": r.score,
                "market_cap": r.market_cap or 0.0,
                "cluster": r.cluster if r.cluster is not None else -1,
                "moat": moat_desc,
            }
        )
    return pd.DataFrame(rows)


def render_radar_screen(results: List[ScoreResult]) -> Optional[str]:
    """Render the radar screen and return the ticker the user drilled into.

    Args:
        results: Scored universe.

    Returns:
        The ticker chosen for a deep dive, or ``None``.
    """
    st.header("🛰️ Radar Screen")
    if not results:
        st.info(
            "No results yet. Set a universe in the sidebar and click "
            "**Refresh Data & Re-score**."
        )
        return None

    df = _results_to_df(results)

    # ---- Filters ----------------------------------------------------- #
    c1, c2, c3 = st.columns(3)
    sectors = sorted(df["sector"].unique().tolist())
    with c1:
        chosen_sectors = st.multiselect("Sector", sectors, default=sectors)
    with c2:
        min_score = st.slider("Minimum score", 0.0, 100.0, 0.0, 1.0)
    with c3:
        caps = df["market_cap"].replace(0, pd.NA).dropna()
        cap_max = float(caps.max()) if not caps.empty else 1.0
        max_cap_b = st.slider(
            "Max market cap ($B)",
            0.0,
            max(1.0, cap_max / 1e9),
            max(1.0, cap_max / 1e9),
            step=0.5,
        )

    mask = (
        df["sector"].isin(chosen_sectors)
        & (df["score"] >= min_score)
        & (df["market_cap"] <= max_cap_b * 1e9)
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No companies match the current filters.")
        return None

    # ---- Scatter ----------------------------------------------------- #
    fdf["market_cap_plot"] = fdf["market_cap"].replace(
        0, fdf["market_cap"].median() or 1
    )
    fig = px.scatter(
        fdf,
        x="market_cap_plot",
        y="score",
        color="sector",
        size=[14] * len(fdf),
        hover_name="name",
        hover_data={
            "ticker": True,
            "score": ":.1f",
            "moat": True,
            "market_cap_plot": False,
            "sector": False,
        },
        log_x=True,
        labels={
            "market_cap_plot": "Market Cap (log $)",
            "score": "FreeAlphaRadar Score",
        },
        title="Score vs. Market Cap — under-the-radar names sit top-left",
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(height=520, legend_title_text="Sector")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Ranked table ------------------------------------------------ #
    st.subheader("Ranked companies")
    table = fdf[["ticker", "name", "sector", "score", "market_cap"]].copy()
    table["market_cap ($B)"] = (table["market_cap"] / 1e9).round(2)
    table = table.drop(columns=["market_cap"]).sort_values("score", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ---- Deep-dive selector ----------------------------------------- #
    choice = st.selectbox(
        "🔬 Open deep dive for:",
        options=["—"] + fdf.sort_values("score", ascending=False)["ticker"].tolist(),
    )
    return None if choice == "—" else choice
