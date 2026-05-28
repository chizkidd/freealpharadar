"""The Deep Dive view for a single company.

Shows the financial trajectory, an SEC risk-factor excerpt with FinBERT
sentiment highlighting, the patent-filing timeline and top assignees, the GDELT
news feed with tone bars, and a waterfall chart decomposing the final score.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from freealpharadar.ml.finbert import FinBERTSentiment
from freealpharadar.pipeline import CompanyData
from freealpharadar.scoring import ScoreResult


def render_deep_dive(
    company: Optional[CompanyData],
    result: Optional[ScoreResult],
    analyzer: Optional[FinBERTSentiment] = None,
) -> None:
    """Render the deep-dive view for one company.

    Args:
        company: The enriched company bundle.
        result: The company's score result.
        analyzer: Shared FinBERT analyser for span highlighting.
    """
    if company is None or result is None:
        st.info("Select a company on the Radar Screen to open its deep dive.")
        return

    st.header(f"🔬 {result.name} ({result.ticker})")
    _render_headline_metrics(result)

    if company.warnings:
        for w in company.warnings:
            st.warning(w, icon="⚠️")

    tabs = st.tabs(
        ["📈 Financials", "📜 SEC Risk", "💡 Patents", "📰 News", "🧮 Score breakdown"]
    )
    with tabs[0]:
        _render_financials(company)
    with tabs[1]:
        _render_sec_risk(company, analyzer)
    with tabs[2]:
        _render_patents(company)
    with tabs[3]:
        _render_news(company)
    with tabs[4]:
        _render_waterfall(result)


def _render_headline_metrics(result: ScoreResult) -> None:
    """Top-line metric row."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{result.score:.1f}")
    cap = result.market_cap or 0
    c2.metric("Market cap", f"${cap / 1e9:.2f}B" if cap else "n/a")
    c3.metric("Sector", result.sector or "Unknown")
    c4.metric("Cluster", str(result.cluster) if result.cluster is not None else "n/a")


def _render_financials(company: CompanyData) -> None:
    """Revenue / gross margin / net income trajectory."""
    income = company.yfinance.get("income_statement", []) or []
    if not income:
        st.info("No income-statement data available.")
        return
    df = pd.DataFrame(income)
    rev_col = _first_col(df, ["Total Revenue", "TotalRevenue", "Revenues"])
    gp_col = _first_col(df, ["Gross Profit", "GrossProfit"])
    ni_col = _first_col(df, ["Net Income", "NetIncome"])
    if "period" in df.columns:
        df = df.sort_values("period")

    if rev_col:
        st.plotly_chart(
            px.bar(df, x="period", y=rev_col, title="Revenue by period"),
            use_container_width=True,
        )
    if rev_col and gp_col:
        df["gross_margin"] = df[gp_col] / df[rev_col]
        st.plotly_chart(
            px.line(
                df,
                x="period",
                y="gross_margin",
                markers=True,
                title="Gross margin trajectory",
            ),
            use_container_width=True,
        )
    if ni_col:
        st.plotly_chart(
            px.bar(df, x="period", y=ni_col, title="Net income by period"),
            use_container_width=True,
        )

    history = company.yfinance.get("history", []) or []
    if history:
        hdf = pd.DataFrame(history)
        st.plotly_chart(
            px.line(hdf, x="date", y="close", title="Price history (5y, monthly)"),
            use_container_width=True,
        )


def _render_sec_risk(
    company: CompanyData, analyzer: Optional[FinBERTSentiment]
) -> None:
    """SEC risk-factor excerpt with sentence-level sentiment highlighting."""
    sections = company.sec.get("sections", {})
    risk = sections.get("risk_factors", "")
    if not risk:
        st.info("No SEC risk-factor text available (try Refresh, or check ticker).")
        return

    derived = company.derived or {}
    sent = derived.get("finbert_sentiment")
    backend = derived.get("finbert_backend", "lexicon")
    if sent is not None:
        st.caption(
            f"Overall risk-section sentiment: **{sent:+.2f}** "
            f"(backend: {backend}). Controversy score: "
            f"**{derived.get('controversy_score')}**"
        )

    analyzer = analyzer or FinBERTSentiment()
    sentences = [s.strip() for s in risk.split(". ") if s.strip()][:40]
    html_parts: List[str] = []
    for sentence in sentences:
        s = analyzer.analyze(sentence)
        color = {
            "positive": "#1b5e20",
            "negative": "#b71c1c",
            "neutral": "#37474f",
        }.get(s.label, "#37474f")
        bg = {
            "positive": "#e8f5e9",
            "negative": "#ffebee",
            "neutral": "#eceff1",
        }.get(s.label, "#eceff1")
        html_parts.append(
            f'<span style="background:{bg};color:{color};padding:1px 3px;'
            f'border-radius:3px;">{sentence}.</span>'
        )
    st.markdown(
        '<div style="line-height:2.0">' + " ".join(html_parts) + "</div>",
        unsafe_allow_html=True,
    )

    flags = company.sec.get("flags", {})
    if flags:
        st.subheader("Parsed qualitative flags")
        st.json(flags)


def _render_patents(company: CompanyData) -> None:
    """Patent timeline and sample titles."""
    patents = company.patents or {}
    counts = patents.get("counts_by_year", [])
    if not counts:
        st.info("No patent data available from PatentsView.")
        return
    cdf = pd.DataFrame(counts)
    st.plotly_chart(
        px.bar(cdf, x="year", y="count", title="Granted patents by year"),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Total patents", patents.get("total_patents", 0))
    growth = patents.get("patent_growth_rate")
    c2.metric("Patent growth rate", f"{growth:+.1%}" if growth is not None else "n/a")
    titles = patents.get("sample_titles", [])
    if titles:
        st.subheader("Sample patent titles")
        for t in titles[:15]:
            st.markdown(f"- {t}")


def _render_news(company: CompanyData) -> None:
    """GDELT news feed with per-article tone bars."""
    gdelt = company.gdelt or {}
    articles = gdelt.get("articles", [])
    if not articles:
        st.info("No recent GDELT news available.")
        return
    avg = gdelt.get("avg_tone")
    st.metric("Average news tone", f"{avg:+.2f}" if avg is not None else "n/a")

    adf = pd.DataFrame(articles)
    if "tone" in adf.columns:
        tdf = adf.dropna(subset=["tone"]).head(25)
        if not tdf.empty:
            fig = px.bar(
                tdf,
                x="tone",
                y=tdf.get("title", tdf.index).astype(str).str.slice(0, 60),
                orientation="h",
                color="tone",
                color_continuous_scale="RdYlGn",
                title="Article tone (red = negative, green = positive)",
            )
            fig.update_layout(height=600, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Headlines")
    for art in articles[:20]:
        tone = art.get("tone")
        tone_str = f" ({tone:+.1f})" if tone is not None else ""
        url = art.get("url", "#")
        title = art.get("title", "(untitled)")
        st.markdown(f"- [{title}]({url}){tone_str} — *{art.get('domain', '')}*")


def _render_waterfall(result: ScoreResult) -> None:
    """Waterfall chart of each factor's contribution to the composite."""
    contribs = [c for c in result.contributions if abs(c.contribution) > 1e-9]
    if not contribs:
        st.info("No non-zero factor contributions to display.")
        return
    contribs = sorted(contribs, key=lambda c: c.contribution, reverse=True)

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(contribs),
            x=[c.label for c in contribs],
            y=[c.contribution for c in contribs],
            connector={"line": {"color": "rgb(150,150,150)"}},
            increasing={"marker": {"color": "#2e7d32"}},
            decreasing={"marker": {"color": "#c62828"}},
        )
    )
    fig.update_layout(
        title="Factor contributions to composite (z-units × weight)",
        height=520,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factor detail")
    rows = [
        {
            "Factor": c.label,
            "Group": c.group,
            "Raw value": c.raw,
            "Z-score": round(c.zscore, 3),
            "Weight": c.weight,
            "Contribution": round(c.contribution, 4),
        }
        for c in result.contributions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column present in ``df``."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
