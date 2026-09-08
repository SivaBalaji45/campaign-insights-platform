"""Campaign Performance Dashboard (Streamlit).

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import run_insight_query, run_weekly_workflow  # noqa: E402
from core import kpis  # noqa: E402
from core.business_rules import evaluate_rules  # noqa: E402

# --- palette (validated categorical + status tokens; see dataviz skill) -----
CHANNEL_COLORS = {
    "Search": "#2a78d6",
    "Social": "#eb6834",
    "Display": "#1baf7a",
    "Video": "#eda100",
    "Email": "#e87ba4",
    "Affiliate": "#008300",
}
STATUS_COLORS = {"critical": "#d03b3b", "warning": "#fab219", "opportunity": "#0ca30c"}
GRID_COLOR = "#e1e0d9"
MUTED = "#898781"
INK = "#52514e"

st.set_page_config(page_title="Campaign Performance Dashboard", layout="wide", page_icon="📊")


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return kpis.load_data()


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def _metric_delta(value):
    if value is None or pd.isna(value):
        return None
    return f"{value:+.1f}% WoW"


def main():
    st.title("📊 Campaign Performance Dashboard")
    st.caption(
        "Synthetic paid-media dataset · AutoGen Insight Agent grounded in the KPI + business-rule layer below · "
        "not vibe-coded numbers — every figure traces back to core/kpis.py"
    )

    with st.sidebar:
        st.header("Filters")
        if st.button("🔄 Refresh data"):
            _load_data.clear()
        df_all = _load_data()

        days = st.select_slider("Trailing window (days)", options=[7, 14, 30], value=7)
        channels = sorted(df_all["channel"].unique())
        selected_channels = st.multiselect("Channels", channels, default=channels)

        st.divider()
        st.header("Weekly Stakeholder Report")
        st.caption("Runs the real two-agent AutoGen team: Insight Agent → Reporting Agent.")
        if st.button("▶️ Generate report now", type="primary"):
            with st.spinner("Running Insight Agent → Reporting Agent…"):
                result = asyncio.run(run_weekly_workflow())
            st.session_state["last_report"] = result
            st.success(f"Report generated — status: **{result.status_flag.upper()}**")

    df = df_all[df_all["channel"].isin(selected_channels)]
    if df.empty:
        st.warning("No data for the selected channels.")
        return

    portfolio = kpis.portfolio_summary(df, days=days)
    campaign_df = kpis.campaign_summary(df, days=days)
    insights = evaluate_rules(campaign_df)
    anomalies = kpis.detect_anomalies(df)

    # --- KPI row ---
    cols = st.columns(6)
    cols[0].metric("Spend", f"${portfolio['spend']:,.0f}", _metric_delta(portfolio.get("spend_wow_pct")))
    cols[1].metric("Revenue", f"${portfolio['revenue']:,.0f}", _metric_delta(portfolio.get("revenue_wow_pct")))
    cols[2].metric("ROAS", f"{portfolio['roas']:.2f}x", _metric_delta(portfolio.get("roas_wow_pct")))
    cols[3].metric("Conversions", f"{portfolio['conversions']:,.0f}", _metric_delta(portfolio.get("conversions_wow_pct")))
    cols[4].metric("CPA", f"${portfolio['cpa']:,.2f}", _metric_delta(portfolio.get("cpa_wow_pct")), delta_color="inverse")
    cols[5].metric("CTR", f"{portfolio['ctr']*100:.2f}%", _metric_delta(portfolio.get("ctr_wow_pct")))

    st.divider()

    # --- trend charts (single axis each, per dataviz guardrail) ---
    trend_days = max(days * 4, 28)
    daily = kpis.rolling_portfolio(df, window=7).tail(trend_days)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["spend"], name="Spend", line=dict(color="#2a78d6", width=2)))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"], name="Revenue", line=dict(color="#1baf7a", width=2)))
        _base_layout(fig, "Spend vs. Revenue (daily)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["roas"], name="ROAS (daily)",
            line=dict(color="#2a78d6", width=1), opacity=0.35,
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["roas_roll7"], name="ROAS (7d avg)",
            line=dict(color="#2a78d6", width=3),
        ))
        fig.add_hline(y=2.0, line_dash="dot", line_color=MUTED, annotation_text="healthy guardrail (2.0x)")
        _base_layout(fig, "Portfolio ROAS (daily + 7d trend)")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        by_channel = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=days - 1))].groupby("channel", as_index=False)["spend"].sum()
        fig = go.Figure(go.Bar(
            x=by_channel["channel"], y=by_channel["spend"],
            marker_color=[CHANNEL_COLORS.get(c, MUTED) for c in by_channel["channel"]],
        ))
        _base_layout(fig, f"Spend by Channel (trailing {days}d)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        roas_by_channel = campaign_df.groupby("channel").apply(
            lambda g: g["revenue"].sum() / g["spend"].sum() if g["spend"].sum() else 0
        ).reset_index(name="roas")
        fig = go.Figure(go.Bar(
            x=roas_by_channel["channel"], y=roas_by_channel["roas"],
            marker_color=[CHANNEL_COLORS.get(c, MUTED) for c in roas_by_channel["channel"]],
        ))
        fig.add_hline(y=2.0, line_dash="dot", line_color=MUTED)
        _base_layout(fig, f"ROAS by Channel (trailing {days}d)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- actionable insights ---
    st.subheader("🎯 Actionable Insights (business-rule engine)")
    if not insights:
        st.success("No business-rule thresholds are currently breached. ✅")
    else:
        severity_order = {"critical": 0, "warning": 1, "opportunity": 2}
        for ins in sorted(insights, key=lambda i: severity_order[i.severity]):
            label = f"**{ins.campaign_name}** ({ins.channel}) — {ins.title}: {ins.recommended_action}"
            if ins.severity == "critical":
                st.error(label, icon="🔴")
            elif ins.severity == "warning":
                st.warning(label, icon="🟡")
            else:
                st.success(label, icon="🟢")

    # --- anomalies ---
    with st.expander(f"📈 Detected anomalies ({len(anomalies)})", expanded=False):
        if anomalies:
            adf = pd.DataFrame([a.__dict__ for a in anomalies[:25]])
            st.dataframe(adf, use_container_width=True, hide_index=True)
        else:
            st.caption("No statistical anomalies detected in the current window.")

    # --- campaign table ---
    st.subheader("Campaign Performance")
    display_cols = [
        "campaign_name", "channel", "objective", "spend", "revenue", "roas",
        "cpa", "ctr", "cvr", "budget_pacing_ratio", "roas_wow_pct", "revenue_wow_pct",
    ]
    st.dataframe(
        campaign_df[display_cols].style.format({
            "spend": "${:,.0f}", "revenue": "${:,.0f}", "roas": "{:.2f}x", "cpa": "${:,.2f}",
            "ctr": "{:.2%}", "cvr": "{:.2%}", "budget_pacing_ratio": "{:.0%}",
            "roas_wow_pct": "{:+.1f}%", "revenue_wow_pct": "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    # --- report preview, if one was just generated ---
    if "last_report" in st.session_state:
        st.divider()
        st.subheader("📨 Latest Weekly Stakeholder Report")
        result = st.session_state["last_report"]
        html_content = Path(result.html_path).read_text()
        st.components.v1.html(html_content, height=650, scrolling=True)
        colA, colB = st.columns(2)
        colA.download_button("Download HTML", html_content, file_name=Path(result.html_path).name)
        colB.download_button("Download Markdown", Path(result.markdown_path).read_text(), file_name=Path(result.markdown_path).name)

    st.divider()

    # --- ask the insight agent ---
    st.subheader("💬 Ask the Insight Agent")
    st.caption("Real AutoGen agent — calls live tools over this same data, then answers with grounded numbers + a predicted outlook.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, content in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("e.g. Which campaigns should we pause this week, and what's the outlook for next week?")
    if question:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Insight Agent is analyzing…"):
                answer = asyncio.run(run_insight_query(question))
            st.markdown(answer)
        st.session_state["chat_history"].append(("assistant", answer))


if __name__ == "__main__":
    main()
