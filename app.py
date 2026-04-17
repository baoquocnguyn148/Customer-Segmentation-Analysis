"""
app.py ── Superstore Customer Intelligence Dashboard
=====================================================
A professional-grade Streamlit application featuring:
  Tab 1 ─ 🎯 Customer Segmentation  (RFM + K-Means)
  Tab 2 ─ 📈 Sales & Profit Forecasting  (Holt-Winters)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from rfm_clustering import (
    load_raw_data,
    compute_rfm,
    run_kmeans,
    cluster_summary,
    compute_elbow,
    load_monthly_data,
    forecast_holt_winters,
    CLUSTER_PALETTE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore · Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Modern Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Import Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0D1117 0%, #0D1B2A 60%, #0A1628 100%);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161B22 0%, #090E1A 100%);
    border-right: 1px solid #21262D;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #161B22 0%, #1A2332 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(79,195,247,0.15);
  }
  [data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    color: #8B949E;
    padding: 10px 20px;
    border-radius: 6px 6px 0 0;
    transition: color 0.2s ease;
  }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #4FC3F7 !important;
    border-bottom-color: #4FC3F7 !important;
  }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8B949E;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262D;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* ── Cluster badge ── */
  .cluster-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
  }

  /* ── Divider ── */
  hr {
    border-color: #21262D !important;
    margin: 24px 0;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 8px;
    overflow: hidden;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared plotly layout
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(color="#E6EDF3", family="Inter"),
    xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    hoverlabel=dict(bgcolor="#161B22", bordercolor="#4FC3F7", font_size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

# Default legend style — spread per chart to avoid duplicate-kwarg errors
LEGEND_DEFAULT = dict(
    bgcolor="rgba(22,27,34,0.8)", bordercolor="#21262D", borderwidth=1, font_size=12
)
LEGEND_INLINE  = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    font_size=11, bgcolor="rgba(0,0,0,0)", borderwidth=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Loading raw transaction data…")
def get_raw():
    return load_raw_data()


@st.cache_data(show_spinner="📐 Computing RFM features…")
def get_rfm(_df):
    return compute_rfm(_df)


@st.cache_data(show_spinner="🤖 Training K-Means model…")
def get_clusters(_rfm, n_clusters):
    return run_kmeans(_rfm, n_clusters)


@st.cache_data(show_spinner="📊 Computing elbow curve…")
def get_elbow(_rfm):
    scaler_tmp = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    X = scaler_tmp.fit_transform(_rfm[["Recency", "Frequency", "Monetary"]])
    return compute_elbow(X, k_range=range(2, 11))


@st.cache_data(show_spinner="📈 Loading monthly data…")
def get_monthly():
    return load_monthly_data()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
      <div style='font-size:2.5rem;'>🛒</div>
      <div style='font-size:1.1rem; font-weight:700; color:#E6EDF3; margin-top:8px;'>
        Customer Intelligence
      </div>
      <div style='font-size:0.75rem; color:#8B949E; margin-top:4px;'>
        Superstore Analytics · 2014–2017
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Segmentation Settings")
    n_clusters = st.slider(
        "Number of Customer Clusters (K)",
        min_value=3, max_value=7, value=4, step=1,
        help="Use the Elbow Curve in the Segmentation tab to find the optimal K.",
    )

    st.markdown("---")
    st.markdown("### 📈 Forecasting Settings")
    forecast_metric = st.selectbox(
        "Metric to Forecast",
        ["total_sales", "total_profit"],
        format_func=lambda x: "💰 Total Sales" if x == "total_sales" else "📊 Total Profit",
    )
    forecast_periods = st.slider("Forecast Horizon (months)", 1, 12, 6)

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem; color:#484F58; text-align:center; line-height:1.7;'>
      Built with Streamlit · scikit-learn · Plotly<br>
      Superstore Dataset · 2014–2017<br>
      <span style='color:#4FC3F7;'>by DA Portfolio Project</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 8px;'>
  <h1 style='font-size:2rem; font-weight:800; color:#E6EDF3; margin:0; letter-spacing:-0.02em;'>
    🛒 Superstore Customer Intelligence
  </h1>
  <p style='color:#8B949E; margin-top:8px; font-size:0.95rem;'>
    RFM Segmentation &nbsp;·&nbsp; K-Means Clustering &nbsp;·&nbsp; Sales Forecasting &nbsp;·&nbsp; 2014–2017 Retail Data
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
df_raw     = get_raw()
df_rfm     = get_rfm(df_raw)
df_seg, X_scaled, scaler, kmeans_model = get_clusters(df_rfm, n_clusters)
df_summary = cluster_summary(df_seg)
df_monthly = get_monthly()

# ─────────────────────────────────────────────────────────────────────────────
# Global KPIs
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Customers",    f"{df_seg['Customer ID'].nunique():,}")
c2.metric("🧾 Total Orders",       f"{df_raw['Order ID'].nunique():,}")
c3.metric("💰 Total Revenue",      f"${df_raw['Sales'].sum():,.0f}")
c4.metric("📊 Total Profit",       f"${df_raw['Profit'].sum():,.0f}")
c5.metric("🏷️ Avg Profit Margin",  f"{(df_raw['Profit'].sum() / df_raw['Sales'].sum()) * 100:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Customer Segmentation", "📈  Sales & Profit Forecasting"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Segmentation
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Cluster KPIs ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cluster Overview</div>', unsafe_allow_html=True)
    cols = st.columns(n_clusters)
    for i, row in df_summary.iterrows():
        color = CLUSTER_PALETTE[int(row["Cluster"]) % len(CLUSTER_PALETTE)]
        with cols[i]:
            st.markdown(f"""
            <div style='
              background:linear-gradient(135deg,#161B22,#1A2332);
              border:1px solid {color}44;
              border-left: 3px solid {color};
              border-radius:10px;
              padding:14px 16px;
              box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            '>
              <div style='font-size:0.85rem; font-weight:700; color:{color}; margin-bottom:6px;'>
                {row["Cluster_Name"]}
              </div>
              <div style='font-size:1.4rem; font-weight:800; color:#E6EDF3;'>{row["Customers"]:,}</div>
              <div style='font-size:0.72rem; color:#484F58; margin-top:4px;'>customers</div>
              <hr style='margin:8px 0; border-color:#21262D;'>
              <div style='display:flex; justify-content:space-between; font-size:0.78rem; color:#8B949E;'>
                <span>Avg $: <b style='color:#E6EDF3;'>${row["Avg_Monetary"]:,}</b></span>
                <span>F: <b style='color:#E6EDF3;'>{row["Avg_Frequency"]:.1f}x</b></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: 3D Scatter + Elbow Curve ──────────────────────────────────────
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown('<div class="section-header">3D RFM Scatter Plot</div>', unsafe_allow_html=True)
        fig_3d = px.scatter_3d(
            df_seg,
            x="Recency", y="Frequency", z="Monetary",
            color="Cluster_Name",
            color_discrete_sequence=CLUSTER_PALETTE[:n_clusters],
            hover_data={"Customer_Name": True, "R_score": True, "F_score": True, "M_score": True, "Cluster_Name": False},
            labels={"Cluster_Name": "Segment"},
            opacity=0.82,
            height=520,
        )
        fig_3d.update_traces(marker_size=4)
        base_kv = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
        fig_3d.update_layout(
            **base_kv,
            title="",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_3d.update_scenes(
            xaxis=dict(title="Recency (days)", gridcolor="#21262D", backgroundcolor="rgba(0,0,0,0)", color="#8B949E"),
            yaxis=dict(title="Frequency (orders)", gridcolor="#21262D", backgroundcolor="rgba(0,0,0,0)", color="#8B949E"),
            zaxis=dict(title="Monetary ($)", gridcolor="#21262D", backgroundcolor="rgba(0,0,0,0)", color="#8B949E"),
            bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_3d, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Elbow Curve — Optimal K</div>', unsafe_allow_html=True)
        elbow_data = get_elbow(df_rfm)
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(
                x=elbow_data["k"], y=elbow_data["inertia"],
                name="Inertia (WCSS)", mode="lines+markers",
                line=dict(color="#F06292", width=2.5),
                marker=dict(size=6, color="#F06292"),
            ), secondary_y=False,
        )
        fig_elbow.add_trace(
            go.Scatter(
                x=elbow_data["k"], y=elbow_data["silhouette"],
                name="Silhouette Score", mode="lines+markers",
                line=dict(color="#4FC3F7", width=2.5, dash="dot"),
                marker=dict(size=6, color="#4FC3F7"),
            ), secondary_y=True,
        )
        fig_elbow.add_vline(
            x=n_clusters, line_width=1.5, line_dash="dash",
            line_color="#FFB74D", opacity=0.7,
            annotation_text=f" K={n_clusters} selected",
            annotation_font_color="#FFB74D",
        )
        base_kv_e = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
        fig_elbow.update_layout(**base_kv_e, title="", height=240, margin=dict(l=0, r=0, t=10, b=0))
        fig_elbow.update_layout(legend=LEGEND_INLINE)
        fig_elbow.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False, gridcolor="#21262D", color="#8B949E")
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True, gridcolor="#21262D", color="#8B949E")
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", gridcolor="#21262D", color="#8B949E")
        st.plotly_chart(fig_elbow, use_container_width=True)

        # ── Radar Chart ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:16px;">Cluster Radar Profile</div>', unsafe_allow_html=True)
        radar_metrics = ["Avg_Recency", "Avg_Frequency", "Avg_Monetary"]
        radar_labels  = ["Recency (days)", "Frequency (orders)", "Monetary ($)"]

        # Normalize each metric for radar (0-1)
        norm = df_summary[radar_metrics].copy()
        norm["Avg_Recency"] = 1 - (norm["Avg_Recency"] - norm["Avg_Recency"].min()) / (norm["Avg_Recency"].max() - norm["Avg_Recency"].min() + 1e-9)
        for col in ["Avg_Frequency", "Avg_Monetary"]:
            norm[col] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)

        fig_radar = go.Figure()
        for i, row in df_summary.iterrows():
            color = CLUSTER_PALETTE[int(row["Cluster"]) % len(CLUSTER_PALETTE)]
            vals  = norm.loc[i, radar_metrics].tolist()
            # Convert hex color to rgba for fill (Plotly requires rgba for transparency)
            r_int = int(color[1:3], 16)
            g_int = int(color[3:5], 16)
            b_int = int(color[5:7], 16)
            fill_rgba = f"rgba({r_int},{g_int},{b_int},0.18)"
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_labels + [radar_labels[0]],
                name=row["Cluster_Name"],
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(color=color, width=2),
            ))
        base_kv_r = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
        fig_radar.update_layout(
            **base_kv_r,
            title="",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262D", color="#8B949E"),
                angularaxis=dict(gridcolor="#21262D", color="#8B949E"),
            ),
            height=270,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Row 2: Revenue Contribution + Segment Distribution ───────────────────
    st.divider()
    col_pie, col_bar = st.columns(2, gap="medium")

    with col_pie:
        st.markdown('<div class="section-header">Revenue Contribution by Segment</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            df_summary,
            names="Cluster_Name",
            values="Total_Revenue",
            color_discrete_sequence=CLUSTER_PALETTE[:n_clusters],
            hole=0.55,
        )
        fig_pie.update_traces(
            textfont_size=12,
            marker=dict(line=dict(color="#0D1117", width=2)),
        )
        fig_pie.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
            title="",
            height=320,
            margin=dict(l=0, r=0, t=10, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown('<div class="section-header">Segment — Avg RFM Breakdown</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        metrics_bar = [("Avg_Recency", "#F06292", "Recency (days, inverted)"),
                       ("Avg_Frequency", "#81C784", "Avg Frequency"),
                       ("Avg_Monetary", "#4FC3F7", "Avg Monetary ($)")]
        for col_key, color, label in metrics_bar:
            fig_bar.add_trace(go.Bar(
                x=df_summary["Cluster_Name"],
                y=df_summary[col_key],
                name=label,
                marker_color=color,
                opacity=0.85,
            ))
        fig_bar.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin",)},
            title="",
            barmode="group",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=LEGEND_INLINE,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Detail Table ──────────────────────────────────────────────────────────
    with st.expander("📋  Cluster Summary Table", expanded=True):
        display_cols = {
            "Cluster_Name":  "Segment",
            "Customers":     "# Customers",
            "Avg_Recency":   "Avg Recency (days)",
            "Avg_Frequency": "Avg Frequency",
            "Avg_Monetary":  "Avg Monetary ($)",
            "Total_Revenue": "Total Revenue ($)",
            "Avg_RFM_Score": "Avg RFM Score",
        }
        st.dataframe(
            df_summary[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("🔍  Customer-level Detail"):
        search_col = st.columns([3, 1])
        search_term = search_col[0].text_input("Search customer name", placeholder="Type customer name…")
        seg_filter  = search_col[1].multiselect(
            "Filter by Segment",
            options=df_seg["Cluster_Name"].unique().tolist(),
            default=df_seg["Cluster_Name"].unique().tolist(),
        )
        display_df = df_seg[df_seg["Cluster_Name"].isin(seg_filter)]
        if search_term:
            display_df = display_df[display_df["Customer_Name"].str.contains(search_term, case=False, na=False)]
        st.dataframe(
            display_df[[
                "Customer_Name", "Segment", "Region",
                "Recency", "Frequency", "Monetary",
                "R_score", "F_score", "M_score", "RFM_Score",
                "Cluster_Name",
            ]].rename(columns={"Cluster_Name": "ML Segment"}),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecasting
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Historical Performance & ML Forecast</div>', unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    m_col = forecast_metric
    last_val     = df_monthly[m_col].iloc[-1]
    avg_val      = df_monthly[m_col].mean()
    max_val      = df_monthly[m_col].max()
    max_month    = df_monthly.loc[df_monthly[m_col].idxmax(), "month_year"].strftime("%b %Y")
    label_name   = "Sales" if m_col == "total_sales" else "Profit"

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric(f"📅 Last Month {label_name}", f"${last_val:,.0f}")
    kc2.metric(f"📊 Historical Average", f"${avg_val:,.0f}")
    kc3.metric(f"🏆 Peak {label_name}", f"${max_val:,.0f}", f"in {max_month}")

    # ── Forecast ──────────────────────────────────────────────────────────────
    series_data = df_monthly.set_index("month_year")[m_col].asfreq("MS").ffill()
    fc_values   = forecast_holt_winters(series_data, periods=forecast_periods)

    last_date    = series_data.index[-1]
    fc_dates     = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_periods, freq="MS")
    fc_series    = pd.Series(fc_values.values, index=fc_dates)

    # ── Main forecast chart ───────────────────────────────────────────────────
    fig_fc = go.Figure()

    fig_fc.add_trace(go.Scatter(
        x=series_data.index, y=series_data.values,
        name=f"Historical {label_name}",
        mode="lines",
        line=dict(color="#4FC3F7", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.08)",
    ))

    # Confidence band (±10% for illustration; a production system would use model CIs)
    ci_upper = fc_series * 1.10
    ci_lower = fc_series * 0.90
    fig_fc.add_trace(go.Scatter(
        x=list(fc_dates) + list(fc_dates[::-1]),
        y=list(ci_upper.values) + list(ci_lower.values[::-1]),
        fill="toself",
        fillcolor="rgba(255,183,77,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band (±10%)",
        showlegend=True,
    ))

    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc_series.values,
        name=f"Forecast {label_name}",
        mode="lines+markers",
        line=dict(color="#FFB74D", width=2.5, dash="dot"),
        marker=dict(size=7, color="#FFB74D", symbol="diamond"),
    ))

    # Bridge connector
    fig_fc.add_trace(go.Scatter(
        x=[series_data.index[-1], fc_dates[0]],
        y=[series_data.values[-1], fc_series.values[0]],
        mode="lines",
        line=dict(color="#FFB74D", width=1.5, dash="dot"),
        showlegend=False,
    ))

    fig_fc.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        title="",
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=LEGEND_INLINE,
        yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D",
                   tickprefix="$", tickformat=",.0f"),
        xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast Table ────────────────────────────────────────────────────────
    with st.expander("📋  Forecast Values Table"):
        fc_df = pd.DataFrame({
            "Month":            fc_dates.strftime("%B %Y"),
            f"Forecasted {label_name} ($)": fc_series.round(0).astype(int).values,
            "Lower Bound ($)":  ci_lower.round(0).astype(int).values,
            "Upper Bound ($)":  ci_upper.round(0).astype(int).values,
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # ── YoY Trend ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Monthly Sales vs Profit — Side-by-Side Historical</div>', unsafe_allow_html=True)

    fig_dual = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Total Sales (USD)", "Total Profit (USD)"),
        vertical_spacing=0.08,
    )
    fig_dual.add_trace(
        go.Scatter(x=df_monthly["month_year"], y=df_monthly["total_sales"],
                   name="Sales", mode="lines", line=dict(color="#4FC3F7", width=2),
                   fill="tozeroy", fillcolor="rgba(79,195,247,0.08)"),
        row=1, col=1,
    )
    fig_dual.add_trace(
        go.Scatter(x=df_monthly["month_year"], y=df_monthly["total_profit"],
                   name="Profit", mode="lines", line=dict(color="#81C784", width=2),
                   fill="tozeroy", fillcolor="rgba(129,199,132,0.08)"),
        row=2, col=1,
    )
    base_kv_d = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
    fig_dual.update_layout(
        **base_kv_d,
        title="",
        height=380, showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig_dual.update_yaxes(gridcolor="#21262D", tickprefix="$", tickformat=",.0f")
    fig_dual.update_xaxes(gridcolor="#21262D", row=2, col=1)
    st.plotly_chart(fig_dual, use_container_width=True)
