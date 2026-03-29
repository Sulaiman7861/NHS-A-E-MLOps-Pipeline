import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.features.builder import build_features
from src.models.detector import (
    detect_zscore, detect_isolation_forest, detect_autoencoder,
    run_all, detect_drift,
)

PARQUET_PATH = "data/processed/ae_combined.parquet"

FEATURE_LABELS = {
    "total_attendances":         "Total Attendances",
    "breach_rate":               "4-Hour Breach Rate",
    "admissions_per_attendance": "Admissions per Attendance",
    "over_4hr_rate":             "Over 4hr Decision-to-Admit Rate",
}

NHS_BLUE      = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_RED       = "#DA291C"
NHS_GREEN     = "#009639"
NHS_LIGHT     = "#F0F4F5"
NHS_WHITE     = "#FFFFFF"
NHS_GREY      = "#768692"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NHS A&E Anomaly Detection",
    page_icon="🏥",
    layout="wide",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Fira Sans', Arial, sans-serif;
    background-color: {NHS_LIGHT};
    color: #231f20;
  }}
  .nhs-header {{
    background-color: {NHS_BLUE};
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
  }}
  .nhs-logo {{
    background-color: {NHS_WHITE};
    color: {NHS_BLUE};
    font-weight: 700;
    font-size: 22px;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 2px;
  }}
  .nhs-header-title {{ color: {NHS_WHITE}; font-size: 22px; font-weight: 600; margin: 0; }}
  .nhs-header-sub   {{ color: #bfd7ed; font-size: 13px; margin: 0; }}
  .nhs-kpi {{
    background: {NHS_WHITE};
    border-left: 6px solid {NHS_BLUE};
    padding: 16px 20px;
    border-radius: 0 4px 4px 0;
    margin-bottom: 8px;
  }}
  .nhs-kpi-value {{ font-size: 36px; font-weight: 700; color: {NHS_BLUE}; line-height: 1.1; }}
  .nhs-kpi-label {{ font-size: 13px; color: {NHS_GREY}; text-transform: uppercase; letter-spacing: 0.5px; }}
  .nhs-h2 {{
    font-size: 20px; font-weight: 700; color: {NHS_DARK_BLUE};
    border-bottom: 3px solid {NHS_BLUE};
    padding-bottom: 6px; margin: 24px 0 12px 0;
  }}
  section[data-testid="stSidebar"] {{ background-color: {NHS_DARK_BLUE}; }}
  section[data-testid="stSidebar"] * {{ color: {NHS_WHITE} !important; }}
  section[data-testid="stSidebar"] label {{ color: #bfd7ed !important; font-size: 13px; text-transform: uppercase; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 0 !important; }}
</style>

<div class="nhs-header">
  <div class="nhs-logo">NHS</div>
  <div>
    <p class="nhs-header-title">A&amp;E Anomaly Detection Dashboard</p>
    <p class="nhs-header-sub">Detecting unusual patterns in A&amp;E attendance and performance — England</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    raw = pd.read_parquet(PARQUET_PATH)
    return build_features(raw)

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"No data found at `{PARQUET_PATH}`. Run `python main.py` first.")
    st.stop()

available_periods = sorted(df["period"].dropna().unique())
period_labels     = [pd.Timestamp(p).strftime("%b %Y") for p in available_periods]
period_map        = dict(zip(period_labels, available_periods))

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")

    method_name = st.selectbox("Detection technique", ["Z-Score", "Isolation Forest", "Autoencoder"])
    metric      = st.selectbox("Metric to visualise", list(FEATURE_LABELS.keys()),
                               index=1,  # default to breach_rate
                               format_func=lambda x: FEATURE_LABELS[x])
    orgs          = sorted(df["org_name"].dropna().unique())
    selected_orgs = st.multiselect("Filter by organisation (blank = all)", options=orgs)

    st.divider()
    st.markdown("### Reference window")
    st.caption("Train models on pre-COVID baseline only. Everything else is scored against it.")

    use_reference = st.toggle("Use reference window", value=len(period_labels) > 1)
    if use_reference and len(period_labels) >= 2:
        # Default start = first available period, default end = Feb 2020 (last pre-COVID month)
        default_end = "Feb 2020" if "Feb 2020" in period_labels else period_labels[len(period_labels) // 2 - 1]
        ref_start_label = st.selectbox("Reference start", period_labels, index=0)
        ref_end_label   = st.selectbox("Reference end", period_labels,
                                       index=period_labels.index(default_end) if default_end in period_labels else max(0, len(period_labels) // 2 - 1))
        ref_start    = period_map[ref_start_label]
        ref_end      = period_map[ref_end_label]
        reference_df = df[df["period"].between(ref_start, ref_end)]
    else:
        reference_df = None

    if method_name == "Z-Score":
        threshold = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)
        results   = detect_zscore(df, threshold=threshold, reference_df=reference_df)
    elif method_name == "Isolation Forest":
        contamination = st.slider("Contamination", 0.01, 0.20, 0.05, 0.01)
        results       = detect_isolation_forest(df, contamination=contamination, reference_df=reference_df)
    else:
        pct     = st.slider("Anomaly threshold percentile", 80, 99, 95, 1)
        results = detect_autoencoder(df, threshold_percentile=pct, reference_df=reference_df)

if selected_orgs:
    results = results[results["org_name"].isin(selected_orgs)]

# ── KPI row ───────────────────────────────────────────────────────────────────

total   = len(results)
flagged = int(results["anomaly"].sum())
pct_str = f"{100 * flagged / total:.1f}%" if total else "0%"

st.markdown('<p class="nhs-h2">Summary</p>', unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
for col, value, label in [(k1, total, "Organisations"), (k2, flagged, "Anomalies Flagged"), (k3, pct_str, "% Flagged")]:
    col.markdown(f"""
    <div class="nhs-kpi">
      <div class="nhs-kpi-value">{value}</div>
      <div class="nhs-kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

# ── Scatter chart — red/blue dots ─────────────────────────────────────────────

st.markdown(f'<p class="nhs-h2">{FEATURE_LABELS[metric]} — Anomalies Highlighted</p>', unsafe_allow_html=True)

plot_df = results.copy()
plot_df["Status"]     = plot_df["anomaly"].map({True: "Anomaly", False: "Normal"})
plot_df["period_str"] = plot_df["period"].dt.strftime("%b %Y")

# Sort x-axis chronologically by mapping each label to its timestamp
chron_order = (
    plot_df[["period", "period_str"]].drop_duplicates()
    .sort_values("period")["period_str"].tolist()
)

fig = px.scatter(
    plot_df,
    x="period_str",
    y=metric,
    color="Status",
    color_discrete_map={"Anomaly": NHS_RED, "Normal": NHS_BLUE},
    hover_data=["org_name", "org_code"],
    labels={"period_str": "Period", metric: FEATURE_LABELS[metric], "Status": ""},
    category_orders={"period_str": chron_order},
    height=440,
)
fig.update_traces(marker=dict(size=9, opacity=0.8))
fig.update_layout(
    plot_bgcolor=NHS_WHITE,
    paper_bgcolor=NHS_WHITE,
    xaxis=dict(title="Period", gridcolor="#e8edee", tickangle=-45),
    yaxis=dict(title=FEATURE_LABELS[metric], gridcolor="#e8edee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    font=dict(family="Fira Sans, Arial, sans-serif"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Method comparison with numbers ────────────────────────────────────────────

st.markdown('<p class="nhs-h2">Comparison — All Three Detection Methods</p>', unsafe_allow_html=True)

all_results = run_all(df, reference_df=reference_df)
if selected_orgs:
    all_results = all_results[all_results["org_name"].isin(selected_orgs)]

summary = (
    all_results.groupby("method")["anomaly"]
    .agg(flagged="sum", total="count")
    .assign(pct=lambda x: (100 * x["flagged"] / x["total"]).round(1))
    .reset_index()
)
summary.columns = ["Method", "Flagged", "Total", "% Flagged"]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(summary, hide_index=True, use_container_width=True)
with col_b:
    fig_cmp = px.bar(
        summary, x="Method", y="Flagged",
        text="Flagged",
        color="Method",
        color_discrete_sequence=[NHS_BLUE, NHS_DARK_BLUE, NHS_GREEN],
        labels={"Flagged": "Anomalies Flagged"},
        height=320,
    )
    fig_cmp.update_traces(texttemplate="%{text} flagged", textposition="outside")
    fig_cmp.update_layout(
        showlegend=False,
        plot_bgcolor=NHS_WHITE, paper_bgcolor=NHS_WHITE,
        font=dict(family="Fira Sans, Arial, sans-serif"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ── Drift detection ───────────────────────────────────────────────────────────

st.markdown('<p class="nhs-h2">Data Drift Detection</p>', unsafe_allow_html=True)

if reference_df is None or len(reference_df) == 0:
    st.info("Enable the reference window in the sidebar to see drift analysis.")
else:
    current_df = df[~df["period"].between(ref_start, ref_end)]
    if current_df.empty:
        st.info("No data outside the reference window to compare against.")
    else:
        drift_report = detect_drift(reference_df, current_df)
        n_drifted    = int(drift_report["drifted"].sum())
        st.caption(
            f"KS test comparing reference window vs all other months. "
            f"**{n_drifted} of {len(drift_report)} features have drifted** (p < 0.05)."
        )
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            disp = drift_report.copy()
            disp["drifted"] = disp["drifted"].map({True: "YES", False: "no"})
            st.dataframe(disp, hide_index=True, use_container_width=True)
        with col_d2:
            fig_drift = px.bar(
                drift_report, x="feature", y="ks_statistic",
                color="drifted",
                color_discrete_map={True: NHS_RED, False: NHS_BLUE},
                labels={"ks_statistic": "KS Statistic", "feature": "Feature", "drifted": "Drifted"},
                title="KS statistic per feature (higher = more drift)",
                height=320,
            )
            fig_drift.add_hline(y=0.3, line_dash="dot", line_color=NHS_RED,
                                annotation_text="Drift threshold")
            fig_drift.update_layout(
                plot_bgcolor=NHS_WHITE, paper_bgcolor=NHS_WHITE,
                font=dict(family="Fira Sans, Arial, sans-serif"),
            )
            st.plotly_chart(fig_drift, use_container_width=True)
        st.caption(
            "**What this means:** Features flagged as drifted have shifted significantly from the "
            "baseline. This is why COVID-era months appear as anomalies — the models were trained "
            "on pre-COVID normality. In production, retrain once the drift stabilises."
        )

# ── Heatmap ───────────────────────────────────────────────────────────────────

st.markdown(f'<p class="nhs-h2">Heatmap — {FEATURE_LABELS[metric]} by Organisation and Month</p>',
            unsafe_allow_html=True)
st.caption("✕ marks anomalies. A dark column = an unusual month across many trusts.")

heatmap_df               = results.copy()
heatmap_df["period_str"] = heatmap_df["period"].dt.strftime("%b %Y")
pivot                    = heatmap_df.pivot_table(index="org_name", columns="period_str", values=metric,    aggfunc="mean")
anomaly_pivot            = heatmap_df.pivot_table(index="org_name", columns="period_str", values="anomaly", aggfunc="max")

period_order  = (heatmap_df[["period", "period_str"]].drop_duplicates()
                 .sort_values("period")["period_str"].tolist())
pivot         = pivot.reindex(columns=[c for c in period_order if c in pivot.columns])
anomaly_pivot = anomaly_pivot.reindex(columns=pivot.columns)
annotations   = anomaly_pivot.applymap(lambda v: "✕" if v == 1 else "")

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    text=annotations.values,
    texttemplate="%{text}",
    colorscale="RdYlGn" if metric != "breach_rate" else "RdYlGn_r",
    colorbar=dict(title=FEATURE_LABELS[metric]),
    hovertemplate="Org: %{y}<br>Period: %{x}<br>Value: %{z:,.2f}<extra></extra>",
))
fig_heat.update_layout(
    height=max(400, len(pivot) * 18),
    plot_bgcolor=NHS_WHITE, paper_bgcolor=NHS_WHITE,
    xaxis=dict(tickangle=-45),
    yaxis=dict(tickfont=dict(size=10)),
    margin=dict(l=250),
    font=dict(family="Fira Sans, Arial, sans-serif"),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Flagged records ───────────────────────────────────────────────────────────

st.markdown(f'<p class="nhs-h2">Flagged Records — {method_name}</p>', unsafe_allow_html=True)

flagged_df   = results[results["anomaly"]].copy()
flagged_df["period"] = flagged_df["period"].dt.strftime("%b %Y")
display_cols = ["period", "org_code", "org_name"] + [
    c for c in ["total_attendances", "breach_rate", "over_4hr_rate"] if c in flagged_df.columns
]
st.dataframe(
    flagged_df[display_cols].sort_values("breach_rate", ascending=False),
    hide_index=True, use_container_width=True,
)
