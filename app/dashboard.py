"""DriveGuard: hard drive failure risk dashboard (Streamlit)."""
from pathlib import Path
import datetime as dt
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import shap
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="DriveGuard", page_icon="💾", layout="wide")


@st.cache_resource
def load_model():
    b = joblib.load(ROOT / "models" / "lgbm_ST12000NM0008.joblib")
    return b["model"], b["features"], shap.TreeExplainer(b["model"])


@st.cache_resource
def load_data():
    df = pl.read_parquet(ROOT / "data" / "dashboard_test.parquet")
    meta = json.loads((ROOT / "data" / "dashboard_meta.json").read_text())
    return df, meta


model, FEATURES, explainer = load_model()
df, meta = load_data()
THR = meta["threshold"]

NAMES = {
    "smart_5": "reallocated (bad) sectors", "smart_187": "uncorrectable errors",
    "smart_188": "command timeouts", "smart_197": "pending sectors",
    "smart_198": "offline uncorrectable sectors",
}


def describe(feat, value):
    for k, name in NAMES.items():
        if feat.startswith(k + "_"):
            if feat.endswith("_diff7"):
                return f"{name} rose by {value:,.0f} in the last 7 days"
            if feat.endswith("_diff30"):
                return f"{name} rose by {value:,.0f} in the last 30 days"
            if feat.endswith("_max30"):
                return f"{name} peaked at {value:,.0f} in the last 30 days"
            if feat.endswith("_days_nonzero30"):
                return f"{name} reported on {value:.0f} of the last 30 days"
            if feat.endswith("_raw"):
                return f"{name} = {value:,.0f}"
    if feat == "age_days":
        return f"drive age {value / 365:.1f} years"
    if feat == "temp_c":
        return f"temperature {value:.0f}°C"
    if feat == "temp_mean7":
        return f"7-day average temperature {value:.0f}°C"
    if feat == "gap_days":
        return f"{value:.0f} days since previous report"
    return f"{feat} = {value}"


# ---------------- Sidebar ----------------
st.sidebar.title("💾 DriveGuard")
st.sidebar.caption("Seagate ST12000NM0008 · Backblaze Drive Stats · test period Apr–Jun 2026")
min_d, max_d = df["date"].min(), df["date"].max()
as_of = st.sidebar.slider("View the fleet as of", min_value=min_d, max_value=max_d,
                          value=min_d + (max_d - min_d) // 2, format="YYYY-MM-DD")
reveal = st.sidebar.checkbox("Reveal what actually happened (hindsight)", value=False)
st.sidebar.markdown(f"**Alert threshold:** {THR:.3f}  \n(set so ~1% of healthy drives are flagged)")

# ---------------- Fleet snapshot ----------------
snap = (df.filter((pl.col("date") <= as_of) & (pl.col("date") > as_of - dt.timedelta(days=7)))
          .sort("date").group_by("serial_number").agg(pl.all().last())
          .sort("score", descending=True))
flagged = snap.filter(pl.col("score") > THR)

st.title("Hard drive failure early-warning")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Drives reporting", f"{snap.height:,}")
c2.metric("Flagged high-risk", f"{flagged.height:,}")
c3.metric("Share of fleet flagged", f"{flagged.height / max(snap.height, 1):.2%}")
if reveal:
    hits = int(flagged["label"].sum())
    c4.metric("Flagged drives that failed ≤30 days", f"{hits}/{flagged.height}")
else:
    c4.metric("As of", str(as_of))

st.subheader("Riskiest drives")
table = snap.head(50).select(
    pl.col("serial_number").alias("Drive"),
    pl.col("date").dt.strftime("%Y-%m-%d").alias("Last report"),
    pl.col("score").round(3).alias("Risk score"),
    pl.col("smart_187_raw").alias("Uncorrectable (187)"),
    pl.col("smart_188_raw").alias("Timeouts (188)"),
    pl.col("smart_5_raw").alias("Bad sectors (5)"),
    pl.col("smart_197_raw").alias("Pending (197)"),
    (pl.col("age_days") / 365).round(1).alias("Age (yrs)"),
    pl.col("days_to_failure").alias("Failed in (days)"),
)
if not reveal:
    table = table.drop("Failed in (days)")
st.dataframe(table.to_pandas(), width="stretch", hide_index=True, height=320)

# ---------------- Drive drill-down ----------------
st.subheader("Drive detail")
options = snap.head(50)["serial_number"].to_list()
if not options:
    st.info("No drives reporting in the week before this date.")
    st.stop()
serial = st.selectbox("Choose a drive", options)

row = snap.filter(pl.col("serial_number") == serial)
x = row.select(FEATURES).to_pandas()
sv = explainer.shap_values(x)
sv = (sv[1] if isinstance(sv, list) else sv)[0]

left, right = st.columns([1, 2])
with left:
    score = row["score"][0]
    status = "🔴 HIGH RISK" if score > THR else "🟢 Normal"
    st.markdown(f"### {status}\nRisk score **{score:.3f}** on {row['date'][0]}")
    st.markdown("**Why:**")
    for i in np.argsort(-sv)[:4]:
        if sv[i] > 0:
            st.markdown(f"- ↑ {describe(FEATURES[i], x.iloc[0, i])}")
    if reveal:
        dtf = row["days_to_failure"][0]
        st.markdown(f"**Outcome:** {'failed ' + str(int(dtf)) + ' days later' if dtf is not None and dtf >= 0 else 'did not fail in the test period'}")

with right:
    hist = df.filter((pl.col("serial_number") == serial) & (pl.col("date") <= as_of)).sort("date").to_pandas()
    fig = px.line(hist, x="date", y="score", title="Risk score over time")
    fig.add_hline(y=THR, line_dash="dash", annotation_text="alert threshold")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    long = hist.melt(id_vars="date", value_vars=["smart_187_raw", "smart_188_raw", "smart_5_raw", "smart_197_raw"],
                     var_name="SMART attribute", value_name="count")
    st.plotly_chart(px.line(long, x="date", y="count", color="SMART attribute", title="SMART error counters"),
                    width="stretch")

# ---------------- Results ----------------
with st.expander("Model performance on the test quarter (Apr–Jun 2026)"):
    st.markdown(f"{meta['drives']:,} drives · {meta['failures']} failures. "
                "Both methods are tuned to flag the same share of healthy drives.")