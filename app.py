import os
from io import BytesIO
from datetime import timezone

import boto3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

# --- Config (defaults match your bucket/key) ---
BUCKET = os.environ.get("BUCKET_NAME", "crypto-forecast-bucket")
KEY = os.environ.get("PRED_KEY", "predictions/latest_all.csv")
AWS_PROFILE = os.environ.get("AWS_PROFILE")  # optional

# --- AWS client ---
@st.cache_resource
def get_s3_client():
    if AWS_PROFILE:
        session = boto3.Session(profile_name=AWS_PROFILE)
    else:
        session = boto3.Session()
    return session.client("s3")

s3 = get_s3_client()

@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_predictions(bucket: str, key: str):
    head = s3.head_object(Bucket=bucket, Key=key)
    last_modified = head["LastModified"]

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    df = pd.read_csv(BytesIO(body))
    return df, last_modified

def parse_dt(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- UI ---
st.title("📈 Crypto Forecast Dashboard (S3: latest_all.csv)")

topL, topM, topR = st.columns([2, 2, 3])
with topL:
    bucket = st.text_input("S3 bucket", value=BUCKET)
with topM:
    key = st.text_input("S3 key", value=KEY)
with topR:
    st.caption("Env vars supported: BUCKET_NAME, PRED_KEY, AWS_PROFILE")

# refresh button
if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()

try:
    df, last_modified = load_predictions(bucket, key)
except s3.exceptions.NoSuchKey:
    st.error(f"File not found: s3://{bucket}/{key}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load from S3: {e}")
    st.stop()

st.subheader("Data source")
st.code(f"s3://{bucket}/{key}", language="text")
st.caption(f"Last modified: {last_modified.astimezone(timezone.utc).isoformat()} (UTC)")

# try parse common datetime columns
df = parse_dt(df, ["date", "prediction_date", "run_time_utc", "run_time", "timestamp", "last_observed_date", "target_date"])

# --- Metrics row ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")

sym_col = first_existing(df, ["symbol", "coin", "asset", "ticker"])
m2.metric("Coins", str(df[sym_col].nunique()) if sym_col else "—")

date_col = first_existing(df, ["date", "prediction_date", "timestamp", "target_date"])
if date_col and df[date_col].notna().any():
    m3.metric("Min date", str(df[date_col].min().date()))
    m4.metric("Max date", str(df[date_col].max().date()))
else:
    m3.metric("Min date", "—")
    m4.metric("Max date", "—")

st.divider()

# --- Layout: overview + detail ---
left, right = st.columns([2, 3])

with left:
    st.subheader("Overview")
    view = df.copy()

    if sym_col:
        symbols = sorted(view[sym_col].dropna().unique().tolist())
        default_syms = symbols[:10] if len(symbols) > 10 else symbols
        chosen = st.multiselect("Filter coins", symbols, default=default_syms)
        view = view[view[sym_col].isin(chosen)]

    # Optional: rank table if we can infer predicted return
    pred_col = first_existing(view, ["pred_close", "prediction", "y_pred", "forecast", "predicted_close"])
    last_col = first_existing(view, ["close", "last_close", "actual_close", "y_true", "target"])

    if pred_col and last_col:
        tmp = view.copy()
        # avoid divide-by-zero
        tmp["pred_return_pct"] = (tmp[pred_col] - tmp[last_col]) / tmp[last_col].replace(0, pd.NA) * 100
        sort_on = st.selectbox("Sort by", ["pred_return_pct", pred_col, last_col], index=0)
        tmp = tmp.sort_values(sort_on, ascending=False, na_position="last")
        st.dataframe(tmp, width='stretch', height=520)
    else:
        st.dataframe(view, width='stretch', height=520)
        st.caption("Tip: If your CSV has both an actual/last close and a predicted close column, I can auto-add return ranking.")

with right:
    st.subheader("Coin detail")

    if sym_col and date_col:
        sym = st.selectbox("Coin", sorted(df[sym_col].dropna().unique().tolist()))
        d = df[df[sym_col] == sym].copy()
        d = d.sort_values(date_col)

        pred_col = first_existing(d, ["pred_close", "prediction", "y_pred", "forecast", "predicted_close"])
        last_col = first_existing(d, ["close", "last_close", "actual_close", "y_true", "target"])

        plot_cols = [c for c in [last_col, pred_col] if c]
        if plot_cols:
            y_min = d[plot_cols].min().min()
            y_max = d[plot_cols].max().max()
            pad = (y_max - y_min) * 0.05  # 5% padding

            fig = go.Figure()

            for col in plot_cols:
                fig.add_trace(
                    go.Scatter(
                        x=d[date_col],
                        y=d[col],
                        mode="lines",
                        name=col
                    )
                )

            fig.update_layout(
                yaxis=dict(range=[y_min - pad, y_max + pad]),
                xaxis_title="Date",
                yaxis_title="Price",
                margin=dict(l=40, r=20, t=30, b=40),
                legend_title_text="Series",
            )

            st.plotly_chart(fig, width='stretch')
        else:
            st.info("I couldn't detect numeric columns for plotting. Tell me your actual/pred column names and I'll wire them.")
    else:
        st.info("To enable the chart, the CSV needs a symbol column (e.g. symbol) and a date column (e.g. date).")

st.divider()
st.caption("Auto-refresh uses a 5-minute cache (ttl=300). Use 'Refresh now' for immediate reload.")