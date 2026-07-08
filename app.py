import streamlit as st
import pandas as pd
import joblib
import numpy as np
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictive Maintenance Risk Dashboard",
    page_icon="⚙️",
    layout="wide"
)
 
# ============================================================
# LOAD MODEL + SCALER (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
 
model, scaler = load_model()

def engineer_features(df):
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * 3.14159 / 60)
    df["wear_torque_ratio"] = df["Tool wear [min]"] / (df["Torque [Nm]"] + 1)
    return df
 
# ============================================================
# RISK BAND LOGIC (your tuned thresholds)
# ============================================================
def risk_category(score):
    if score <= 15:
        return "Safe"
    elif score <= 25:
        return "Watch closely"
    elif score <= 75:
        return "High risk"
    else:
        return "Critical"
 

ACTION_MAP = {
    "Safe": "Keep running",
    "Watch closely": "Schedule inspection",
    "High risk": "Inspect soon",
    "Critical": "Stop and repair immediately"
}
 
COLOR_MAP = {
    "Safe": "#2ecc71",
    "Watch closely": "#f1c40f",
    "High risk": "#e67e22",
    "Critical": "#e74c3c"
}
 
REQUIRED_COLUMNS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
 
# ============================================================
# SIDEBAR — machine identity + upload
# ============================================================
st.sidebar.title("⚙️ Controls")

machine_names = st.sidebar.text_area(
    "Optional: Machine names (one per row, matching upload order)",
    placeholder="Crusher 1\nRaw Mill 2\nKiln 1\n..."
)
 
uploaded_file = st.sidebar.file_uploader(
    "Upload machine sensor readings (CSV)", type="csv"
)
 
st.sidebar.markdown("---")
st.sidebar.caption(
    "Required columns: " + ", ".join(REQUIRED_COLUMNS)
)
 
# ============================================================
# MAIN HEADER
# ============================================================
st.title("🏭 Predictive Maintenance & Reliability Early-Warning Dashboard")
st.caption("DCP Engineering Challenge — Track 2: Predictive Maintenance and Reliability Early-Warning System")

# ============================================================
# MAIN LOGIC
# ============================================================
if uploaded_file is None:
    st.info("👈 Upload a CSV of machine sensor readings to generate risk scores.")
    with st.expander("See expected CSV format"):
        example = pd.DataFrame({
            "Type": ["M", "L", "H"],
            "Air temperature [K]": [298.1, 298.2, 298.4],
            "Process temperature [K]": [308.6, 308.7, 309.0],
            "Rotational speed [rpm]": [1551, 1408, 1400],
            "Torque [Nm]": [42.8, 46.3, 60.1],
            "Tool wear [min]": [0, 3, 210]
        })
        st.dataframe(example)
    st.stop()
 
# --- Read uploaded file ---
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()
 
# --- Validate columns ---
missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing:
    st.error(f"Missing required column(s): {missing}")
    st.stop()
 
# --- Attach machine names ---
if machine_names.strip():
    names_list = [n.strip() for n in machine_names.split("\n") if n.strip()]
    if len(names_list) == len(raw_df):
        raw_df.insert(0, "Machine", names_list)
    else:
        st.warning("Number of machine names doesn't match number of rows — using generic IDs instead.")
        raw_df.insert(0, "Machine", [f"Machine {i+1}" for i in range(len(raw_df))])
else:
    raw_df.insert(0, "Machine", [f"Machine {i+1}" for i in range(len(raw_df))])
 
# --- Feature engineering ---
df_fe = engineer_features(raw_df)
 
# --- One-hot encode Type to match training ---
df_fe = pd.get_dummies(df_fe, columns=["Type"], dtype=int)
 
# Ensure all Type dummy columns the model expects exist (fill missing with 0)
for col in ["Type_H", "Type_L", "Type_M"]:
    if col not in df_fe.columns:
        df_fe[col] = 0
 
# --- Build feature matrix in the exact order the model was trained on ---
feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None
if feature_cols is not None:
    X_new = df_fe[feature_cols]
else:
    st.error("Scaler is missing feature name metadata — re-save scaler with a fitted DataFrame.")
    st.stop()
 
# --- Scale + predict ---
X_scaled = scaler.transform(X_new)
failure_prob = model.predict_proba(X_scaled)[:, 1]
 
# --- Build results table ---
results = raw_df[["Machine"] + REQUIRED_COLUMNS].copy()
results["risk_score"] = (failure_prob * 100).round(1)
results["risk_level"] = results["risk_score"].apply(risk_category)
results["recommended_action"] = results["risk_level"].map(ACTION_MAP)
results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)
 
# ============================================================
# SUMMARY METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
counts = results["risk_level"].value_counts()
col1.metric("🟢 Safe", int(counts.get("Safe", 0)))
col2.metric("🟡 Watch closely", int(counts.get("Watch closely", 0)))
col3.metric("🟠 High risk", int(counts.get("High risk", 0)))
col4.metric("🔴 Critical", int(counts.get("Critical", 0)))
 
st.markdown("---")
 
# ============================================================
# CRITICAL ALERTS BANNER
# ============================================================
critical = results[results["risk_level"] == "Critical"]
if len(critical) > 0:
    st.error(f"🚨 {len(critical)} machine(s) flagged CRITICAL — immediate attention required: "
              + ", ".join(critical["Machine"].tolist()))
 
# ============================================================
# MAIN TABLE (color-coded)
# ============================================================
st.subheader("Machine Risk Overview")
 
def highlight_risk(row):
    color = COLOR_MAP.get(row["risk_level"], "white")
    return [f"background-color: {color}; color: white" if col == "risk_level" else "" for col in row.index]
 
st.dataframe(
    results.style.apply(highlight_risk, axis=1),
    use_container_width=True,
    height=450
)
 
# ============================================================
# DOWNLOAD BUTTON
# ============================================================
csv_out = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download risk report (CSV)",
    data=csv_out,
    file_name="risk_report.csv",
    mime="text/csv"
)
 
# ============================================================
# RISK SCORE DISTRIBUTION CHART
# ============================================================
st.subheader("Risk Score Distribution")
st.bar_chart(results.set_index("Machine")["risk_score"])
 
