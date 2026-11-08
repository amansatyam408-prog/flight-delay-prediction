"""
dashboard/app.py — STEP 10/13.3: Streamlit dashboard (API mode + standalone mode).
Local:  terminal 1: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
        terminal 2: python -m streamlit run dashboard/app.py   -> http://localhost:8501
Standalone (e.g. Streamlit Cloud): if the API is not reachable, the model is loaded
in-process via src.predict using committed models/ files — no API needed.
Cloud main file path: dashboard/app.py (forward slash!)
"""
import os
import sys
from datetime import time

import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)                                             # so `import src...` works
API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Predictive Aviation Analytics & Flight Delay Mitigation Engine")
st.caption("XGBoost on 5.7M flights (U.S. DOT 2015) + real weather (Open-Meteo) + airport-traffic features. "
           "ROC-AUC 0.754 | MAE ≈ 18.9 min")

# ---------- mode detection: API first, standalone fallback ----------
try:
    api_ok = requests.get(f"{API}/health", timeout=2).json().get("status") == "ok"
except Exception:
    api_ok = False

if api_ok:
    airlines = requests.get(f"{API}/airlines", timeout=5).json()
    airports = requests.get(f"{API}/airports", timeout=5).json()
else:
    import joblib
    _L = joblib.load(os.path.join(ROOT, "models", "lookups.pkl"))
    airlines = _L["AIRLINES"]
    airports = _L["AIRPORTS"]
    st.info("🛰️ Standalone mode (no API): predicting in-process with the committed XGBoost model. "
            "Locally you can also run the FastAPI backend — then this banner disappears.")

airline_opts = {f"{a['AIRLINE']} ({a['IATA_CODE']})": a["IATA_CODE"] for a in airlines}
airport_opts = {f"{a['CITY']}, {a['STATE']} — {a['AIRPORT']} ({a['IATA_CODE']})": a["IATA_CODE"]
                for a in airports}

# ---------- input form ----------
st.header("🔎 Enter flight details")
with st.form("flight_form"):
    c1, c2, c3 = st.columns(3)
    airline_label = c1.selectbox("Airline", list(airline_opts))
    origin_label = c2.selectbox("Origin airport", list(airport_opts))
    dest_label = c3.selectbox("Destination airport", list(airport_opts))
    c4, c5, c6 = st.columns(3)
    date = c4.date_input("Flight date", value=pd.Timestamp("2015-12-20").date())
    dep_time = c5.time_input("Scheduled departure", value=time(19, 15))
    arr_time = c6.time_input("Scheduled arrival", value=time(20, 30))
    sched_time = st.number_input("Scheduled flight duration (min)", min_value=10, max_value=1200, value=135)
    submitted = st.form_submit_button("🔮 Predict delay risk")

# ---------- predict (API or in-process) ----------
if submitted:
    if origin_label == dest_label:
        st.error("Origin and destination must differ.")
    else:
        payload = {
            "airline": airline_opts[airline_label],
            "origin": airport_opts[origin_label],
            "dest": airport_opts[dest_label],
            "date": str(date),
            "sched_dep": dep_time.hour * 100 + dep_time.minute,
            "sched_arr": arr_time.hour * 100 + arr_time.minute,
            "sched_time": int(sched_time),
        }
        res = None
        if api_ok:
            with st.spinner("Calling the prediction API..."):
                r = requests.post(f"{API}/predict", json=payload, timeout=30)
            if r.status_code == 200:
                res = r.json()
            else:
                st.error(f"API returned {r.status_code}: {r.text}")
        else:
            from src.predict import predict_flight
            with st.spinner("Running XGBoost model in-process..."):
                res = predict_flight(**payload)

        if res:
            p = res["delay_probability"]
            st.subheader(f"Result: {res['airline']} {res['route']} on {res['date']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Delay probability (>15 min)", f"{p:.1%}")
            m2.metric("Estimated delay duration", f"{res['predicted_delay_min']:.0f} min")
            badge = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}[res["risk_category"]]
            m3.metric("Risk category", badge)
            st.progress(min(p, 1.0), text=f"P(delay > 15 min) = {p:.1%}")
            if res["risk_category"] == "High":
                st.error("⚠️ HIGH delay risk — alert dispatchers early, consider rebooking options.")
            elif res["risk_category"] == "Medium":
                st.warning("🟡 Medium risk — monitor weather and airport congestion.")
            else:
                st.success("🟢 Low risk — flight expected to be close to schedule.")
            with st.expander("Raw prediction response"):
                st.json(res)

# ---------- insights (Step 7 figures) ----------
st.header("📊 Insights from the 2015 test data")
figs = ["roc_curve", "feature_importance", "delay_by_hour", "delay_by_month",
        "delay_by_airline", "confusion_matrix", "actual_vs_predicted"]
cols = st.columns(2)
for i, f in enumerate(figs):
    with cols[i % 2]:
        st.image(os.path.join(ROOT, "docs", "figures", f + ".png"), caption=f, width="stretch")

st.caption("Model: XGBoost (ROC-AUC 0.754, MAE 18.9 min) | Data: U.S. DOT BTS 2015 + Open-Meteo archive weather | "
           "No post-flight features used (leakage-free)")