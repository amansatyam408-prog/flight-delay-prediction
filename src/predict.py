"""
src/predict.py — STEP 8/13.2: prediction pipeline (deployment-friendly).

Turns RAW flight info (airline, route, date, times) into the exact same 23
features the models were trained on, then returns:
    delay_probability   P(arrival delay > 15 min)   (XGBoost classifier)
    predicted_delay_min estimated delay in minutes  (XGBoost regressor)
    risk_category       Low / Medium / High

Deployment mode: loads the small models/lookups.pkl (committed, ~0.9 MB).
Local fallback: if lookups.pkl is missing, rebuilds tables from the big parquet
(run python src/export_lookups.py once to create it).

Demo:   python src/predict.py
API:    from src.predict import predict_flight
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
M = lambda p: os.path.join(BASE, "models", p)                        # models/...

# ---------- load everything once at startup ----------
FEATS = json.load(open(M("feature_list.json")))
ENC = joblib.load(M("encoders.pkl"))
CLF = xgb.XGBClassifier(); CLF.load_model(M("xgb_classifier.json"))
REG = xgb.XGBRegressor(); REG.load_model(M("xgb_regressor.json"))

if os.path.exists(M("lookups.pkl")):
    _L = joblib.load(M("lookups.pkl"))          # small, committed lookup tables
    TRAFFIC_DAY, TRAFFIC_HOUR, TRAFFIC_ARR = _L["TRAFFIC_DAY"], _L["TRAFFIC_HOUR"], _L["TRAFFIC_ARR"]
    GLOBAL_TRAFFIC = _L["GLOBAL_TRAFFIC"]
    WX_COLS, WX_MEDIAN, WX_BY_KEY = _L["WX_COLS"], _L["WX_MEDIAN"], _L["WX_BY_KEY"]
    AIRPORT_LL = _L["AIRPORT_LL"]
    print("Loaded lookups from models/lookups.pkl")
else:                                            # local fallback: big files
    _m = pd.read_parquet(os.path.join(BASE, "data/processed/flights_merged.parquet"),
                         columns=["ORIGIN_AIRPORT", "SCHED_HOUR", "DESTINATION_AIRPORT",
                                  "origin_dep_per_day", "origin_dep_per_hour", "dest_arr_per_day"])
    TRAFFIC_DAY = _m.groupby("ORIGIN_AIRPORT")["origin_dep_per_day"].mean()
    TRAFFIC_HOUR = _m.groupby(["ORIGIN_AIRPORT", "SCHED_HOUR"])["origin_dep_per_hour"].mean()
    TRAFFIC_ARR = _m.groupby("DESTINATION_AIRPORT")["dest_arr_per_day"].mean()
    GLOBAL_TRAFFIC = float(_m["origin_dep_per_day"].mean())
    del _m
    _w = pd.read_parquet(os.path.join(BASE, "data/raw/weather_2015.parquet"))
    WX_COLS = [c for c in _w.columns if c not in ("ORIGIN_AIRPORT", "DATE")]
    WX_MEDIAN = _w[WX_COLS].median()
    WX_BY_KEY = _w.set_index(["ORIGIN_AIRPORT", "DATE"])
    AIRPORT_LL = pd.read_parquet(os.path.join(BASE, "data/processed/airports.parquet")
                                 ).set_index("IATA_CODE")[["LATITUDE", "LONGITUDE"]]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points (DISTANCE feature)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _enc(col, val):
    """Encode a category; unseen value -> 'unknown' code (same as training)."""
    m = ENC[col]
    return m.get(str(val), len(m))


def build_features(airline, origin, dest, date, sched_dep, sched_arr, sched_time, distance=None):
    """Build a 1-row DataFrame with exactly the 23 training features, in order."""
    date = pd.Timestamp(date)
    hour = int(sched_dep) // 100
    minute = int(sched_dep) % 100
    arr_hour = int(sched_arr) // 100
    dow = int(date.day_of_week) + 1        # pandas Monday=0 -> DOT Monday=1

    if distance is None:
        try:
            la1, lo1 = AIRPORT_LL.loc[origin]
            la2, lo2 = AIRPORT_LL.loc[dest]
            distance = haversine_km(la1, lo1, la2, lo2)
        except KeyError:
            distance = 1000.0

    has_w = 0
    wvals = WX_MEDIAN
    try:
        wvals = WX_BY_KEY.loc[(origin, date)][WX_COLS]
        has_w = 1
    except KeyError:
        pass

    f = {
        "MONTH": date.month, "DAY": date.day, "DAY_OF_WEEK": dow,
        "SCHEDULED_TIME": float(sched_time), "DISTANCE": float(distance),
        "SCHED_HOUR": hour, "SCHED_MIN": minute, "SCHED_ARR_HOUR": arr_hour,
        "IS_WEEKEND": int(dow in (6, 7)),
        "origin_dep_per_day": float(TRAFFIC_DAY.get(origin, GLOBAL_TRAFFIC)),
        "origin_dep_per_hour": float(TRAFFIC_HOUR.get((origin, hour), GLOBAL_TRAFFIC)),
        "dest_arr_per_day": float(TRAFFIC_ARR.get(dest, GLOBAL_TRAFFIC)),
        "has_weather": has_w,
        "AIRLINE_enc": _enc("AIRLINE", airline),
        "ORIGIN_AIRPORT_enc": _enc("ORIGIN_AIRPORT", origin),
        "DESTINATION_AIRPORT_enc": _enc("DESTINATION_AIRPORT", dest),
    }
    for c in WX_COLS:
        f[c] = float(wvals[c])
    return pd.DataFrame([[f[k] for k in FEATS]], columns=FEATS)


def risk_category(p):
    return "High" if p >= 0.60 else ("Medium" if p >= 0.30 else "Low")


def predict_flight(airline, origin, dest, date, sched_dep, sched_arr, sched_time, distance=None):
    X = build_features(airline, origin, dest, date, sched_dep, sched_arr, sched_time, distance)
    p = float(CLF.predict_proba(X)[0, 1])
    mins = float(REG.predict(X)[0])
    return {
        "airline": airline, "route": f"{origin}->{dest}", "date": str(date),
        "delay_probability": round(p, 4),
        "predicted_delay_min": round(mins, 1),
        "delayed_over_15min": bool(p >= 0.5),
        "risk_category": risk_category(p),
    }


if __name__ == "__main__":
    examples = [
        dict(airline="UA", origin="ORD", dest="LAX", date="2015-07-04", sched_dep=1830, sched_arr=2110, sched_time=280),
        dict(airline="DL", origin="ATL", dest="JFK", date="2015-01-15", sched_dep=800, sched_arr=1030, sched_time=150),
        dict(airline="WN", origin="DAL", dest="PHX", date="2015-12-20", sched_dep=1915, sched_arr=2030, sched_time=135),
    ]
    for e in examples:
        r = predict_flight(**e)
        print(f"{r['airline']} {r['route']} on {r['date']}  ->  "
              f"P(delay>15m)={r['delay_probability']:.3f}  "
              f"est. delay={r['predicted_delay_min']} min  risk={r['risk_category']}")