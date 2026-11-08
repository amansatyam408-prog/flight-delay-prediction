"""
src/export_lookups.py — STEP 13.1: export SMALL lookup tables for deployment.
The full merged dataset (5.7M rows) is too big to commit, so we precompute the
tiny tables src/predict.py needs into one file: models/lookups.pkl (~1-2 MB).
Run from project root:  python src/export_lookups.py
"""
import os
import joblib
import pandas as pd

_m = pd.read_parquet(
    "data/processed/flights_merged.parquet",
    columns=["ORIGIN_AIRPORT", "SCHED_HOUR", "DESTINATION_AIRPORT",
             "origin_dep_per_day", "origin_dep_per_hour", "dest_arr_per_day"],
)
lookups = {
    "TRAFFIC_DAY": _m.groupby("ORIGIN_AIRPORT")["origin_dep_per_day"].mean(),
    "TRAFFIC_HOUR": _m.groupby(["ORIGIN_AIRPORT", "SCHED_HOUR"])["origin_dep_per_hour"].mean(),
    "TRAFFIC_ARR": _m.groupby("DESTINATION_AIRPORT")["dest_arr_per_day"].mean(),
    "GLOBAL_TRAFFIC": float(_m["origin_dep_per_day"].mean()),
}
del _m

weather = pd.read_parquet("data/raw/weather_2015.parquet")
weather["DATE"] = pd.to_datetime(weather["DATE"])     # match pd.Timestamp lookup keys
wx_cols = [c for c in weather.columns if c not in ("ORIGIN_AIRPORT", "DATE")]
lookups["WX_COLS"] = wx_cols
lookups["WX_MEDIAN"] = weather[wx_cols].median()
lookups["WX_BY_KEY"] = weather.set_index(["ORIGIN_AIRPORT", "DATE"])

airports = pd.read_parquet("data/processed/airports.parquet")
lookups["AIRPORT_LL"] = airports.set_index("IATA_CODE")[["LATITUDE", "LONGITUDE"]]

lookups["AIRLINES"] = pd.read_parquet("data/processed/airlines.parquet").to_dict("records")
lookups["AIRPORTS"] = airports.to_dict("records")

joblib.dump(lookups, "models/lookups.pkl")
print("Saved models/lookups.pkl   size = %.2f MB" % (os.path.getsize("models/lookups.pkl") / 1e6))
print("  TRAFFIC_DAY:", len(lookups["TRAFFIC_DAY"]), "airports |",
      "TRAFFIC_HOUR:", len(lookups["TRAFFIC_HOUR"]), "rows |",
      "WX_BY_KEY:", len(lookups["WX_BY_KEY"]), "rows |",
      "AIRPORT_LL:", len(lookups["AIRPORT_LL"]), "airports")