"""
src/merge_data.py — STEP 4b: combine flights + weather + airport traffic.
Traffic = real congestion computed from the flights data (departures per
origin airport per day/hour, arrivals at destination per day).
Run from project root:  python src/merge_data.py
Output: data/processed/flights_merged.parquet
"""
import pandas as pd

def main():
    flights = pd.read_parquet("data/processed/flights_clean.parquet")
    weather = pd.read_parquet("data/raw/weather_2015.parquet")

    # date column from YEAR/MONTH/DAY
    flights["DATE"] = pd.to_datetime(
        flights.rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"})[["year", "month", "day"]])
    flights["SCHED_HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100   # 1435 -> 14

    # ---- airport traffic features (groupby = "count rows per group") ----
    dep_day  = flights.groupby(["ORIGIN_AIRPORT", "DATE"]).size().rename("origin_dep_per_day")
    dep_hour = flights.groupby(["ORIGIN_AIRPORT", "DATE", "SCHED_HOUR"]).size().rename("origin_dep_per_hour")
    arr_day  = flights.groupby(["DESTINATION_AIRPORT", "DATE"]).size().rename("dest_arr_per_day")
    flights = flights.merge(dep_day,  on=["ORIGIN_AIRPORT", "DATE"], how="left")
    flights = flights.merge(dep_hour, on=["ORIGIN_AIRPORT", "DATE", "SCHED_HOUR"], how="left")
    flights = flights.merge(arr_day,  on=["DESTINATION_AIRPORT", "DATE"], how="left")

    # ---- weather merge on (origin airport, date) ----
    flights = flights.merge(weather, on=["ORIGIN_AIRPORT", "DATE"], how="left")

    # airports outside top-30: fill weather with overall median + flag
    wcols = [c for c in weather.columns if c not in ("ORIGIN_AIRPORT", "DATE")]
    flights["has_weather"] = flights[wcols[0]].notna().astype(int)
    for c in wcols:
        flights[c] = flights[c].fillna(flights[c].median())

    out = "data/processed/flights_merged.parquet"
    flights.to_parquet(out, index=False)
    print("Saved ->", out, " shape =", flights.shape)
    print("columns:", list(flights.columns))
    print("weather coverage (share with real weather):", round(flights["has_weather"].mean(), 3))

if __name__ == "__main__":
    main()