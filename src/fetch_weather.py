"""
src/fetch_weather.py — STEP 4a: fetch REAL 2015 daily weather via Open-Meteo Archive API.
No API key needed. Run from project root:  python src/fetch_weather.py
Output: data/raw/weather_2015.parquet
"""
import os, time
import pandas as pd
import requests

TOP_N_AIRPORTS = 30
OUT = "data/raw/weather_2015.parquet"
API = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
              "snowfall_sum", "wind_speed_10m_max", "wind_gusts_10m_max", "weather_code"]

def fetch_airport_weather(iata, lat, lon):
    params = {"latitude": lat, "longitude": lon, "start_date": "2015-01-01",
              "end_date": "2015-12-31", "daily": ",".join(DAILY_VARS), "timezone": "UTC"}
    for attempt in range(3):                      # retry on network errors
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()                  # raise if HTTP error
            d = r.json()["daily"]                 # JSON -> dict -> daily table
            df = pd.DataFrame(d)
            df["DATE"] = pd.to_datetime(df["time"])
            df["ORIGIN_AIRPORT"] = iata
            return df.drop(columns=["time"])
        except Exception as e:
            print(f"    retry {attempt+1}/3 for {iata}: {e}")
            time.sleep(2)
    return None

def main():
    airports = pd.read_parquet("data/processed/airports.parquet")
    flights = pd.read_parquet("data/processed/flights_clean.parquet", columns=["ORIGIN_AIRPORT"])
    top = flights["ORIGIN_AIRPORT"].value_counts().head(TOP_N_AIRPORTS).index.tolist()
    print(f"Fetching 2015 daily weather for top {len(top)} origin airports...")
    frames = []
    for i, iata in enumerate(top, 1):
        row = airports[airports["IATA_CODE"] == iata]
        if row.empty:
            print(f"  [{i}/{len(top)}] {iata}: no lat/lon, skipped"); continue
        lat, lon = float(row.iloc[0]["LATITUDE"]), float(row.iloc[0]["LONGITUDE"])
        w = fetch_airport_weather(iata, lat, lon)
        if w is not None:
            frames.append(w)
            print(f"  [{i}/{len(top)}] {iata}: {len(w)} days OK")
        time.sleep(0.2)                              # be polite to the free API
    weather = pd.concat(frames, ignore_index=True)
    os.makedirs("data/raw", exist_ok=True)
    weather.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}  shape={weather.shape}")

if __name__ == "__main__":
    main()