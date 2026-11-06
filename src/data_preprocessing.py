"""
src/data_preprocessing.py  —  STEP 3: Data Cleaning & Preprocessing
Run from the PROJECT ROOT:  python src/data_preprocessing.py
"""
import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Columns only known AFTER the flight -> using them as inputs = DATA LEAKAGE.
# (ARRIVAL_DELAY is kept ONLY to build the targets, never as an input feature.)
POST_FLIGHT_COLUMNS = [
    "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
    "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
CONSTANT_COLUMNS = ["CANCELLED", "DIVERTED", "CANCELLATION_REASON"]  # constant after filtering

def load_raw_data():
    # low_memory=False + dtype=str fixes the mixed-type warning from Step 2
    flights = pd.read_csv(
        os.path.join(RAW_DIR, "flights.csv"),
        low_memory=False,
        dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str, "TAIL_NUMBER": str},
    )
    airlines = pd.read_csv(os.path.join(RAW_DIR, "airlines.csv"))
    airports = pd.read_csv(os.path.join(RAW_DIR, "airports.csv"), dtype={"IATA_CODE": str})
    print("Loaded raw shapes:", flights.shape, airlines.shape, airports.shape)
    return flights, airlines, airports

def clean_flights(flights):
    df = flights.copy()
    # 1) remove cancelled & diverted flights (no arrival delay to predict)
    before = len(df)
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    print(f"Removed {before - len(df):,} cancelled/diverted flights")
    # 2) drop rows where the target is missing
    before = len(df)
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    print(f"Dropped {before - len(df):,} rows with missing ARRIVAL_DELAY")
    # 3) targets: classification label + regression target (ARRIVAL_DELAY kept)
    df["DELAYED_15"] = (df["ARRIVAL_DELAY"] > 15).astype(int)
    # 4) drop leakage + constant columns
    drop_cols = [c for c in POST_FLIGHT_COLUMNS + CONSTANT_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} post-flight/constant columns (leakage protection)")
    return df.reset_index(drop=True)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    flights, airlines, airports = load_raw_data()
    clean = clean_flights(flights)
    out = os.path.join(PROCESSED_DIR, "flights_clean.parquet")
    clean.to_parquet(out, index=False)
    print(f"Saved -> {out}  shape={clean.shape}")
    sample = clean.sample(n=500_000, random_state=42)   # for fast notebook experiments
    sample.to_parquet(os.path.join(PROCESSED_DIR, "flights_sample_500k.parquet"), index=False)
    print(f"Saved -> data/processed/flights_sample_500k.parquet  shape={sample.shape}")
    airlines.to_parquet(os.path.join(PROCESSED_DIR, "airlines.parquet"), index=False)
    airports.to_parquet(os.path.join(PROCESSED_DIR, "airports.parquet"), index=False)
    print("\nClass balance of DELAYED_15 (1 = delayed > 15 min):")
    print(clean["DELAYED_15"].value_counts(normalize=True).round(4))

if __name__ == "__main__":
    main()