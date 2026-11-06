"""
src/feature_engineering.py — STEP 5: Feature Engineering & Selection
Run from project root:  python src/feature_engineering.py
Outputs: data/processed/train.parquet, data/processed/test.parquet
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MAX_ROWS = None  # set to 1_000_000 for a quick run on a slow laptop

TARGET_CLF = "DELAYED_15"      # classification target (1 = delay > 15 min)
TARGET_REG = "ARRIVAL_DELAY"   # regression target (minutes)
CAT_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
DROP_COLS = ["YEAR", "TAIL_NUMBER", "FLIGHT_NUMBER", "DATE",
             "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", TARGET_CLF, TARGET_REG]

def engineer(df):
    df = df.copy()
    df["SCHED_MIN"] = df["SCHEDULED_DEPARTURE"] % 100      # 1435 -> 35
    df["SCHED_ARR_HOUR"] = df["SCHEDULED_ARRIVAL"] // 100   # 2210 -> 22
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    return df

def main():
    df = pd.read_parquet("data/processed/flights_merged.parquet")
    if MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        print(f"Using a {MAX_ROWS:,}-row sample for a quick run")
    df = engineer(df)

    # Split FIRST (stratified) -> no leakage; 82/18 balance kept in both sets
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_CLF])
    print("train:", train.shape, " test:", test.shape)

    # Label-encode categoricals fitted on TRAIN only; unseen in test -> "unknown" code
    for col in CAT_COLS:
        le = LabelEncoder().fit(train[col].astype(str))
        mapping = {c: i for i, c in enumerate(le.classes_)}
        unk = len(mapping)
        train[col + "_enc"] = train[col].astype(str).map(mapping)
        test[col + "_enc"] = test[col].astype(str).map(mapping).fillna(unk).astype(int)
        print(f"  {col}: {len(mapping)} categories")

    feature_cols = [c for c in train.columns if c not in DROP_COLS + CAT_COLS]
    print(f"\nFinal feature list ({len(feature_cols)}):")
    print(feature_cols)

    keep = feature_cols + [TARGET_CLF, TARGET_REG]
    train[keep].to_parquet("data/processed/train.parquet", index=False)
    test[keep].to_parquet("data/processed/test.parquet", index=False)
    print("\nSaved data/processed/train.parquet", train[keep].shape)
    print("Saved data/processed/test.parquet ", test[keep].shape)

if __name__ == "__main__":
    main()