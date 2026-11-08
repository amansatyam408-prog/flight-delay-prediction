import pandas as pd

flights = pd.read_csv("data/raw/flights.csv", dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str})
airlines = pd.read_csv("data/raw/airlines.csv")
airports = pd.read_csv("data/raw/airports.csv")

print("flights  shape:", flights.shape)
print("airlines shape:", airlines.shape)
print("airports shape:", airports.shape)
print("\nflights columns:", list(flights.columns))
print("\nflights dtypes:\n", flights.dtypes)
print("\nfirst 3 rows:\n", flights.head(3))
print("\nmissing values (top 10):\n", flights.isnull().sum().sort_values(ascending=False).head(10))
print("\nARRIVAL_DELAY stats:\n", flights["ARRIVAL_DELAY"].describe())