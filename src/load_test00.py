import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
prices = pd.read_csv(ROOT / "data" / "raw" / "all_stocks_5yr.csv")
cons   = pd.read_csv(ROOT / "data" / "raw" / "constituents.csv")

print(prices.head())
print(prices.columns)
print(prices.shape)

print(cons.head())
print(cons.columns)
print(cons.shape)