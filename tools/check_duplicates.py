import pandas as pd

# Check duplicates in CSV data
df = pd.read_csv("data/screening_us.csv")
print(f"Total rows: {len(df)}")
print(f'Unique tickers: {df["Ticker"].nunique()}')
print(f'Duplicate tickers: {len(df) - df["Ticker"].nunique()}')

# Check what happens with load_from_csv
from finance_ml import data
from pathlib import Path

all_stocks = data.load_from_csv(Path("data"), limit=500)
print(f"\nAfter load_from_csv:")
print(f"Total rows: {len(all_stocks)}")
if "ticker" in all_stocks.columns:
    print(f'Unique tickers: {all_stocks["ticker"].nunique()}')
    print(f'Duplicate tickers: {len(all_stocks) - all_stocks["ticker"].nunique()}')

# Check what preprocess does
all_stocks_normalized = data.normalize_columns(all_stocks)
print(f"\nAfter normalize_columns:")
print(f"Total rows: {len(all_stocks_normalized)}")

all_stocks_preprocessed = data.preprocess(all_stocks_normalized)
print(f"\nAfter preprocess:")
print(f"Total rows: {len(all_stocks_preprocessed)}")
print(f'Has ticker column: {"ticker" in all_stocks_preprocessed.columns}')
print(f"Columns in preprocessed: {len(all_stocks_preprocessed.columns)}")
if "ticker" in all_stocks_preprocessed.columns:
    print(f'Unique tickers: {all_stocks_preprocessed["ticker"].nunique()}')
    print(f'Ticker null count: {all_stocks_preprocessed["ticker"].isnull().sum()}')
    print(f'Ticker dtype: {all_stocks_preprocessed["ticker"].dtype}')
    print(f'Sample ticker values: {all_stocks_preprocessed["ticker"].head(10).tolist()}')
else:
    print("WARNING: ticker column was removed by preprocess()!")
    print("Available columns:", list(all_stocks_preprocessed.columns)[:20])
