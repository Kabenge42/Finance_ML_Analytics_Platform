import pandas as pd

# Load the Excel file
df = pd.read_excel(r"outputs\prediction_analyst_comparison_report.xlsx")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check for negative predictions
if "Predicted_Price_Target" in df.columns:
    neg_mask = df["Predicted_Price_Target"] < 0
    print(f"\nNegative predictions count: {neg_mask.sum()}")
    if neg_mask.sum() > 0:
        print(f'Min value: {df[neg_mask]["Predicted_Price_Target"].min()}')
        print(f'Max negative value: {df[neg_mask]["Predicted_Price_Target"].max()}')
        print(f"\nSample negative predictions:")
        print(
            df[neg_mask][
                ["Ticker", "Sector", "Last_Price", "Predicted_Price_Target", "Analyst_Price_Target"]
            ].head(10)
        )

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics for Predicted_Price_Target:")
if "Predicted_Price_Target" in df.columns:
    print(df["Predicted_Price_Target"].describe())
