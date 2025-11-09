#!/usr/bin/env python3
"""Quick analysis of prediction coverage."""

import csv

csv_path = "outputs/regression/regression_predictions_full.csv"

total_rows = 0
non_null_preds = 0
null_preds = 0

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        y_pred = row.get("y_pred", "").strip()
        if y_pred and y_pred != "":
            non_null_preds += 1
        else:
            null_preds += 1

print(f"Total stocks: {total_rows}")
print(f"Non-null predictions: {non_null_preds}")
print(f"Null predictions: {null_preds}")
print(f"Coverage: {non_null_preds/total_rows*100:.1f}%")

if null_preds > 0:
    print(f"\n⚠ Found {null_preds} stocks WITHOUT predictions")
    print("This indicates a data flow problem in the pipeline.")
