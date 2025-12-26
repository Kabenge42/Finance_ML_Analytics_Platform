#!/usr/bin/env python3
"""Compare CSV columns with SQL schema to identify mismatches."""

# Read CSV header
with open('data/screening_us.csv', 'r', encoding='utf-8') as f:
    csv_header = f.readline().strip()

csv_cols = [col.strip() for col in csv_header.split(',')]

print(f"CSV has {len(csv_cols)} columns\n")
print("=" * 80)
print("CSV COLUMN ORDER (first 50):")
print("=" * 80)
for i, col in enumerate(csv_cols[:50], 1):
    print(f"{i:3d}: {col}")

print("\n" + "=" * 80)
print("CSV COLUMN ORDER (columns 51-100):")
print("=" * 80)
for i, col in enumerate(csv_cols[50:100], 51):
    print(f"{i:3d}: {col}")

print("\n" + "=" * 80)
print("ALL CSV COLUMNS (for reference):")
print("=" * 80)
for i, col in enumerate(csv_cols, 1):
    print(f"{i:3d}: {col}")
