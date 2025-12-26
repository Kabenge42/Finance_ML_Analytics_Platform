import csv

with open('data/screening_us.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, col in enumerate(header):
        print(f"{i}: {col}")
