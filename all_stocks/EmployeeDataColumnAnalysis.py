import csv
from pathlib import Path
from collections import OrderedDict

DATA_DIR = Path("data")
FILES = [
    "screening_us.csv",
    "screening_eu.csv",
    "screening_apac.csv",
    "screening_rotw.csv",
]


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


headers = {}
for fn in FILES:
    p = DATA_DIR / fn
    headers[fn] = read_header(p)

# Baseline: US (you can change this)
baseline_file = "screening_us.csv"
baseline = headers[baseline_file]

baseline_set = set(baseline)

print(f"Baseline: {baseline_file} -> {len(baseline)} columns\n")

# 1) New/removed columns per file
for fn, cols in headers.items():
    s = set(cols)
    added = [c for c in cols if c not in baseline_set]
    removed = [c for c in baseline if c not in s]
    print(f"== {fn} ==")
    print(f"  columns: {len(cols)}")
    print(f"  added vs {baseline_file}: {len(added)}")
    for c in added[:25]:
        print(f"    + {c}")
    if len(added) > 25:
        print(f"    ... +{len(added) - 25} more")
    print(f"  removed vs {baseline_file}: {len(removed)}")
    for c in removed[:25]:
        print(f"    - {c}")
    if len(removed) > 25:
        print(f"    ... -{len(removed) - 25} more")
    print()


# 2) Order drift: same columns but different order
def order_drift(a: list[str], b: list[str]) -> list[tuple[str, int, int]]:
    # only compare columns that exist in both
    idx_a = {c: i for i, c in enumerate(a)}
    idx_b = {c: i for i, c in enumerate(b)}
    common = [c for c in a if c in idx_b]
    drift = []
    for c in common:
        if idx_a[c] != idx_b[c]:
            drift.append((c, idx_a[c], idx_b[c]))
    return drift


for fn, cols in headers.items():
    if fn == baseline_file:
        continue
    drift = order_drift(baseline, cols)
    print(f"Order drift vs {baseline_file} for {fn}: {len(drift)} columns")
    for c, i0, i1 in drift[:30]:
        print(f"  {c!r}: {i0} -> {i1}")
    if len(drift) > 30:
        print(f"  ... {len(drift) - 30} more")
    print()

# 3) Highlight employee-related columns (your example)
keywords = [
    "employee",
    "employees",
    "full time",
    "full-time",
    "fte",
    "headcount",
    "staff",
]


def find_cols(cols: list[str]) -> list[str]:
    low = [(c, c.lower()) for c in cols]
    out = []
    for c, cl in low:
        if any(k in cl for k in keywords):
            out.append(c)
    return out


print("Employee-related columns by file:")
for fn, cols in headers.items():
    emp = find_cols(cols)
    print(f"- {fn}: {emp}")
