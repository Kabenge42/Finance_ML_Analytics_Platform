#!/usr/bin/env python3
"""
Lightweight SQLite CSV importer with chunked loading.

Usage examples (from project root):
- python tools/import_sqlite.py --db equities.sqlite --data-dir data --chunksize 2000
- python tools/import_sqlite.py --db equities.sqlite --regions US,EU

Notes:
- Requires the SQLite schema to be created first:
  sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"
- This script inserts with INSERT OR IGNORE to honor the UNIQUE("Ticker","Region") index.
- Empty strings are converted to NULL. Region is backfilled per file if missing.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Optional progress bar; falls back to no-op if tqdm isn't installed
try:
    from tqdm import tqdm  # type: ignore
except Exception:

    def tqdm(iterable, **kwargs):
        return iterable


REGION_TO_FILE = {
    "US": "screening_us.csv",
    "EU": "screening_eu.csv",
    "APAC": "screening_apac.csv",
    "ROTW": "screening_rotw.csv",
}


def quote_identifier(name: str) -> str:
    """Quote a SQL identifier with double quotes, escaping internal quotes."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def iter_regions(selection: Optional[str]) -> List[str]:
    if not selection:
        return list(REGION_TO_FILE.keys())
    parts = [p.strip().upper() for p in selection.split(",") if p.strip()]
    # Validate and preserve order as given by user, but keep only known keys
    valid = [p for p in parts if p in REGION_TO_FILE]
    if not valid:
        raise ValueError(f"No valid regions in {selection!r}. Valid: {list(REGION_TO_FILE)}")
    return valid


def chunk_insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert a DataFrame into the equities table using executemany with INSERT OR IGNORE.

    Returns number of rows attempted (for logging). Uses the DataFrame's column order.
    """
    # Ensure column order is exactly as in the CSV/file
    columns = list(df.columns)
    # Convert empty strings to None (NULL in SQLite) without making a full copy
    df.replace({"": None}, inplace=True)

    # Build SQL
    cols_sql = ", ".join(quote_identifier(c) for c in columns)
    placeholders = ",".join(["?"] * len(columns))
    sql = f"INSERT OR IGNORE INTO equities ({cols_sql}) VALUES ({placeholders})"

    # Prepare data as list of tuples using vectorized access for performance
    records = [tuple(row) for row in df.values]
    if not records:
        return 0

    try:
        with conn:
            conn.executemany(sql, records)
    except sqlite3.Error as e:
        # Provide context for debugging; re-raise to allow caller handling
        raise sqlite3.Error(f"executemany failed for {len(records)} records into equities: {e}")
    return len(records)


def import_region(
    conn: sqlite3.Connection,
    data_dir: Path,
    region: str,
    chunksize: int,
) -> int:
    """Import one region's CSV into SQLite in chunks. Returns attempted row count."""
    csv_path = data_dir / REGION_TO_FILE[region]
    if not csv_path.exists():
        print(f"[WARN] CSV not found for region {region}: {csv_path}")
        return 0

    total = 0
    try:
        # dtype=str preserves original strings; na_filter=False keeps empty strings
        chunks = pd.read_csv(
            csv_path, dtype=str, chunksize=chunksize, encoding="utf-8", na_filter=False
        )
        for chunk in tqdm(chunks, desc=f"Importing {region}", unit="chunk"):
            # Ensure Region is correctly set/backfilled
            if "Region" in chunk.columns:
                # Replace empty strings with None then fill with region
                chunk["Region"] = chunk["Region"].replace({"": None}).fillna(region)
            else:
                chunk["Region"] = region
            try:
                total += chunk_insert_dataframe(conn, chunk)
            except sqlite3.Error as e:
                print(f"[ERROR] Database error inserting chunk for {region}: {e}")
                raise
    except pd.errors.ParserError as e:
        print(f"[ERROR] CSV parsing error for {region} at {csv_path}: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error importing {region}: {e}")
        raise
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked SQLite CSV importer")
    parser.add_argument("--db", default="equities.sqlite", help="Path to SQLite database file")
    parser.add_argument("--data-dir", default="data", help="Directory containing regional CSVs")
    parser.add_argument(
        "--regions", default=None, help="Comma-separated subset of regions: US,EU,APAC,ROTW"
    )
    parser.add_argument("--chunksize", type=int, default=2000, help="CSV read chunk size")
    args = parser.parse_args()

    db_path = Path(args.db)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return 2

    regions = iter_regions(args.regions)

    print(f"[INFO] Connecting to SQLite DB: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        # Fastest settings for bulk insert on local DB
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")

        grand_total = 0
        for r in regions:
            print(f"[INFO] Importing region {r}...")
            count = import_region(conn, data_dir, r, args.chunksize)
            grand_total += count
            print(f"[INFO]   Attempted rows inserted for {r}: {count}")

        print(f"[INFO] Done. Total attempted rows inserted: {grand_total}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
