"""
Script to load equities data from CSV files into PostgreSQL database.
Handles data type conversion and cleaning for the equities table.
"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


def convert_to_date(value):
    """Convert string date to PostgreSQL DATE format."""
    if pd.isna(value) or value == '':
        return None
    try:
        # Try parsing common date formats
        for fmt in ['%b-%d-%Y', '%Y-%m-%d', '%m/%d/%Y']:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None
    except:
        return None


def convert_to_numeric(value):
    """Convert string to numeric, handling empty strings and special cases."""
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def convert_to_text(value):
    """Convert to text, handling None and empty strings."""
    if pd.isna(value) or value == '':
        return None
    return str(value).strip()


def load_csv_to_postgres(csv_file, conn):
    """Load a single CSV file into the equities table."""
    print(f"\nProcessing {csv_file}...")

    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")

    # Get column names from the CSV
    columns = df.columns.tolist()

    # Prepare data for insertion
    records = []

    for idx, row in df.iterrows():
        record = []
        for col in columns:
            value = row[col]

            # Date columns
            if col in ['Last Updated', 'Income Statement Report Date', 'Next Earnings']:
                record.append(convert_to_date(value))
            # Text columns (first 18 columns are mostly text)
            elif col in ['Ticker', 'ISIN', 'Name', 'Description', 'Exchange', 'Unit',
                        'Sector', 'Industry', 'Style Class', 'Next Earnings (Status)',
                        'Size Class', 'Flag', 'Region', 'Country', 'Trading Country']:
                record.append(convert_to_text(value))
            # All other columns are NUMERIC
            else:
                record.append(convert_to_numeric(value))

        records.append(tuple(record))

    # Create INSERT query using psycopg2.sql to safely handle special characters in column names
    # This prevents issues with columns containing /, %, (, ), etc.
    insert_query = sql.SQL(
        "INSERT INTO equities ({}) VALUES %s ON CONFLICT DO NOTHING"
    ).format(
        sql.SQL(', ').join([sql.Identifier(col) for col in columns])
    )

    # Create placeholders template for execute_values
    placeholders = sql.SQL('({})').format(
        sql.SQL(', ').join([sql.Placeholder()] * len(columns))
    )

    # Insert data in batches
    cur = conn.cursor()
    try:
        # Convert SQL objects to strings for execute_values
        execute_values(
            cur,
            insert_query.as_string(conn),
            records,
            template=placeholders.as_string(conn),
            page_size=1000
        )
        conn.commit()
        print(f"  Inserted {len(records)} records")
    except Exception as e:
        conn.rollback()
        print(f"  Error inserting records: {e}")
        raise
    finally:
        cur.close()


def main():
    """Main function to load all CSV files."""
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': 'bItcfiTg142!'  # Update with your password
    }

    # Determine project root directory
    # Script is in tools/ subdirectory, so project root is parent directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'

    # CSV files to load (using absolute paths)
    csv_files = [
        data_dir / 'screening_us.csv',
        data_dir / 'screening_eu.csv',
        data_dir / 'screening_apac.csv',
        data_dir / 'screening_rotw.csv'
    ]

    # Connect to database
    try:
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        print("Connected successfully!")

        # Load each CSV file
        total_before = 0
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM equities")
        total_before = cur.fetchone()[0]
        cur.close()
        print(f"\nRecords in database before import: {total_before}")

        for csv_file in csv_files:
            if csv_file.exists():
                load_csv_to_postgres(str(csv_file), conn)
            else:
                print(f"\nWarning: {csv_file} not found, skipping...")

        # Show final count
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM equities")
        total_after = cur.fetchone()[0]
        cur.close()

        print(f"\n{'='*60}")
        print(f"Records in database after import: {total_after}")
        print(f"New records added: {total_after - total_before}")
        print(f"{'='*60}")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
