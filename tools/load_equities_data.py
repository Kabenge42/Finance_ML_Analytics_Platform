"""
Script to load equities data from CSV files into PostgreSQL database.
Handles data type conversion and cleaning for the equities table.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


# Constants for column classification (optimized for fast lookup)
DATE_COLUMNS = frozenset(['Last Updated', 'Income Statement Report Date', 'Next Earnings'])
TEXT_COLUMNS = frozenset([
    'Ticker', 'ISIN', 'Name', 'Description', 'Exchange', 'Unit',
    'Sector', 'Industry', 'Style Class', 'Next Earnings (Status)',
    'Size Class', 'Region', 'Country', 'Trading Country'
])


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


def _prepare_records_for_insert(df: pd.DataFrame) -> Tuple[List[str], List[tuple]]:
    """
    Prepare records from DataFrame with optimized type conversion.

    Segregates data transformation logic and optimizes by mapping columns
    to converter functions once (pre-loop) instead of checking column names
    against lists for every cell value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with equity data

    Returns
    -------
    columns : List[str]
        List of column names
    records : List[tuple]
        List of tuples, each representing a row with converted values
    """
    columns = df.columns.tolist()
    converters = []

    # Map columns to converters once to avoid repetitive checks
    for col in columns:
        if col in DATE_COLUMNS:
            converters.append(convert_to_date)
        elif col in TEXT_COLUMNS:
            converters.append(convert_to_text)
        else:
            converters.append(convert_to_numeric)

    records = []
    for _, row in df.iterrows():
        # Apply converters by position
        record = tuple(converter(row[col]) for col, converter in zip(columns, converters))
        records.append(record)

    return columns, records


def _execute_db_insert(conn: psycopg2.extensions.connection, columns: List[str], records: List[tuple]) -> None:
    """
    Execute batch insert into PostgreSQL with proper transaction management.

    Isolates database interaction logic (SQL query construction, transaction
    management, and batch execution) following Single Responsibility Principle.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Active database connection
    columns : List[str]
        List of column names to insert
    records : List[tuple]
        List of tuples containing row data

    Raises
    ------
    Exception
        If insert operation fails, rolls back transaction and re-raises
    """
    # Create INSERT query using psycopg2.sql safely
    insert_query = sql.SQL(
        "INSERT INTO equities ({}) VALUES %s ON CONFLICT DO NOTHING"
    ).format(
        sql.SQL(', ').join([sql.Identifier(col) for col in columns])
    )

    # Create placeholders template
    placeholders = sql.SQL('({})').format(
        sql.SQL(', ').join([sql.Placeholder()] * len(columns))
    )

    cur = conn.cursor()
    try:
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


def load_csv_to_postgres(csv_file: str, conn: psycopg2.extensions.connection) -> None:
    """
    Load a single CSV file into the equities table.

    High-level orchestrator function that coordinates data loading,
    preparation, and insertion following the Extract Method pattern.

    Parameters
    ----------
    csv_file : str
        Path to CSV file to load
    conn : psycopg2.extensions.connection
        Active database connection

    Raises
    ------
    Exception
        If CSV loading or database insertion fails
    """
    print(f"\nProcessing {csv_file}...")

    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")

    # Prepare data and execute insert
    columns, records = _prepare_records_for_insert(df)
    _execute_db_insert(conn, columns, records)


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
