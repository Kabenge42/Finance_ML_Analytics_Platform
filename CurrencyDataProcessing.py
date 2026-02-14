import os
import psycopg2

# --- Configuration ---
CSV_PATH = r"C:/Users/markm/PycharmProjects/Finance_Analytics_Platform/snb_data/snb-data-devkum.csv"

# Load DB_URL from environment or env file
if "DB_URL" not in os.environ:
    env_file = "environment_variables.txt"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

DB_URL = os.environ.get("DB_URL", "postgresql://user:password@localhost:5432/postgres")

# psycopg2 does not understand SQLAlchemy dialect suffixes like "+psycopg2"
if DB_URL.startswith("postgresql+"):
    DB_URL = "postgresql" + DB_URL[DB_URL.index("://") :]

conn = psycopg2.connect(DB_URL)
conn.autocommit = False
cur = conn.cursor()

try:
    # --- Recreate table ---
    cur.execute("DROP TABLE IF EXISTS currencies;")
    cur.execute("""
                CREATE TABLE currencies (
                                            "Date"           TEXT,
                                            D0               TEXT,
                                            D1               TEXT,
                                            currency         TEXT,
                                            unit             NUMERIC,
                                            "Value"          DOUBLE PRECISION,
                                            reference_date   DATE
                );
                """)

    # --- Create/replace helper functions ---
    cur.execute("""
                CREATE OR REPLACE FUNCTION parse_year_month_to_end_of_month(date_text TEXT)
                    RETURNS DATE AS $$
                DECLARE year_val INTEGER; month_val INTEGER;
                BEGIN
                    IF date_text IS NULL OR TRIM(date_text) = '' THEN RETURN NULL; END IF;
                    IF date_text !~ '^\\d{4}-\\d{2}$' THEN RETURN NULL; END IF;
                    year_val  := SPLIT_PART(date_text, '-', 1)::INTEGER;
                    month_val := SPLIT_PART(date_text, '-', 2)::INTEGER;
                    IF year_val < 1900 OR year_val > 2100 OR month_val < 1 OR month_val > 12 THEN RETURN NULL; END IF;
                    RETURN (MAKE_DATE(year_val, month_val, 1) + INTERVAL '1 month - 1 day')::DATE;
                END; $$ LANGUAGE plpgsql IMMUTABLE;
                """)

    cur.execute("""
                CREATE OR REPLACE FUNCTION extract_currency(d1_value TEXT)
                    RETURNS TEXT AS $$
                BEGIN RETURN REGEXP_REPLACE(TRIM(d1_value), '[0-9]+[A-Za-z]*$', ''); END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """)

    cur.execute("""
                CREATE OR REPLACE FUNCTION extract_unit(d1_value TEXT)
                    RETURNS NUMERIC AS $$
                DECLARE num_part TEXT;
                BEGIN
                    num_part := SUBSTRING(TRIM(d1_value) FROM '([0-9]+)');
                    IF num_part IS NULL OR num_part = '' THEN RETURN NULL; END IF;
                    RETURN num_part::NUMERIC;
                END; $$ LANGUAGE plpgsql IMMUTABLE;
                """)

    # --- Staging table + client-side COPY ---
    cur.execute(
        'CREATE TEMP TABLE currencies_staging ("Date" TEXT, D0 TEXT, D1 TEXT, "Value" TEXT);'
    )

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        cur.copy_expert(
            """COPY currencies_staging ("Date", D0, D1, "Value")
               FROM STDIN WITH (FORMAT CSV, DELIMITER ';', HEADER TRUE, QUOTE '"')""",
            f,
        )

    # --- Transform and insert ---
    cur.execute("""
                INSERT INTO currencies ("Date", D0, D1, currency, unit, "Value", reference_date)
                SELECT TRIM(s."Date"), TRIM(s.D0), TRIM(s.D1),
                       extract_currency(s.D1), extract_unit(s.D1),
                       NULLIF(TRIM(s."Value"), '')::DOUBLE PRECISION,
                       parse_year_month_to_end_of_month(TRIM(s."Date"))
                FROM currencies_staging s;
                """)

    cur.execute("DROP TABLE IF EXISTS currencies_staging;")
    conn.commit()

    # --- Verify ---
    cur.execute("""
                SELECT "Date", D0, D1, currency, unit, "Value", reference_date
                FROM currencies LIMIT 20;
                """)
    rows = cur.fetchall()
    print(f"✅ Imported successfully. Sample ({len(rows)} rows):")
    for r in rows:
        print(r)

except Exception as e:
    conn.rollback()
    print(f"❌ Error: {e}")
finally:
    cur.close()
    conn.close()
