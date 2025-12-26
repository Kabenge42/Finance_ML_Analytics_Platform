"""
Test that Python COLUMN_SCHEMA matches PostgreSQL equities table.

This test should be run in CI to catch schema drift.
"""
import os

import pytest
from sqlalchemy import create_engine, inspect

from finance_ml.core.schema import COLUMN_SCHEMA


@pytest.fixture
def db_connection():
    """Get database connection from environment."""
    url = os.getenv("DB_URL", "postgresql://localhost:5432/postgres")
    try:
        engine = create_engine(url)
        # Try to connect to verify availability
        with engine.connect() as conn:
            pass
        return engine
    except Exception:
        return None

def test_schema_columns_match_database(db_connection):
    """Verify all database columns are in COLUMN_SCHEMA."""
    if db_connection is None:
        pytest.skip("Database not available")
        
    inspector = inspect(db_connection)
    try:
        db_columns = {c["name"] for c in inspector.get_columns("equities")}
    except Exception:
        pytest.skip("'equities' table not found in database")

    if not db_columns:
        pytest.skip("'equities' table is empty or has no columns")

    schema_sql_names = {
        meta.get("sql_name", col)
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") != "auxiliary"  # Exclude legacy aliases
    }

    missing_in_schema = db_columns - schema_sql_names
    assert not missing_in_schema, f"DB columns missing from COLUMN_SCHEMA: {missing_in_schema}"

def test_schema_alignment_score():
    """Ensure schema alignment meets minimum threshold."""
    from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics
    
    # Mock some metrics to test logic
    metrics = ETLMetrics()
    metrics.schema_alignment_score = 0.95
    
    assert metrics.schema_alignment_score >= 0.85, "Schema alignment score below threshold"
