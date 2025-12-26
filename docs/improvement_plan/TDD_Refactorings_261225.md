# Comprehensive Codebase Analysis: Project Structure Reorganization & Schema Alignment

## Executive Summary

Based on examination of the codebase against the PostgreSQL `postgres.public.equities` schema and the Python
`COLUMN_SCHEMA`, I've identified several areas requiring reorganization, synchronization, and modularization.

---

## 1. Schema Alignment Analysis

### 1.1 Database Schema vs Python COLUMN_SCHEMA Discrepancies

| PostgreSQL `equities` Column         | Python Normalized Name             | Issue                       |
|--------------------------------------|------------------------------------|-----------------------------|
| `Trading Country`                    | `trading_country`                  | ✅ Aligned                   |
| `P/E (NTM)`                          | `p_e_ntm`                          | ✅ Aligned                   |
| `EPS GAAP Est Avg Rev % (FY1E - 1M)` | `eps_gaap_est_avg_rev_pct_fy1e_1m` | ⚠️ Complex normalization    |
| `Full Time Employees (FQ)`           | `full_time_employees_fq`           | ⚠️ Missing in COLUMN_SCHEMA |

**Recommended Action**: Create automated schema synchronization tool.

### 1.2 SQL-to-Python Column Mapping Gaps

The SQL schema has **296 columns** while `COLUMN_SCHEMA` has **555 entries** (including derived columns). Key
discrepancies:

```
Missing from COLUMN_SCHEMA (sampled):
- eps_gaap_est_avg_ntm
- eps_gaap_est_avg_fy1e
- ebitda_est_avg_fy1e
- several FTE (full-time employee) columns
```

---

## 2. Project Structure Reorganization

### 2.1 Current Structure Issues

```
Current (Fragmented):
├── finance_ml/
│   ├── ml_workflow/
│   │   ├── data/schema.py          ← COLUMN_SCHEMA definition
│   │   ├── eda/phase93_categories.py  ← DUPLICATE categories
│   │   ├── preprocessing/etl.py    ← Schema validation logic
│   │   └── features/advanced.py    ← Feature generators
│   └── dashboards/earnings_widgets.py ← CATEGORY_COLORS duplicate
├── import_equities_data.sql        ← SQL aliases (should be auto-generated)
├── create_equities_schema.sql      ← Manual schema (should derive from COLUMN_SCHEMA)
└── [20+ scattered test files at root level]
```

### 2.2 Proposed Reorganized Structure

```
Proposed (Unified):
├── finance_ml/
│   ├── core/                       ← NEW: Core abstractions
│   │   ├── __init__.py
│   │   ├── schema.py               ← Consolidated schema (single source of truth)
│   │   ├── constants.py            ← All constants (colors, thresholds)
│   │   └── types.py                ← Type definitions (DType, Role, PresetName)
│   │
│   ├── data/                       ← NEW: Data layer (unified)
│   │   ├── __init__.py
│   │   ├── loaders.py              ← CSV/DB loaders
│   │   ├── catalog.py              ← Data catalog
│   │   ├── versioning.py           ← Schema versioning
│   │   └── sql/                    ← AUTO-GENERATED SQL
│   │       ├── create_equities_schema.sql
│   │       └── import_equities_data.sql
│   │
│   ├── etl/                        ← NEW: ETL subpackage
│   │   ├── __init__.py
│   │   ├── pipeline.py             ← ETLPipeline class
│   │   ├── config.py               ← ETLConfig, all config dataclasses
│   │   ├── stages/                 ← Stage modules
│   │   │   ├── extraction.py
│   │   │   ├── dtype_casting.py
│   │   │   ├── imputation.py
│   │   │   ├── transformation.py
│   │   │   └── validation.py
│   │   └── metrics.py              ← ETLMetrics
│   │
│   ├── features/                   ← Refactored features
│   │   ├── __init__.py
│   │   ├── api.py                  ← build_features entry point
│   │   ├── core.py                 ← Basic feature functions
│   │   ├── advanced/               ← Split advanced.py (currently 4000+ lines)
│   │   │   ├── __init__.py
│   │   │   ├── valuation.py
│   │   │   ├── profitability.py
│   │   │   ├── momentum.py
│   │   │   ├── quality.py
│   │   │   ├── earnings.py         ← GAAP vs adjusted analytics
│   │   │   └── employment.py
│   │   └── registry.py             ← Feature category registry
│   │
│   ├── dashboards/                 ← Cleaner dashboard structure
│   │   ├── __init__.py
│   │   ├── config.py               ← PLOTLY_TEMPLATE, CATEGORY_COLORS (import from core)
│   │   ├── widgets/
│   │   │   ├── earnings.py
│   │   │   ├── portfolio.py
│   │   │   └── correlation.py
│   │   └── apps/
│   │       ├── dash_app.py
│   │       └── streamlit_app.py
│   │
│   └── ml_workflow/                ← Slimmed ML workflow
│       ├── eda/
│       ├── classification/
│       ├── regression/
│       └── evaluation/
│
├── scripts/                        ← NEW: Utility scripts
│   ├── generate_sql_schema.py      ← Auto-generate SQL from COLUMN_SCHEMA
│   ├── validate_schema_sync.py     ← CI check for schema alignment
│   └── analyze_imputation.py
│
├── tests/                          ← Consolidated tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── [notebooks at root - OK]
```

---

## 3. Refactoring Tasks

### Task 1: Create Unified Schema Module

**File**: `finance_ml/core/schema.py`

```python
"""
Unified schema module - Single Source of Truth.

This module is the ONLY place where column definitions exist.
All other modules MUST import from here.
"""

from __future__ import annotations
from typing import Dict, List, Literal, TypedDict, Set
from dataclasses import dataclass

# Type definitions
DType = Literal["float", "int", "string", "category", "datetime64[ns]", "bool"]
Role = Literal["id", "target", "target_fallback", "date", "categorical",
"auxiliary", "feature", "price", "market_value", "ratio",
"percentage", "count", "label"]


class ColumnMeta(TypedDict):
    dtype: DType
    role: Role
    sql_name: str  # Original SQL column name
    description: str  # Column description


# Master schema - auto-generates SQL
COLUMN_SCHEMA: Dict[str, ColumnMeta] = {
    "ticker": {
        "dtype": "string",
        "role": "id",
        "sql_name": "Ticker",
        "description": "Stock ticker symbol"
    },
    # ... all columns
}

# Phase 9.3 categories - derived from COLUMN_SCHEMA
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
    # Auto-populated from feature generators
}

# Visualization constants
CATEGORY_COLORS: Dict[str, str] = {
    "Momentum & Technical": "#3498db",
    "Valuation Ratios": "#375a7f",
    # ... all categories
}


def get_sql_column_name(normalized_name: str) -> str:
    """Get original SQL column name from normalized Python name."""
    meta = COLUMN_SCHEMA.get(normalized_name)
    return meta["sql_name"] if meta else normalized_name


def generate_sql_schema() -> str:
    """Generate CREATE TABLE statement from COLUMN_SCHEMA."""
    lines = ["CREATE TABLE IF NOT EXISTS equities ("]
    for col_name, meta in COLUMN_SCHEMA.items():
        sql_name = meta["sql_name"]
        sql_type = {
            "float": "NUMERIC",
            "int": "INTEGER",
            "string": "TEXT",
            "category": "TEXT",
            "datetime64[ns]": "DATE",
            "bool": "BOOLEAN"
        }[meta["dtype"]]
        lines.append(f'  "{sql_name}" {sql_type},')
    lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
    lines.append(");")
    return "\n".join(lines)
```

### Task 2: Split `advanced.py` into Modular Components

**Current**: `advanced.py` is 4000+ lines with 35+ functions.

**Proposed Split**:

```python
# finance_ml/features/advanced/__init__.py
"""
Advanced feature engineering - modular implementation.
"""
from .valuation import (
    engineer_valuation_ratios,
    engineer_valuation_timeseries_features,
)
from .profitability import (
    engineer_profitability_ratios,
    engineer_margin_trends,
)
from .momentum import (
    engineer_momentum_features,
    engineer_technical_analysis_features,
)
from .quality import (
    engineer_accounting_quality_features,
    engineer_financial_distress_features,
    engineer_composite_scores,
)
from .earnings import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics,
)
from .employment import (
    engineer_employee_productivity_features,
    engineer_employment_dynamics_features,
)
from .comprehensive import build_comprehensive_features

__all__ = [
    "engineer_valuation_ratios",
    "engineer_valuation_timeseries_features",
    # ... all exports
    "build_comprehensive_features",
]
```

### Task 3: Extract ETL Config Classes

**Current**: `etl.py` has 13 dataclass configs mixed with pipeline logic (~3500 lines).

**Proposed**: `finance_ml/etl/config.py`

```python
"""ETL configuration dataclasses - extracted from etl.py."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Literal
import pandas as pd


@dataclass
class DataExtractionConfig:
    """Configuration for data extraction stage."""
    limit: Optional[int] = None
    normalize_column_names: bool = True
    source_type: Literal["csv", "database", "all_stocks"] = "csv"


@dataclass
class SchemaValidationConfig:
    """Configuration for schema validation."""
    validate_schema: bool = True
    require_target_column: bool = True
    drop_rows_with_missing_critical_fields: bool = False
    validate_schema_alignment: bool = True
    schema_alignment_threshold: float = 0.85


# ... all other config classes

@dataclass
class ETLConfig:
    """Master ETL configuration - combines all stage configs."""
    extraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
    validation: SchemaValidationConfig = field(default_factory=SchemaValidationConfig)

    # ... all stage configs

    @classmethod
    def for_production(cls) -> "ETLConfig":
        """Factory for production-ready configuration."""
        return cls(
            validation=SchemaValidationConfig(
                validate_schema=True,
                validate_schema_alignment=True,
                schema_alignment_threshold=0.95,
            ),
            # ... production settings
        )

    @classmethod
    def for_development(cls) -> "ETLConfig":
        """Factory for fast development iteration."""
        return cls(
            extraction=DataExtractionConfig(limit=1000),
            # ... dev settings
        )
```

### Task 4: Auto-Generate SQL from Schema

**File**: `scripts/generate_sql_schema.py`

```python
#!/usr/bin/env python
"""
Generate SQL schema files from COLUMN_SCHEMA.

This ensures SQL and Python schemas are always synchronized.
Run: python scripts/generate_sql_schema.py
"""

from pathlib import Path
from finance_ml.core.schema import COLUMN_SCHEMA, generate_sql_schema

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # Generate CREATE TABLE
    schema_sql = generate_sql_schema()
    schema_path = PROJECT_ROOT / "create_equities_schema.sql"
    schema_path.write_text(schema_sql)
    print(f"✓ Generated: {schema_path}")

    # Generate IMPORT statement
    import_sql = generate_import_sql()
    import_path = PROJECT_ROOT / "import_equities_data.sql"
    import_path.write_text(import_sql)
    print(f"✓ Generated: {import_path}")


def generate_import_sql() -> str:
    """Generate COPY FROM CSV statement with column mapping."""
    columns = [meta["sql_name"] for meta in COLUMN_SCHEMA.values()]
    return f"""
-- Auto-generated from COLUMN_SCHEMA
COPY equities ({', '.join(f'"{c}"' for c in columns)})
FROM '/path/to/data.csv'
WITH (FORMAT csv, HEADER true, NULL '');
"""


if __name__ == "__main__":
    main()
```

### Task 5: Eliminate Duplicate Definitions

**Issue**: `PHASE93_FEATURE_CATEGORIES` is defined in both:

- `finance_ml/ml_workflow/data/schema.py`
- `finance_ml/ml_workflow/eda/phase93_categories.py`

**Solution**: Single source in `schema.py`, re-export from `phase93_categories.py`:

```python
# finance_ml/ml_workflow/eda/phase93_categories.py
"""
Phase 9.3 Feature Category Registry - Re-exports from canonical source.
"""
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_CATEGORIES


# Only helper functions defined here
def get_feature_category(column_name: str) -> str | None:
    """Look up which category a feature belongs to."""
    for category, features in PHASE93_FEATURE_CATEGORIES.items():
        if column_name in features:
            return category
    return None

# ... other helper functions
```

### Task 6: Consolidate Root-Level Test Files

**Issue**: 20+ test files at project root instead of `tests/`.

```
Move to tests/unit/:
- test_zero_imputation_simple.py → tests/unit/test_imputation_zero.py
- test_datetime_fix.py → tests/unit/test_dtypes_datetime.py
- analyze_*.py → scripts/ (not tests)
```

### Task 7: Create Schema Sync Validation

**File**: `tests/integration/test_schema_sync.py`

```python
"""
Test that Python COLUMN_SCHEMA matches PostgreSQL equities table.

This test should be run in CI to catch schema drift.
"""
import pytest
from sqlalchemy import create_engine, inspect

from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name


@pytest.fixture
def db_connection():
    """Get database connection from environment."""
    import os
    url = os.getenv("DB_URL", "postgresql://localhost:5432/postgres")
    return create_engine(url)


def test_schema_columns_match_database(db_connection):
    """Verify all database columns are in COLUMN_SCHEMA."""
    inspector = inspect(db_connection)
    db_columns = {c["name"] for c in inspector.get_columns("equities")}

    schema_sql_names = {
        meta.get("sql_name", col)
        for col, meta in COLUMN_SCHEMA.items()
        if meta["role"] != "auxiliary"  # Exclude legacy aliases
    }

    missing_in_schema = db_columns - schema_sql_names
    assert not missing_in_schema, f"DB columns missing from COLUMN_SCHEMA: {missing_in_schema}"


def test_schema_alignment_score():
    """Ensure schema alignment meets minimum threshold."""
    from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline
    # ... validation logic
```

---

## 4. Migration Plan

### Phase 1: Schema Consolidation (Week 1)

1. Create `finance_ml/core/schema.py` with unified definitions
2. Update all imports to use new canonical source
3. Add deprecation warnings to old locations
4. Generate SQL files from Python schema

### Phase 2: ETL Modularization (Week 2)

1. Extract config dataclasses to `finance_ml/etl/config.py`
2. Split ETLPipeline stages into separate modules
3. Ensure backward compatibility via re-exports

### Phase 3: Feature Engineering Split (Week 3)

1. Split `advanced.py` into domain modules
2. Update `api.py` imports
3. Add feature registry for auto-discovery

### Phase 4: Test Consolidation (Week 4)

1. Move root test files to `tests/`
2. Consolidate duplicate fixtures
3. Add schema sync integration tests

### Phase 5: Dashboard Cleanup (Week 5)

1. Extract widget modules from `earnings_widgets.py`
2. Unify color/template constants
3. Remove duplicate CATEGORY_COLORS definitions

---

## 5. CI/CD Recommendations

```yaml
# .github/workflows/schema-sync.yml
name: Schema Synchronization Check

on: [ push, pull_request ]

jobs:
  schema-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: pip install -e .[dev]

      - name: Validate Schema Alignment
        run: python scripts/validate_schema_sync.py

      - name: Check SQL Generation
        run: |
          python scripts/generate_sql_schema.py
          git diff --exit-code create_equities_schema.sql
```

---

## Summary of Refactoring Tasks

| Priority  | Task                           | Files Affected      | Complexity |
|-----------|--------------------------------|---------------------|------------|
| 🔴 High   | Unified schema module          | 15+ imports         | Medium     |
| 🔴 High   | Eliminate duplicate categories | 3 files             | Low        |
| 🟡 Medium | Split advanced.py              | 1 → 7 files         | High       |
| 🟡 Medium | Extract ETL configs            | etl.py → config.py  | Medium     |
| 🟢 Low    | Auto-generate SQL              | New script          | Low        |
| 🟢 Low    | Consolidate root tests         | 20+ files → tests/  | Low        |
| 🟢 Low    | Dashboard widget split         | earnings_widgets.py | Medium     |

This reorganization will:

1. ✅ Eliminate "unknown column" warnings
2. ✅ Ensure SQL ↔ Python schema synchronization
3. ✅ Improve maintainability with smaller modules
4. ✅ Enable automated schema drift detection
5. ✅ Reduce code duplication across the codebase