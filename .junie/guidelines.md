Finance ML Analytics Platform — Development Guidelines

TL;DR (Quick Start)

- Python: 3.10, 3.11, or 3.12 recommended.
- Create venv: python -m venv .venv && .venv\Scripts\activate
- Upgrade pip: python -m pip install --upgrade pip setuptools wheel
- Install deps: pip install -r requirements.txt
- PostgreSQL: install and start local Postgres; run create_equities_schema.sql to create the equities table.
- Data: Import the CSVs from data/ into the equities table, tagging each by Region.
- Notebook: open ml_finance_model_main.ipynb and run cells in order.
- Tests: python -m unittest -v

1) Build and Configuration Instructions
A. Prerequisites
- OS: Windows 10/11 (tested), macOS, or Linux.
- Python: 3.10, 3.11, or 3.12. Use pyenv or the official installer. Avoid mixing Conda with venv in the same project.
- Git: optional but recommended for version control.
- PostgreSQL: 15+ recommended. Local DB will be used for data loading. JDBC URL in this project: jdbc:postgresql://localhost:5432/postgres
- JDK (optional): Only if you use JVM tools or JDBC from Java-based tooling.
- Jupyter: installed with pip (notebook or jupyterlab) if you’ll run the .ipynb.

B. Virtual environment setup
- Create and activate a virtual environment:
  - Windows (PowerShell):
    - python -m venv .venv
    - .venv\Scripts\Activate.ps1
  - macOS/Linux (bash):
    - python3 -m venv .venv
    - source .venv/bin/activate
- Upgrade packaging tools:
  - python -m pip install --upgrade pip setuptools wheel

C. Install Python dependencies
- Install all packages:
  - pip install -r requirements.txt
- Notes:
  - TensorFlow is heavy; CPU-only install is fine for this project. If you have an NVIDIA GPU and want acceleration, follow the official TensorFlow GPU install docs and ensure CUDA/cuDNN compatibility. If installation is problematic, you can temporarily comment out tensorflow in requirements.txt and proceed; the core workflow primarily uses scikit-learn and gradient boosting libraries.
  - Database access from Python: psycopg2-binary and SQLAlchemy are now included in requirements.txt for PostgreSQL
    connectivity. Most data movement can be done via psql or GUI tools, but these libraries enable direct database
    access from Python scripts and notebooks.

D. Environment variables
- See environment_variables.txt. You can export them in your shell or create a .env for tools that auto-load it.
- Defaults in repo:
  - TF_CPP_MIN_LOG_LEVEL=2 (reduce TensorFlow log verbosity)
- Optional environment variables (examples):
  - DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS, MEMORY_LIMIT
  - DB_URL (SQLAlchemy connection URL for database access, e.g., postgresql+psycopg2://postgres:@localhost:5432/postgres)

E. PostgreSQL setup
- Install PostgreSQL (Windows installer from postgresql.org). Ensure psql is on PATH.
- Start PostgreSQL service and verify access with user postgres (adjust user/password if needed).
- Schema creation:
  - Run the provided SQL script to create the equities table:
    - Windows (PowerShell):
      - psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
    - You will be prompted for the postgres password if required.
- Connection details:
  - JDBC: jdbc:postgresql://localhost:5432/postgres (user: postgres)
  - Driver: org.postgresql.Driver (42.7.x)
  - DataGrip/IDEA DataSource template (for convenience):
    - <data-source source="LOCAL" name="postgres@localhost" ... jdbc-url="jdbc:postgresql://localhost:5432/postgres" user-name="postgres" jdbc-driver="org.postgresql.Driver"/>

F. Loading data from CSVs into PostgreSQL
- The data CSV files live in data/ and represent screenings by region: screening_us.csv, screening_eu.csv, screening_apac.csv, screening_rotw.csv.
- Recommended approach: Use the comprehensive import script that handles all regions with proper NULL handling:
  - psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
  - This script uses staging tables, proper NULL handling (NULL '', ENCODING 'UTF8'), and provides validation at each step.
- Optional but recommended: Validate CSV data quality before import:
  - python validate_csv_import.py
  - This validates schema, checks for missing values, and identifies data quality issues.
- Alternative manual approach using psql and a staging table (if you need more control):
  1) Create a temporary staging table with identical columns to equities. The main equities table already includes a "Region" column.
  2) For each CSV, import with \copy (client-side) so paths work reliably on Windows:
     - Example for US region with proper NULL handling:
       - psql -h localhost -p 5432 -U postgres -d postgres -c "CREATE TEMP TABLE equities_staging (LIKE equities)"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "\copy equities_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "UPDATE equities_staging SET \"Region\"='US' WHERE \"Region\" IS NULL"
       - psql -h localhost -p 5432 -U postgres -d postgres -c "INSERT INTO equities SELECT * FROM equities_staging ON CONFLICT DO NOTHING"
- Critical import parameters:
  - NULL '' - Treats empty strings as NULL values (essential to avoid import errors)
  - ENCODING 'UTF8' - Ensures proper character handling
  - HEADER true - Skips the CSV header row
- Tip: Because many column names use spaces and punctuation, always quote identifiers in SQL as shown in create_equities_schema.sql.

G. Accessing the database from Python (optional)
- Install optional deps: pip install psycopg2-binary SQLAlchemy
- Code snippet to query into pandas:
  - from sqlalchemy import create_engine
  - import pandas as pd
  - engine = create_engine('postgresql+psycopg2://postgres:@localhost:5432/postgres')  # add password if needed
  - query = 'SELECT * FROM equities WHERE "Region" IN (\'US\', \'EU\', \'APAC\', \'ROTW\')'
  - all_stocks = pd.read_sql(query, engine)
  - # Normalize column names for Pythonic usage
  - all_stocks.columns = (
  -     all_stocks.columns
  -     .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
  -     .str.strip('_')
  -     .str.lower()
  - )

2) Data Pipeline aligned to the new all_stocks dataframe
   The notebook (ml_finance_model_main.ipynb) implements the pipeline below. Use all_stocks as the single, unified
   dataframe sourced from PostgreSQL across regions.

A. Loading and preprocessing
- Source: equities table in PostgreSQL populated from the four CSVs.
- Recommended filters: drop rows with missing Ticker, Sector, Last_Price, and critical financials needed for targets.
- Type coercion: convert numeric columns with pd.to_numeric(errors='coerce'), then handle NaNs.
- Deduplication: drop_duplicates on Ticker if needed; for multiple listings, choose Trading_Country or Exchange priority.
- Train/validation split: by time if you have snapshot dates; otherwise stratify by Sector or Region to maintain balance.

B. Exploratory Data Analysis (EDA)
- Global overview: row counts per Region and Sector; missingness matrix; distributions for key metrics (Market Cap, EV, P/E, EBITDA, margins).
- Correlations: compute Pearson/Spearman; watch for multicollinearity (e.g., EV, Market Cap, Total Assets).
- Sector-level slices: summary statistics and outlier detection per Sector, Region.
- Target relationship: analyze distribution of Price_Target versus Last_Price; consider Price_Target_YTD_Ago for drift.

C. Advanced feature engineering with sector-specific optimizations
- Baseline transforms:
  - Ratios: EV/EBITDA, Net_Debt/EBITDA, P_E, P_B, Margin features, Revenue CAGR, Volatility windows.
  - Winsorize or robust scaling per Sector to mitigate outliers.
  - One-hot encode Sector, Industry; optionally target-encode Industry with CV to avoid leakage.
- Sector-specific enrichments:
  - Financials: TBV-related metrics, P_TBV, ROE.
  - Energy/Materials: Asset turnover, CAPEX intensity, EBITDA margin stability.
  - Tech/Healthcare: R&D intensity, growth proxies, volatility profile.
- Interactions: Region x Sector and Size_Class x Sector interactions.
- Feature selection: Boruta/Shap-based pruning to reduce noise.

D. Multi-class classification of financial events
- Define classes from events using available columns, for example:
  - 0: Neutral, 1: Positive catalyst, 2: Negative catalyst (construct from changes in Price_Target, Analyst_Rating, and Volatility spikes). Ensure label creation only uses information available at training time.
- Models: LightGBM/XGBoost/CatBoost classifiers with class_weight balancing or scale_pos_weight.
- Validation: Grouped CV by Sector or Ticker to prevent leakage across folds.
- Outputs: Use predicted class probabilities as meta-features for the regression stage.

E. Sector-optimized regression models enhanced with classification features
- Strategy: Train one regressor per Sector for Price_Target or Price_Target_Median using Gradient Boosting (XGBoost/LightGBM/CatBoost) and/or linear models with elastic nets as benchmarks.
- Features: Core engineered features + classification probabilities + key categorical encodings.
- Calibration: Consider quantile models to estimate uncertainty bands.
- Stacking: Optionally stack sector models with a meta-learner trained on out-of-fold predictions.

F. Model evaluation and error analysis
- Metrics: MAE, RMSE, MAPE, R2; compute per Sector and overall.
- Diagnostics: Residual plots per Sector, error histograms, performance by Market_Cap buckets.
- Ablation: Feature-importance-based ablations and SHAP analysis to validate drivers.

G. Identification of under/overvalued stocks with visualization
- Mispricing score: (Predicted_Target - Last_Price) / Last_Price.
- Ranking: Top-N undervalued/overvalued per Sector and Region.
- Visuals: bar charts, scatter plots of Predicted_Target vs Last_Price, and sector heatmaps; use matplotlib/seaborn/plotly.

H. Comprehensive analytics of prediction results
- Confusion matrices for event classifier; PR/ROC curves.
- Regression error buckets: by Sector, Region, and Volatility.
- Stability checks across snapshots (if multiple dates exist).
- Export: write CSV/Excel of predictions and analytics; consider xlsxwriter for formatted reports.

2.5) Running as a Python script
In addition to the notebook-first workflow, you can run a lightweight script version of the pipeline.

A. Script: ml_finance_model_main.py
- Python script with CLI for batch processing:
  - `--data-source {auto|csv|db}` — Data source selection (default: auto)
  - `--db-url <url>` — Database connection string (or use DB_URL env var)
  - `--limit <n>` — Limit rows for testing
  - `--out-dir <path>` — Output directory (default: outputs)
  - `--dry-run` — Skip model training

B. Usage examples
- Windows (PowerShell):
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
- macOS/Linux (bash):
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs

C. Data source selection
- --data-source auto (default) tries DB first if DB_URL (or --db-url) is provided and SQLAlchemy is available, otherwise falls back to CSVs in data/.
- --data-source db forces DB; provide --db-url or set DB_URL env var (e.g., postgresql+psycopg2://postgres:@localhost:5432/postgres).
- --data-source csv forces CSV fallback and combines the four region files.

D. Outputs
- Artifacts are written to the outputs/ directory (e.g., eda_summary.json, regression_predictions.csv).
- Environment variables recognized: DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS, DB_URL.

3) Testing Information
A. How to run tests
- We use Python’s built-in unittest to avoid extra dependencies.
- Run all tests from project root:
  - python -m unittest -v

B. How to add tests
- Create files under tests/ named test_*.py with unittest.TestCase classes.
- Keep tests isolated from external services:
  - Prefer testing pure functions and small utilities.
  - For DB-related code, stub or mock the connection; keep integration tests optional and locally configured via env vars.
- Use small, deterministic samples; avoid loading full CSVs unless necessary.

C. Test suite overview
The project includes a comprehensive test suite with the following test modules (27 total):

- tests/test_analytics.py — Analytics and stock ranking tests
- tests/test_build_features.py — Feature building pipeline
- tests/test_classification.py — Event classification model tests
- tests/test_cli.py — Command-line interface tests
- tests/test_coverage_smoke.py — Smoke test for coverage validation
- tests/test_data_quality.py — Data validation and quality checks
- tests/test_eda.py — Exploratory data analysis utilities
- tests/test_features.py — Feature engineering functions
- tests/test_finance_ml_config.py — Configuration management tests
- tests/test_finance_ml_data.py — Data loading module tests
- tests/test_finance_ml_eval.py — Evaluation and analytics module tests
- tests/test_finance_ml_features.py — Features module tests
- tests/test_finance_ml_models.py — Models module tests
- tests/test_improvement_plan_revision.py — Development plan validation
- tests/test_loaders.py — CSV and database loading functions
- tests/test_logging.py — Logging configuration tests
- tests/test_notebook_config.py — Notebook configuration tests
- tests/test_notebook_enhancements.py — Notebook enhancements validation
- tests/test_portfolio_optimization.py — Portfolio optimization tests
- tests/test_preprocess_and_training.py — Preprocessing and training workflows
- tests/test_regression.py — Regression model evaluation
- tests/test_repository_setup.py — Validates repository basics (required files, SQL schema, environment config)
- tests/test_risk_metrics.py — Risk metrics calculation tests
- tests/test_setup_environment.py — Setup script validation
- tests/test_sqlite_import.py — SQLite import functionality (header removal, NULL handling, region backfilling)
- tests/test_sql_scripts.py — SQL script validation tests
- tests/test_validate_csv_import.py — CSV validation (schema validation, data quality checks)
- tests/test_visualizations.py — Visualization functions tests

4) Additional Development Information
A. Code style and quality
- Python style: PEP 8 (black-like formatting), type hints where possible.
- Logging: prefer Python logging over prints. The notebook initializes a logger fallback if the custom one isn’t available.
- Reproducibility: set RANDOM_SEED where applicable; record library versions for experiments.

B. Notebook hygiene
- Keep data loading, feature engineering, and modeling steps modularized into functions within the notebook to ease reuse.
- Avoid hard-coding paths; use Path from pathlib and environment variables.
- Save intermediate artifacts (processed datasets, models) with versioned filenames.

C. Performance tips
- Use joblib parallelism judiciously (n_jobs) and monitor memory for large models.
- Start with a smaller feature set and scale up; use downsampling or stratification for class imbalance.

D. Development roadmap
- See IMPROVEMENT_PLAN.md for a phased development roadmap aligned to this project.
- The plan covers 8 phases including: foundations, data ingestion/validation, EDA/feature engineering, classification models, regression models, analytics/reporting, testing/CI, and packaging/modularity.
- Immediate next steps and risk mitigation strategies are documented.

E. Troubleshooting
- TensorFlow install issues: set TF_CPP_MIN_LOG_LEVEL=2; try CPU-only; verify Visual C++ Redistributable on Windows.
- Postgres import quoting: many column names have spaces; keep using double quotes in SQL to avoid errors.
- Large CSVs on Windows: prefer \copy from psql to avoid server-side file permission issues.

Appendix: IDE Data Source Settings (for JetBrains tools)
- You can configure a Data Source using the following (adapt as needed):
  #DataSourceSettings#
  #LocalDataSource: postgres@localhost
  #BEGIN#
  <data-source source="LOCAL" name="postgres@localhost" uuid="3091f81a-c800-413c-b86b-f760950418f7"><database-info product="PostgreSQL" version="18.0" jdbc-version="4.2" driver-name="PostgreSQL JDBC Driver" driver-version="42.7.3" dbms="POSTGRES" exact-version="18.0" exact-driver-version="42.7"><identifier-quote-string>&quot;</identifier-quote-string></database-info><case-sensitivity plain-identifiers="lower" quoted-identifiers="exact"/><driver-ref>postgresql</driver-ref><synchronize>true</synchronize><jdbc-driver>org.postgresql.Driver</jdbc-driver><jdbc-url>jdbc:postgresql://localhost:5432/postgres</jdbc-url><jdbc-additional-properties><property name="com.intellij.clouds.kubernetes.db.host.port"/><property name="com.intellij.clouds.kubernetes.db.enabled" value="false"/><property name="com.intellij.clouds.kubernetes.db.container.port"/></jdbc-additional-properties><secret-storage>master_key</secret-storage><user-name>postgres</user-name><schema-mapping><introspection-scope><node kind="database" qname="@"><node kind="schema" qname="@"/></node></introspection-scope></schema-mapping><working-dir>$ProjectFileDir$</working-dir></data-source>
  #END#

Versioning
- When modifying modeling behavior materially, bump MODEL_VERSION (e.g., v8_3) and document in this file what changed (features, labels, metrics).



---

Addendum: Key Features, Main Scripts, CLI, and Testing/Coverage

This addendum supplements the core guidelines with quick-reference details aligned with the README and packaging configuration. It is intended to speed up onboarding and future development.

Key Features (aligns with README)

- 📊 Data Management: Load from PostgreSQL or CSV, with validation and quality checks
- 🔧 Feature Engineering: Financial ratios, margins, volatility, revenue CAGR, and more
- 🤖 ML Models: Event classification, sector-optimized regression, quantile models, stacking ensembles
- 📈 Analytics: Mispricing scores, stock ranking, interactive visualizations
- ⚙️ Configuration: Flexible config via Anaconda (optional) or venv/pip with environment variables and CLI options;
  avoid mixing Conda with venv
- 🧪 Tested: Comprehensive unit tests with good coverage
- 🚀 CLI: Three command-line tools for different workflows (see CLI Tools below)

Finance_ML_Analytics_Platform — Main Scripts

- ml_finance_model_main.ipynb — Primary notebook for end-to-end workflow and exploration.
- ml_finance_model_main.py — Lightweight script version with a CLI for batch runs and automation.

CLI Tools (installed via pyproject’s console scripts)
- finance-ml — Primary pipeline runner (data load, preprocess, features, models, outputs).
  Example: finance-ml --data-source auto --limit 5000 --output-dir outputs
- finance-ml-analyze — EDA/analytics-only workflows.
  Example: finance-ml-analyze --data-source csv --output-dir outputs
- finance-ml-validate — Validation-only workflows (schema checks, data quality, etc.).
  Example: finance-ml-validate --data-source csv --output-dir outputs
Notes:
- The script ml_finance_model_main.py provides equivalent capabilities via Python directly:
  - python ml_finance_model_main.py --data-source auto --limit 5000 --out-dir outputs
- Data sources: --data-source auto|csv|db as described earlier. Use DB_URL or --db-url for database access.

Testing and Coverage
- Test runner (unittest):
  - python -m unittest -v
- Creating new tests:
  - Add files under tests/ named test_*.py and use unittest.TestCase.
  - Keep tests isolated from external services; prefer pure functions and mocks.
- Coverage (option A: coverage.py):
  1) pip install coverage
  2) coverage run -m unittest -v
  3) coverage report -m
  4) coverage html  # optional HTML report under htmlcov/
- Coverage (option B: pytest + pytest-cov):
  1) pip install pytest pytest-cov
  2) pytest --cov=finance_ml --cov-report=term-missing
- Demonstration test included: tests/test_coverage_smoke.py — a minimal smoke test that verifies arithmetic and imports the finance_ml package. Use it as a template for adding more tests and to confirm the coverage commands above work end-to-end.

Documentation and Artifact Update Checklist
When making significant changes, ensure the following files are updated accordingly (as applicable):
- environment_variables.txt — Add or adjust env vars; keep TF_CPP_MIN_LOG_LEVEL=2 default.
- Pipfile — Reflect dependency changes for Pipenv users.
- requirements.txt — Keep constraints aligned with pyproject.toml; consider optional extras.
- README.md — Key Features, setup, CLI usage, examples, and links.
- ml_finance_model_main.ipynb — Ensure cells align with the current pipeline functions and APIs.
- ml_finance_model_main.py — Keep CLI options and defaults in sync with package functions.
- IMPROVEMENT_PLAN.md — Log changes to features/labels/metrics; note version bumps.
- pyproject.toml — Update version, console scripts, optional dependencies, and tooling configs.
- qodana.yaml — Static analysis and quality gates configuration.
- import_equities_data.sql — Data loading procedure updates (staging, NULL handling, validation steps).
- create_equities_schema.sql — Schema changes; ensure quoted identifiers and ownership.
- .aiassistant/rules/promt_rules.md — Assistant and prompt rules if the dev workflow changes.
- LICENSE — Copyright years and ownership.

Notes on Configuration Consistency
- Prefer environment variables and CLI flags for configuration in code and notebooks.
- Avoid hard-coded paths; use pathlib.Path and env vars (DATA_DIR, MODEL_DIR, OUTPUT_DIR, etc.).
- Keep the notebook and script aligned with the finance_ml package APIs to reduce duplication.

Versioning Reminder
- When modifying modeling behavior materially, bump MODEL_VERSION (e.g., v8_3) and record changes (features, labels, metrics) in this file and IMPROVEMENT_PLAN.md.

## SQLite Local Setup and Data Load

For quick local testing without PostgreSQL, you can use SQLite.

1) Create the SQLite schema:

- sqlite3 equities.sqlite ".read create_equities_schema_sqlite.sql"

2) Import CSVs into SQLite using the provided CLI SQL script (recommended):

- sqlite3 equities.sqlite ".read import_equities_data_sqlite.sql"

This import script:

- Uses per‑region staging tables (US, EU, APAC, ROTW)
- Deletes header rows when imported as data
- Backfills missing Region values per file
- Inserts with INSERT OR IGNORE honoring the UNIQUE("Ticker","Region") index
- Prints a post‑import validation summary

3) Python alternative: chunked importer for very large CSVs:

- python tools/import_sqlite.py --db equities.sqlite --data-dir data --chunksize 2000
- python tools/import_sqlite.py --db equities.sqlite --regions US,EU

The Python importer (tools/import_sqlite.py) features:

- Configurable chunk size for memory-efficient processing of large CSVs
- Automatic header detection and removal
- NULL handling (converts empty strings to NULL)
- Per-region import with automatic Region column backfilling
- UNIQUE("Ticker","Region") constraint enforcement for deduplication
- Comprehensive test coverage via tests/test_sqlite_import.py

4) Data validation before import (recommended):

- python validate_csv_import.py

This validation script:

- Checks schema compliance and critical column presence (Ticker, Sector, Last Price, etc.)
- Detects non-numeric values in numeric columns
- Identifies data quality issues before database import
- Produces detailed validation reports per region
- Comprehensive test coverage via tests/test_validate_csv_import.py

Notes:

- The Python importer uses pandas (already listed in requirements.txt). It reads CSVs with dtype=str, converts empty
  strings to NULLs, and backfills Region.
- sqlite3 is included with Python; no extra dependency is required.
- All SQLite import paths are fully tested with ≥80% coverage targets.
