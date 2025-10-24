---
apply: always
---

# Finance ML Analytics Platform — Project Rules

**Version**: 0.3.0  
**Last Updated**: 2025-10-24

## Project Overview

Finance ML Analytics Platform is a professional, modular Python package for equity screening, feature engineering, and machine learning models across global regions (US, EU, APAC, ROTW).

**Key capabilities**:
- Unified data pipeline with PostgreSQL integration + CSV fallback
- Modular Python package (`finance_ml`) with clean, tested, reusable code
- Interactive Jupyter notebook for exploration and prototyping
- CLI tools for batch processing and automation
- Comprehensive unit tests with good coverage
- Configuration management via environment variables, JSON, or YAML


## Technology Stack

### Language and Runtime
- **Language**: Python 3.10–3.11 (3.11 recommended)
- **Notebook Environment**: Jupyter Notebook or JupyterLab
- **Database**: PostgreSQL 15+ (local instance)

### Package Managers
- **Primary**: pip + venv (see `requirements.txt`)
- **Alternative**: pipenv (see `Pipfile` aligned with Python 3.11)

### Core Dependencies
**Data Science**:
- numpy>=1.24.0,<2.0.0
- pandas>=2.0.0,<3.0.0
- scipy>=1.10.0,<2.0.0
- statsmodels>=0.14.0,<1.0.0

**Machine Learning**:
- scikit-learn>=1.3.0,<2.0.0
- imbalanced-learn>=0.11.0,<1.0.0
- tensorflow>=2.13.0,<3.0.0
- scikeras>=0.12.0,<1.0.0

**Gradient Boosting**:
- xgboost>=2.0.0,<3.0.0
- lightgbm>=4.0.0,<5.0.0
- catboost>=1.2.0,<2.0.0

**Feature Engineering & Explainability**:
- boruta>=0.3.0
- shap>=0.42.0,<1.0.0

**Visualization**:
- matplotlib>=3.7.0,<4.0.0
- seaborn>=0.12.0,<1.0.0
- plotly>=5.14.0,<6.0.0

**Utilities**:
- tqdm>=4.65.0,<5.0.0
- joblib>=1.3.0,<2.0.0
- numba>=0.57.0,<1.0.0
- Pillow>=10.0.0,<11.0.0
- xlsxwriter>=3.1.0,<4.0.0
- psutil>=5.9.0,<6.0.0

**Optional (for database access)**:
- psycopg2-binary (not in requirements.txt; install separately if needed)
- SQLAlchemy (not in requirements.txt; install separately if needed)

### Testing Framework
- **Built-in**: unittest (Python standard library)
- **Test Runner**: `python -m unittest -v`
- **Coverage**: `.coverage` file indicates coverage tracking is in use


## Requirements and Prerequisites

**Operating System**:
- Windows 10/11 (tested)
- macOS
- Linux

**Software**:
- Python 3.10 or 3.11 (use pyenv or official installer; avoid mixing Conda with venv)
- PostgreSQL 15+ (local instance)
- Git (optional but recommended)
- psql command-line tool (included with PostgreSQL; must be on PATH)

**Hardware**:
- TensorFlow works with CPU-only; GPU acceleration optional (CUDA/cuDNN setup required)


## Quick Start Setup

### 1. Virtual Environment
**Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**macOS/Linux (bash)**:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Optional Database Client Libraries
If you need to query PostgreSQL from Python:
```bash
pip install psycopg2-binary SQLAlchemy
```

### 3. PostgreSQL Setup
Create the equities table:
```powershell
# Windows (PowerShell)
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

### 4. Data Import
Load regional CSVs from `data/` into PostgreSQL:
```powershell
# Windows (PowerShell)
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
```

Optional: Validate CSV data quality before import:
```powershell
python validate_csv_import.py
```

### 5. Run Tests
```bash
python -m unittest -v
```

### 6. Start Working
**Notebook** (primary workflow):
```bash
jupyter notebook
# Open ml_finance_model_v8_2.ipynb and run cells in order
```

**CLI**:
```bash
# Quick analysis
finance-ml-analyze --data-source auto --limit 1000 -v

# Full pipeline
finance-ml --data-source auto --limit 5000 --out-dir outputs
```


## Entry Points

### 1. Interactive Notebook (Primary)
**File**: `ml_finance_model_v8_2.ipynb`
- Main Jupyter notebook for exploration, EDA, feature engineering, and modeling
- Imports from `finance_ml` package (modular design)
- Run cells in order
- Outputs: model diagnostics, ranking tables, optional CSV/Excel exports

### 2. Python Script (Legacy)
**File**: `ml_finance_model_v8_2.py`
- Command-line interface for batch processing
- Now uses `finance_ml` package internally (refactored in Phase 7)

**Usage**:
```bash
python ml_finance_model_v8_2.py --data-source auto --limit 5000 --out-dir outputs
```

**Options**:
- `--data-source {auto|csv|db}` — Data source selection (default: auto)
- `--db-url <url>` — Database connection string (or use DB_URL env var)
- `--limit <n>` — Limit rows for testing
- `--out-dir <path>` — Output directory (default: outputs)
- `--dry-run` — Skip model training

### 3. CLI Tools (Modern)
Three console commands installed via `pyproject.toml` entry points:

**a) `finance-ml`** — Main analysis pipeline
```bash
finance-ml --data-source auto --limit 5000
finance-ml --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres
finance-ml --config config.json
```

**b) `finance-ml-analyze`** — Quick EDA and data profiling
```bash
finance-ml-analyze --data-source csv --data-dir ./data
finance-ml-analyze --data-source db --limit 1000 -v
```

**c) `finance-ml-validate`** — Data validation and quality checks
```bash
finance-ml-validate --data-source csv --data-dir ./data
finance-ml-validate --data-source db -v
```


## Available Scripts and Utilities

### Database Setup Scripts
- **`create_equities_schema.sql`** — SQL script to initialize the equities table in PostgreSQL
  - Creates table with 234 columns (many with spaces/punctuation; always use double quotes in SQL)
  - Sets ownership to `postgres` user
  - Run via: `psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql`

- **`import_equities_data.sql`** — Comprehensive import script for all regional CSVs
  - Handles US, EU, APAC, ROTW regions with staging tables
  - Proper NULL handling (`NULL ''`), encoding (`ENCODING 'UTF8'`), and validation
  - Run via: `psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql`

### Data Validation Scripts
- **`validate_csv_import.py`** — CSV data quality validator
  - Validates schema, checks for missing values, identifies data quality issues
  - Run before importing data to catch problems early

### Environment Setup Scripts
- **`setup_environment.py`** — Automated environment setup utility
  - Checks prerequisites (Python, PostgreSQL, Git)
  - Creates virtual environment
  - Installs dependencies
  - Sets up database and loads CSV data
  - Configures environment variables
  - Runs tests
  - Provides activation instructions

### Notebook Utilities
- **`analyze_notebook.py`** — Notebook structure analyzer
  - Counts cells, previews content, searches for functions
  
- **`update_notebook.py`** — Notebook synchronizer
  - Extracts TDD functions from .py files and inserts into .ipynb
  
- **`verify_notebook.py`** — Notebook verification utility
  - Checks for presence of TDD functions

- **`refactor_notebook.py`** — Notebook refactoring tool
  - TODO: Document purpose and usage

- **`update_notebook_imports.py`** — Import updater
  - TODO: Document purpose and usage


## Environment Variables

See `environment_variables.txt` for complete documentation.

### Core Configuration
```bash
# Logging Configuration
TF_CPP_MIN_LOG_LEVEL=2          # TensorFlow log level: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Directory Paths
DATA_DIR=data                    # Data directory
MODEL_DIR=models                 # Saved models directory
CACHE_DIR=.cache                 # Cache directory
OUTPUT_DIR=outputs               # Output directory

# Database Configuration
DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres

# Model Configuration
MODEL_VERSION=v8_2               # Model version identifier
RANDOM_SEED=42                   # Random seed for reproducibility

# Performance Settings
N_JOBS=-1                        # Number of parallel jobs (-1=all cores)
MEMORY_LIMIT=8GB                 # Maximum memory allocation

# Logging
LOG_LEVEL=INFO                   # Logging level
```

### Setting Environment Variables

**Windows (PowerShell)**:
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:DB_URL = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
```

**macOS/Linux (bash)**:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
export DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres
```

**Using .env file**:
Create a `.env` file in the project root with the variables above. Tools that support `.env` auto-loading will pick them up.


## Database Configuration

### Connection Details
- **JDBC URL**: `jdbc:postgresql://localhost:5432/postgres`
- **Driver**: `org.postgresql.Driver` (version 42.7.x)
- **User**: `postgres`
- **Database**: `postgres`
- **Schema**: `public`
- **Table**: `equities`

### Important Notes
- Many column names contain spaces and special characters
- **Always use double quotes** when referencing column names in SQL
- Example: `SELECT "Ticker", "Last Price", "Market Cap" FROM equities WHERE "Region" = 'US'`

### Data Import Critical Parameters
- **`NULL ''`** — Treats empty strings as NULL values (essential to avoid import errors)
- **`ENCODING 'UTF8'`** — Ensures proper character handling
- **`HEADER true`** — Skips the CSV header row

### Regional Data Files
- `data/screening_us.csv` — US equities
- `data/screening_eu.csv` — European equities
- `data/screening_apac.csv` — Asia-Pacific equities
- `data/screening_rotw.csv` — Rest of World equities


## Testing

### Test Framework
- **Framework**: Python's built-in `unittest`
- **Location**: `tests/` directory
- **Count**: Comprehensive unit tests
- **Coverage**: Tracked in `.coverage` file

### Running Tests
```bash
# Run all tests with verbose output
python -m unittest -v

# Run specific test module
python -m unittest tests.test_finance_ml_data

# Run with pytest (if installed)
pytest tests/ -v --cov=finance_ml

# Discover and run all tests
python -m unittest discover -s tests -v
```

### Test Suite Modules
- **`test_repository_setup.py`** — Validates repository basics
  - Key files exist (requirements.txt, SQL scripts, environment config, CSVs)
  - SQL file contains CREATE TABLE equities and sets OWNER TO postgres
  - environment_variables.txt includes TF_CPP_MIN_LOG_LEVEL=2
  - CSVs are non-empty and have a header line

- **`test_data_quality.py`** — Data validation and quality checks
- **`test_loaders.py`** — CSV and database loading functions
- **`test_features.py`** — Feature engineering functions
- **`test_build_features.py`** — Feature building pipeline
- **`test_eda.py`** — Exploratory data analysis utilities
- **`test_preprocess_and_training.py`** — Preprocessing and training workflows
- **`test_regression.py`** — Regression model evaluation
- **`test_classification.py`** — Event classification model tests
- **`test_analytics.py`** — Analytics and stock ranking tests
- **`test_finance_ml_data.py`** — finance_ml.data module tests
- **`test_finance_ml_features.py`** — finance_ml.features module tests
- **`test_finance_ml_models.py`** — finance_ml.models module tests
- **`test_finance_ml_eval.py`** — finance_ml.eval module tests

### Writing New Tests
- Create files under `tests/` named `test_*.py` with `unittest.TestCase` classes
- Keep tests isolated from external services
- Prefer testing pure functions and small utilities
- For DB-related code, stub or mock the connection
- Use small, deterministic samples; avoid loading full CSVs unless necessary


## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                     # Main Python package (v0.3.0)
│   ├── __init__.py                # Package exports and version
│   ├── data.py                    # Data loading, normalization, validation
│   ├── features.py                # Feature engineering functions
│   ├── models.py                  # ML models (classification, regression, ensembles)
│   ├── eval.py                    # Analytics, visualizations, reporting
│   ├── config.py                  # Configuration management
│   └── cli.py                     # Command-line interface
│
├── tests/                         # Unit tests (see tests/ for modules)
│   ├── test_finance_ml_data.py
│   ├── test_finance_ml_features.py
│   ├── test_finance_ml_models.py
│   ├── test_finance_ml_eval.py
│   ├── test_repository_setup.py
│   └── ... (10+ test modules)
│
├── data/                          # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── .github/workflows/             # CI/CD pipelines
│   └── tests.yml                  # Automated testing workflow
│
├── .aiassistant/rules/            # AI assistant rules
│   └── promt_rules.md             # This file
│
├── .junie/                        # Junie AI guidelines
│   └── guidelines.md              # Development guidelines
│
├── outputs/                       # Output artifacts
│   ├── eda_summary.json
│   ├── regression_predictions.csv
│   └── ... (generated files)
│
├── ml_finance_model_v8_2.ipynb   # Interactive Jupyter notebook
├── ml_finance_model_v8_2.py      # Legacy script (uses finance_ml package)
│
├── pyproject.toml                # Modern Python packaging configuration
├── setup.py                      # Backward-compatible setup
├── requirements.txt              # Core dependencies
├── Pipfile                       # Pipenv dependencies (Python 3.11)
│
├── CHANGELOG.md                  # Version history
├── README.md                     # Project documentation
├── IMPROVEMENT_PLAN.md           # Development roadmap (8 phases)
├── REFACTORING_COMPLETE.md       # Refactoring documentation
├── TDD_IMPLEMENTATION_COMPLETE.md # TDD implementation summary
│
├── create_equities_schema.sql    # PostgreSQL schema setup
├── import_equities_data.sql      # Data import script
├── environment_variables.txt     # Environment configuration examples
│
├── setup_environment.py          # Automated environment setup
├── validate_csv_import.py        # CSV data quality validator
├── analyze_notebook.py           # Notebook structure analyzer
├── update_notebook.py            # Notebook synchronizer
├── verify_notebook.py            # Notebook verification utility
├── refactor_notebook.py          # Notebook refactoring tool
└── update_notebook_imports.py    # Import updater
```

### Package Modules (`finance_ml`)

#### `finance_ml.data`
Data loading, normalization, and validation functions.
- `load_from_csv()`, `load_from_db()`: Multi-source data loading
- `preprocess()`: Data cleaning and normalization
- `normalize_columns()`: Column name normalization
- `validate_schema()`, `check_missing_values()`: Quality checks
- `detect_outliers_iqr()`, `validate_numeric_ranges()`: Outlier detection
- `infer_region_from_filename()`: Region inference from CSV filenames

#### `finance_ml.features`
Feature engineering for financial data.
- `engineer_basic_ratios()`: EV/EBITDA, P/E, P/B ratios
- `engineer_margin_features()`: Gross, operating, net margins
- `engineer_volatility_features()`: Price volatility windows
- `engineer_revenue_cagr()`: Revenue growth metrics
- `build_features_and_target()`: Complete feature pipeline

#### `finance_ml.models`
Machine learning models for classification and regression.
- `create_event_labels()`: Event label creation
- `train_event_classifier()`: Multi-class event classification
- `train_and_evaluate_regression()`: Baseline regression models
- `train_and_evaluate_regression_by_sector()`: Sector-optimized models
- `train_quantile_regression()`: Uncertainty estimation
- `predict_quantile_regression()`: Quantile predictions
- `train_quantile_regression_by_sector()`: Sector-specific quantile models
- `train_stacking_ensemble()`: Meta-learner stacking
- `train_stacking_ensemble_by_sector()`: Sector-specific ensembles
- `build_regression_pipeline()`: Regression pipeline builder

#### `finance_ml.eval`
Analytics, visualizations, and reporting.
- `simple_eda()`: Exploratory data analysis
- `calculate_mispricing_score()`: Valuation analysis
- `rank_undervalued_stocks()`, `rank_overvalued_stocks()`: Stock ranking
- `rank_stocks_by_sector()`: Sector-based ranking
- `create_sector_heatmap()`: Sector performance heatmap
- `create_interactive_prediction_plot()`: Interactive visualizations
- `create_region_sector_heatmap()`: Region-sector analysis
- `export_predictions_to_excel()`: Excel reporting

#### `finance_ml.config`
Configuration management system.
- `FinanceMLConfig`: Configuration dataclass
- `load_config()`: Load from environment/JSON/YAML
- `get_config()`, `set_config()`: Global config management

#### `finance_ml.cli`
Command-line interface tools.
- `finance-ml`: Main analysis pipeline
- `finance-ml-analyze`: Quick data analysis
- `finance-ml-validate`: Data validation


## Configuration Management

### Configuration Sources (Priority Order)
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file (JSON/YAML)
4. Default values (lowest priority)

### Configuration File Formats

**config.json**:
```json
{
  "data_dir": "data",
  "output_dir": "outputs",
  "db_url": "postgresql+psycopg2://postgres:@localhost:5432/postgres",
  "model_version": "v8_2",
  "random_seed": 42,
  "n_jobs": -1
}
```

**config.yaml**:
```yaml
data_dir: data
output_dir: outputs
db_url: postgresql+psycopg2://postgres:@localhost:5432/postgres
model_version: v8_2
random_seed: 42
n_jobs: -1
```

### Loading Config in Python
```python
from finance_ml import load_config

config = load_config("config.json")
config.apply_to_env()
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Finance ML Analytics Platform Contributors


## Development Guidelines

See `.junie/guidelines.md` for comprehensive development guidelines covering:
- Build and configuration instructions
- Data pipeline aligned to all_stocks dataframe
- Advanced feature engineering with sector-specific optimizations
- Multi-class classification of financial events
- Sector-optimized regression models
- Model evaluation and error analysis
- Identification of under/overvalued stocks
- Comprehensive analytics of prediction results
- Testing information
- Code style and quality standards
- Troubleshooting tips

See `IMPROVEMENT_PLAN.md` for phased development roadmap (8 phases):
1. Foundations
2. Data Ingestion/Validation
3. EDA/Feature Engineering
4. Classification Models
5. Regression Models
6. Analytics/Reporting
7. Testing/CI
8. Packaging/Modularity


## Troubleshooting

### TensorFlow Installation Issues
- Project primarily uses scikit-learn and gradient boosting libraries
- CPU-only TensorFlow is fine for this project
- If installation is problematic, you can temporarily comment out `tensorflow` in requirements.txt
- Ensure Visual C++ Redistributable is installed on Windows
- Set `TF_CPP_MIN_LOG_LEVEL=2` to reduce log verbosity

### PostgreSQL Issues
- **Column quoting**: Many columns contain spaces/punctuation; always use double quotes in SQL
- **Import errors**: Use `\copy` (client-side) from psql to avoid server-side file permission issues on Windows
- **NULL handling**: Always use `NULL ''` parameter in COPY/\copy commands to treat empty strings as NULL

### Data Import Issues
- **"relation equities does not exist"**: Run `create_equities_schema.sql` first
- **"invalid input syntax for type numeric"**: Check CSV for invalid numeric values; `NULL ''` should handle empty strings
- **"could not open file for reading"**: Ensure you're running psql from the project root directory, or use absolute paths
- **Permission denied errors**: Use `\copy` (client-side) instead of `COPY` (server-side)


## TODOs and Open Items

### Documentation
- [ ] Document purpose and usage of `refactor_notebook.py`
- [ ] Document purpose and usage of `update_notebook_imports.py`
- [ ] Create detailed `CONTRIBUTING.md` file with comprehensive contribution guidelines
- [ ] Update GitHub username/organization in badge URLs in README.md

### Nice to Have
- [ ] Document recommended IDE setup (DataGrip/PyCharm configuration)
- [ ] Add examples of config file usage in practice
- [ ] Document performance benchmarks for different data sizes
- [ ] Add troubleshooting section for common notebook issues


## Version History

- **v0.3.0** (Current) — Modular package, CLI tools, configuration management, CI/CD
- Earlier versions documented in `CHANGELOG.md`

---

*This file is automatically applied by AI assistants to understand project structure and conventions.*
