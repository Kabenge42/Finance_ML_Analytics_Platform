# Finance ML Analytics Platform — Improvement Plan

**Version 0.4.0** — Last Updated: 2025-10-27

This document provides a comprehensive overview of the Finance ML Analytics Platform project, including its technology stack, setup instructions, project structure, and phased development roadmap.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Entry Points and Scripts](#entry-points-and-scripts)
6. [Environment Variables](#environment-variables)
7. [Testing](#testing)
8. [Project Structure](#project-structure)
9. [Code Quality Improvement Tasks](#code-quality-improvement-tasks)
10. [Development Phases](#development-phases)
11. [License](#license)

---

## Project Overview

Finance ML Analytics Platform is a professional, modular Python package for equity screening, feature engineering, and machine learning models across global regions (US, EU, APAC, ROTW).

### Key Features
- **Unified data pipeline**: PostgreSQL integration + CSV fallback for multi-region equity data
- **Modular Python package** (`finance_ml`): Clean, tested, reusable code for data loading, feature engineering, modeling, and analytics
- **Interactive notebook**: Jupyter-based workflow (`ml_finance_model_v8_2.ipynb`) for exploration and prototyping
- **CLI tools**: Three command-line interfaces for batch processing and automation
- **Production-ready**: Modern packaging, comprehensive tests, configuration management

### Use Cases
- Quantitative equity analysis and valuation
- Multi-class event classification for financial catalysts
- Sector-optimized regression models for price target prediction
- Under/overvalued stock identification with mispricing scores
- Interactive data visualization and Excel reporting

---

## Technology Stack

<!-- SQLite parity section will be appended by tools/apply_improvement_plan_updates.py if missing -->

### Language
- **Python**: 3.10 or 3.11 (required)

### Package Managers
- **pip + venv** (primary): Dependencies in `requirements.txt`
- **pipenv** (optional): Alternative via `Pipfile` (aligned to Python 3.11)

### Database
- **PostgreSQL**: 15+ recommended (local instance)
- **Driver**: psycopg2 / SQLAlchemy (optional, for Python DB access)

### Core Frameworks & Libraries
#### Machine Learning
- **scikit-learn** (>=1.3.0): Core ML framework
- **XGBoost** (>=2.0.0): Gradient boosting
- **LightGBM** (>=4.0.0): Gradient boosting
- **CatBoost** (>=1.2.0): Gradient boosting
- **TensorFlow** (>=2.13.0): Deep learning (optional)
- **imbalanced-learn** (>=0.11.0): Class balancing
- **SHAP** (>=0.42.0): Model interpretation

#### Data Science
- **NumPy** (>=1.24.0): Numerical computing
- **pandas** (>=2.0.0): Data manipulation
- **SciPy** (>=1.10.0): Scientific computing
- **statsmodels** (>=0.14.0): Statistical models

#### Visualization
- **matplotlib** (>=3.7.0): Static plots
- **seaborn** (>=0.12.0): Statistical visualization
- **plotly** (>=5.14.0): Interactive plots

#### Utilities
- **joblib** (>=1.3.0): Parallel processing
- **tqdm** (>=4.65.0): Progress bars
- **xlsxwriter** (>=3.1.0): Excel export
- **psutil** (>=5.9.0): System monitoring

### Notebook Environment
- **Jupyter Notebook** or **JupyterLab**

### Development Tools (Optional)
- **pytest**: Testing framework
- **black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter
- **mypy**: Type checker

### Build System
- **setuptools**: Modern packaging via `pyproject.toml`

---

## Requirements

### System Requirements
- **OS**: Windows 10/11 (tested), macOS, or Linux
- **Python**: 3.10 or 3.11 (3.12 not yet tested)
- **PostgreSQL**: 15+ (local instance with psql on PATH)
- **Git**: Optional but recommended for version control

### Python Dependencies
See `requirements.txt` for complete list. Core dependencies:
- numpy, pandas, scipy, statsmodels
- scikit-learn, imbalanced-learn
- xgboost, lightgbm, catboost
- tensorflow, scikeras (optional for deep learning)
- shap, matplotlib, seaborn, plotly
- tqdm, joblib, xlsxwriter, psutil

### Optional Dependencies (via pyproject.toml)
- **dev**: pytest, pytest-cov, black, flake8, mypy, isort
- **tensorflow**: tensorflow, scikeras
- **database**: psycopg2-binary, SQLAlchemy
- **advanced-features**: boruta, numba
- **all**: Includes all optional groups

---

## Setup and Installation

### 1. Create Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade Packaging Tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies
**Basic installation:**
```bash
pip install -r requirements.txt
```

**Package installation (editable mode):**
```bash
# Install package in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"
```

**Optional database client libraries:**
```bash
pip install psycopg2-binary SQLAlchemy
```

### 4. PostgreSQL Setup
**Install and start PostgreSQL**, then create the equities table:

**Windows (PowerShell):**
```powershell
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

**macOS/Linux (bash):**
```bash
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

### 5. Load Data into PostgreSQL
**Recommended: Use comprehensive import script**
```powershell
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
```

This script:
- Handles all four regions (US, EU, APAC, ROTW)
- Uses staging tables for safe imports
- Applies proper NULL handling (`NULL ''`, `ENCODING 'UTF8'`)
- Provides validation and statistics

**Optional: Validate CSV data quality before import**
```bash
python validate_csv_import.py
```

### 6. Set Environment Variables
See `environment_variables.txt` for examples. Key variables:
```bash
# Windows (PowerShell)
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:DB_URL = "postgresql+psycopg2://postgres:@localhost:5432/postgres"

# macOS/Linux (bash)
export TF_CPP_MIN_LOG_LEVEL=2
export DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres
```

---

## Entry Points and Scripts

### Main Entry Points

#### 1. Jupyter Notebook (Primary)
**File**: `ml_finance_model_v8_2.ipynb`

Start Jupyter and open the notebook:
```bash
jupyter notebook
# or
jupyter lab
```

Run cells in order for interactive exploration, modeling, and analysis.

#### 2. Python Script
**File**: `ml_finance_model_v8_2.py`

Command-line script for batch processing:
```bash
python ml_finance_model_v8_2.py --data-source auto --limit 5000 --out-dir outputs
```

Options:
- `--data-source {auto|csv|db}` — Data source (default: auto)
- `--db-url <url>` — Database connection string
- `--limit <n>` — Limit rows for testing
- `--out-dir <path>` — Output directory (default: outputs)
- `--dry-run` — Skip model training

#### 3. CLI Tools (Package Console Scripts)
Three command-line tools available after package installation:

**a) `finance-ml` — Main Analysis Pipeline**
```bash
finance-ml --data-source auto --limit 5000
finance-ml --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres
finance-ml --data-source csv --data-dir ./data --dry-run
```

**b) `finance-ml-analyze` — Quick Data Analysis**
```bash
finance-ml-analyze --data-source csv --data-dir ./data
finance-ml-analyze --data-source db --limit 1000 -v
```

**c) `finance-ml-validate` — Data Validation**
```bash
finance-ml-validate --data-source csv --data-dir ./data -v
finance-ml-validate --data-source db --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres
```

### Database Scripts

#### create_equities_schema.sql
Creates the PostgreSQL `equities` table with proper schema (243 lines).

**Usage:**
```bash
psql -h localhost -p 5432 -U postgres -d postgres -f create_equities_schema.sql
```

**Features:**
- Drops existing table if present
- Defines 200+ columns with appropriate data types (TEXT, NUMERIC, DATE)
- Sets table ownership to `postgres`
- Includes table comments

#### import_equities_data.sql
Comprehensive import script for all regional CSV files (250 lines).

**Usage:**
```bash
psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
```

**Features:**
- Imports all four regions: US, EU, APAC, ROTW
- Uses staging tables for safe imports
- Applies proper NULL handling (`NULL ''`, `ENCODING 'UTF8'`)
- Provides validation and row count statistics
- Includes troubleshooting notes

### Utility Scripts

#### setup_environment.py
Automated environment setup utility.

**Features:**
- Checks prerequisites (Python, PostgreSQL, Git)
- Creates virtual environment
- Installs dependencies
- Sets up database and loads CSV data
- Configures environment variables
- Runs tests
- Provides activation instructions

#### validate_csv_import.py
CSV data quality validator.

**Usage:**
```bash
python validate_csv_import.py
```

**Features:**
- Schema validation
- Missing value checks
- Data quality issue identification
- Uses validation functions from `finance_ml.data`

#### analyze_notebook.py
Notebook structure analyzer.

**Features:**
- Counts cells by type (code, markdown)
- Previews cell content
- Searches for function definitions

#### update_notebook.py
Notebook synchronizer.

**Features:**
- Extracts functions from Python modules
- Inserts/updates functions in notebook cells
- Maintains notebook structure

#### verify_notebook.py
Notebook verification utility.

**Features:**
- Checks for presence of expected functions
- Validates notebook structure
- Reports missing components

---

## Environment Variables

Configuration via `environment_variables.txt` (examples) or shell export.

### Core Variables

#### Logging
```bash
TF_CPP_MIN_LOG_LEVEL=2    # TensorFlow log level: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
LOG_LEVEL=INFO            # Python logging level
```

#### Directory Paths
```bash
DATA_DIR=data             # Directory containing CSV files
MODEL_DIR=models          # Directory for saved models
CACHE_DIR=.cache          # Cache directory
OUTPUT_DIR=outputs        # Output directory for artifacts
```

#### Database Configuration
```bash
DB_URL=postgresql+psycopg2://postgres:password@localhost:5432/postgres
```

Format: `postgresql+psycopg2://user:password@host:port/database`

#### Model Configuration
```bash
MODEL_VERSION=v8_2        # Model version identifier
RANDOM_SEED=42            # Random seed for reproducibility
```

#### Performance Settings
```bash
N_JOBS=-1                 # Number of parallel jobs (-1 = all cores)
MEMORY_LIMIT=8GB          # Maximum memory allocation
```

#### External API Keys (if needed)
```bash
ALPHA_VANTAGE_API_KEY=your_api_key_here
FINANCIAL_API_KEY=your_api_key_here
```

### Setting Variables

**Windows (PowerShell):**
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:DB_URL = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
$env:RANDOM_SEED = "42"
```

**macOS/Linux (bash):**
```bash
export TF_CPP_MIN_LOG_LEVEL=2
export DB_URL=postgresql+psycopg2://postgres:@localhost:5432/postgres
export RANDOM_SEED=42
```

### Configuration Files

Alternatively, use JSON or YAML config files:

**config.json:**
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

Load in Python:
```python
from finance_ml import load_config
config = load_config("config.json")
config.apply_to_env()
```

---

## Testing

### Test Framework
- **unittest** (built-in, primary)
- **pytest** (optional, for advanced features)

### Running Tests

**Run all tests:**
```bash
python -m unittest -v
```

**Run with pytest (if installed):**
```bash
pytest tests/ -v --cov=finance_ml
```

**Run specific test module:**
```bash
python -m unittest tests.test_finance_ml_data -v
```

### Test Suite Overview

**18 test modules** with **comprehensive coverage** covering:

#### Core Package Tests
- `test_finance_ml_data.py` — Data loading, normalization, validation
- `test_finance_ml_features.py` — Feature engineering functions
- `test_finance_ml_models.py` — Classification, regression, ensembles
- `test_finance_ml_eval.py` — Analytics, visualizations, reporting
- `test_finance_ml_config.py` — Configuration management
- `test_cli.py` — Command-line interface

#### Component Tests
- `test_repository_setup.py` — Repository structure validation
- `test_data_quality.py` — Data validation and quality checks
- `test_loaders.py` — CSV and database loading
- `test_features.py` — Feature engineering utilities
- `test_build_features.py` — Feature pipeline
- `test_eda.py` — Exploratory data analysis
- `test_preprocess_and_training.py` — Preprocessing workflows
- `test_regression.py` — Regression models
- `test_classification.py` — Event classification
- `test_analytics.py` — Stock ranking and mispricing
- `test_visualizations.py` — Plotting functions

### Code Quality Tools

**Format code:**
```bash
black finance_ml tests
```

**Sort imports:**
```bash
isort finance_ml tests
```

**Type checking:**
```bash
mypy finance_ml --ignore-missing-imports
```

**Linting:**
```bash
flake8 finance_ml
```

### CI/CD

Currently, no CI workflows are included in this repository (no .github/workflows directory).

TODO:

- Add a GitHub Actions workflow for automated tests across Windows, Ubuntu, and macOS on Python 3.10 and 3.11.
- Optionally add code quality checks (black, isort, flake8, mypy) and coverage reporting.

You can run equivalent checks locally:

```bash
# Run tests
python -m unittest discover -s tests -v

# Check code formatting
black --check finance_ml tests

# Check import sorting
isort --check-only finance_ml tests

# Lint code
flake8 finance_ml

# Type checking
mypy finance_ml --ignore-missing-imports
```

---

## Project Structure

```
Finance_ML_Analytics_Platform/
├── finance_ml/                    # Main Python package (v0.3.0)
│   ├── __init__.py               # Package exports and version
│   ├── data.py                   # Data loading, normalization, validation (355 lines, 12 functions)
│   ├── features.py               # Feature engineering functions (182 lines, 6 functions)
│   ├── models.py                 # ML models: classification, regression, ensembles (517 lines, 10 functions)
│   ├── eval.py                   # Analytics, visualizations, reporting (398 lines, 9 functions)
│   ├── config.py                 # Configuration management (200+ lines)
│   └── cli.py                    # Command-line interface (300+ lines)
│
├── tests/                        # Unit tests (see tests/ for modules)
│   ├── test_finance_ml_data.py
│   ├── test_finance_ml_features.py
│   ├── test_finance_ml_models.py
│   ├── test_finance_ml_eval.py
│   ├── test_finance_ml_config.py
│   ├── test_cli.py
│   ├── test_repository_setup.py
│   ├── test_data_quality.py
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_build_features.py
│   ├── test_eda.py
│   ├── test_preprocess_and_training.py
│   ├── test_regression.py
│   ├── test_classification.py
│   ├── test_analytics.py
│   └── test_visualizations.py
│
├── data/                         # Regional equity data (CSV files)
│   ├── screening_us.csv
│   ├── screening_eu.csv
│   ├── screening_apac.csv
│   └── screening_rotw.csv
│
├── ml_finance_model_v8_2.ipynb  # Interactive Jupyter notebook (main entry point)
├── ml_finance_model_v8_2.py     # Python script with CLI (uses finance_ml package)
│
├── pyproject.toml               # Modern Python packaging configuration (PEP 621)
├── setup.py                     # Backward-compatible setup script
├── requirements.txt             # Core dependencies (pip)
├── Pipfile                      # Alternative dependencies (pipenv, Python 3.11)
│
├── create_equities_schema.sql   # PostgreSQL schema setup (243 lines)
├── import_equities_data.sql     # Data import script (250 lines)
├── environment_variables.txt    # Environment configuration examples
│
├── setup_environment.py         # Automated environment setup
├── validate_csv_import.py       # CSV data quality validator
├── analyze_notebook.py          # Notebook structure analyzer
├── update_notebook.py           # Notebook synchronizer
├── verify_notebook.py           # Notebook verification utility
│
├── README.md                    # Project overview and documentation
├── IMPROVEMENT_PLAN.md          # This file: improvement plan and roadmap
├── CHANGELOG.md                 # Version history
│
├── .junie/guidelines.md         # Development guidelines
├── .gitignore                   # Git ignore rules
│
└── outputs/                     # Generated artifacts (created at runtime)
```

### Package Modules Detail

#### `finance_ml.data` (355 lines, 12 functions)
- `setup_logging()`, `get_env()`: Utilities
- `load_from_csv()`, `load_from_db()`: Multi-source data loading
- `normalize_columns()`, `infer_region_from_filename()`: Data normalization
- `preprocess()`: Data cleaning and preprocessing
- `validate_schema()`, `check_missing_values()`: Quality checks
- `detect_outliers_iqr()`, `validate_numeric_ranges()`: Outlier detection
- `simple_eda()`: Exploratory data analysis

#### `finance_ml.features` (182 lines, 6 functions)
- `engineer_basic_ratios()`: EV/EBITDA, P/E, P/B ratios
- `engineer_margin_features()`: Gross, operating, net margins
- `engineer_volatility_features()`: Price volatility windows
- `engineer_revenue_cagr()`: Revenue growth metrics
- `build_features_and_target()`: Complete feature pipeline
- `create_event_labels()`: Event classification labels

#### `finance_ml.models` (517 lines, 10 functions)
- `train_event_classifier()`: Multi-class event classification (LightGBM/XGBoost)
- `build_regression_pipeline()`: Preprocessing and model selection
- `train_and_evaluate_regression()`: Baseline regression models
- `train_and_evaluate_regression_by_sector()`: Sector-optimized models
- `train_quantile_regression()`: Quantile regression for uncertainty
- `predict_quantile_regression()`: Quantile predictions
- `train_quantile_regression_by_sector()`: Sector-specific quantile models
- `train_stacking_ensemble()`: Meta-learner stacking
- `train_stacking_ensemble_by_sector()`: Sector-specific stacking

#### `finance_ml.eval` (398 lines, 9 functions)
- `calculate_mispricing_score()`: Valuation analysis
- `rank_undervalued_stocks()`: Top undervalued stocks
- `rank_overvalued_stocks()`: Top overvalued stocks
- `rank_stocks_by_sector()`: Sector-level rankings
- `export_predictions_to_excel()`: Excel reporting
- `create_sector_heatmap()`: Sector performance heatmap
- `create_interactive_prediction_plot()`: Interactive scatter plots
- `create_region_sector_heatmap()`: Region-sector analysis

#### `finance_ml.config` (200+ lines)
- `FinanceMLConfig`: Configuration dataclass
- `load_config()`: Load from environment/JSON/YAML
- `get_config()`, `set_config()`, `reset_config()`: Global config management

#### `finance_ml.cli` (300+ lines)
- `main()`: Main analysis pipeline CLI
- `analyze_main()`: Quick analysis CLI
- `validate_main()`: Data validation CLI
- `_load_data()`: Helper for data loading

---

## Code Quality Improvement Tasks

Based on recent PyCharm inspection analysis, the following improvements have been identified to enhance code quality,
maintainability, and adherence to Python best practices.

### Priority 1: Critical Issues (High Impact)

#### 1.1 Type Hints and Type Checking

**Status**: 🔴 High Priority  
**Effort**: Medium (2-3 days)  
**Impact**: Improves IDE support, reduces runtime errors, enhances documentation

**Tasks**:

- [ ] Add comprehensive type hints to all public functions in `finance_ml/` modules
- [ ] Add type hints to function parameters and return types
- [ ] Use `typing` module for complex types (List, Dict, Optional, Union, Tuple)
- [ ] Enable and configure `mypy` for strict type checking
- [ ] Fix all type checker warnings identified in inspection results
- [ ] Add type hints to CLI interface functions in `finance_ml/cli.py`

**Example**:

#### 1.7 Notebook Code Quality and Consistency

**Status**: 🟡 Medium-High Priority  
**Effort**: Small (1 day)  
**Impact**: Improves notebook reliability, maintainability, and user experience

**Context**: Code review of `ml_finance_model_main.ipynb` identified 10 issues affecting code quality, execution
reliability, and best practices adherence.

**Tasks**:

- [ ] **Import Organization** (Critical)
  - [ ] Add missing `NotebookConfig` to finance_ml imports (line 79-91)
  - [ ] Move `Path` import from line 159 to main imports section
  - [ ] Consolidate all imports into single cell at notebook top
  - [ ] Test cell execution order independence

- [ ] **Configuration Management** (Critical)
  - [ ] Fix config mutation anti-pattern (lines 155-164)
  - [ ] Use proper config initialization with output_dir parameter
  - [ ] Document configuration best practices in notebook
  - [ ] Add validation that config is immutable after load

- [ ] **Package Bug Fixes** (Critical)
  - [ ] Fix `simple_eda()` AttributeError: use `.dtypes` not `.dtype`
  - [ ] Add unit test to catch this regression
  - [ ] Remove AttributeError workaround from notebook after fix
  - [ ] Update finance_ml.eval module with proper DataFrame dtype handling

- [ ] **Type Safety and Validation** (High)
  - [ ] Add isinstance checks for load_stock_data() return value
  - [ ] Improve error messages with type information
  - [ ] Add validation checkpoints throughout pipeline
  - [ ] Document expected types in docstrings

- [ ] **Error Handling** (High)
  - [ ] Flatten nested try-except blocks (lines 193-225)
  - [ ] Use consistent error handling pattern across cells
  - [ ] Log errors with full stack traces (exc_info=True)
  - [ ] Provide user-friendly error messages

- [ ] **Code Cleanup** (Medium)
  - [ ] Replace redundant feature flag variables with direct cfg access
  - [ ] Add deprecation comments for backward-compatible variables
  - [ ] Update logger name from `__name__` to 'finance_ml_notebook'
  - [ ] Remove duplicate imports and unused variables

- [ ] **Documentation** (Medium)
  - [ ] Complete any truncated markdown cells
  - [ ] Add execution order notes to notebook header
  - [ ] Document configuration options in notebook
  - [ ] Add troubleshooting section for common issues

**Testing Checklist**:

- [ ] Restart kernel and run all cells in order - no errors
- [ ] Run cells out of order - appropriate error messages
- [ ] Test with missing dependencies - graceful degradation
- [ ] Verify all feature flags work correctly
- [ ] Check output directory creation and artifact saving

**Success Criteria**:

- No NameError or AttributeError when running notebook top-to-bottom
- All imports consolidated in single cell
- Config is properly initialized without mutation
- simple_eda() works without catching AttributeError
- Type validation provides clear error messages
- Error handling is consistent and informative

**Related Issues**:

- Links to Phase 9.1-9.7 implementation tasks
- Dependencies on finance_ml package bug fixes
- Integration with notebook utilities module

**References**:

- `ml_finance_model_main.ipynb` - main notebook
- `finance_ml/eval.py` - simple_eda() function
- `finance_ml/notebook_utils.py` - utility functions
- `finance_ml/notebook_config.py` - NotebookConfig class

---

## Immediate Next Steps (Current Sprint)

### Notebook Quality Improvements

**Priority**: High - These improvements will significantly enhance notebook reliability and maintainability

**Estimated Total**: 1 day of focused work

#### 1. Fix Critical Import Issues (30 minutes)

- Add NotebookConfig to imports
- Move Path to main imports
- Test import cell independence

#### 2. Fix Configuration Anti-Pattern (1 hour)

- Refactor config initialization
- Remove config mutation
- Add validation tests

#### 3. Fix simple_eda Bug (2 hours)

- Locate and fix .dtype/.dtypes issue in finance_ml/eval.py
- Add unit test
- Remove workaround from notebook
- Test EDA pipeline end-to-end

#### 4. Add Type Safety (1 hour)

- Add isinstance checks
- Improve error messages
- Add validation docstrings

#### 5. Refactor Error Handling (2 hours)

- Flatten nested try-except
- Standardize error pattern
- Add proper logging

---

### Phase 0 — Foundations and Housekeeping (Week 0) ✓
- Align environment and packaging ✓
- Repository hygiene ✓
- Configuration ✓
- Automated setup script created ✓
- Notebook utility scripts created ✓

### Phase 1 — Data Ingestion and Validation (Week 1) ✓
- Data source unification ✓
- Data validation ✓
- Deterministic region tagging ✓

### Phase 2 — EDA and Feature Engineering (Week 2) ✓
- EDA automation ✓
- Feature engineering ✓

### Phase 3 — Modeling: Classification (Week 3) ✓
- Event classification labels ✓
- Models ✓
- Validation and artifacts ✓

### Phase 4 — Modeling: Sector-optimized Regression (Week 4) ✓
- Baseline ✓
- Validation ✓
- Uncertainty and stacking ✓

### Phase 5 — Analytics and Reporting (Week 5) ✓
- Mispricing score ✓
- Exports ✓
- Visualizations ✓

### Phase 6 — Testing, Reproducibility, and CI (Week 5–6) ✓
- Unit tests ✓
- Reproducibility ✓
- CI ✓

### Phase 7 — Packaging and Modularity (Week 6+) ✓
- Code structure (TDD implementation) ✓
- Modern packaging ✓
- Config management ✓

### Phase 8 — Documentation and Versioning (Ongoing) ✓
- Documentation ✓
- Versioning ✓

### Phase 9 — Advanced Stock Prediction ML System (In Progress)

**Business Objective**: Predict Stock Price Targets for all stocks in the `all_stocks` dataframe  
**Target Variable**: "Predicted Price Target" for regression modeling  
**Primary Notebook**: `ml_finance_model_main.ipynb` (v8_3/v0.3.0)  
**Reference Report**: `Stock_Prediction_Analysis_Report_20250806_131704.xlsx`  
**Alignment**: ML Project Checklist (`reference material/ml-project-checklist.md`)  
**Reference Materials**: 19 Jupyter notebooks (01-19) covering ML landscape through deployment at scale

This phase implements a sophisticated Stock Prediction Machine Learning System with 8 key workflow steps, integrating
advanced techniques from industry-standard ML references:

#### 9.1 Loading and Preprocessing Financial Data from Multiple Regions

**Reference**: `13_loading_and_preprocessing_data.ipynb`, `02_end_to_end_machine_learning_project.ipynb`  
**Data Source**: `all_stocks` dataframe from PostgreSQL/CSV (US, EU, APAC, ROTW)  
**Schema**: 234 columns mapped via `COLUMN_MAPPING_SUMMARY.md` with LTM/NTM/FY time periods

**Implementation Strategy**:

- [x] **Core Implementation** (Complete - Phase 9.1 TDD Implementation)
  - [x] Add robust outlier detection: IQR by sector, z-score thresholding, isolation forest
  - [x] Implement winsorization (1-99th percentile) for extreme values by sector
  - [x] Add data quality scoring: completeness, consistency, validity checks
  - [x] Implement temporal validation and time-aware splits
  - [x] Add support for temporal features (snapshot dates, quarterly reporting cycles)
  - [x] Implement time-series cross-validation (expanding window, rolling window)
  - [x] Ensure no data leakage: strict past→future splits, group by ticker in CV
  - [x] Add data versioning and lineage tracking
  - [x] Implement data versioning with timestamp and hash-based tracking
  - [x] Add provenance tracking: source, transformations, quality metrics
  - [x] Create data catalog with metadata (schema, statistics, data quality)
  - [x] Handle missing values: sector-specific imputation (median/mean)
  - [x] Implement feature scaling pipelines: StandardScaler, RobustScaler, MinMaxScaler by sector
  - [x] **Testing**: Integration tests for multi-region data loading with edge cases (56 tests passing)
  - [x] **Documentation**: Preprocessing decisions documented in PHASE_9_1_TDD_IMPLEMENTATION_SUMMARY.md

- [x] **Phase 9.1 Enhancements** (Complete - v0.3.0, 2025-10-30)
  - [x] **KNN Imputation with Sector-Aware Logic** (`impute_missing_values_knn_sector`)
    - [x] Sector-specific neighbor-based imputation preserving sector characteristics
    - [x] Configurable k neighbors with automatic adjustment for small sectors
    - [x] Fallback to global KNN when sector column missing
    - [x] Test coverage: 6/6 tests passing
    - [x] Notebook integration: Section 9.1.6.1 with usage examples
  - [x] **Regularized Target Encoding** (`RegularizedTargetEncoder`)
    - [x] Cross-validated target encoding with smoothing regularization
    - [x] Prevents overfitting and handles rare categories gracefully
    - [x] sklearn-compatible transformer (BaseEstimator, TransformerMixin)
    - [x] Test coverage: 5/5 tests passing
    - [x] Notebook integration: Section 9.1.6.2 with usage examples
  - [x] **Custom Financial Transformers** (`FinancialRatioTransformer`, `SafeDivisionTransformer`,
    `ValuationRatioTransformer`)
    - [x] Safe division handling for financial ratios (division by zero, infinities, negatives)
    - [x] sklearn-compatible pipeline integration
    - [x] Automatic NaN/Inf handling with configurable strategies
    - [x] Test coverage: 6/6 tests passing
    - [x] Notebook integration: Section 9.1.6.3 with usage examples
  - [x] **Data Quality Dashboard** (`generate_data_quality_dashboard`, `export_profiling_report`)
    - [x] Interactive HTML reports with comprehensive profiling
    - [x] Multi-method support: ydata-profiling, sweetviz, minimal fallback
    - [x] Missing value analysis, distributions, correlations, data quality warnings
    - [x] Test coverage: 3/4 tests passing (1 requires optional dependency)
    - [x] Notebook integration: Section 9.1.6.4 with usage examples
  - [x] **Summary Documentation**: PHASE_9_1_ENHANCEMENTS_SUMMARY.md with API docs and integration guide
  - [x] **Notebook Integration**: Section 9.1.6 with 218 lines of comprehensive examples

- [ ] **Future Enhancements** (Deferred to future phases)
  - [ ] TensorFlow Dataset API patterns (optional TensorFlow dependency)
    - [ ] Implement data pipeline with prefetching and caching for large datasets
    - [ ] Note: Skipped in current phase to avoid optional dependency; can be implemented when TensorFlow is required
  - [ ] Advanced categorical encoding extensions
    - [ ] Frequency encoding, hash encoding, CatBoost encoding
  - [ ] Iterative imputation (MICE) as alternative to KNN

#### 9.2 Exploratory Data Analysis of Financial Metrics

**Reference**: `02_end_to_end_machine_learning_project.ipynb`, `08_dimensionality_reduction.ipynb`,
`09_unsupervised_learning.ipynb`  
**Current Implementation**: `ml_finance_model_main.ipynb` cells with matplotlib/seaborn/plotly visualizations

**Implementation Strategy**:

- [x] Enhance `finance_ml.eval.simple_eda()` with comprehensive statistical analysis (Phase 9.2 Integration -
  2025-10-30)
    - [x] Add automated correlation analysis: Pearson, Spearman, Kendall tau, distance correlation (Phase 9.2
      Continuation - 2025-10-30)
    - [x] Implement feature importance via random forest, mutual information, SHAP values (integrated with target_column
      parameter)
    - [x] Add distribution analysis: normality tests (Shapiro-Wilk, Kolmogorov-Smirnov), skewness, kurtosis
    - [x] Implement outlier detection visualization: box plots, violin plots, scatter with z-scores (Phase 9.2
      Continuation - 2025-10-30)
    - [x] Add multivariate analysis: PCA visualization, t-SNE, UMAP for high-dimensional exploration (Phase 9.2
      Continuation - 2025-10-30)
- [x] Sector and region-specific benchmarking (Phase 9.2 Benchmarking - 2025-10-30)
    - [x] Create sector-wise distribution comparisons (P/E, P/B, EV/EBITDA, margins)
    - [x] Add regional valuation metric comparisons with statistical significance tests
    - [x] Implement peer group analysis within sectors
    - [x] Add time-series trend analysis for key metrics (if temporal data available)
    - [x] Notebook integration: Added 6 demonstration cells showcasing all benchmarking functions (2025-10-30)
- [ ] Automated EDA report generation
  - [ ] Integrate pandas-profiling for comprehensive data profiling
  - [ ] Generate HTML/PDF reports with executive summary, data quality, correlations, distributions
  - [ ] Add custom financial metric dashboards (valuation, profitability, growth, leverage)
  - [ ] Implement alert system for data quality issues and anomalies
- [ ] Interactive EDA dashboards
  - [ ] Create Plotly Dash or Streamlit dashboard with drill-down capabilities
  - [ ] Add interactive filters: sector, region, market cap, valuation ranges
  - [ ] Implement comparison tools: stock vs. sector average, peer comparisons
  - [ ] Add real-time data quality monitoring dashboard
- [ ] Statistical hypothesis testing framework
  - [ ] Implement sector mean comparison tests (ANOVA, Kruskal-Wallis)
  - [ ] Add region performance comparison tests (t-tests, Mann-Whitney U)
  - [ ] Test for market efficiency hypotheses using price/target relationships
- [ ] **Testing**: Add tests for EDA report generation, statistical calculations, visualization outputs
- [ ] **Documentation**: Document EDA insights and their implications for modeling
- [x] **Bug Fixes**: Fix simple_eda() AttributeError
    - [x] Replace incorrect `.dtype` access with `.dtypes` on DataFrames
    - [x] Add unit test to prevent regression
    - [x] Update notebook to remove AttributeError workaround
    - [x] Document DataFrame dtype handling best practices (see finance_ml/eval.py lines 189-201)

#### 9.3 Advanced Feature Engineering with Sector-Specific Optimizations ✓

**Reference**: `02_end_to_end_machine_learning_project.ipynb`, `13_loading_and_preprocessing_data.ipynb`  
**Data Schema**: 234 columns from `COLUMN_MAPPING_SUMMARY.md` with LTM/NTM/FY variants  
**Module**: `finance_ml.advanced_features` (978 lines, 13 functions)  
**Status**: ✅ COMPLETE (2025-10-31)

**Implementation Results**:

- [x] **Core Implementation** (Complete - Phase 9.3 TDD Implementation)
  - [x] Comprehensive financial ratios: valuation, profitability, leverage, liquidity, efficiency, growth
  - [x] Sector-specific features for Financials, Energy/Materials, Technology, Healthcare, Consumer, Industrials,
    Utilities
  - [x] Temporal features: fiscal quarter, month, year, days since reference
  - [x] Market microstructure: volatility (30/60/90d), momentum, moving averages, price range
  - [x] Non-linear transforms: log, square root, inverse for skewed distributions
  - [x] Feature interactions: pairwise interactions and polynomial features (degree 2-3)
  - [x] Relative value features: sector median deviation, z-scores, percentile ranks
  - [x] Feature selection: mutual information, Random Forest, SHAP, RFE with cross-validation
  - [x] Orchestrator function: `build_comprehensive_features()` for full pipeline
  - [x] **Testing**: 88 comprehensive unit tests (100% passing)
  - [x] **Coverage**: 93% code coverage (exceeds 80% requirement by 13 percentage points)
  - [x] **Documentation**: PHASE_9_3_TDD_IMPLEMENTATION_SUMMARY.md

**Key Features Implemented**:

1. **Financial Ratio Engineering** (6 functions, 29 tests):
  - `engineer_valuation_ratios()` - P/E, P/B, P/S, EV/EBITDA, EV/Sales, PEG, Dividend Yield
  - `engineer_profitability_ratios()` - ROE, ROA, ROIC, Gross/Operating/Net Margins
  - `engineer_leverage_ratios()` - Debt/Equity, Net Debt/EBITDA, Interest Coverage, Debt/Assets
  - `engineer_liquidity_ratios()` - Current, Quick, Cash ratios, Working Capital/Sales
  - `engineer_efficiency_ratios()` - Asset/Inventory/Receivables Turnover, Revenue/Employee
  - `engineer_growth_metrics()` - Revenue/EPS/EBITDA Growth YoY

2. **Sector-Specific Features** (1 function, 19 tests):
  - **Financials**: TBV, P/TBV, Net Interest Margin, Efficiency Ratio
  - **Energy/Materials**: CAPEX intensity, Asset turnover
  - **Technology**: R&D intensity, SG&A efficiency, Rule of 40, Cash burn rate
  - **Healthcare**: R&D/Revenue ratio
  - **Consumer**: Inventory days, Marketing efficiency
  - **Industrials**: CAPEX/Depreciation, Working capital efficiency
  - **Utilities**: Dividend payout ratio

3. **Advanced Features** (4 functions, 14 tests):
  - `engineer_temporal_features()` - Fiscal quarter, month, year, days since reference
  - `engineer_market_microstructure_features()` - Volatility, momentum, moving averages, price range
  - `engineer_nonlinear_transforms()` - Log, sqrt, inverse transforms
  - `create_feature_interactions()` - Pairwise interactions, polynomial features

4. **Relative Value & Selection** (4 functions, 13 tests):
  - `create_relative_value_features()` - Sector median deviation, z-scores, percentiles
  - `calculate_feature_importance_mutual_info()` - Mutual information-based importance
  - `calculate_feature_importance_rf()` - Random Forest-based importance
  - `calculate_feature_importance_shap()` - SHAP value-based importance
  - `calculate_feature_importance_rfe()` - Recursive Feature Elimination

5. **Pipeline Orchestration** (1 function, 3 tests):
  - `build_comprehensive_features()` - End-to-end feature engineering pipeline

**Test Coverage Summary**:

- **Total Tests**: 88 (55 original + 33 expanded)
- **Pass Rate**: 100% (88/88 passing)
- **Code Coverage**: 93% (350 statements, 327 covered, 23 missed)
- **Execution Time**: ~41 seconds
- **Test Categories**: Normal operation (64%), Edge cases (27%), Error handling (9%)

**Files Modified**:

- `finance_ml/advanced_features.py` (978 lines)
- `tests/test_advanced_features.py` (1000 lines, 88 tests)
- `docs/improvement_plan/PHASE_9_3_TDD_IMPLEMENTATION_SUMMARY.md` (282 lines)

#### 9.4 Multi-Class Classification of Financial Events (Sophisticated Models)

**Reference**: `03_classification.ipynb`, `07_ensemble_learning_and_random_forests.ipynb`,
`10_neural_nets_with_keras.ipynb`, `11_training_deep_neural_networks.ipynb`  
**Purpose**: Classify financial events/catalysts to use as meta-features in regression models  
**Classes**: Neutral (0), Positive Catalyst (1), Negative Catalyst (2) - expandable to more granular classes

**Implementation Strategy**:

- [ ] Enhance event label creation in `finance_ml.models`
  - [ ] Expand `create_event_labels()` with sophisticated event definitions:
    - [ ] Price momentum events (breakouts, breakdowns, volatility spikes)
    - [ ] Analyst rating changes (upgrades, downgrades, initiation)
    - [ ] Valuation events (undervalued, overvalued based on P/E, P/B percentiles)
    - [ ] Fundamental events (earnings surprises, margin expansion/contraction)
    - [ ] Market events (sector rotation, regional trends)
  - [ ] Implement multi-class labels with confidence scores
  - [ ] Add temporal validation: ensure labels use only past information (no leakage)
  - [ ] Create balanced label sets with stratification by sector/region
- [ ] Implement diverse classification models
  - [ ] **Gradient Boosting Ensembles**:
    - [ ] XGBoost with hyperparameter tuning (learning rate, max_depth, subsample, colsample)
    - [ ] LightGBM with categorical feature support and early stopping
    - [ ] CatBoost with built-in categorical encoding and ordered boosting
    - [ ] Compare model performances with cross-validation
  - [ ] **Random Forest and Bagging**:
    - [ ] Random Forest with feature importance analysis
    - [ ] Extra Trees for variance reduction
    - [ ] Bagging with diverse base estimators
  - [ ] **Neural Network Classifiers** (TensorFlow/Keras):
    - [ ] Feedforward DNN with batch normalization and dropout (ref: notebook 10, 11)
    - [ ] Architecture: Input → Dense(256) + BN + Dropout → Dense(128) + BN + Dropout → Dense(64) → Output(3)
    - [ ] Activation: ReLU for hidden layers, softmax for output
    - [ ] Optimizer: Adam with learning rate scheduling
    - [ ] Regularization: L1/L2, dropout (0.3-0.5), early stopping
    - [ ] Class weights for imbalanced data
  - [ ] **Support Vector Machines** (ref: notebook 05):
    - [ ] SVM with RBF and polynomial kernels
    - [ ] One-vs-Rest and One-vs-One strategies
  - [ ] **Advanced Ensemble Methods** (ref: notebook 07):
    - [ ] Voting classifier (soft/hard voting across diverse models)
    - [ ] Stacking classifier with meta-learner (logistic regression or XGBoost)
    - [ ] Blending with holdout validation set
- [ ] Handle class imbalance
  - [ ] Implement SMOTE (Synthetic Minority Over-sampling Technique)
  - [ ] Add ADASYN (Adaptive Synthetic Sampling)
  - [ ] Use class weights in model training
  - [ ] Apply under-sampling for majority class (random, Tomek links, NearMiss)
  - [ ] Combine over/under-sampling strategies
- [ ] Model evaluation and selection
  - [ ] Implement stratified k-fold cross-validation (grouped by sector/ticker)
  - [ ] Evaluate with precision, recall, F1-score, AUC-ROC, AUC-PR per class
  - [ ] Create comprehensive confusion matrices with visualization
  - [ ] Add classification reports with per-class metrics
  - [ ] Implement learning curves to diagnose bias/variance
- [ ] Feature importance and interpretation
  - [ ] Extract feature importance from tree-based models
  - [ ] Compute SHAP values for model explainability
  - [ ] Analyze per-class feature importance
  - [ ] Identify key drivers for each event type
- [ ] Export classification outputs for regression
  - [ ] Generate class probabilities (3 probabilities per stock)
  - [ ] Create binary indicators for each class
  - [ ] Add confidence scores (max probability)
  - [ ] Export as meta-features for downstream regression models
- [ ] **Testing**: Add comprehensive tests for event labeling, model training, class imbalance handling, probability
  export
- [ ] **Documentation**: Document event definitions, model architectures, hyperparameters, and interpretation guidelines

#### 9.5 Sector-Optimized Regression Models Enhanced with Classification Features

**Reference**: `04_training_linear_models.ipynb`, `07_ensemble_learning_and_random_forests.ipynb`,
`10_neural_nets_with_keras.ipynb`, `11_training_deep_neural_networks.ipynb`,
`12_custom_models_and_training_with_tensorflow.ipynb`, `19_training_and_deploying_at_scale.ipynb`  
**Target Variable**: "Predicted Price Target" for all stocks in `all_stocks` dataframe  
**Strategy**: Train sector-specific models + global model with classification meta-features

**Implementation Strategy**:

- [ ] Integrate classification features into regression pipeline
  - [ ] Add classification probabilities (3 features: P(Neutral), P(Positive), P(Negative))
  - [ ] Include predicted event class as categorical feature
  - [ ] Add classification confidence score (max probability)
  - [ ] Create interaction features: classification_probs × valuation_metrics
- [ ] Implement diverse regression model architectures
  - [ ] **Linear Models** (ref: notebook 04):
    - [ ] Ridge regression with alpha tuning (L2 regularization)
    - [ ] Lasso regression for feature selection (L1 regularization)
    - [ ] Elastic Net combining L1 and L2 penalties
    - [ ] Polynomial regression (degree 2-3) with regularization
    - [ ] Bayesian Ridge for uncertainty estimation
  - [ ] **Gradient Boosting Models**:
    - [ ] XGBoost regressor with early stopping and CV-based tuning
    - [ ] LightGBM regressor with dart/goss boosting
    - [ ] CatBoost regressor with categorical feature encoding
    - [ ] Histogram-based gradient boosting (sklearn HistGradientBoostingRegressor)
  - [ ] **Random Forests and Tree Ensembles** (ref: notebook 07):
    - [ ] Random Forest regressor with feature importance
    - [ ] Extra Trees regressor for variance reduction
    - [ ] Gradient Boosting Regressor (sklearn)
  - [ ] **Neural Network Regressors** (ref: notebooks 10, 11, 12):
    - [ ] Feedforward DNN: Input → Dense(512, ReLU) + BN + Dropout → Dense(256, ReLU) + BN + Dropout → Dense(128,
      ReLU) + Dropout → Dense(1, linear)
    - [ ] Advanced techniques:
      - [ ] Batch normalization for stable training
      - [ ] Dropout (0.2-0.4) for regularization
      - [ ] L2 regularization on weights
      - [ ] Learning rate scheduling (ReduceLROnPlateau, ExponentialDecay)
      - [ ] Early stopping with patience
      - [ ] Gradient clipping to prevent exploding gradients
    - [ ] Loss functions: MSE, MAE, Huber (robust to outliers)
    - [ ] Optimizer: Adam with beta tuning or AdamW with weight decay
    - [ ] Advanced architectures:
      - [ ] Residual connections (ResNet-style) for deep networks
      - [ ] Wide & Deep architecture (linear + deep branches)
      - [ ] Attention mechanisms for feature weighting
- [ ] Sector-specific model optimization
  - [ ] Train separate models for each major sector (Financials, Technology, Healthcare, Consumer, Energy, Industrials,
    Utilities)
  - [ ] Implement sector-specific feature selection (relevant features per sector)
  - [ ] Apply sector-specific hyperparameter tuning
  - [ ] Create sector-specific preprocessing pipelines (scaling, outlier treatment)
  - [ ] Ensemble sector models with global fallback for rare sectors
- [ ] Automated hyperparameter optimization
  - [ ] Implement Optuna for Bayesian optimization
  - [ ] Use GridSearchCV/RandomizedSearchCV for baseline tuning
  - [ ] Apply Hyperband for efficient early stopping
  - [ ] Create optimization study with cross-validation
  - [ ] Track hyperparameter experiments with MLflow or Weights & Biases
  - [ ] Implement automated model selection based on validation metrics
- [ ] Advanced ensemble methods (ref: notebook 07)
  - [ ] **Stacking Regressor**:
    - [ ] Base models: XGBoost, LightGBM, CatBoost, Random Forest, Neural Network
    - [ ] Meta-learner: Ridge, Lasso, or simple Linear Regression
    - [ ] Use out-of-fold predictions for meta-features
    - [ ] Implement sector-specific stacking ensembles
  - [ ] **Voting Regressor**:
    - [ ] Combine diverse models with weighted averaging
    - [ ] Optimize weights with validation set
  - [ ] **Blending**:
    - [ ] Train models on different data splits
    - [ ] Combine predictions with holdout-based weighting
- [ ] Quantile regression for uncertainty estimation
  - [ ] Implement quantile regression for prediction intervals (5th, 25th, 50th, 75th, 95th percentiles)
  - [ ] Use LightGBM/XGBoost with quantile objectives
  - [ ] Train separate models for each quantile
  - [ ] Combine with conformal prediction for calibrated intervals
  - [ ] Provide confidence bands: [lower_bound, prediction, upper_bound]
- [ ] Model persistence and versioning (ref: notebook 19)
  - [ ] Save trained models with joblib/pickle (sklearn models) or SavedModel (TensorFlow)
  - [ ] Implement model versioning with timestamps and metadata
  - [ ] Track model lineage: features used, hyperparameters, training data version
  - [ ] Create model registry for production deployment
  - [ ] Add model signature validation (input/output schemas)
  - [ ] Implement A/B testing framework for model comparison
- [ ] **Testing**: Add comprehensive tests for model training, sector-specific logic, ensemble methods, quantile
  regression, model persistence
- [ ] **Documentation**: Document model architectures, hyperparameters, training procedures, and sector-specific
  strategies

#### 9.6 Model Evaluation and Error Analysis

**Reference**: `02_end_to_end_machine_learning_project.ipynb`, ML Project Checklist step 6  
**Purpose**: Comprehensive model performance assessment and error diagnosis  
**Metrics**: MAE, RMSE, MAPE, R², residual analysis, SHAP values

**Implementation Strategy**:

- [ ] Enhance `finance_ml.eval` with comprehensive evaluation framework
  - [ ] **Core Regression Metrics**:
    - [ ] MAE (Mean Absolute Error) - interpretable dollar error
    - [ ] RMSE (Root Mean Squared Error) - penalizes large errors
    - [ ] MAPE (Mean Absolute Percentage Error) - relative error
    - [ ] R² (Coefficient of Determination) - variance explained
    - [ ] Median Absolute Error - robust to outliers
    - [ ] Max Error - worst-case performance
  - [ ] **Sector and Region-specific Metrics**:
    - [ ] Compute metrics by sector (7 major sectors)
    - [ ] Compute metrics by region (US, EU, APAC, ROTW)
    - [ ] Compute metrics by market cap buckets (Large, Mid, Small)
    - [ ] Compute metrics by valuation quartiles (High P/E, Low P/E)
    - [ ] Create performance heatmaps (Sector × Region)
  - [ ] **Residual Analysis**:
    - [ ] Plot residuals vs. predicted values (check homoscedasticity)
    - [ ] Q-Q plots for normality assessment
    - [ ] Histogram of residuals with normality tests
    - [ ] Residuals vs. features to detect non-linearities
    - [ ] Identify systematic bias patterns (over/under-prediction by segment)
  - [ ] **Error Bucketing and Segmentation**:
    - [ ] Group errors by market cap: Large (>$10B), Mid ($2-10B), Small (<$2B)
    - [ ] Group errors by volatility: Low, Medium, High (percentile-based)
    - [ ] Group errors by sector and analyze sector-specific challenges
    - [ ] Identify outlier predictions (>3 std dev from mean error)
    - [ ] Analyze prediction errors for undervalued vs. overvalued stocks
- [ ] Model interpretation and explainability
  - [ ] **SHAP (SHapley Additive exPlanations)**:
    - [ ] Compute SHAP values for tree-based models and neural networks
    - [ ] Create SHAP summary plots (global feature importance)
    - [ ] Generate SHAP dependence plots (feature interactions)
    - [ ] Create SHAP waterfall plots for individual predictions
    - [ ] Analyze SHAP values by sector (sector-specific drivers)
  - [ ] **LIME (Local Interpretable Model-agnostic Explanations)**:
    - [ ] Apply LIME for local explanations of individual predictions
    - [ ] Compare LIME and SHAP for consistency
  - [ ] **Feature Importance**:
    - [ ] Extract feature importance from tree-based models
    - [ ] Compute permutation importance for all models
    - [ ] Rank features by importance and stability across folds
- [ ] Cross-validation framework
  - [ ] **Time-series aware cross-validation** (if temporal data available):
    - [ ] Implement expanding window CV (train on past, test on future)
    - [ ] Implement rolling window CV with fixed train/test sizes
    - [ ] Ensure strict temporal ordering (no future data in training)
  - [ ] **Stratified cross-validation**:
    - [ ] Stratify by sector to maintain sector balance in folds
    - [ ] Stratify by region for balanced geographic representation
  - [ ] **Grouped cross-validation**:
    - [ ] Group by ticker to prevent data leakage (all observations of a ticker in same fold)
  - [ ] **Custom cross-validation**:
    - [ ] Implement custom splitters for financial data
    - [ ] Support multiple CV strategies with comparison
- [ ] Model comparison and selection
  - [ ] Create model comparison dashboard (table with metrics per model)
  - [ ] Implement statistical significance tests (paired t-test, Wilcoxon)
  - [ ] Generate learning curves (training size vs. performance)
  - [ ] Create validation curves (hyperparameter vs. performance)
  - [ ] Implement automated model selection based on validation metrics and business rules
  - [ ] Compare sector-specific models vs. global model performance
- [ ] Bias-variance diagnosis
  - [ ] Analyze training vs. validation performance (detect overfitting/underfitting)
  - [ ] Create bias-variance decomposition plots
  - [ ] Identify optimal model complexity
- [ ] **Testing**: Add comprehensive tests for metric calculations, residual analysis, SHAP computation, CV strategies
- [ ] **Documentation**: Document evaluation methodology, interpretation guidelines, and model selection criteria

#### 9.7 Identification of Under/Overvalued Stocks with Visualization

**Reference**: `02_end_to_end_machine_learning_project.ipynb`, ML Project Checklist step 7  
**Purpose**: Identify investment opportunities based on predicted price targets  
**Output**: Ranked lists, interactive dashboards, valuation reports

**Implementation Strategy**:

- [ ] Enhance stock valuation analysis in `finance_ml.eval`
  - [ ] **Mispricing Score Calculation**:
    - [ ] Base formula: `(Predicted_Target - Last_Price) / Last_Price * 100` (% upside/downside)
    - [ ] Add confidence intervals from quantile regression (lower/upper bounds)
    - [ ] Compute risk-adjusted mispricing: `(Expected_Return - Risk_Free_Rate) / Volatility`
    - [ ] Apply sector-relative adjustments (compare to sector median)
  - [ ] **Valuation Categories**:
    - [ ] Strong Buy: Mispricing > +20% with high confidence
    - [ ] Buy: Mispricing +10% to +20%
    - [ ] Hold: Mispricing -10% to +10%
    - [ ] Sell: Mispricing -20% to -10%
    - [ ] Strong Sell: Mispricing < -20% with high confidence
    - [ ] Apply sector-specific thresholds (volatile sectors get wider bands)
  - [ ] **Sector-Relative Valuation**:
    - [ ] Calculate z-scores for P/E, P/B, EV/EBITDA within sector
    - [ ] Identify stocks trading at discount/premium to sector
    - [ ] Compute percentile ranks within sector and peer group
  - [ ] **Multi-Factor Screening**:
    - [ ] Combine valuation, quality (ROE, margins), growth (revenue CAGR)
    - [ ] Apply custom scoring formulas (e.g., Value Score = Valuation × Quality)
    - [ ] Filter by liquidity, market cap, sector preferences
- [ ] Automated stock screening and ranking
  - [ ] **Top Undervalued Stocks** (buy opportunities):
    - [ ] Rank by mispricing score (highest upside first)
    - [ ] Filter by sector, region, market cap
    - [ ] Add quality filters (profitability, leverage, growth)
    - [ ] Export top 20-50 stocks with detailed metrics
  - [ ] **Top Overvalued Stocks** (sell/short opportunities):
    - [ ] Rank by negative mispricing (highest downside first)
    - [ ] Identify potential shorts or portfolio exits
  - [ ] **Sector Leaders and Laggards**:
    - [ ] Identify best/worst stocks within each sector
    - [ ] Compare to sector benchmarks
- [ ] Interactive valuation dashboards
  - [ ] **Plotly Dash or Streamlit Dashboard**:
    - [ ] Scatter plot: Current Price vs. Predicted Target (color by sector)
    - [ ] Interactive filters: sector, region, market cap, valuation categories
    - [ ] Click on stock to see detailed profile and SHAP explanation
    - [ ] Comparison tools: stock vs. peers, sector averages
  - [ ] **Valuation Heatmaps**:
    - [ ] Sector × Region heatmap with average mispricing
    - [ ] Market cap × Sector heatmap with opportunity counts
    - [ ] Correlation heatmap: valuation metrics vs. predicted returns
  - [ ] **Time Series Tracking** (if temporal data available):
    - [ ] Track mispricing scores over time
    - [ ] Identify persistent opportunities or mean reversion
- [ ] PDF report generation
  - [ ] Create professional stock recommendation reports with ReportLab
  - [ ] Include: executive summary, top opportunities, risk warnings, model explanation
  - [ ] Add charts: valuation scatter, sector breakdown, confidence intervals
  - [ ] Customize reports by client preferences
- [ ] **Testing**: Add tests for mispricing calculations, ranking logic, filtering, dashboard components
- [ ] **Documentation**: Document valuation methodology, screening criteria, and interpretation guidelines

#### 9.8 Comprehensive Analytics: Predicted vs. Analyst Price Target

**Reference**: `Stock_Prediction_Analysis_Report_20250806_131704.xlsx`, ML Project Checklist step 7  
**Purpose**: Compare model predictions against analyst consensus targets  
**Output**: Excel reports, dashboards, accuracy metrics, investment insights

**Implementation Strategy**:

- [ ] Create dedicated analytics module in `finance_ml.eval`
  - [ ] **Prediction vs. Analyst Target Comparison**:
    - [ ] Calculate differences: `Model_Target - Analyst_Target`
    - [ ] Compute agreement rate: % of stocks where model and analysts agree (same direction)
    - [ ] Identify disagreement opportunities: stocks where model significantly differs
    - [ ] Analyze directional accuracy: % where model correctly predicts direction vs. current price
  - [ ] **Accuracy Metrics**:
    - [ ] Mean Absolute Error: `|Model_Target - Actual_Future_Price|` (if available)
    - [ ] Directional accuracy: % correct up/down predictions
    - [ ] Hit rate by confidence level (high confidence predictions more accurate?)
    - [ ] Calibration: predicted upside vs. realized upside
  - [ ] **Agreement/Disagreement Analysis**:
    - [ ] Segment by sector: which sectors show highest model-analyst agreement?
    - [ ] Segment by region: regional differences in prediction accuracy
    - [ ] Analyze by stock characteristics: large vs. small cap, value vs. growth
    - [ ] Identify systematic biases: does model consistently over/under-predict vs. analysts?
  - [ ] **Temporal Tracking** (if multiple snapshots available):
    - [ ] Track prediction changes over time
    - [ ] Analyze model stability vs. analyst target revisions
    - [ ] Identify early signals: model predicts before analyst consensus shifts
- [ ] Generate comprehensive Excel reports (match Stock_Prediction_Analysis_Report format)
  - [ ] **Sheet 1: Executive Summary**:
    - [ ] Overall statistics: total stocks, avg predicted target, avg analyst target
    - [ ] Model performance metrics: MAE, RMSE, R², directional accuracy
    - [ ] Top opportunities summary: count by valuation category
    - [ ] Sector breakdown: performance by sector
  - [ ] **Sheet 2: Detailed Stock List**:
    - [ ] Columns: Ticker, Sector, Region, Current Price, Predicted Target, Analyst Target, Mispricing %, Confidence
      Interval, Valuation Category, Model-Analyst Difference
    - [ ] Sortable and filterable
    - [ ] Conditional formatting for quick insights (green=buy, red=sell)
  - [ ] **Sheet 3: Top Opportunities (Undervalued)**:
    - [ ] Top 50 stocks by mispricing score
    - [ ] Detailed metrics: fundamentals, valuation ratios, growth metrics
    - [ ] SHAP feature importance for each stock
  - [ ] **Sheet 4: Risk Analysis (Overvalued)**:
    - [ ] Top 50 overvalued stocks (potential sells/shorts)
    - [ ] Risk indicators: high leverage, declining margins, negative growth
  - [ ] **Sheet 5: Prediction Accuracy**:
    - [ ] Model vs. Analyst comparison metrics
    - [ ] Error distribution by sector, region, market cap
    - [ ] Agreement/disagreement analysis
  - [ ] **Sheet 6: Sector Analysis**:
    - [ ] Sector-wise performance: avg mispricing, opportunity count, avg metrics
    - [ ] Sector rankings by attractiveness
  - [ ] **Sheet 7: Model Interpretation**:
    - [ ] Global feature importance (SHAP summary)
    - [ ] Sector-specific feature importance
    - [ ] Model methodology summary
  - [ ] **Formatting**: Use xlsxwriter for professional formatting, charts, conditional formatting
- [ ] Create interactive prediction comparison dashboards
  - [ ] **Streamlit or Plotly Dash Dashboard**:
    - [ ] Main view: scatter plot of Predicted vs. Analyst Targets
    - [ ] Color by agreement/disagreement magnitude
    - [ ] Drill-down: click stock for detailed view
    - [ ] Side-by-side comparison: model features vs. analyst rationale
    - [ ] Real-time filtering and sorting
  - [ ] **Prediction Accuracy Dashboard**:
    - [ ] Error distribution plots
    - [ ] Performance over time (if temporal data)
    - [ ] Sector/region performance comparison
- [ ] Automated report scheduling and distribution
  - [ ] Schedule weekly/monthly report generation
  - [ ] Email distribution with summary highlights
  - [ ] Automated alerts for high-conviction opportunities
  - [ ] Version control for reports (track changes over time)
- [ ] **Testing**: Add comprehensive tests for comparison calculations, Excel report generation, dashboard components
- [ ] **Documentation**: Document analytics methodology, report interpretation guide, and update procedures

#### Technology Stack Enhancements for Phase 9

**Core ML and Deep Learning**:

- **TensorFlow/Keras** (>=2.13.0): Neural networks, custom models, advanced architectures
- **PyTorch** (optional): Alternative deep learning framework for research models
- **Scikit-learn** (>=1.3.0): Traditional ML algorithms, preprocessing, pipelines
- **XGBoost, LightGBM, CatBoost**: Gradient boosting implementations
- **Imbalanced-learn** (>=0.11.0): Class balancing techniques (SMOTE, ADASYN)

**Hyperparameter Optimization**:

- **Optuna** (>=3.0.0): Bayesian optimization with pruning
- **Hyperopt**: Tree-structured Parzen estimator optimization
- **Ray Tune** (optional): Distributed hyperparameter tuning at scale

**Model Interpretation and Explainability**:

- **SHAP** (>=0.42.0): SHapley Additive exPlanations for model interpretation
- **LIME** (>=0.2.0): Local Interpretable Model-agnostic Explanations
- **ELI5**: Model explanation library
- **InterpretML**: Microsoft's interpretability toolkit

**Dashboarding and Visualization**:

- **Streamlit** (>=1.25.0): Rapid dashboard prototyping
- **Plotly Dash**: Production-grade interactive dashboards
- **Plotly** (>=5.14.0): Interactive visualizations
- **Matplotlib, Seaborn**: Static visualizations

**Report Generation**:

- **ReportLab**: Professional PDF report generation
- **xlsxwriter** (>=3.1.0): Excel file creation with formatting
- **openpyxl**: Excel file manipulation
- **python-pptx**: PowerPoint report generation (optional)

**Feature Selection and Engineering**:

- **Boruta** (optional): All-relevant feature selection
- **mlxtend**: Feature selection utilities
- **Feature-engine**: Feature engineering library

**Time Series and Temporal Analysis**:

- **statsmodels** (>=0.14.0): Statistical modeling and time series
- **prophet** (optional): Facebook's time series forecasting
- **arch**: GARCH models for volatility forecasting

**Experiment Tracking and MLOps**:

- **MLflow** (>=2.5.0): Experiment tracking, model registry, deployment
- **Weights & Biases** (optional): Experiment tracking and collaboration
- **DVC** (optional): Data version control

**Data Profiling and Quality**:

- **pandas-profiling** (ydata-profiling): Automated EDA reports
- **sweetviz**: Comparative data analysis
- **great_expectations**: Data validation framework

**Advanced Libraries**:

- **NumPy** (>=1.24.0): Numerical computing
- **Pandas** (>=2.0.0): Data manipulation
- **SciPy** (>=1.10.0): Scientific computing
- **Joblib** (>=1.3.0): Parallel processing and model persistence

#### Advanced ML Techniques from Reference Notebooks (Future Enhancements)

**Reference**: Notebooks 14-18 covering advanced deep learning architectures  
**Status**: Future research and experimentation (Phase 10+)  
**Purpose**: Explore cutting-edge techniques for potential performance improvements

##### 14. Deep Computer Vision with CNNs (Future Research)

**Reference**: `14_deep_computer_vision_with_cnns.ipynb`  
**Potential Applications**:

- [ ] **Chart Pattern Recognition**: Train CNNs on candlestick chart images to detect technical patterns
  - [ ] Convert OHLC data to candlestick chart images
  - [ ] Use ResNet, EfficientNet, or Vision Transformer (ViT) for pattern classification
  - [ ] Identify breakouts, reversals, head-and-shoulders, triangles
- [ ] **Financial Report Image Analysis**: Extract information from earnings report charts/tables
- [ ] **Correlation Heatmap Analysis**: Use CNNs to identify complex correlation patterns
- **Feasibility**: Medium - requires image data generation from time series
- **Priority**: Low - traditional features likely sufficient for tabular financial data

##### 15. Processing Sequences using RNNs and CNNs (Future Research)

**Reference**: `15_processing_sequences_using_rnns_and_cnns.ipynb`  
**Potential Applications**:

- [ ] **Time-Series Price Prediction**: Use LSTMs/GRUs to model temporal dependencies in prices
  - [ ] Multi-step ahead forecasting with LSTM encoder-decoder
  - [ ] Bidirectional RNNs for context-aware predictions
  - [ ] Stateful RNNs for online learning
- [ ] **Sequential Financial Metrics**: Model quarterly earnings sequences
  - [ ] Predict next quarter's metrics from historical sequence
  - [ ] Detect trend changes (growth → decline)
- [ ] **1D CNNs for Time Series**: Apply temporal convolutional networks (TCN) to financial sequences
- **Feasibility**: High - if temporal/sequential data becomes available
- **Priority**: Medium - valuable for quarterly reporting cycles

##### 16. NLP with RNNs and Attention (Future Research)

**Reference**: `16_nlp_with_rnns_and_attention.ipynb`  
**Potential Applications**:

- [ ] **Earnings Call Transcript Analysis**: Extract sentiment from CEO/CFO comments
  - [ ] Use LSTM + Attention for sentiment classification
  - [ ] Identify forward-looking statements, risk factors
  - [ ] Extract key topics with attention weights
- [ ] **News Sentiment Analysis**: Analyze financial news articles for stock sentiment
  - [ ] Use pre-trained BERT/FinBERT for financial text
  - [ ] Aggregate sentiment scores as features
- [ ] **Analyst Report Mining**: Extract price targets, ratings, key insights from analyst PDFs
- [ ] **Transformer Models**: Apply BERT, GPT, or domain-specific models (FinBERT, BloombergGPT)
- **Feasibility**: High - if text data (earnings transcripts, news, analyst reports) available
- **Priority**: Medium-High - NLP features can significantly improve predictions

##### 17. Autoencoders, GANs, and Diffusion Models (Future Research)

**Reference**: `17_autoencoders_gans_and_diffusion_models.ipynb`  
**Potential Applications**:

- [ ] **Autoencoders for Anomaly Detection**:
  - [ ] Train autoencoder on "normal" financial metrics
  - [ ] Detect outliers/anomalies (potential fraud, unusual patterns)
  - [ ] Use reconstruction error as anomaly score
  - [ ] Variational Autoencoder (VAE) for probabilistic anomaly detection
- [ ] **Dimensionality Reduction**:
  - [ ] Use autoencoders to compress 234 features into dense representations
  - [ ] Compare with PCA, t-SNE, UMAP
  - [ ] Use latent representations as meta-features
- [ ] **GANs for Synthetic Data Generation**:
  - [ ] Generate synthetic financial data for rare sectors/regions
  - [ ] Address class imbalance with synthetic examples
  - [ ] Create stress-test scenarios (e.g., recession conditions)
  - [ ] Conditional GANs (cGAN) for sector-specific data generation
- [ ] **Diffusion Models** (Experimental):
  - [ ] Generate realistic financial metric distributions
  - [ ] Sample-based uncertainty estimation
- **Feasibility**: Medium - requires careful validation (synthetic data quality)
- **Priority**: Low-Medium - useful for data augmentation and anomaly detection

##### 18. Reinforcement Learning (Future Research)

**Reference**: `18_reinforcement_learning.ipynb`  
**Potential Applications**:

- [ ] **Portfolio Optimization with RL**:
  - [ ] Agent learns optimal portfolio allocation strategy
  - [ ] State: current portfolio, market conditions, stock features
  - [ ] Action: buy/sell/hold decisions, position sizes
  - [ ] Reward: Sharpe ratio, returns, risk-adjusted performance
  - [ ] Algorithms: DQN, A3C, PPO, DDPG, SAC
- [ ] **Trading Strategy Learning**:
  - [ ] Learn when to enter/exit positions based on predictions
  - [ ] Incorporate transaction costs, slippage
  - [ ] Multi-agent RL for market simulation
- [ ] **Dynamic Feature Selection**:
  - [ ] RL agent learns which features to use per stock/sector
  - [ ] Adaptive model selection (which model to use when)
- [ ] **Hyperparameter Optimization with RL**:
  - [ ] Use RL for sequential hyperparameter search (alternative to Optuna)
- **Feasibility**: Medium-High - requires careful reward design and extensive testing
- **Priority**: Medium - valuable for portfolio optimization and trading strategies

##### Implementation Considerations for Advanced Techniques

- [ ] **Data Requirements**: Assess availability of temporal, text, and alternative data
- [ ] **Computational Resources**: Deep learning models require GPUs for training
- [ ] **Interpretability Trade-offs**: Balance model complexity with explainability requirements
- [ ] **Validation Rigor**: Extensive backtesting and out-of-sample validation for advanced models
- [ ] **Incremental Adoption**: Start with simpler techniques, gradually add complexity if proven beneficial
- [ ] **Benchmarking**: Compare advanced models against gradient boosting baselines
- [ ] **Production Constraints**: Consider inference latency, memory, maintenance costs

#### Testing and Validation Strategy

- [ ] Add integration tests for complete workflow (end-to-end)
- [ ] Implement model performance regression tests
- [ ] Add data drift detection tests
- [ ] Create benchmark datasets for reproducibility
- [ ] Implement continuous model monitoring framework

#### Documentation for Phase 9

- [ ] Create detailed workflow documentation for each step
- [ ] Add Jupyter notebook tutorials for each component
- [ ] Create API reference documentation (Sphinx)
- [ ] Add model card documentation for deployed models
- [ ] Create user guide for stock prediction system

#### Success Criteria

- ✅ All 8 workflow steps fully implemented and tested
- ✅ Test coverage >90% for new modules
- ✅ Comprehensive Excel reports matching reference format
- ✅ Interactive dashboards for real-time analysis
- ✅ Automated model evaluation and selection
- ✅ Production-ready prediction system

#### ML Project Checklist Alignment

Phase 9 aligns with the standard ML project framework (reference: `reference material/ml-project-checklist.md`):

**1. Frame the problem and look at the big picture** ✓

- Business objective: Predict Stock Price Targets for portfolio optimization and investment decisions
- Problem type: Supervised learning (regression + multi-class classification)
- Performance measure: MAE, RMSE, R² for regression; F1-score for classification
- Success metric: Prediction accuracy vs. analyst targets, actionable stock recommendations

**2. Get the data** ✓

- Data sources: PostgreSQL database with multi-region equity data (US, EU, APAC, ROTW)
- Data format: CSV files (screening_us.csv, screening_eu.csv, screening_apac.csv, screening_rotw.csv)
- Data size: 200+ columns, thousands of stocks across regions
- Implemented in: Phase 1 (Data Ingestion) and Phase 9.1 (Enhanced Preprocessing)

**3. Explore the data** → Phase 9.2

- EDA automation with `finance_ml.eval.simple_eda()`
- Statistical analysis, correlation matrices, distribution testing
- Sector-specific metric benchmarking
- Interactive dashboards for exploratory analysis

**4. Prepare the data** → Phase 9.1, 9.3

- Data cleaning and quality checks (Phase 9.1)
- Outlier detection and treatment by sector
- Feature engineering with sector-specific optimizations (Phase 9.3)
- Feature scaling, encoding, and transformation pipelines

**5. Explore many different models** → Phase 9.4, 9.5

- Event classification: Gradient boosting (XGBoost, LightGBM, CatBoost), Neural Networks
- Regression: Linear models, ensemble methods, deep learning, quantile regression
- Sector-optimized models with classification features integration
- Hyperparameter optimization with Optuna/GridSearchCV

**6. Fine-tune your models** → Phase 9.5, 9.6

- Automated hyperparameter optimization (Optuna, GridSearchCV)
- Cross-validation framework (time-series aware, grouped by sector)
- Model ensembling and stacking
- Prediction interval estimation with quantile regression

**7. Present your solution** → Phase 9.7, 9.8

- Comprehensive Excel reports (Stock_Prediction_Analysis_Report format)
- Interactive dashboards with Plotly/Streamlit
- Valuation analysis: undervalued/overvalued stock identification
- Predicted vs. Analyst Target comparison analytics
- PDF report generation for stakeholders

**8. Launch, monitor, and maintain** → Phase 9 Infrastructure

- Model persistence and versioning
- Automated report scheduling and distribution
- Continuous model monitoring and performance tracking
- Data drift detection and alerting
- Retraining pipeline automation

### Risk and Mitigation
- **Heavy dependencies** (TensorFlow, LightGBM) can be fragile on Windows.
  - **Mitigation**: Keep them optional in usage paths; provide scikit-learn fallbacks.
- **Data leakage** via target creation or temporal splits.
  - **Mitigation**: Implement time-based or grouped CV; enforce non-leaky feature generation.
- **DB availability** during tests or runs.
  - **Mitigation**: CSV fallback and mocked DB in tests.

### Completed Milestones (2025-10-24)

**Session 5 — Portfolio Optimization Implementation (TDD):**
- Portfolio optimization module: 0% → 89% coverage
- Created `finance_ml/portfolio_optimization.py` with Modern Portfolio Theory implementation
- Created `tests/test_portfolio_optimization.py` with 30 comprehensive test cases
- Implemented 9 functions: portfolio metrics, efficient frontier, optimization algorithms, rebalancing
- Total tests: 267 → 297
- All tests passing (1 skipped)
- Module fully integrated into finance_ml package

**Session 4 — TDD Coverage Improvement:**
- CLI module testing: 0% → 98% coverage
- Created `tests/test_cli.py` with 27 comprehensive test cases
- Fixed API mismatches between cli.py and data.py
- Total tests: 173 → 200
- Overall package coverage: 72% → 90%

**Session 3 — Final Refactoring:**
- GitHub Actions workflow created
- Multi-OS/Python testing configured
- pyproject.toml with console_scripts entry points
- Configuration management (`finance_ml/config.py`)
- CLI interface (`finance_ml/cli.py`)
- Notebook integration updated
- CHANGELOG.md created
- Package version v0.3.0

**Session 2 — TDD Refactoring:**
- All modules implemented with strict TDD methodology
- `finance_ml/data.py`, `finance_ml/features.py`, `finance_ml/models.py`, `finance_ml/eval.py`
- Test suite: 144 tests passing (5 skipped)
- Backward compatibility maintained

### Current Status: Production-Ready ✓

The Finance ML Analytics Platform is now a production-ready Python package with:
- ✅ Modular architecture (`finance_ml` package)
- ✅ Comprehensive test suite (144+ tests)
- ✅ CLI tools (3 console commands)
- ✅ Configuration management
- ✅ CI/CD pipeline
- ✅ Modern packaging (`pyproject.toml`)
- ✅ Complete documentation

### Future Enhancements (TODO)

#### Documentation
- [ ] Add `/docs` directory with detailed usage examples
- [ ] Create architecture diagrams
- [ ] Add FAQ section
- [ ] Create tutorial notebooks for common workflows
- [ ] Add API reference documentation (Sphinx or MkDocs)

#### Features
- [ ] Add time-series analysis for temporal patterns
- [x] Implement portfolio optimization algorithms (✓ Session 5 - TDD)
- [x] Add risk metrics (VaR, CVaR, Sharpe ratio) (✓ Session 4 - TDD)
- [ ] Create dashboard for real-time monitoring (Streamlit/Dash)
- [ ] Add support for alternative data sources (APIs)

#### Testing
- [ ] Increase test coverage to >95%
- [ ] Add integration tests for end-to-end workflows
- [ ] Add performance benchmarks
- [ ] Add stress tests for large datasets

#### Infrastructure
- [ ] Docker containerization for reproducibility
- [ ] Cloud deployment guide (AWS/Azure/GCP)
- [ ] Database migration scripts for schema updates
- [ ] Add logging to file with rotation

#### Packaging
- [ ] Publish to PyPI for easy installation
- [ ] Create conda package for conda-forge
- [ ] Add wheels for common platforms

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Copyright**: (c) 2025 Finance ML Analytics Platform Contributors

**License Details**:
- Type: MIT License
- File: `LICENSE` (in repository root)
- Also declared in: `pyproject.toml`, `README.md`

---

## Contributing

Contributions are welcome! This project follows standard open-source contribution practices.

### Current Contribution Guidelines

See README.md for current contribution guidelines including:
- How to contribute workflow (fork, branch, PR)
- Code style requirements (PEP 8, black, isort, type hints)
- Testing requirements (unit tests, coverage maintenance)
- Code quality checks (black, flake8, mypy, isort)

### TODO

**Action Required**: Create detailed `CONTRIBUTING.md` file with comprehensive guidelines for:
- Detailed development workflow
- Testing requirements and best practices
- Pull request process and templates
- Issue reporting templates
- Code of conduct
- Review process and timelines

---

## References

- **README.md**: Project overview and quick start
- **CHANGELOG.md**: Version history and changes
- **.junie/guidelines.md**: Detailed development guidelines
- **pyproject.toml**: Package configuration
- **requirements.txt**: Core dependencies
- **environment_variables.txt**: Environment configuration examples

---

**Document Version**: 2.0 (2025-10-24)  
**Project Version**: 0.3.0  
**Maintainers**: Finance ML Team



---

## Notebook Enhancements Guide (v8_3 proposed)

A focused, notebook-first plan for upgrading ml_finance_model_v8_2.ipynb is provided in NOTEBOOK_ENHANCEMENTS.md. It
covers:

- Building the unified all_stocks dataframe from PostgreSQL or CSV with validation and preprocessing
- Feature engineering (financial ratios, margins, volatility, revenue CAGR, categorical encodings)
- ML models (multi-class event classification, sector-optimized regression, quantile models, stacking ensembles)
- Analytics and reporting (mispricing scores, stock ranking, interactive plots, Excel export)

Reference artifact for reporting/layout:

- reports/Stock_Prediction_Analysis_Report_20250806_131704.xlsx

When implementing these enhancements in the notebook, bump MODEL_VERSION to v8_3 and record changes here and in
README.md.

### New Section: Robust SQLite Ingestion Path and Parity With PostgreSQL

#### Problem summary

- SQLite shell `.import` treats the first row as data (no automatic header skip).
- Empty strings are not coerced to NULL by default, causing downstream issues.
- Errors are not isolated across regions; limited validation and mapping safeguards.

#### Proposed approach

- Use per-region TEMP staging tables with generic `col1..colN` schema.
- Delete the header row explicitly and map with `NULLIF(colN, '')` into `equities`.
- Wrap each region import in its own transaction with `.bail on` for fail-fast.
- Default missing "Region" per file (US/EU/APAC/ROTW) and rely on `UNIQUE("Ticker","Region")`.
- Provide a Python importer alternative with chunking using pandas for reliability.

#### Tasks (to be tracked under Phase 2 — Data Ingestion and Validation)

1) SQLite import hardening (shell-based) ✓

- [x] Add `import_equities_data_sqlite.sql` with:
    - [x] `.bail on`, `.echo on`
    - [x] TEMP staging `col1..colN` per region
    - [x] Explicit header-row deletion
    - [x] `NULLIF` mapping to `equities` and default Region per file
    - [x] `INSERT OR IGNORE` for deduplication via `UNIQUE("Ticker","Region")`
    - [x] Per-region transactions and basic validation summaries

2) Python import alternative for SQLite ✓

- [x] Create `tools/import_sqlite.py` that:
    - [x] Reads CSVs with `dtype=str`, normalizes empty strings to `None`
    - [x] Backfills Region, supports `--chunksize` and per-region selection
    - [x] Appends with de-duplication via the unique index or temp-table merge

3) Validation utilities parity (Partial)

- [ ] Validate header matches expected columns, sample numeric fields, per-region counts.
- [ ] Emit a machine-readable JSON report for CI.

4) Documentation updates (Partial)

- [ ] README: add a "SQLite local path" subsection with exact commands and caveats
  (header handling, NULLs, `.bail on`).

5) Tests for SQLite path ✓ **COMPLETED 2025-10-26**

- [x] Add `tests/test_sqlite_import.py`:
    - [x] Build a temp SQLite DB, apply schema, import tiny CSV fixtures
    - [x] Assert header removed, empty strings mapped to NULL, Region backfilled
    - [x] Ensure `UNIQUE("Ticker","Region")` prevents duplicates
    - [x] Test error handling paths (CSV parser errors, database errors)
    - [x] Test main CLI function with various scenarios
    - [x] Test tqdm fallback functionality
    - [x] **Coverage achieved: 97% for tools/import_sqlite.py (37 tests passing)**

#### Rationale

- Eliminates header-as-data issues; enforces consistent NULL semantics.
- Improves debuggability via explicit mapping and per-region transactions.
- Python importer provides a robust, cross-platform alternative with chunked loads.
