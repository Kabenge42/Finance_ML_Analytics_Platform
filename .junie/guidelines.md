# Finance ML Analytics Platform — Development Guidelines

This document provides essential information for setting up, developing, and testing the Finance ML Analytics Platform.

## 1. Build/Configuration Instructions

### Prerequisites

- **Python**: Version 3.12 to 3.14 is required.
- **Operating System**: Windows is primarily supported (PowerShell scripts provided).
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, and `pyproject.toml`.

### Automated Environment Setup

The easiest way to set up the development environment is by using the provided setup script:

```powershell
python tools\setup_environment.py
```

This script performs the following:

- Detects the Python version.
- Creates a virtual environment (`venv`).
- Upgrades `pip`.
- Installs all necessary dependencies from `requirements.txt`.
- Configures environment variables.
- (Optional) Sets up the PostgreSQL database and loads initial data.

### Manual Setup

If you prefer manual setup:

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## 2. Testing Information

### Configuring and Running Tests

The project uses `pytest` as the primary testing framework, along with the built-in `unittest` module.

#### Running All Tests

To run the full test suite (which may take some time):

```powershell
pytest
```

#### Running Fast Tests

For quick verification during development, use the fast test runner:

```powershell
python tools\run_fast_tests.py
```

This script runs a subset of tests focused on utility modules, avoiding heavy ML training.

#### Running Specific Tests

You can run individual test files using `pytest` or `python`:

```powershell
pytest tests\test_etl_unified_pipeline.py
python tests\test_junie_demo.py
```

### Guidelines for Adding New Tests

- **Location**: Place new tests in the `tests/` directory.
- **Naming**: Test files should be named `test_*.py`. Test classes should inherit from `unittest.TestCase` or use
  `pytest` style functions.
- **Isolation**: Use small, deterministic samples. Avoid external service dependencies; use mocks or stubs where
  necessary.
- **Coverage**: Aim for ≥80% test coverage for new code.

### Testing Demonstration

A simple test `tests\test_junie_demo.py` was created to demonstrate the process:

```python
import unittest
from pathlib import Path

class TestJunieDemo(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(1 + 1, 2)
        
    def test_project_structure(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertTrue((project_root / "finance_ml").exists())
```

**Execution Result:**

```
> python tests\test_junie_demo.py
..
----------------------------------------------------------------------
Ran 2 tests in 0.000s
OK
```

## 3. Additional Development Information

### Code Style

The project follows standard Python styling conventions, enforced by:

- **Black**: Code formatting.
- **Flake8**: Linting and style guide enforcement.
- **Isort**: Import sorting.
- **Mypy**: Static type checking.

### Architecture

- **Unified Schema Module**: Use `finance_ml.core.schema` as the single source of truth for column definitions (
  `COLUMN_SCHEMA`).
- **Modular Feature Engineering**: Feature functions are organized by domain in `finance_ml/features/advanced/` (
  valuation, profitability, momentum, etc.).
- **Modular ETL**: ETL configuration and pipeline stages are decoupled in `finance_ml/etl/`.
- **Notebook Imports**: Update Jupyter notebooks to use the new modular import patterns.
- **Module Support**: Reusable logic is extracted into the `finance_ml/` package, with `finance_ml.core` containing
  shared constants and schema.
- **ML Workflow**: Follows an 8-phase process (Phase 9.1–9.8) covering data loading, preprocessing, feature engineering,
  and model evaluation.

### Key Directories

- `finance_ml/`: Core library code.
- `tests/`: Comprehensive test suite.
- `docs/`: Detailed documentation, including `code_guidelines.md`.
- `tools/`: Utility scripts for environment setup, data processing, and fast testing.
- `outputs/`: Generated reports, visualizations, and model artifacts.
