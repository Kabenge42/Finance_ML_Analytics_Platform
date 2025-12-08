"""Notebook Phase 9.1 refactor compliance tests.

These tests enforce the refactoring requirements described in the issue:
- Remove legacy Phase 9.1 scaffolding blocks
- Keep a single unified 6-step imputation call
- Introduce shared validation helpers and additional config constants
"""

import json
from pathlib import Path


def _load_notebook_text(notebook_path: Path) -> tuple[list[str], str]:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = [
        "".join(c.get("source", []))
        for c in notebook.get("cells", [])
        if c.get("cell_type") == "code"
    ]
    all_text = "\n".join(code_cells)
    return code_cells, all_text


def test_removes_legacy_phase91_scaffolding():
    """Legacy scaffolding around all_stocks_raw/normalize_columns(a) must be removed."""

    code_cells, all_text = _load_notebook_text(Path("ml_finance_model_main.ipynb"))

    forbidden_markers = [
        "if 'all_stocks_raw' in globals():",
        "normalize_columns(a)",
        "Data preprocessing scaffolding warning",
    ]

    offending = [m for m in forbidden_markers if m in all_text]
    assert not offending, f"Legacy Phase 9.1 scaffolding still present: {offending}"


def test_single_unified_imputation_call():
    """The enhanced 6-step imputation should be invoked only once (Phase 9.1)."""

    code_cells, all_text = _load_notebook_text(Path("ml_finance_model_main.ipynb"))

    imputation_calls = all_text.count("apply_enhanced_imputation_strategy_6step(")
    assert (
        imputation_calls == 1
    ), f"Expected a single unified imputation call, found {imputation_calls} occurrences"


def test_validation_helpers_defined():
    """Shared validation helpers should be defined for reuse across phases."""

    _, all_text = _load_notebook_text(Path("ml_finance_model_main.ipynb"))

    expected_helpers = [
        "def assert_df_has_columns",
        "def assert_no_missing",
        "def assert_price_columns_preserved",
        "def require_dataframe",
    ]

    missing = [h for h in expected_helpers if h not in all_text]
    assert not missing, f"Missing validation helpers: {missing}"


def test_additional_config_constants_present():
    """New configuration constants must be centralized in the config cell (magic numbers removed)."""

    code_cells, all_text = _load_notebook_text(Path("ml_finance_model_main.ipynb"))

    expected_constants = [
        "DISAGREEMENT_THRESHOLD",
        "TOP_N_RANKINGS",
        "TOP_N_PORTFOLIO_CANDIDATES",
        "SAFETY_RAILS_THRESHOLDS",
        "MAX_PORTFOLIO_WEIGHT",
        "MAX_SHARPE_THRESHOLD",
        "MAX_RETURN_THRESHOLD",
        "RUN_DEMO_SECTIONS",
    ]

    missing = [c for c in expected_constants if c not in all_text]
    assert not missing, f"Missing refactor constants: {missing}"
