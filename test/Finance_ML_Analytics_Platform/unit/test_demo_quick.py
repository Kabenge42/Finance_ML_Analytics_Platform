"""
Ultra-fast smoke test to validate local environment and imports.

Phase 7 Restructuring: This module demonstrates the use of pytest with
shared fixtures from conftest.py.

Run with pytest (Windows PowerShell):
  pytest tests/unit/test_demo_quick.py -v

Run with unittest (backward compatible):
  python -m unittest tests.unit.test_demo_quick -v
"""

from __future__ import annotations

import re

import pytest


class TestDemoQuickBasic:
    """Basic smoke tests without fixtures."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert 2 + 2 == 4
        assert abs(10 / 4 - 2.5) < 1e-9

    def test_package_import_and_version(self):
        """Test that finance_ml package can be imported with valid version."""
        import finance_ml

        assert hasattr(finance_ml, "__version__")
        # Accepts semantic-like versions such as 0.4.1 or v9_10
        version = str(finance_ml.__version__)
        assert re.match(r"^(v?\d+[._]\d+(?:[._]\d+)?)$", version)


class TestDemoQuickWithFixtures:
    """Tests demonstrating conftest.py fixture usage."""

    def test_sample_financial_df_fixture(self, sample_financial_df):
        """Test that sample_financial_df fixture is properly injected."""
        assert sample_financial_df is not None
        assert len(sample_financial_df) == 10
        assert "ticker" in sample_financial_df.columns
        assert "sector" in sample_financial_df.columns
        assert "last_price" in sample_financial_df.columns

    def test_sample_predictions_df_fixture(self, sample_predictions_df):
        """Test that sample_predictions_df fixture has standardized schema."""
        required_cols = ["y_true", "y_pred", "pred_p10", "pred_p50", "pred_p90"]
        for col in required_cols:
            assert col in sample_predictions_df.columns

    def test_default_config_fixture(self, default_config):
        """Test that default_config fixture provides expected keys."""
        assert "random_seed" in default_config
        assert default_config["random_seed"] == 42
        assert "test_size" in default_config
        assert "cv_folds" in default_config

    def test_temp_output_dir_fixture(self, temp_output_dir):
        """Test that temp_output_dir fixture creates a valid directory."""
        assert temp_output_dir.exists()
        assert temp_output_dir.is_dir()


# Backward compatibility: unittest-style tests still work
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
