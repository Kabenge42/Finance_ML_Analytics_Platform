"""
Tests for train_sector_specific_models smart feature handling.

This suite verifies that the function gracefully handles feature_cols provided
as a dict or list, performs smart extraction (preferring 'all_features'),
combines feature types as fallback, deduplicates, validates against the
DataFrame, and logs clear messages.
"""

import logging
import numpy as np
import pandas as pd
import pytest

from finance_ml.advanced_models import train_sector_specific_models


@pytest.fixture()
def toy_df():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Finance", "Finance"] * 10,
            "feature1": rng.normal(size=40),
            "feature2": rng.normal(size=40),
            "target": rng.normal(size=40) + 100.0,  # keep positive-ish
        }
    )
    return df


def test_feature_cols_dict_uses_all_features_with_logging(toy_df, caplog):
    caplog.set_level(logging.INFO, logger="finance_ml.advanced_models")

    feature_cols_dict = {
        "numeric_features": ["feature1"],
        "categorical_features": [],
        "classification_features": [],
        "all_features": ["feature1", "feature2"],
    }

    sector_models, results = train_sector_specific_models(
        df=toy_df,
        feature_cols=feature_cols_dict,
        target_col="target",
        sector_col="sector",
        min_samples=5,
        random_state=0,
    )

    # Models exist for both sectors
    assert set(sector_models.keys()) == set(toy_df["sector"].unique())

    # Logging contains extraction info
    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "feature_cols is a dict" in logs or "dictionary" in logs
    assert "Using 'all_features'" in logs or "Using all_features" in logs
    assert "Final feature count" in logs

    # Models should have been trained on 2 features
    for m in sector_models.values():
        assert getattr(m, "n_features_in_", 2) == 2


def test_feature_cols_dict_combines_types_and_deduplicates_with_validation(toy_df, caplog):
    caplog.set_level(logging.INFO, logger="finance_ml.advanced_models")

    # 'all_features' missing; also include a duplicate and a missing feature
    feature_cols_dict = {
        "numeric_features": ["feature1", "feature2", "feature2"],
        "categorical_features": ["missing_feature"],
        "classification_features": [],
    }

    sector_models, results = train_sector_specific_models(
        df=toy_df,
        feature_cols=feature_cols_dict,
        target_col="target",
        sector_col="sector",
        min_samples=5,
        random_state=0,
    )

    # Expect 2 features actually used after dedup and validation
    for m in sector_models.values():
        assert getattr(m, "n_features_in_", 2) == 2

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "Combined feature types" in logs
    assert "After deduplication" in logs
    assert "features not in DataFrame" in logs or "not in DataFrame" in logs


def test_feature_cols_list_with_missing_gets_filtered_and_warned(toy_df, caplog):
    caplog.set_level(logging.INFO, logger="finance_ml.advanced_models")

    feature_cols = ["feature1", "missing_feature"]

    sector_models, results = train_sector_specific_models(
        df=toy_df,
        feature_cols=feature_cols,
        target_col="target",
        sector_col="sector",
        min_samples=5,
        random_state=0,
    )

    # Only 1 valid feature should be used
    for m in sector_models.values():
        assert getattr(m, "n_features_in_", 1) == 1

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "Warning" in logs or "features not in DataFrame" in logs


def test_error_when_no_valid_features_remain(toy_df):
    with pytest.raises(ValueError) as exc:
        train_sector_specific_models(
            df=toy_df,
            feature_cols=["unknown1", "unknown2"],
            target_col="target",
            sector_col="sector",
            min_samples=5,
            random_state=0,
        )
    assert "No valid feature columns" in str(exc.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
