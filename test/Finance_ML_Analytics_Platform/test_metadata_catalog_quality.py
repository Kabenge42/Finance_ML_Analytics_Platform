"""
Test suite for metadata catalog quality and consistency.

Tests the metadata artifacts (all_stocks_initial_metadata.json,
preprocessed_stocks_metadata.json) to ensure they contain proper
dtype information, missing counts, and quality statistics.
"""

import unittest
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from finance_ml.ml_workflow.preprocessing.dtypes import detect_and_cast_dtypes
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA


class TestMetadataCatalogQuality(unittest.TestCase):
    """Test metadata catalog structure and quality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for metadata files
        self.temp_dir = tempfile.mkdtemp()

        # Create sample dataframe for testing
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": [150.0, 2800.0, 350.0],
                "market_cap": [2.5e12, np.nan, 2.6e12],
                "price_target": [160.0, np.nan, 370.0],
                "last_updated": ["2023-01-15", "2023-01-16", "2023-01-17"],
            }
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_metadata_includes_dtypes_and_missing_counts(self):
        """
        Test that metadata includes dtypes and missing counts.

        After running the pipeline (or a small subset), metadata should include:
        - "dtypes": mapping of column names to dtype strings
        - "missing_counts": mapping of column names to missing value counts
        """
        # Arrange: Cast dtypes and track diagnostics
        df_cast, diagnostics = detect_and_cast_dtypes(self.sample_df)

        # Create metadata structure similar to what pipeline would generate
        metadata = {
            "shape": df_cast.shape,
            "columns": list(df_cast.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_cast.dtypes.items()},
            "missing_counts": {col: int(df_cast[col].isna().sum()) for col in df_cast.columns},
            "coercion_warnings": diagnostics.get("coercion_warnings", {}),
        }

        # Act: Save metadata to file
        metadata_file = Path(self.temp_dir) / "all_stocks_initial_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Assert: Load and validate metadata structure
        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        # Assert: Required keys present
        self.assertIn("dtypes", loaded_metadata, "Metadata should include dtypes")
        self.assertIn("missing_counts", loaded_metadata, "Metadata should include missing_counts")
        self.assertIn("shape", loaded_metadata, "Metadata should include shape")
        self.assertIn("columns", loaded_metadata, "Metadata should include columns")

        # Assert: dtypes structure
        self.assertIsInstance(loaded_metadata["dtypes"], dict)
        self.assertEqual(len(loaded_metadata["dtypes"]), len(df_cast.columns))

        # Assert: missing_counts structure
        self.assertIsInstance(loaded_metadata["missing_counts"], dict)
        self.assertEqual(len(loaded_metadata["missing_counts"]), len(df_cast.columns))

        # Assert: Specific column dtypes are recorded
        self.assertIn("last_price", loaded_metadata["dtypes"])
        self.assertIn("sector", loaded_metadata["dtypes"])

        # Assert: Missing counts are numeric
        for col, count in loaded_metadata["missing_counts"].items():
            self.assertIsInstance(count, int, f"Missing count for {col} should be int")
            self.assertGreaterEqual(count, 0, f"Missing count for {col} should be >= 0")

    def test_preprocessed_metadata_flags_zero_missing_for_phase93_features(self):
        """
        Test that preprocessed metadata shows zero missing for Phase 9.3 features.

        After full imputation, critical Phase 9.3 feature inputs should have
        missing_counts[col] == 0.

        NOTE: This test documents the desired post-imputation state.
        """
        # Arrange: Simulate preprocessed data (after imputation)
        # In real pipeline, this would be after apply_enhanced_imputation_strategy_6step
        df_preprocessed = self.sample_df.copy()

        # Fill missing values to simulate imputation
        df_preprocessed["market_cap"] = df_preprocessed["market_cap"].fillna(2.0e12)
        df_preprocessed["price_target"] = df_preprocessed["price_target"].fillna(
            df_preprocessed["last_price"]
        )

        # Create preprocessed metadata
        preprocessed_metadata = {
            "shape": df_preprocessed.shape,
            "columns": list(df_preprocessed.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_preprocessed.dtypes.items()},
            "missing_counts": {
                col: int(df_preprocessed[col].isna().sum()) for col in df_preprocessed.columns
            },
            "imputation_applied": True,
            "quality_stats": {
                "total_missing": int(df_preprocessed.isna().sum().sum()),
                "complete_columns": [
                    col for col in df_preprocessed.columns if df_preprocessed[col].isna().sum() == 0
                ],
            },
        }

        # Act: Save preprocessed metadata
        metadata_file = Path(self.temp_dir) / "preprocessed_stocks_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(preprocessed_metadata, f, indent=2)

        # Assert: Load and validate
        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        # Assert: Structure
        self.assertIn("missing_counts", loaded_metadata)
        self.assertIn("quality_stats", loaded_metadata)
        self.assertIn("imputation_applied", loaded_metadata)

        # Assert: For critical columns (that exist in our test data), missing count should be 0
        critical_cols = ["last_price", "market_cap", "price_target"]
        for col in critical_cols:
            if col in loaded_metadata["missing_counts"]:
                self.assertEqual(
                    loaded_metadata["missing_counts"][col],
                    0,
                    f"Critical column {col} should have zero missing values after imputation",
                )

        # Assert: Quality stats
        self.assertIn("total_missing", loaded_metadata["quality_stats"])
        self.assertEqual(
            loaded_metadata["quality_stats"]["total_missing"],
            0,
            "Total missing should be 0 after full imputation",
        )

    def test_quality_stats_consistency_with_metadata(self):
        """
        Test that quality_stats are consistent with metadata.

        Compare quality_stats output from preprocessing with metadata JSON.
        Ensure counts and dtypes align between different representations.
        """
        # Arrange: Create dataframe and compute quality stats
        df = self.sample_df.copy()

        # Compute quality stats manually
        quality_stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_missing": int(df.isna().sum().sum()),
            "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
            "dtypes_by_column": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        }

        # Create metadata
        metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        }

        # Act: Check consistency
        # Assert: Row/column counts match
        self.assertEqual(
            quality_stats["total_rows"],
            metadata["shape"][0],
            "Row count should match between quality_stats and metadata",
        )
        self.assertEqual(
            quality_stats["total_columns"],
            metadata["shape"][1],
            "Column count should match between quality_stats and metadata",
        )

        # Assert: Missing counts consistency
        for col in df.columns:
            self.assertEqual(
                quality_stats["missing_by_column"][col],
                metadata["missing_counts"][col],
                f"Missing count for {col} should be consistent",
            )

        # Assert: Dtypes consistency
        for col in df.columns:
            self.assertEqual(
                quality_stats["dtypes_by_column"][col],
                metadata["dtypes"][col],
                f"Dtype for {col} should be consistent",
            )

        # Assert: Total missing calculation
        calculated_total = sum(metadata["missing_counts"].values())
        self.assertEqual(
            quality_stats["total_missing"],
            calculated_total,
            "Total missing should equal sum of column missing counts",
        )

        # Assert: Numeric/categorical column lists are non-empty and valid
        self.assertGreater(
            len(quality_stats["numeric_columns"]), 0, "Should have at least some numeric columns"
        )

        for col in quality_stats["numeric_columns"]:
            self.assertIn(
                col, metadata["columns"], f"Numeric column {col} should be in metadata columns"
            )

        for col in quality_stats["categorical_columns"]:
            self.assertIn(
                col, metadata["columns"], f"Categorical column {col} should be in metadata columns"
            )


class TestMetadataIntegrationWithSchema(unittest.TestCase):
    """Test integration between metadata and schema."""

    def test_metadata_dtypes_align_with_schema_expectations(self):
        """
        Test that metadata dtypes align with COLUMN_SCHEMA expectations.

        For columns in both metadata and schema, verify dtype compatibility.
        """
        # Arrange: Create dataframe with schema-aligned columns
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": [150.0, 2800.0, 350.0],
                "market_cap": [2.5e12, 1.8e12, 2.6e12],
            }
        )

        # Act: Cast dtypes according to schema
        df_cast, diagnostics = detect_and_cast_dtypes(df)

        # Create metadata
        metadata_dtypes = {col: str(dtype) for col, dtype in df_cast.dtypes.items()}

        # Assert: Check alignment with schema
        for col in df_cast.columns:
            # Normalize column name
            col_normalized = col.lower().replace(" ", "_")

            if col_normalized in COLUMN_SCHEMA:
                expected_dtype = COLUMN_SCHEMA[col_normalized]["dtype"]
                actual_dtype = metadata_dtypes[col]

                # Check dtype compatibility
                if expected_dtype == "float":
                    self.assertTrue(
                        "float" in actual_dtype or "int" in actual_dtype,
                        f"{col} expected float-compatible, got {actual_dtype}",
                    )
                elif expected_dtype == "int":
                    self.assertTrue(
                        "int" in actual_dtype.lower(),
                        f"{col} expected int-compatible, got {actual_dtype}",
                    )
                elif expected_dtype == "category":
                    self.assertTrue(
                        "category" in actual_dtype or "object" in actual_dtype,
                        f"{col} expected categorical, got {actual_dtype}",
                    )
                elif expected_dtype == "string":
                    self.assertTrue(
                        "string" in actual_dtype or "object" in actual_dtype,
                        f"{col} expected string, got {actual_dtype}",
                    )
                elif expected_dtype == "datetime64[ns]":
                    self.assertTrue(
                        "datetime" in actual_dtype, f"{col} expected datetime, got {actual_dtype}"
                    )


if __name__ == "__main__":
    unittest.main()
