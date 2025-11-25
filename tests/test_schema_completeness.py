"""
Test module for COLUMN_SCHEMA completeness and integrity.

TDD Implementation: Data Normalization Fixes (2025-11-24)
Phase 1.2 - Schema Completeness Tests

Purpose:
- Verify COLUMN_SCHEMA has all Phase 9.3 feature columns
- Validate schema integrity (dtype, role fields)
- Ensure critical columns are present
- Detect missing or malformed schema entries

Test Coverage:
- Phase 9.3 feature categories coverage
- Critical financial columns presence
- Schema entry integrity (dtype and role fields)
- Numeric/categorical/date column lists
"""

import unittest
from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_INPUTS,
    get_expected_dtype,
    get_column_role,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
)


class TestSchemaCompleteness(unittest.TestCase):
    """Test COLUMN_SCHEMA completeness for Phase 9.3 features."""

    def test_phase93_momentum_features_in_schema(self):
        """Test Phase 9.3 momentum feature columns are in COLUMN_SCHEMA."""
        if 'momentum' in PHASE93_FEATURE_INPUTS:
            momentum_cols = PHASE93_FEATURE_INPUTS['momentum']
            missing = [col for col in momentum_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 momentum features missing from COLUMN_SCHEMA: {missing}"
            )

    def test_phase93_valuation_features_in_schema(self):
        """Test Phase 9.3 valuation feature columns are in COLUMN_SCHEMA."""
        if 'valuation' in PHASE93_FEATURE_INPUTS:
            valuation_cols = PHASE93_FEATURE_INPUTS['valuation']
            missing = [col for col in valuation_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 valuation features missing from COLUMN_SCHEMA: {missing}"
            )

    def test_phase93_profitability_features_in_schema(self):
        """Test Phase 9.3 profitability feature columns are in COLUMN_SCHEMA."""
        if 'profitability' in PHASE93_FEATURE_INPUTS:
            profitability_cols = PHASE93_FEATURE_INPUTS['profitability']
            missing = [col for col in profitability_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 profitability features missing from COLUMN_SCHEMA: {missing}"
            )

    def test_phase93_quality_risk_features_in_schema(self):
        """Test Phase 9.3 quality/risk feature columns are in COLUMN_SCHEMA."""
        if 'quality_risk' in PHASE93_FEATURE_INPUTS:
            quality_cols = PHASE93_FEATURE_INPUTS['quality_risk']
            missing = [col for col in quality_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 quality/risk features missing from COLUMN_SCHEMA: {missing}"
            )

    def test_phase93_cash_flow_features_in_schema(self):
        """Test Phase 9.3 cash flow feature columns are in COLUMN_SCHEMA."""
        if 'cash_flow' in PHASE93_FEATURE_INPUTS:
            cash_flow_cols = PHASE93_FEATURE_INPUTS['cash_flow']
            missing = [col for col in cash_flow_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 cash flow features missing from COLUMN_SCHEMA: {missing}"
            )

    def test_phase93_growth_features_in_schema(self):
        """Test Phase 9.3 growth feature columns are in COLUMN_SCHEMA."""
        if 'growth' in PHASE93_FEATURE_INPUTS:
            growth_cols = PHASE93_FEATURE_INPUTS['growth']
            missing = [col for col in growth_cols if col not in COLUMN_SCHEMA]
            self.assertEqual(
                len(missing),
                0,
                f"Phase 9.3 growth features missing from COLUMN_SCHEMA: {missing}"
            )


class TestCriticalColumnsPresence(unittest.TestCase):
    """Test critical columns required for ML pipeline are present."""

    CRITICAL_COLUMNS = [
        'ticker',
        'sector',
        'region',
        'last_price',
        'price_target',
        'market_cap',
        'enterprise_value',
        'ebitda_ltm',
        'total_revenues_ltm',
        'p_e_ltm',
    ]

    def test_critical_columns_all_present(self):
        """Test all critical columns are in COLUMN_SCHEMA."""
        missing = [col for col in self.CRITICAL_COLUMNS if col not in COLUMN_SCHEMA]
        self.assertEqual(
            len(missing),
            0,
            f"Critical columns missing from COLUMN_SCHEMA: {missing}"
        )

    def test_critical_columns_have_dtype(self):
        """Test critical columns have dtype information."""
        for col in self.CRITICAL_COLUMNS:
            with self.subTest(column=col):
                if col in COLUMN_SCHEMA:
                    dtype = get_expected_dtype(col)
                    self.assertIsNotNone(
                        dtype,
                        f"Critical column '{col}' missing dtype information"
                    )

    def test_critical_columns_have_role(self):
        """Test critical columns have role information."""
        for col in self.CRITICAL_COLUMNS:
            with self.subTest(column=col):
                if col in COLUMN_SCHEMA:
                    role = get_column_role(col)
                    self.assertIsNotNone(
                        role,
                        f"Critical column '{col}' missing role information"
                    )


class TestSchemaIntegrity(unittest.TestCase):
    """Test COLUMN_SCHEMA integrity (all entries well-formed)."""

    def test_all_schema_entries_have_dtype(self):
        """Test every COLUMN_SCHEMA entry has a 'dtype' field."""
        missing_dtype = []
        for key, value in COLUMN_SCHEMA.items():
            if not isinstance(value, dict) or 'dtype' not in value:
                missing_dtype.append(key)

        self.assertEqual(
            len(missing_dtype),
            0,
            f"Schema entries missing 'dtype' field: {missing_dtype}"
        )

    def test_all_schema_entries_have_role(self):
        """Test every COLUMN_SCHEMA entry has a 'role' field."""
        missing_role = []
        for key, value in COLUMN_SCHEMA.items():
            if not isinstance(value, dict) or 'role' not in value:
                missing_role.append(key)

        self.assertEqual(
            len(missing_role),
            0,
            f"Schema entries missing 'role' field: {missing_role}"
        )

    def test_dtype_values_are_valid(self):
        """Test dtype values are one of expected types."""
        valid_dtypes = {'float', 'int', 'str', 'string', 'datetime', 'datetime64[ns]', 'object', 'category'}
        invalid = []
        for key, value in COLUMN_SCHEMA.items():
            if isinstance(value, dict) and 'dtype' in value:
                dtype = value['dtype']
                if dtype not in valid_dtypes:
                    invalid.append((key, dtype))

        self.assertEqual(
            len(invalid),
            0,
            f"Schema entries with invalid dtype: {invalid}"
        )

    def test_role_values_are_valid(self):
        """Test role values are one of expected roles."""
        valid_roles = {'feature', 'identifier', 'id', 'target', 'target_fallback', 'metadata', 'auxiliary', 'datetime', 'date', 'categorical'}
        invalid = []
        for key, value in COLUMN_SCHEMA.items():
            if isinstance(value, dict) and 'role' in value:
                role = value['role']
                if role not in valid_roles:
                    invalid.append((key, role))

        self.assertEqual(
            len(invalid),
            0,
            f"Schema entries with invalid role: {invalid}"
        )


class TestSchemaHelperFunctions(unittest.TestCase):
    """Test schema helper functions return valid results."""

    def test_list_numeric_feature_cols_returns_list(self):
        """Test list_numeric_feature_cols() returns a list."""
        result = list_numeric_feature_cols()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "No numeric feature columns found")

    def test_list_categorical_cols_returns_list(self):
        """Test list_categorical_cols() returns a list."""
        result = list_categorical_cols()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "No categorical columns found")

    def test_list_date_cols_returns_list(self):
        """Test list_date_cols() returns a list."""
        result = list_date_cols()
        self.assertIsInstance(result, list)
        # Date columns may be empty in some schemas, so just check it's a list
        self.assertIsInstance(result, list)

    def test_numeric_features_have_numeric_dtype(self):
        """Test numeric feature columns have float or int dtype."""
        numeric_cols = list_numeric_feature_cols()
        for col in numeric_cols[:10]:  # Sample first 10
            with self.subTest(column=col):
                dtype = get_expected_dtype(col)
                self.assertIn(
                    dtype,
                    ['float', 'int'],
                    f"Numeric feature '{col}' has non-numeric dtype: {dtype}"
                )

    def test_categorical_cols_have_appropriate_dtype(self):
        """Test categorical columns have str, category, or object dtype."""
        categorical_cols = list_categorical_cols()
        for col in categorical_cols:
            with self.subTest(column=col):
                dtype = get_expected_dtype(col)
                self.assertIn(
                    dtype,
                    ['str', 'category', 'object'],
                    f"Categorical column '{col}' has inappropriate dtype: {dtype}"
                )


if __name__ == '__main__':
    unittest.main()
