import unittest

import numpy as np
import pandas as pd

from finance_ml.etl.stages.imputation import apply_pre_imputation_business_fills, run_imputation_stage
from finance_ml.etl.stages.sanitization import apply_business_rule_zero_fills, run_sanitization_stage


class TestBusinessRuleImputation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "dividend_record_amount": [1.0, np.nan, 2.0],
            "dividend_record_currency": ["USD", np.nan, "EUR"],
            "dividend_record_frequency": ["Quarterly", np.nan, "Annual"],
            "num_strong_buys_ratings": [5, np.nan, 10],
            "randd_expenses_ltm": [100.0, np.nan, 200.0],
            "goodwill_fq": [50.0, np.nan, 100.0],
            "impairment_of_goodwill_fq": [10.0, np.nan, 20.0],
            "other_col": [1.0, np.nan, 3.0]
        })
        # Make categorical
        self.df["dividend_record_currency"] = self.df["dividend_record_currency"].astype("category")
        self.df["dividend_record_frequency"] = self.df["dividend_record_frequency"].astype("category")

    def test_sanitization_zero_fills(self):
        """Test business rule zero fills in sanitization stage."""
        result = apply_business_rule_zero_fills(self.df)
        
        # Check zero fills
        self.assertEqual(result["dividend_record_amount"].iloc[1], 0)
        self.assertEqual(result["num_strong_buys_ratings"].iloc[1], 0)
        self.assertEqual(result["goodwill_fq"].iloc[1], 0)
        self.assertEqual(result["impairment_of_goodwill_fq"].iloc[1], 0)

        # R&D is recurring -> should be NaN in sanitization (handled later in imputation)
        self.assertTrue(np.isnan(result["randd_expenses_ltm"].iloc[1]))
        
        # Check categorical fill
        self.assertEqual(result["dividend_record_currency"].iloc[1], "N/A")
        self.assertEqual(result["dividend_record_frequency"].iloc[1], "None")
        
        # other_col should remain NaN
        self.assertTrue(np.isnan(result["other_col"].iloc[1]))

    def test_imputation_pre_fills(self):
        """Test business rule pre-fills in imputation stage."""
        result = apply_pre_imputation_business_fills(self.df)
        
        # Check zero fills
        self.assertEqual(result["dividend_record_amount"].iloc[1], 0)
        self.assertEqual(result["num_strong_buys_ratings"].iloc[1], 0)
        
        # Check categorical fill
        self.assertEqual(result["dividend_record_currency"].iloc[1], "N/A")
        
        # randd_expenses_ltm is NOT in pre_imputation_zero_fill_columns 
        # (check definition in imputation.py: it only includes dividend, analyst, and zero_imputation_columns)
        # zero_imputation_columns are non-recurring exceptional items. R&D is recurring.
        # So R&D should NOT be filled here (it will be filled by main imputation later)
        self.assertTrue(np.isnan(result["randd_expenses_ltm"].iloc[1]))
    
    def test_run_sanitization_stage(self):
        result = run_sanitization_stage(self.df, apply_business_rules=True)
        self.assertEqual(result["dividend_record_amount"].iloc[1], 0)
        
    def test_run_imputation_stage(self):
        # We need sector and price columns for 6step fallback
        df_imp = self.df.copy()
        df_imp["sector"] = ["Tech", "Tech", "Tech"]
        df_imp["last_price"] = [100.0, 100.0, 100.0]
        
        # Make sure other required columns for 6-step exist or it handles them gracefully
        # 6-step usually requires more context but should handle partial data
        
        result = run_imputation_stage(
            df_imp, 
            strategy="6step",
            apply_pre_imputation_fills=True
        )
        self.assertEqual(result["dividend_record_amount"].iloc[1], 0)
        self.assertEqual(result["dividend_record_currency"].iloc[1], "N/A")

if __name__ == "__main__":
    unittest.main()
