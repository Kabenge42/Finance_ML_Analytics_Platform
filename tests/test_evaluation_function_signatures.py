"""
Test evaluation function signatures to prevent parameter mismatches.

Following code_guidelines.md §6.2 - Python Script/Module Review Checklist.
These tests validate that evaluation functions have correct parameter signatures
to prevent TypeError in notebook and script usage.

Test-Driven Development (TDD) approach for issue:
"Implement Parameter Mismatch TypeError Fixes with strict TDD"
"""

import inspect
import unittest

from finance_ml.ml_workflow.evaluation import (
    safety_rails_sensitivity_app,
    estimate_sector_bias,
    plot_metrics_by_sector_time,
    create_sector_bias_dashboard,
    compute_stacking_contributions,
    meta_error_maps,
    build_lineage_json,
)


class TestEvaluationFunctionSignatures(unittest.TestCase):
    """Test that evaluation function signatures match expected parameters."""

    def test_safety_rails_sensitivity_app_signature(self):
        """Verify safety_rails_sensitivity_app accepts data_df parameter (not df_raw)."""
        sig = inspect.signature(safety_rails_sensitivity_app)
        params = sig.parameters.keys()
        
        # Correct parameter should exist
        self.assertIn('data_df', params, 
                      "safety_rails_sensitivity_app should accept 'data_df' parameter")
        
        # Incorrect parameter should NOT exist
        self.assertNotIn('df_raw', params,
                        "safety_rails_sensitivity_app should NOT accept 'df_raw' parameter")
        
        # Verify required parameters
        self.assertIn('output_dir', params)
        
        # Verify optional parameters
        self.assertIn('default_lower_pct', params)
        self.assertIn('default_upper_pct', params)
        self.assertIn('thresholds', params)

    def test_estimate_sector_bias_signature(self):
        """Verify estimate_sector_bias does NOT accept column name parameters."""
        sig = inspect.signature(estimate_sector_bias)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('predictions_df', params,
                      "estimate_sector_bias should accept 'predictions_df' parameter")
        self.assertIn('output_dir', params)
        self.assertIn('model_version', params)
        
        # Column name parameters should NOT exist (hardcoded internally)
        self.assertNotIn('y_true_col', params,
                        "estimate_sector_bias should NOT accept 'y_true_col' parameter")
        self.assertNotIn('y_pred_col', params,
                        "estimate_sector_bias should NOT accept 'y_pred_col' parameter")
        self.assertNotIn('y_pred_calibrated_col', params,
                        "estimate_sector_bias should NOT accept 'y_pred_calibrated_col' parameter")
        self.assertNotIn('sector_col', params,
                        "estimate_sector_bias should NOT accept 'sector_col' parameter")

    def test_plot_metrics_by_sector_time_signature(self):
        """Verify plot_metrics_by_sector_time has correct parameter names."""
        sig = inspect.signature(plot_metrics_by_sector_time)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('predictions_df', params,
                      "plot_metrics_by_sector_time should accept 'predictions_df' parameter")
        self.assertNotIn('metrics_history', params,
                        "plot_metrics_by_sector_time should NOT accept 'metrics_history' parameter")
        
        self.assertIn('date_col', params,
                      "plot_metrics_by_sector_time should accept 'date_col' parameter")
        self.assertNotIn('snapshot_date_col', params,
                        "plot_metrics_by_sector_time should NOT accept 'snapshot_date_col' parameter")
        
        self.assertIn('output_dir', params)

    def test_create_sector_bias_dashboard_signature(self):
        """Verify create_sector_bias_dashboard does NOT accept bias_dict parameter."""
        sig = inspect.signature(create_sector_bias_dashboard)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('predictions_df', params,
                      "create_sector_bias_dashboard should accept 'predictions_df' parameter")
        self.assertIn('output_dir', params)
        
        # bias_dict should NOT exist (calculated internally)
        self.assertNotIn('bias_dict', params,
                        "create_sector_bias_dashboard should NOT accept 'bias_dict' parameter")

    def test_compute_stacking_contributions_signature(self):
        """Verify compute_stacking_contributions does NOT accept y_true parameter."""
        sig = inspect.signature(compute_stacking_contributions)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('base_predictions', params,
                      "compute_stacking_contributions should accept 'base_predictions' parameter")
        self.assertIn('meta_predictions', params,
                      "compute_stacking_contributions should accept 'meta_predictions' parameter")
        self.assertIn('output_dir', params)
        
        # y_true should NOT exist (not used in this function)
        self.assertNotIn('y_true', params,
                        "compute_stacking_contributions should NOT accept 'y_true' parameter")

    def test_meta_error_maps_signature(self):
        """Verify meta_error_maps has correct parameter names."""
        sig = inspect.signature(meta_error_maps)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('predictions_df', params,
                      "meta_error_maps should accept 'predictions_df' parameter")
        self.assertIn('output_dir', params)
        self.assertIn('feature_cols', params,
                      "meta_error_maps should accept 'feature_cols' parameter")
        
        # Column name parameters should NOT exist (auto-detected from df)
        self.assertNotIn('error_col', params,
                        "meta_error_maps should NOT accept 'error_col' parameter")
        self.assertNotIn('sector_col', params,
                        "meta_error_maps should NOT accept 'sector_col' parameter")

    def test_build_lineage_json_signature(self):
        """Verify build_lineage_json accepts model_info dict (not separate parameters)."""
        sig = inspect.signature(build_lineage_json)
        params = sig.parameters.keys()
        
        # Correct parameters should exist
        self.assertIn('model_info', params,
                      "build_lineage_json should accept 'model_info' parameter")
        self.assertIn('output_dir', params)
        self.assertIn('model_version', params)
        
        # Individual component parameters should NOT exist (bundled in model_info)
        self.assertNotIn('datasets', params,
                        "build_lineage_json should NOT accept 'datasets' parameter")
        self.assertNotIn('features', params,
                        "build_lineage_json should NOT accept 'features' parameter")
        self.assertNotIn('models', params,
                        "build_lineage_json should NOT accept 'models' parameter")
        self.assertNotIn('artifacts', params,
                        "build_lineage_json should NOT accept 'artifacts' parameter")
        self.assertNotIn('metrics', params,
                        "build_lineage_json should NOT accept 'metrics' parameter")


class TestParameterNamingConventions(unittest.TestCase):
    """Test parameter naming conventions across evaluation modules."""
    
    def test_data_parameter_naming_consistency(self):
        """Verify consistent naming of data parameters across functions."""
        # Functions that accept DataFrames should use consistent naming
        
        # safety_rails uses data_df
        sig1 = inspect.signature(safety_rails_sensitivity_app)
        self.assertIn('data_df', sig1.parameters)
        
        # calibration functions use predictions_df
        sig2 = inspect.signature(estimate_sector_bias)
        self.assertIn('predictions_df', sig2.parameters)
        
        sig3 = inspect.signature(plot_metrics_by_sector_time)
        self.assertIn('predictions_df', sig3.parameters)
        
        sig4 = inspect.signature(create_sector_bias_dashboard)
        self.assertIn('predictions_df', sig4.parameters)
        
        sig5 = inspect.signature(meta_error_maps)
        self.assertIn('predictions_df', sig5.parameters)
    
    def test_output_dir_parameter_consistency(self):
        """Verify all functions use 'output_dir' (not out_dir, save_dir, etc.)."""
        functions_to_test = [
            safety_rails_sensitivity_app,
            estimate_sector_bias,
            plot_metrics_by_sector_time,
            create_sector_bias_dashboard,
            compute_stacking_contributions,
            meta_error_maps,
            build_lineage_json,
        ]
        
        for func in functions_to_test:
            sig = inspect.signature(func)
            params = sig.parameters.keys()
            
            self.assertIn('output_dir', params,
                         f"{func.__name__} should use 'output_dir' parameter")
            self.assertNotIn('out_dir', params,
                            f"{func.__name__} should NOT use 'out_dir' parameter")
            self.assertNotIn('save_dir', params,
                            f"{func.__name__} should NOT use 'save_dir' parameter")
    
    def test_column_name_parameter_suffix(self):
        """Verify column name parameters use '_col' suffix."""
        # plot_metrics_by_sector_time should use date_col (not date_column, date_field)
        sig = inspect.signature(plot_metrics_by_sector_time)
        params = sig.parameters.keys()
        
        self.assertIn('date_col', params,
                      "Column name parameters should use '_col' suffix")
        self.assertNotIn('date_column', params)
        self.assertNotIn('date_field', params)


if __name__ == "__main__":
    unittest.main()
