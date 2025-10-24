"""
Tests for finance_ml.cli module.

Following strict TDD methodology:
1. Red phase: Write failing tests
2. Green phase: Tests pass (code already exists)
3. Refactor phase: Improve test quality

Coverage target: ≥80% for cli.py module
"""
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import tempfile
import pandas as pd

from finance_ml.cli import main, analyze_main, validate_main, _load_data
from finance_ml.config import FinanceMLConfig


class TestLoadDataHelper(unittest.TestCase):
    """Test the _load_data helper function."""

    def test_load_data_from_csv_explicit(self):
        """Should load from CSV when source is 'csv'."""
        config = FinanceMLConfig(data_dir=Path("data"))
        
        with patch('finance_ml.cli.load_from_csv') as mock_load_csv:
            mock_load_csv.return_value = pd.DataFrame({'ticker': ['AAPL']})
            
            result = _load_data("csv", config, limit=100)
            
            mock_load_csv.assert_called_once_with(Path("data"), limit=100)
            self.assertEqual(len(result), 1)

    def test_load_data_from_db_explicit(self):
        """Should load from DB when source is 'db' and db_url is set."""
        config = FinanceMLConfig(db_url="postgresql://localhost/test")
        
        with patch('finance_ml.cli.load_from_db') as mock_load_db:
            mock_load_db.return_value = pd.DataFrame({'ticker': ['AAPL', 'MSFT']})
            
            result = _load_data("db", config, limit=50)
            
            mock_load_db.assert_called_once_with("postgresql://localhost/test", limit=50)
            self.assertEqual(len(result), 2)

    def test_load_data_db_without_url_raises_error(self):
        """Should raise ValueError when source is 'db' but db_url is not set."""
        config = FinanceMLConfig(db_url=None)
        
        with self.assertRaises(ValueError) as ctx:
            _load_data("db", config)
        
        self.assertIn("DB_URL not set", str(ctx.exception))

    def test_load_data_auto_with_db_url_set(self):
        """Should handle auto-detection when db_url is set."""
        config = FinanceMLConfig(db_url="postgresql://localhost/test")
        
        # Mock both possible paths - auto will choose one based on SQLAlchemy availability
        with patch('finance_ml.cli.load_from_db') as mock_load_db:
            with patch('finance_ml.cli.load_from_csv') as mock_load_csv:
                mock_load_db.return_value = pd.DataFrame({'ticker': ['AAPL']})
                mock_load_csv.return_value = pd.DataFrame({'ticker': ['MSFT']})
                
                result = _load_data("auto", config)
                
                # Should successfully load data (from either source)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, pd.DataFrame)
                # At least one loader should be called
                call_count = mock_load_db.call_count + mock_load_csv.call_count
                self.assertGreater(call_count, 0)

    def test_load_data_auto_falls_back_to_csv_when_no_db_url(self):
        """Should auto-detect and use CSV when db_url is not set."""
        config = FinanceMLConfig(data_dir=Path("data"), db_url=None)
        
        with patch('finance_ml.cli.load_from_csv') as mock_load_csv:
            mock_load_csv.return_value = pd.DataFrame({'ticker': ['AAPL']})
            
            result = _load_data("auto", config)
            
            mock_load_csv.assert_called_once()
            self.assertEqual(len(result), 1)

    def test_load_data_csv_with_limit(self):
        """Should pass limit parameter when loading from CSV."""
        config = FinanceMLConfig(data_dir=Path("data"))
        
        with patch('finance_ml.cli.load_from_csv') as mock_load_csv:
            mock_load_csv.return_value = pd.DataFrame({'ticker': ['AAPL', 'MSFT']})
            
            result = _load_data("csv", config, limit=100)
            
            mock_load_csv.assert_called_once_with(Path("data"), limit=100)
            self.assertEqual(len(result), 2)


class TestMainFunction(unittest.TestCase):
    """Test the main() CLI entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output = Path(self.temp_dir) / "outputs"
        
    def test_main_with_csv_source_and_dry_run(self):
        """Should load CSV data and skip training in dry-run mode."""
        test_args = [
            'finance-ml',
            '--data-source', 'csv',
            '--data-dir', 'data',
            '--output-dir', str(self.temp_output),
            '--dry-run'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.simple_eda') as mock_eda:
                        mock_load.return_value = pd.DataFrame({
                            'ticker': ['AAPL', 'MSFT'],
                            'sector': ['Tech', 'Tech'],
                            'last_price': [150.0, 300.0]
                        })
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once()
                        mock_preprocess.assert_called_once()
                        mock_eda.assert_called_once()

    def test_main_with_skip_eda_flag(self):
        """Should skip EDA when --skip-eda flag is set."""
        test_args = [
            'finance-ml',
            '--data-source', 'csv',
            '--dry-run',
            '--skip-eda'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.simple_eda') as mock_eda:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = main()
                        
                        self.assertEqual(result, 0)
                        mock_eda.assert_not_called()

    def test_main_trains_models_when_not_dry_run(self):
        """Should train models when dry-run is not set."""
        test_args = [
            'finance-ml',
            '--data-source', 'csv',
            '--skip-eda'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.train_and_evaluate_regression') as mock_train:
                        with patch('finance_ml.cli.train_and_evaluate_regression_by_sector') as mock_sector:
                            mock_load.return_value = pd.DataFrame({
                                'ticker': ['AAPL'], 'sector': ['Tech'],
                                'last_price': [150.0], 'price_target': [160.0]
                            })
                            mock_preprocess.return_value = mock_load.return_value
                            mock_train.return_value = {'mae': 10.0}
                            mock_sector.return_value = {'Tech': {'mae': 10.0}}
                            
                            result = main()
                            
                            self.assertEqual(result, 0)
                            mock_train.assert_called_once()
                            mock_sector.assert_called_once()

    def test_main_skips_sector_models_with_flag(self):
        """Should skip sector models when --skip-sector-models flag is set."""
        test_args = [
            'finance-ml',
            '--data-source', 'csv',
            '--skip-eda',
            '--skip-sector-models'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.train_and_evaluate_regression') as mock_train:
                        with patch('finance_ml.cli.train_and_evaluate_regression_by_sector') as mock_sector:
                            mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                            mock_preprocess.return_value = mock_load.return_value
                            mock_train.return_value = {'mae': 10.0}
                            
                            result = main()
                            
                            self.assertEqual(result, 0)
                            mock_train.assert_called_once()
                            mock_sector.assert_not_called()

    def test_main_loads_config_from_file_when_provided(self):
        """Should load configuration from file when --config is provided."""
        config_file = Path(self.temp_dir) / "test_config.json"
        config_file.write_text('{"data_dir": "custom_data", "random_seed": 99}')
        
        test_args = [
            'finance-ml',
            '--config', str(config_file),
            '--dry-run',
            '--skip-eda'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                    mock_preprocess.return_value = mock_load.return_value
                    
                    result = main()
                    
                    self.assertEqual(result, 0)

    def test_main_overrides_config_with_cli_args(self):
        """Should override config values with command-line arguments."""
        test_args = [
            'finance-ml',
            '--data-dir', 'custom_data',
            '--n-jobs', '4',
            '--seed', '42',
            '--dry-run',
            '--skip-eda'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.load_config') as mock_load_config:
                        mock_config = FinanceMLConfig()
                        mock_load_config.return_value = mock_config
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = main()
                        
                        self.assertEqual(result, 0)
                        self.assertEqual(mock_config.n_jobs, 4)
                        self.assertEqual(mock_config.random_seed, 42)

    def test_main_returns_1_on_exception(self):
        """Should return 1 when an exception occurs."""
        test_args = ['finance-ml', '--dry-run']
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli._load_data', side_effect=Exception("Test error")):
                result = main()
                
                self.assertEqual(result, 1)

    def test_main_with_verbose_flag_sets_debug_logging(self):
        """Should set DEBUG logging level when --verbose flag is used."""
        test_args = [
            'finance-ml',
            '--verbose',
            '--dry-run',
            '--skip-eda'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.setup_logging') as mock_logging:
                with patch('finance_ml.cli._load_data') as mock_load:
                    with patch('finance_ml.cli.preprocess') as mock_preprocess:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = main()
                        
                        mock_logging.assert_called_once_with(level="DEBUG")


class TestAnalyzeMain(unittest.TestCase):
    """Test the analyze_main() CLI entry point."""

    def test_analyze_main_with_csv_source(self):
        """Should load CSV data and run EDA."""
        test_args = [
            'finance-ml-analyze',
            '--data-source', 'csv',
            '--data-dir', 'data'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.simple_eda') as mock_eda:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = analyze_main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once()
                        mock_preprocess.assert_called_once()
                        mock_eda.assert_called_once()

    def test_analyze_main_with_db_source_and_url(self):
        """Should load from database when source is db and url is provided."""
        test_args = [
            'finance-ml-analyze',
            '--data-source', 'db',
            '--db-url', 'postgresql://localhost/test'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_db') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.simple_eda') as mock_eda:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = analyze_main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once_with('postgresql://localhost/test', limit=None)

    def test_analyze_main_with_db_source_but_no_url_returns_1(self):
        """Should return 1 when db source is selected but no URL provided."""
        test_args = [
            'finance-ml-analyze',
            '--data-source', 'db'
        ]
        
        with patch.object(sys, 'argv', test_args):
            result = analyze_main()
            
            self.assertEqual(result, 1)

    def test_analyze_main_with_limit(self):
        """Should pass limit parameter to data loader."""
        test_args = [
            'finance-ml-analyze',
            '--data-source', 'csv',
            '--limit', '1000'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.preprocess') as mock_preprocess:
                    with patch('finance_ml.cli.simple_eda') as mock_eda:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_preprocess.return_value = mock_load.return_value
                        
                        result = analyze_main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once_with(Path('data'), limit=1000)

    def test_analyze_main_returns_1_on_exception(self):
        """Should return 1 when an exception occurs."""
        test_args = ['finance-ml-analyze']
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv', side_effect=Exception("Test error")):
                result = analyze_main()
                
                self.assertEqual(result, 1)

    def test_analyze_main_with_verbose_flag(self):
        """Should set DEBUG logging when verbose flag is used."""
        test_args = [
            'finance-ml-analyze',
            '--verbose'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.setup_logging') as mock_logging:
                with patch('finance_ml.cli.load_from_csv') as mock_load:
                    with patch('finance_ml.cli.preprocess') as mock_preprocess:
                        with patch('finance_ml.cli.simple_eda') as mock_eda:
                            mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                            mock_preprocess.return_value = mock_load.return_value
                            
                            result = analyze_main()
                            
                            mock_logging.assert_called_once_with(level="DEBUG")


class TestValidateMain(unittest.TestCase):
    """Test the validate_main() CLI entry point."""

    def test_validate_main_with_valid_data(self):
        """Should return 0 when data validation passes."""
        test_args = [
            'finance-ml-validate',
            '--data-source', 'csv'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.validate_schema') as mock_validate:
                    with patch('finance_ml.cli.check_missing_values') as mock_check:
                        mock_load.return_value = pd.DataFrame({
                            'ticker': ['AAPL'],
                            'sector': ['Tech'],
                            'last_price': [150.0]
                        })
                        mock_validate.return_value = True
                        mock_check.return_value = {}
                        
                        result = validate_main()
                        
                        self.assertEqual(result, 0)
                        mock_validate.assert_called_once()
                        mock_check.assert_called_once()

    def test_validate_main_returns_1_on_schema_validation_failure(self):
        """Should return 1 when schema validation fails."""
        test_args = ['finance-ml-validate']
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.validate_schema') as mock_validate:
                    mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                    mock_validate.return_value = False
                    
                    result = validate_main()
                    
                    self.assertEqual(result, 1)

    def test_validate_main_reports_missing_values(self):
        """Should report missing values when detected."""
        test_args = ['finance-ml-validate']
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.validate_schema') as mock_validate:
                    with patch('finance_ml.cli.check_missing_values') as mock_check:
                        mock_load.return_value = pd.DataFrame({
                            'ticker': ['AAPL'],
                            'sector': ['Tech'],
                            'last_price': [150.0]
                        })
                        mock_validate.return_value = True
                        mock_check.return_value = {
                            'price_target': {'count': 5, 'percent': 10.0}
                        }
                        
                        result = validate_main()
                        
                        self.assertEqual(result, 0)
                        mock_check.assert_called_once()

    def test_validate_main_with_db_source(self):
        """Should load from database when source is db."""
        test_args = [
            'finance-ml-validate',
            '--data-source', 'db',
            '--db-url', 'postgresql://localhost/test'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_db') as mock_load:
                with patch('finance_ml.cli.validate_schema') as mock_validate:
                    with patch('finance_ml.cli.check_missing_values') as mock_check:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_validate.return_value = True
                        mock_check.return_value = {}
                        
                        result = validate_main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once()

    def test_validate_main_with_db_source_but_no_url_returns_1(self):
        """Should return 1 when db source is selected but no URL provided."""
        test_args = [
            'finance-ml-validate',
            '--data-source', 'db'
        ]
        
        with patch.object(sys, 'argv', test_args):
            result = validate_main()
            
            self.assertEqual(result, 1)

    def test_validate_main_returns_1_on_exception(self):
        """Should return 1 when an exception occurs."""
        test_args = ['finance-ml-validate']
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv', side_effect=Exception("Test error")):
                result = validate_main()
                
                self.assertEqual(result, 1)

    def test_validate_main_with_limit(self):
        """Should pass limit parameter to data loader."""
        test_args = [
            'finance-ml-validate',
            '--limit', '500'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('finance_ml.cli.load_from_csv') as mock_load:
                with patch('finance_ml.cli.validate_schema') as mock_validate:
                    with patch('finance_ml.cli.check_missing_values') as mock_check:
                        mock_load.return_value = pd.DataFrame({'ticker': ['AAPL']})
                        mock_validate.return_value = True
                        mock_check.return_value = {}
                        
                        result = validate_main()
                        
                        self.assertEqual(result, 0)
                        mock_load.assert_called_once_with(Path('data'), limit=500)


if __name__ == '__main__':
    unittest.main()
