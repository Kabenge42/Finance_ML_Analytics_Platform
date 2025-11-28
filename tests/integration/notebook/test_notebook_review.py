"""
Unit tests for notebook_review static analyzer.

Tests the notebook function signature validation functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path

from finance_ml.ml_workflow.quality.notebook_review import (
    parse_notebook,
    extract_function_calls,
    check_notebook_function_signatures,
    get_function_signature,
)


class TestNotebookReview(unittest.TestCase):
    """Test notebook review static analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_parse_notebook_extracts_code_cells(self):
        """Test that parse_notebook extracts code from code cells."""
        # Create a minimal notebook
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["print('hello')\n", "x = 1"]},
                {"cell_type": "markdown", "source": ["# Header"]},
                {"cell_type": "code", "source": ["y = 2"]},
            ]
        }

        notebook_path = Path(self.temp_dir) / "test.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        code_cells = parse_notebook(notebook_path)

        self.assertEqual(len(code_cells), 2)
        self.assertIn("print('hello')", code_cells[0])
        self.assertIn("y = 2", code_cells[1])

    def test_extract_function_calls_finds_calls(self):
        """Test that extract_function_calls finds function calls in code."""
        code = """
def foo():
    pass

foo()
bar(x=1, y=2)
obj.method()
"""
        calls = extract_function_calls(code)

        # Should find foo(), bar(), and method()
        func_names = [name for name, _, _ in calls]
        self.assertIn("foo", func_names)
        self.assertIn("bar", func_names)
        self.assertIn("method", func_names)

    def test_get_function_signature_returns_signature(self):
        """Test that get_function_signature retrieves function signature."""
        sig = get_function_signature(
            "finance_ml.ml_workflow.evaluation", "safety_rails_sensitivity_app"
        )

        self.assertIsNotNone(sig)
        self.assertIn("data_df", sig.parameters)
        self.assertIn("output_dir", sig.parameters)

    def test_get_function_signature_returns_none_for_invalid(self):
        """Test that get_function_signature returns None for invalid function."""
        sig = get_function_signature("nonexistent.module", "nonexistent_function")

        self.assertIsNone(sig)

    def test_check_notebook_detects_invalid_parameter(self):
        """Test that check_notebook_function_signatures detects invalid parameters."""
        # Create notebook with invalid parameter
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from finance_ml.ml_workflow.evaluation import safety_rails_sensitivity_app\n",
                        "safety_rails_sensitivity_app(\n",
                        "    df_raw=data,\n",  # Wrong parameter name
                        "    output_dir='outputs'\n",
                        ")",
                    ],
                }
            ]
        }

        notebook_path = Path(self.temp_dir) / "test_invalid.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        report = check_notebook_function_signatures(notebook_path)

        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertGreater(report["summary"]["issues_found"], 0)

        # Check that the issue is about df_raw parameter
        issues = report["issues"]
        self.assertTrue(any("df_raw" in str(issue) for issue in issues))

    def test_check_notebook_passes_valid_parameters(self):
        """Test that check_notebook_function_signatures passes valid parameters."""
        # Create notebook with valid parameters
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from finance_ml.ml_workflow.evaluation import safety_rails_sensitivity_app\n",
                        "safety_rails_sensitivity_app(\n",
                        "    data_df=data,\n",  # Correct parameter name
                        "    output_dir='outputs'\n",
                        ")",
                    ],
                }
            ]
        }

        notebook_path = Path(self.temp_dir) / "test_valid.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        report = check_notebook_function_signatures(notebook_path)

        self.assertEqual(report["summary"]["status"], "PASS")
        self.assertEqual(report["summary"]["issues_found"], 0)

    def test_check_notebook_handles_multiple_functions(self):
        """Test validation of multiple different functions."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "from finance_ml.ml_workflow.evaluation import (\n",
                        "    estimate_sector_bias,\n",
                        "    plot_metrics_by_sector_time\n",
                        ")\n",
                        "\n",
                        "# This should pass\n",
                        "estimate_sector_bias(\n",
                        "    predictions_df=df,\n",
                        "    output_dir='outputs',\n",
                        "    model_version='v9_9'\n",
                        ")\n",
                        "\n",
                        "# This should fail - wrong parameter name\n",
                        "plot_metrics_by_sector_time(\n",
                        "    metrics_history=df,\n",  # Should be predictions_df
                        "    output_dir='outputs'\n",
                        ")\n",
                    ],
                }
            ]
        }

        notebook_path = Path(self.temp_dir) / "test_multiple.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        report = check_notebook_function_signatures(notebook_path)

        # Should find issue with plot_metrics_by_sector_time
        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertGreater(report["summary"]["issues_found"], 0)

        # Check we found the metrics_history issue
        issues = report["issues"]
        self.assertTrue(
            any(
                issue["function"] == "plot_metrics_by_sector_time"
                and issue["parameter"] == "metrics_history"
                for issue in issues
            )
        )

    def test_extract_function_calls_handles_syntax_errors(self):
        """Test that extract_function_calls handles syntax errors gracefully."""
        code = "this is not valid python syntax { [ ("

        # Should not raise exception
        calls = extract_function_calls(code)

        # Should return empty list for syntax errors
        self.assertEqual(calls, [])

    def test_check_notebook_with_custom_functions(self):
        """Test checking custom function list."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "import pandas as pd\n",
                        "df = pd.read_csv('file.csv', sep=',', header=0)\n",
                    ],
                }
            ]
        }

        notebook_path = Path(self.temp_dir) / "test_custom.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        # Check pandas read_csv function
        custom_functions = {"read_csv": "pandas"}

        report = check_notebook_function_signatures(
            notebook_path, target_functions=custom_functions
        )

        # Should check the read_csv call
        self.assertGreater(report["summary"]["calls_checked"], 0)


if __name__ == "__main__":
    unittest.main()
