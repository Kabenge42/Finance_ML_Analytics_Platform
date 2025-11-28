"""
Ultra-fast smoke test to validate local environment and imports.

Run alone (Windows PowerShell):
  python -m unittest tests.test_demo_quick -v
"""

from __future__ import annotations

import re
import unittest


class TestDemoQuick(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(2 + 2, 4)
        self.assertAlmostEqual(10 / 4, 2.5)

    def test_package_import_and_version(self):
        import finance_ml

        self.assertTrue(hasattr(finance_ml, "__version__"))
        # Accepts semantic-like versions such as 0.4.1 or v9_9
        version = str(finance_ml.__version__)
        self.assertRegex(version, r"^(v?\d+[._]\d+(?:[._]\d+)?)$")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
