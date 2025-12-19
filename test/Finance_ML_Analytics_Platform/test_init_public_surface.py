"""Tests for the top-level finance_ml public surface (Phase 6.2).

These tests validate the *minimal* guarantees we intend to keep at the
package root while the broader API is gradually streamlined in favour of
``finance_ml.api``.

The goals:
- Ensure that ``finance_ml`` exposes ``__version__`` and a stable
  ``api`` facade.
- Confirm that key configuration / notebook classes remain importable
  from the top level for backward compatibility.
"""

from __future__ import annotations

import importlib
import unittest


class TestTopLevelAPI(unittest.TestCase):
    def test_basic_symbols_present(self):
        import finance_ml

        # Version must exist (smoke-tested elsewhere too)
        self.assertTrue(hasattr(finance_ml, "__version__"))

        # New clean facade should be accessible as attribute
        self.assertTrue(hasattr(finance_ml, "api"))

        # Package should define a curated __all__ describing first-class exports
        self.assertTrue(hasattr(finance_ml, "__all__"))

        # Key config / notebook classes remain importable from root
        curated = [
            "__version__",
            "api",
            "FinanceMLConfig",
            "load_config",
            "NotebookConfig",
            "simple_eda",
        ]

        for name in curated:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(finance_ml, name),
                    msg=f"finance_ml is expected to expose {name} at top level",
                )

        # __all__ exists to document the intended public surface, but we
        # do not assert exact contents here yet because the legacy export
        # list is still being gradually reduced. Backward-compatible
        # attributes may exist without being part of the long-term
        # supported API.

    def test_api_attribute_matches_module(self):
        import finance_ml
        import finance_ml.api as api_module

        # The attribute finance_ml.api should reference the same module
        # object that you get from importing finance_ml.api directly.
        self.assertIs(finance_ml.api, api_module)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
