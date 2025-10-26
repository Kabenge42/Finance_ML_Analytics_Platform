import unittest
from pathlib import Path

from tools.apply_improvement_plan_updates import (
    apply_sqlite_parity_section,
    has_sqlite_section,
    SQLITE_SECTION_TITLE,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "IMPROVEMENT_PLAN.md"


class TestImprovementPlanRevision(unittest.TestCase):
    def setUp(self):
        # Read original for restoration in tearDown
        self.original_full = PLAN_PATH.read_text(encoding="utf-8")
        # Ensure tests start from a state without the new section for the first test
        if has_sqlite_section(self.original_full):
            # Remove the section block if present to force a failing-first scenario
            start = self.original_full.find(SQLITE_SECTION_TITLE)
            if start != -1:
                # remove from start title to end of file for test precondition
                trimmed = self.original_full[: start].rstrip() + "\n"
                PLAN_PATH.write_text(trimmed, encoding="utf-8")
        # else leave file as-is

    def tearDown(self):
        # Restore original content regardless of test changes
        PLAN_PATH.write_text(self.original_full, encoding="utf-8")

    def test_sqlite_parity_section_is_added(self):
        text = PLAN_PATH.read_text(encoding="utf-8")
        self.assertFalse(
            has_sqlite_section(text),
            msg="Precondition: SQLite parity section should be absent before applying",
        )
        modified = apply_sqlite_parity_section(PLAN_PATH)
        self.assertTrue(modified, msg="The plan file should be modified on first apply")
        updated = PLAN_PATH.read_text(encoding="utf-8")
        self.assertIn(
            SQLITE_SECTION_TITLE,
            updated,
            msg="The new SQLite parity section title must be present in the plan",
        )
        # Sanity check a few key lines exist
        for needle in [
            "#### Problem summary",
            "#### Proposed approach",
            "SQLite shell `.import` treats the first row as data",
            "per-region TEMP staging tables",
            "INSERT OR IGNORE",
            "tests/test_sqlite_import.py",
        ]:
            self.assertIn(needle, updated)

    def test_idempotent_apply_does_not_duplicate_section(self):
        # First application
        first = apply_sqlite_parity_section(PLAN_PATH)
        self.assertTrue(first)
        # Second application should be a no-op
        second = apply_sqlite_parity_section(PLAN_PATH)
        self.assertFalse(second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
