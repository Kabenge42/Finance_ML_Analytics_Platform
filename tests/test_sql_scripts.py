import unittest
from pathlib import Path


class TestSQLScriptsConsistency(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        # Paths
        self.pg_schema = self.root / "create_equities_schema.sql"
        self.pg_import = self.root / "import_equities_data.sql"
        self.sqlite_schema = self.root / "create_equities_schema_sqlite.sql"
        self.sqlite_import = self.root / "import_equities_data_sqlite.sql"

    def test_sql_files_exist(self):
        for p in [self.pg_schema, self.pg_import, self.sqlite_schema, self.sqlite_import]:
            self.assertTrue(p.exists(), f"Missing SQL file: {p}")

    def test_sqlite_schema_has_no_pg_specific_syntax(self):
        text = self.sqlite_schema.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        # Must create equities table
        self.assertIn("create table equities", lowered)
        # Should not include PG-only items
        self.assertNotIn("tablespace", lowered)
        self.assertNotIn("owner to postgres", lowered)
        self.assertNotIn("comment on table", lowered)
        # Should include a unique index for ticker+region
        self.assertRegex(lowered, r"create\s+unique\s+index[\s\S]*ticker\"?,\s*\"?region")
        # Should quote at least one spaced column name (sanity check for quoting consistency)
        self.assertIn('"Last Price"'.lower(), lowered)

    def test_sqlite_import_uses_sqlite_cli_commands_and_not_pg(self):
        text = self.sqlite_import.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        # Uses sqlite meta-commands
        self.assertIn(".mode csv", lowered)
        self.assertIn(".headers on", lowered)
        # Does not contain PG meta-commands or blocks
        for forbidden in ["\\copy", "do $$", "raise exception", "owner to postgres", "tablespace"]:
            self.assertNotIn(forbidden, lowered)
        # Uses INSERT OR IGNORE semantics (sqlite) and not ON CONFLICT DO NOTHING in this file
        # Build the phrase dynamically to avoid tooling mis-parsing
        phrase = "insert" + " or " + "ignore into equities"
        self.assertIn(phrase, lowered)
        self.assertNotIn("on conflict do nothing", lowered)

    def test_pg_import_has_no_sqlite_dot_commands(self):
        text = self.pg_import.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        # Contains PG features
        self.assertIn("on conflict do nothing", lowered)
        self.assertIn("\\copy", lowered)
        # Must not include sqlite dot-commands
        self.assertNotIn(".mode", lowered)
        self.assertNotIn(".headers", lowered)
        self.assertNotIn(".import", lowered)

    def test_consistent_sqlite_import_format(self):
        text = self.sqlite_import.read_text(encoding="utf-8", errors="ignore")
        # Ensure there are exactly four .import lines (one per region)
        import_lines = [ln for ln in text.splitlines() if ln.strip().startswith(".import")]
        self.assertEqual(
            4,
            len(import_lines),
            f"Expected 4 .import lines, found {len(import_lines)}: {import_lines}",
        )
        # Ensure no per-line flags like --csv or --skip in .import lines (we configure mode globally)
        for ln in import_lines:
            self.assertNotRegex(ln, r"--(csv|skip)")
        # Ensure each region has header deletion to guard against header row being imported
        regions = {
            "us": "equities_staging_us",
            "eu": "equities_staging_eu",
            "apac": "equities_staging_apac",
            "rotw": "equities_staging_rotw",
        }
        lowered = text.lower()
        for region, staging in regions.items():
            pattern = rf"delete\s+from\s+{staging}\s+where\s+\"ticker\"\s*=\s*'ticker'"
            self.assertRegex(lowered, pattern, f"Missing header deletion step for region {region}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
