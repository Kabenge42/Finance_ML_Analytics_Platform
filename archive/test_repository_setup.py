import unittest
from pathlib import Path


class TestRepositorySetup(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_required_files_exist(self):
        required = [
            self.root / "requirements.txt",
            self.root / "create_equities_schema.sql",
            self.root / "environment_variables.txt",
            self.root / "data" / "screening_us.csv",
            self.root / "data" / "screening_eu.csv",
            self.root / "data" / "screening_apac.csv",
            self.root / "data" / "screening_rotw.csv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        self.assertFalse(missing, f"Missing required files: {missing}")

    def test_sql_contains_required_statements(self):
        sql_path = self.root / "create_equities_schema.sql"
        text = sql_path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        self.assertIn("create table equities", lowered)
        self.assertIn("owner to postgres", lowered)

    def test_env_file_has_tf_log_level(self):
        env_path = self.root / "environment_variables.txt"
        text = env_path.read_text(encoding="utf-8", errors="ignore")
        # normalize whitespace and comments
        lines = [
            l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")
        ]
        normalized = {l.replace(" ", "") for l in lines}
        self.assertIn("TF_CPP_MIN_LOG_LEVEL=2", normalized)

    def test_csvs_have_header_and_are_not_empty(self):
        data_dir = self.root / "data"
        csvs = [
            data_dir / "screening_us.csv",
            data_dir / "screening_eu.csv",
            data_dir / "screening_apac.csv",
            data_dir / "screening_rotw.csv",
        ]
        for csv in csvs:
            size = csv.stat().st_size
            self.assertGreater(size, 0, f"CSV appears empty: {csv}")
            # Read the first non-empty line as header and assert it has commas
            with csv.open("r", encoding="utf-8", errors="ignore") as f:
                header = ""
                for line in f:
                    if line.strip():
                        header = line
                        break
            self.assertTrue("," in header, f"CSV header seems malformed (no commas): {csv}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
