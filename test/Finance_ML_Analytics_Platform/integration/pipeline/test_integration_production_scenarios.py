import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SQLITE_SCHEMA = (
    Path(__file__).resolve().parents[1] / "create_equities_schema_sqlite.sql"
).read_text(encoding="utf-8")


def _init_sqlite_db(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SQLITE_SCHEMA)
        # Seed a few rows across regions
        rows = [
            ("USTK1", "Technology", 10.0, 12.0, "US"),
            ("EUTK1", "Healthcare", 11.0, 13.0, "EU"),
            ("APTK1", "Financials", 9.5, 10.5, "APAC"),
            ("ROTK1", "Energy", 7.0, 7.8, "ROTW"),
        ]
        con.executemany(
            'INSERT INTO equities ("Ticker","Sector","Last Price","Price Target","Region") VALUES (?,?,?,?,?)',
            rows,
        )
        con.commit()
    finally:
        con.close()


@unittest.skipUnless(
    os.environ.get("RUN_DB_INTEGRATION", "0") == "1",
    "Set RUN_DB_INTEGRATION=1 to enable DB integration test",
)
class TestIntegrationProductionScenarios(unittest.TestCase):
    def test_cli_runs_with_sqlite_db_url(self):
        try:
            import sqlalchemy  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dependency guard
            self.skipTest(f"SQLAlchemy not available: {e}")

        with tempfile.TemporaryDirectory() as td_db, tempfile.TemporaryDirectory() as td_out:
            db_file = Path(td_db) / "equities.sqlite"
            _init_sqlite_db(db_file)

            out_dir = Path(td_out)
            env = os.environ.copy()
            env["DB_URL"] = f"sqlite:///{db_file}"
            env["FINANCE_ML_FAST_TEST"] = "1"

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "ml_finance_model_main.py"),
                "--data-source",
                "db",
                "--limit",
                "100",
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                text=True,
                check=False,
            )

            self.assertEqual(
                proc.returncode,
                0,
                msg=f"DB CLI pipeline failed with code {proc.returncode}. Output:\n{proc.stdout}",
            )

            # EDA artifact should exist
            eda_path = out_dir / "eda_summary.json"
            self.assertTrue(eda_path.exists(), msg=f"Missing {eda_path}\nOutput:\n{proc.stdout}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
