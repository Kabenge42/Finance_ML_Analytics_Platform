import io
import sys
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# Module under test
import setup_environment as se


def make_fake_version(major=3, minor=13, micro=9):
    # Minimal object exposing .major/.minor/.micro
    return types.SimpleNamespace(major=major, minor=minor, micro=micro)


class TestEnvironmentSetup(unittest.TestCase):
    def setUp(self):
        # Default args for tests; most steps are not executed directly
        self.args = Namespace(
            recreate_venv=False,
            skip_db=True,
            skip_data_load=False,
            db_host="localhost",
            db_port=5432,
            db_user="postgres",
            db_name="postgres",
            db_password="",
            install_db_libs=False,
            skip_tests=True,
            force=False,
        )

    def _make_setup(self):
        # Patch color to avoid ANSI codes in assertions
        with patch.object(se.Color, "supports_color", return_value=False):
            # Stub subprocess.run called during __init__ in _detect_python_command
            def fake_run(cmd, **kwargs):
                # Simulate successful `python --version` query
                if isinstance(cmd, list) and "--version" in cmd:

                    class Res:
                        returncode = 0
                        stdout = "Python 3.13.9\n"
                        stderr = ""

                    if kwargs.get("check") and kwargs.get("capture_output"):
                        return Res()
                    return Res()

                # Default fallback
                class Res:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Res()

            with patch("subprocess.run", side_effect=fake_run):
                return se.EnvironmentSetup(self.args)

    def test_python_version_check_accepts_3_13_plus(self):
        setup = self._make_setup()
        # Place required files/dirs in a temp location and point project_root there
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # create required structure
            (root / "data").mkdir()
            (root / "tests").mkdir()
            for fname in [
                "requirements.txt",
                "create_equities_schema.sql",
                "environment_variables.txt",
            ]:
                (root / fname).write_text("# placeholder\n", encoding="utf-8")

            setup.project_root = root

            buf = io.StringIO()
            with patch.object(sys, "version_info", make_fake_version(3, 13, 9)):
                with redirect_stdout(buf):
                    ok = setup.check_prerequisites()
            self.assertTrue(
                ok, msg=f"Prerequisites should pass on Python 3.13: output={buf.getvalue()}"
            )
            self.assertFalse(any("Python 3.10+ required" in e for e in setup.errors))

    def test_run_command_missing_executable_check_false_does_not_raise(self):
        setup = self._make_setup()
        # When the executable is missing, with check=False, it should not raise and return (-1, '', err)
        code, out, err = setup._run_command(
            ["definitely-not-a-real-exe-xyz"], check=False, capture_output=True
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(out, "")
        self.assertTrue(err)
        # Ensure no error got appended implicitly
        self.assertEqual(setup.errors, [])

    def test_run_command_missing_executable_check_true_raises_and_tracks_error(self):
        setup = self._make_setup()
        with self.assertRaises(FileNotFoundError):
            setup._run_command(["definitely-not-a-real-exe-xyz"], check=True)
        # Error should be recorded with a meaningful message
        self.assertTrue(any("Command not found" in e for e in setup.errors))

    def test_postgres_check_is_skipped_when_flag_set(self):
        # Ensure psql is not invoked when --skip-db is used
        setup = self._make_setup()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "tests").mkdir()
            for fname in [
                "requirements.txt",
                "create_equities_schema.sql",
                "environment_variables.txt",
            ]:
                (root / fname).write_text("# placeholder\n", encoding="utf-8")
            setup.project_root = root

            calls = []

            def spy_run(cmd, *args, **kwargs):
                calls.append(cmd)
                # Simulate success for all commands
                return 0, "OK", ""

            with patch.object(setup, "_run_command", side_effect=spy_run):
                ok = setup.check_prerequisites()

            self.assertTrue(ok)
            # Ensure no psql call was made
            self.assertFalse(any(cmd and cmd[0] == "psql" for cmd in calls))

    def test_postgres_missing_is_warning_and_blocks_when_not_forced(self):
        # When skip_db=False and psql is missing, prerequisites should fail unless --force is set
        self.args.skip_db = False
        setup = self._make_setup()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "tests").mkdir()
            for fname in [
                "requirements.txt",
                "create_equities_schema.sql",
                "environment_variables.txt",
            ]:
                (root / fname).write_text("# placeholder\n", encoding="utf-8")
            setup.project_root = root

            def fake_run(cmd, *args, **kwargs):
                if cmd and cmd[0] == "psql":
                    return -1, "", "FileNotFoundError: psql"
                if cmd and cmd[:3] == [setup.python_cmd, "-m", "pip"]:
                    return 0, "pip 24.0 from x (python 3.13)", ""
                return 0, "", ""

            with patch.object(setup, "_run_command", side_effect=fake_run):
                ok = setup.check_prerequisites()

            self.assertFalse(ok)
            # Now force=True should allow continuation
            self.args.force = True
            setup_forced = self._make_setup()
            setup_forced.project_root = root
            with patch.object(setup_forced, "_run_command", side_effect=fake_run):
                ok_forced = setup_forced.check_prerequisites()
            self.assertTrue(ok_forced)

    def test_ascii_output_markers_and_error_summary_non_empty(self):
        setup = self._make_setup()
        # Force a couple of errors to be present
        setup._print_error("First error message")
        setup._print_error("Second error message")
        buf = io.StringIO()
        with redirect_stdout(buf):
            setup.print_activation_instructions()
        out = buf.getvalue()
        # Check ASCII markers
        self.assertIn("[ERROR] First error message", out)
        self.assertIn("[ERROR] Second error message", out)
        # Ensure messages are enumerated and non-empty
        # Fixed: The actual format includes both the number and [ERROR] prefix
        self.assertIn("1. [ERROR] First error message", out)
        self.assertIn("2. [ERROR] Second error message", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
