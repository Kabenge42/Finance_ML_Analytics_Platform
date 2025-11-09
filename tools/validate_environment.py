#!/usr/bin/env python3
"""
Environment Validation Script for Finance ML Analytics Platform
Detects and reporting environment conflicts, duplicate directories, and configuration issues.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class EnvironmentValidator:
    """Validates the project environment for conflicts and issues."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []

    def check_python_version(self) -> bool:
        """Check if Python version is supported."""
        version_info = sys.version_info
        current = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        if version_info.major != 3 or version_info.minor < 12:
            self.issues.append(
                {
                    "type": "python_version",
                    "severity": "critical",
                    "message": f"Python {current} detected. Requires Python 3.12 or 3.13",
                    "current": current,
                    "required": "3.12 or 3.13",
                }
            )
            return False
        elif version_info.minor > 13:
            self.warnings.append(
                {
                    "type": "python_version",
                    "severity": "warning",
                    "message": f"Python {current} not tested. Recommended: 3.12 or 3.13",
                    "current": current,
                }
            )

        return True

    def check_duplicate_venvs(self) -> List[Path]:
        """Find duplicate or conflicting virtual environment directories."""
        venv_patterns = [".venv", ".venv1", "venv", "env", "anaconda_projects"]
        found_venvs = []

        for pattern in venv_patterns:
            venv_path = self.project_root / pattern
            if venv_path.exists() and venv_path.is_dir():
                found_venvs.append(venv_path)

        if len(found_venvs) > 1:
            self.issues.append(
                {
                    "type": "duplicate_venvs",
                    "severity": "high",
                    "message": f"Multiple virtual environments found: {[str(v.name) for v in found_venvs]}",
                    "paths": [str(v) for v in found_venvs],
                    "recommendation": "Keep only one active virtual environment",
                }
            )

        return found_venvs

    def check_active_environment(self) -> Tuple[bool, str]:
        """Check which virtual environment is currently active."""
        virtual_env = os.environ.get("VIRTUAL_ENV")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")

        if virtual_env and conda_env:
            self.issues.append(
                {
                    "type": "conflicting_envs",
                    "severity": "critical",
                    "message": "Both venv and conda environments are active",
                    "venv": virtual_env,
                    "conda": conda_env,
                }
            )
            return False, "conflict"

        if virtual_env:
            # Check if it's in the project directory
            if not virtual_env.startswith(str(self.project_root)):
                self.warnings.append(
                    {
                        "type": "external_venv",
                        "severity": "warning",
                        "message": "Virtual environment is outside project directory",
                        "path": virtual_env,
                    }
                )
            return True, "venv"

        if conda_env:
            return True, "conda"

        self.warnings.append(
            {
                "type": "no_active_env",
                "severity": "warning",
                "message": "No virtual environment is currently active",
            }
        )
        return False, "none"

    def check_pyproject_config(self) -> bool:
        """Validate pyproject.toml configuration."""
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            self.issues.append(
                {
                    "type": "missing_config",
                    "severity": "high",
                    "message": "pyproject.toml not found",
                    "path": str(pyproject_path),
                }
            )
            return False

        try:
            import tomli
        except ImportError:
            try:
                import tomllib as tomli
            except ImportError:
                self.warnings.append(
                    {
                        "type": "missing_dependency",
                        "severity": "warning",
                        "message": "Cannot parse pyproject.toml (tomli/tomllib not available)",
                    }
                )
                return True

        try:
            with open(pyproject_path, "rb") as f:
                config = tomli.load(f)

            # Check Python version requirement
            requires_python = config.get("project", {}).get("requires-python")
            if requires_python:
                if "3.13" not in requires_python:
                    self.warnings.append(
                        {
                            "type": "python_support",
                            "severity": "warning",
                            "message": f"pyproject.toml requires-python may not include 3.13: {requires_python}",
                            "current": requires_python,
                        }
                    )

        except Exception as e:
            self.warnings.append(
                {
                    "type": "config_parse_error",
                    "severity": "warning",
                    "message": f"Error parsing pyproject.toml: {str(e)}",
                }
            )

        return True

    def check_git_ignored_files(self) -> None:
        """Check for environment directories not in .gitignore."""
        gitignore_path = self.project_root / ".gitignore"

        if not gitignore_path.exists():
            self.warnings.append(
                {
                    "type": "missing_gitignore",
                    "severity": "warning",
                    "message": ".gitignore not found",
                }
            )
            return

        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()

        venv_patterns = [".venv", ".venv1", "venv/", "env/", "anaconda_projects/"]
        missing_patterns = []

        for pattern in venv_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)

        if missing_patterns:
            self.warnings.append(
                {
                    "type": "gitignore_incomplete",
                    "severity": "warning",
                    "message": "Some environment directories may not be in .gitignore",
                    "missing": missing_patterns,
                }
            )

    def check_installed_packages(self) -> None:
        """Check for package installation issues."""
        try:
            import pkg_resources

            # Check for conflicting versions
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

            # Critical packages to check
            critical_packages = ["numpy", "pandas", "scikit-learn", "tensorflow", "torch"]
            missing_critical = []

            for package in critical_packages:
                if package not in installed_packages:
                    missing_critical.append(package)

            if missing_critical:
                self.warnings.append(
                    {
                        "type": "missing_packages",
                        "severity": "info",
                        "message": f"Critical packages not installed: {missing_critical}",
                    }
                )

        except ImportError:
            self.warnings.append(
                {
                    "type": "cannot_check_packages",
                    "severity": "info",
                    "message": "Cannot check installed packages (pkg_resources not available)",
                }
            )

    def generate_cleanup_script(self, venvs_to_remove: List[Path]) -> str:
        """Generate a cleanup script for duplicate environments."""
        if not venvs_to_remove:
            return ""

        script_lines = [
            "#!/usr/bin/env python3",
            '"""Auto-generated cleanup script for duplicate virtual environments."""',
            "",
            "import shutil",
            "from pathlib import Path",
            "",
            "# Directories to remove:",
        ]

        for venv in venvs_to_remove:
            script_lines.append(f"# - {venv}")

        script_lines.extend(["", "def cleanup():", "    paths_to_remove = ["])

        for venv in venvs_to_remove:
            script_lines.append(f'        Path(r"{venv}"),')

        script_lines.extend(
            [
                "    ]",
                "",
                "    for path in paths_to_remove:",
                "        if path.exists():",
                "            print(f'Removing {path}...')",
                "            shutil.rmtree(path)",
                "            print(f'[+] Removed {path}')",
                "        else:",
                "            print(f'[!] Path not found: {path}')",
                "",
                "if __name__ == '__main__':",
                "    response = input('This will remove the directories listed above. Continue? (yes/no): ')",
                "    if response.lower() == 'yes':",
                "        cleanup()",
                "        print('Cleanup complete!')",
                "    else:",
                "        print('Cleanup cancelled.')",
            ]
        )

        return "\n".join(script_lines)

    def validate(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("FINANCE ML ANALYTICS PLATFORM - ENVIRONMENT VALIDATION")
        print("=" * 70)
        print()

        # Run all checks
        python_ok = self.check_python_version()
        duplicate_venvs = self.check_duplicate_venvs()
        active_ok, env_type = self.check_active_environment()
        config_ok = self.check_pyproject_config()
        self.check_git_ignored_files()
        self.check_installed_packages()

        # Report results
        print(f"[*] Project Root: {self.project_root}")
        print(f"[*] Python Version: {sys.version.split()[0]}")
        print(f"[*] Active Environment: {env_type}")
        print()

        # Critical issues
        if self.issues:
            print("[!] CRITICAL ISSUES:")
            print("-" * 70)
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. [{issue['severity'].upper()}] {issue['message']}")
                for key, value in issue.items():
                    if key not in ["type", "severity", "message"]:
                        print(f"   {key}: {value}")
                print()

        # Warnings
        if self.warnings:
            print("[!] WARNINGS:")
            print("-" * 70)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. [{warning['severity'].upper()}] {warning['message']}")
                for key, value in warning.items():
                    if key not in ["type", "severity", "message"]:
                        print(f"   {key}: {value}")
                print()

        # Success message
        if not self.issues and not self.warnings:
            print("[+] All checks passed! Environment is properly configured.")
            print()

        # Generate cleanup script if needed
        if len(duplicate_venvs) > 1:
            print("[*] CLEANUP RECOMMENDATION:")
            print("-" * 70)
            print("Multiple virtual environments detected. Keep only one:")
            for venv in duplicate_venvs:
                print(f"  - {venv.name}")
            print()

            # Ask which to keep
            print("Generating cleanup script...")
            cleanup_script_path = self.project_root / "cleanup_environments.py"

            # Suggest keeping .venv, removing others
            venvs_to_remove = [v for v in duplicate_venvs if v.name != ".venv"]
            if not venvs_to_remove:
                venvs_to_remove = duplicate_venvs[1:]  # Keep first, remove rest

            cleanup_script = self.generate_cleanup_script(venvs_to_remove)

            with open(cleanup_script_path, "w", encoding="utf-8") as f:
                f.write(cleanup_script)

            print(f"[+] Cleanup script generated: {cleanup_script_path}")
            print(f"    Run with: python {cleanup_script_path}")
            print()

        # Summary
        print("=" * 70)
        print("SUMMARY:")
        print(f"  Critical Issues: {len(self.issues)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Status: {'[!] FAILED' if self.issues else '[+] PASSED'}")
        print("=" * 70)

        return len(self.issues) == 0


def main():
    """Main entry point."""
    validator = EnvironmentValidator()
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
