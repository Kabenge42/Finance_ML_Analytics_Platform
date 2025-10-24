#!/usr/bin/env python3
"""
Finance ML Analytics Platform - Automated Environment Setup Script

This script automates the complete environment recreation process including:
- Virtual environment creation and activation
- Python dependency installation
- PostgreSQL database setup and verification
- Data loading from regional CSV files
- Environment variable configuration
- Test execution and validation

Supports: Windows, macOS, and Linux
Python: 3.10 or 3.11
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class Color:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class EnvironmentSetup:
    """Automated environment setup for Finance ML Analytics Platform"""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.venv_path = self.project_root / '.venv'
        self.platform = platform.system()
        self.python_cmd = self._detect_python_command()
        self.errors = []

    def _detect_python_command(self) -> str:
        """Detect the appropriate Python command for this system"""
        if self.platform == 'Windows':
            return 'python'
        else:
            # Try python3 first, fall back to python
            try:
                subprocess.run(['python3', '--version'],
                             check=True, capture_output=True)
                return 'python3'
            except (subprocess.CalledProcessError, FileNotFoundError):
                return 'python'

    def _print_section(self, message: str):
        """Print a formatted section header"""
        print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
        print(f"{Color.BOLD}{Color.BLUE}{message}{Color.END}")
        print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}\n")

    def _print_success(self, message: str):
        """Print a success message"""
        print(f"{Color.GREEN}✓ {message}{Color.END}")

    def _print_warning(self, message: str):
        """Print a warning message"""
        print(f"{Color.YELLOW}⚠ {message}{Color.END}")

    def _print_error(self, message: str):
        """Print an error message"""
        print(f"{Color.RED}✗ {message}{Color.END}")
        self.errors.append(message)

    def _run_command(self, cmd: List[str], check: bool = True,
                     capture_output: bool = False, shell: bool = False) -> Tuple[int, str, str]:
        """
        Run a command and return exit code, stdout, stderr

        Args:
            cmd: Command as list of strings
            check: Raise exception on non-zero exit
            capture_output: Capture stdout/stderr
            shell: Run command through shell

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        try:
            if shell:
                cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
                result = subprocess.run(cmd_str, shell=True,
                                       capture_output=capture_output,
                                       text=True, check=check)
            else:
                result = subprocess.run(cmd, capture_output=capture_output,
                                       text=True, check=check)
            return result.returncode, result.stdout if capture_output else '', \
                   result.stderr if capture_output else ''
        except subprocess.CalledProcessError as e:
            if not check:
                return e.returncode, e.stdout if capture_output else '', \
                       e.stderr if capture_output else ''
            raise

    def check_prerequisites(self) -> bool:
        """Check if required prerequisites are installed"""
        self._print_section("Checking Prerequisites")

        all_ok = True

        # Check Python version
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor in (10, 11):
            self._print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} found")
        else:
            self._print_error(f"Python 3.10 or 3.11 required, found {version_info.major}.{version_info.minor}.{version_info.micro}")
            all_ok = False

        # Check pip
        try:
            code, stdout, _ = self._run_command([self.python_cmd, '-m', 'pip', '--version'],
                                                capture_output=True)
            if code == 0:
                self._print_success(f"pip found: {stdout.strip()}")
            else:
                self._print_error("pip not found or not working")
                all_ok = False
        except Exception as e:
            self._print_error(f"pip check failed: {e}")
            all_ok = False

        # Check PostgreSQL (if database setup requested)
        if not self.args.skip_db:
            try:
                code, stdout, _ = self._run_command(['psql', '--version'],
                                                    capture_output=True, check=False)
                if code == 0:
                    self._print_success(f"PostgreSQL found: {stdout.strip()}")
                else:
                    self._print_warning("PostgreSQL 'psql' not found in PATH (use --skip-db to skip database setup)")
                    if not self.args.force:
                        all_ok = False
            except FileNotFoundError:
                self._print_warning("PostgreSQL 'psql' not found in PATH (use --skip-db to skip database setup)")
                if not self.args.force:
                    all_ok = False

        # Check required directories
        required_dirs = ['data', 'tests']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self._print_success(f"Directory '{dir_name}/' exists")
            else:
                self._print_warning(f"Directory '{dir_name}/' not found")

        # Check required files
        required_files = ['requirements.txt', 'create_equities_schema.sql', 'environment_variables.txt']
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self._print_success(f"File '{file_name}' exists")
            else:
                self._print_error(f"Required file '{file_name}' not found")
                all_ok = False

        return all_ok

    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        self._print_section("Creating Virtual Environment")

        if self.venv_path.exists() and not self.args.recreate_venv:
            self._print_warning(f"Virtual environment already exists at {self.venv_path}")
            self._print_warning("Use --recreate-venv to recreate it")
            return True

        if self.venv_path.exists() and self.args.recreate_venv:
            self._print_warning(f"Removing existing virtual environment at {self.venv_path}")
            import shutil
            shutil.rmtree(self.venv_path)

        try:
            self._print_success(f"Creating virtual environment at {self.venv_path}")
            self._run_command([self.python_cmd, '-m', 'venv', str(self.venv_path)])
            self._print_success("Virtual environment created successfully")
            return True
        except Exception as e:
            self._print_error(f"Failed to create virtual environment: {e}")
            return False

    def _get_venv_python(self) -> str:
        """Get the path to the Python executable in the virtual environment"""
        if self.platform == 'Windows':
            return str(self.venv_path / 'Scripts' / 'python.exe')
        else:
            return str(self.venv_path / 'bin' / 'python')

    def _get_venv_pip(self) -> List[str]:
        """Get the command to run pip in the virtual environment"""
        venv_python = self._get_venv_python()
        return [venv_python, '-m', 'pip']

    def upgrade_pip(self) -> bool:
        """Upgrade pip, setuptools, and wheel"""
        self._print_section("Upgrading Package Tools")

        try:
            pip_cmd = self._get_venv_pip()
            self._print_success("Upgrading pip, setuptools, and wheel...")
            self._run_command(pip_cmd + ['install', '--upgrade', 'pip', 'setuptools', 'wheel'])
            self._print_success("Package tools upgraded successfully")
            return True
        except Exception as e:
            self._print_error(f"Failed to upgrade package tools: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install Python dependencies from requirements.txt"""
        self._print_section("Installing Python Dependencies")

        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self._print_error("requirements.txt not found")
            return False

        try:
            pip_cmd = self._get_venv_pip()
            self._print_success(f"Installing dependencies from {requirements_file}")
            self._run_command(pip_cmd + ['install', '-r', str(requirements_file)])
            self._print_success("Dependencies installed successfully")

            # Install optional database libraries if requested
            if self.args.install_db_libs:
                self._print_success("Installing optional database libraries (psycopg2-binary, SQLAlchemy)")
                self._run_command(pip_cmd + ['install', 'psycopg2-binary', 'SQLAlchemy'])
                self._print_success("Database libraries installed successfully")

            return True
        except Exception as e:
            self._print_error(f"Failed to install dependencies: {e}")
            return False

    def setup_database(self) -> bool:
        """Setup PostgreSQL database and load data"""
        if self.args.skip_db:
            self._print_warning("Skipping database setup (--skip-db flag)")
            return True

        self._print_section("Setting Up PostgreSQL Database")

        # Database connection parameters
        db_host = self.args.db_host
        db_port = self.args.db_port
        db_user = self.args.db_user
        db_name = self.args.db_name

        # Check if schema file exists
        schema_file = self.project_root / 'create_equities_schema.sql'
        if not schema_file.exists():
            self._print_error(f"Schema file not found: {schema_file}")
            return False

        # Create equities table
        try:
            self._print_success(f"Creating equities table in database '{db_name}'")
            psql_cmd = [
                'psql',
                '-h', db_host,
                '-p', str(db_port),
                '-U', db_user,
                '-d', db_name,
                '-f', str(schema_file)
            ]

            if self.args.db_password:
                env = os.environ.copy()
                env['PGPASSWORD'] = self.args.db_password
                subprocess.run(psql_cmd, env=env, check=True)
            else:
                self._run_command(psql_cmd)

            self._print_success("Database table created successfully")
        except Exception as e:
            self._print_error(f"Failed to create database table: {e}")
            if not self.args.force:
                return False

        # Load CSV data
        if not self.args.skip_data_load:
            return self.load_csv_data(db_host, db_port, db_user, db_name)

        return True

    def load_csv_data(self, db_host: str, db_port: int,
                      db_user: str, db_name: str) -> bool:
        """Load CSV data into PostgreSQL"""
        self._print_section("Loading CSV Data")

        data_dir = self.project_root / 'data'
        if not data_dir.exists():
            self._print_warning(f"Data directory not found: {data_dir}")
            return True

        # Regional CSV files
        regions = {
            'US': 'screening_us.csv',
            'EU': 'screening_eu.csv',
            'APAC': 'screening_apac.csv',
            'ROTW': 'screening_rotw.csv'
        }

        env = os.environ.copy()
        if self.args.db_password:
            env['PGPASSWORD'] = self.args.db_password

        for region_code, csv_file in regions.items():
            csv_path = data_dir / csv_file
            if not csv_path.exists():
                self._print_warning(f"CSV file not found: {csv_path} (skipping)")
                continue

            try:
                self._print_success(f"Loading {region_code} data from {csv_file}")

                # Using psql \copy for client-side copy (works better on Windows)
                copy_cmd = f"\\copy equities FROM '{csv_path.absolute()}' WITH (FORMAT csv, HEADER true)"

                psql_cmd = [
                    'psql',
                    '-h', db_host,
                    '-p', str(db_port),
                    '-U', db_user,
                    '-d', db_name,
                    '-c', copy_cmd
                ]

                if self.args.db_password:
                    subprocess.run(psql_cmd, env=env, check=True)
                else:
                    self._run_command(psql_cmd)

                # Update Region column if needed
                update_cmd = f"UPDATE equities SET \"Region\"='{region_code}' WHERE \"Region\" IS NULL"
                psql_update = [
                    'psql',
                    '-h', db_host,
                    '-p', str(db_port),
                    '-U', db_user,
                    '-d', db_name,
                    '-c', update_cmd
                ]

                if self.args.db_password:
                    subprocess.run(psql_update, env=env, check=True)
                else:
                    self._run_command(psql_update)

                self._print_success(f"Loaded {region_code} data successfully")
            except Exception as e:
                self._print_error(f"Failed to load {region_code} data: {e}")
                if not self.args.force:
                    return False

        return True

    def setup_environment_variables(self) -> bool:
        """Setup environment variables"""
        self._print_section("Setting Up Environment Variables")

        env_file = self.project_root / 'environment_variables.txt'
        if not env_file.exists():
            self._print_warning(f"Environment variables file not found: {env_file}")
            return True

        # Create .env file from environment_variables.txt
        env_target = self.project_root / '.env'

        if env_target.exists() and not self.args.force:
            self._print_warning(f".env file already exists at {env_target}")
            self._print_warning("Use --force to overwrite")
            return True

        try:
            with open(env_file, 'r') as src:
                content = src.read()

            with open(env_target, 'w') as dst:
                dst.write(content)

            self._print_success(f"Created .env file at {env_target}")
            self._print_warning("Remember to update .env with your specific values (DB_URL, API keys, etc.)")
            return True
        except Exception as e:
            self._print_error(f"Failed to create .env file: {e}")
            return False

    def run_tests(self) -> bool:
        """Run test suite to validate setup"""
        if self.args.skip_tests:
            self._print_warning("Skipping tests (--skip-tests flag)")
            return True

        self._print_section("Running Test Suite")

        try:
            venv_python = self._get_venv_python()
            self._print_success("Running unittest discovery...")
            self._run_command([venv_python, '-m', 'unittest', 'discover', '-v'])
            self._print_success("All tests passed")
            return True
        except Exception as e:
            self._print_error(f"Tests failed: {e}")
            if not self.args.force:
                return False
            return True

    def print_activation_instructions(self):
        """Print instructions for activating the virtual environment"""
        self._print_section("Setup Complete!")

        if self.errors:
            self._print_warning(f"\nSetup completed with {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  - {error}")
            print()

        print(f"{Color.BOLD}To activate the virtual environment:{Color.END}\n")

        if self.platform == 'Windows':
            print(f"  PowerShell:")
            print(f"    {Color.GREEN}.venv\\Scripts\\Activate.ps1{Color.END}\n")
            print(f"  Command Prompt:")
            print(f"    {Color.GREEN}.venv\\Scripts\\activate.bat{Color.END}\n")
        else:
            print(f"  Bash/Zsh:")
            print(f"    {Color.GREEN}source .venv/bin/activate{Color.END}\n")

        print(f"{Color.BOLD}To start working with the project:{Color.END}\n")
        print(f"  1. Activate the virtual environment (see above)")
        print(f"  2. Start Jupyter: {Color.GREEN}jupyter notebook{Color.END}")
        print(f"  3. Open: {Color.GREEN}ml_finance_model_v8_2.ipynb{Color.END}")
        print(f"\n  Or run the script: {Color.GREEN}python ml_finance_model_v8_2.py --help{Color.END}\n")

        if not self.args.skip_db:
            print(f"{Color.BOLD}Database connection:{Color.END}")
            print(f"  Host: {self.args.db_host}")
            print(f"  Port: {self.args.db_port}")
            print(f"  Database: {self.args.db_name}")
            print(f"  User: {self.args.db_user}\n")

        print(f"{Color.BOLD}Next steps:{Color.END}")
        print(f"  - Review and update .env with your specific configuration")
        print(f"  - Check the README.md for detailed usage instructions")
        print(f"  - Review IMPROVEMENT_PLAN.md for development roadmap\n")

    def run(self) -> int:
        """Run the complete setup process"""
        print(f"\n{Color.BOLD}{Color.BLUE}Finance ML Analytics Platform - Environment Setup{Color.END}")
        print(f"{Color.BOLD}Platform: {self.platform}{Color.END}")
        print(f"{Color.BOLD}Python: {self.python_cmd}{Color.END}")
        print(f"{Color.BOLD}Project: {self.project_root}{Color.END}\n")

        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Virtual Environment", self.create_virtual_environment),
            ("Package Tools", self.upgrade_pip),
            ("Dependencies", self.install_dependencies),
            ("Database", self.setup_database),
            ("Environment Variables", self.setup_environment_variables),
            ("Tests", self.run_tests),
        ]

        for step_name, step_func in steps:
            if not step_func():
                if not self.args.force:
                    self._print_error(f"\nSetup failed at step: {step_name}")
                    self._print_warning("Use --force to continue despite errors")
                    return 1

        self.print_activation_instructions()
        return 0 if not self.errors else 1


def main():
    parser = argparse.ArgumentParser(
        description='Automated environment setup for Finance ML Analytics Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup with defaults
  python setup_environment.py

  # Skip database setup
  python setup_environment.py --skip-db

  # Recreate virtual environment
  python setup_environment.py --recreate-venv

  # Custom database connection
  python setup_environment.py --db-host localhost --db-port 5432 --db-user postgres

  # Force continue on errors
  python setup_environment.py --force
        """
    )

    # Virtual environment options
    parser.add_argument('--recreate-venv', action='store_true',
                       help='Recreate virtual environment if it exists')

    # Database options
    parser.add_argument('--skip-db', action='store_true',
                       help='Skip database setup')
    parser.add_argument('--skip-data-load', action='store_true',
                       help='Skip loading CSV data into database')
    parser.add_argument('--db-host', default='localhost',
                       help='Database host (default: localhost)')
    parser.add_argument('--db-port', type=int, default=5432,
                       help='Database port (default: 5432)')
    parser.add_argument('--db-user', default='postgres',
                       help='Database user (default: postgres)')
    parser.add_argument('--db-name', default='postgres',
                       help='Database name (default: postgres)')
    parser.add_argument('--db-password', default='',
                       help='Database password (optional, will prompt if needed)')

    # Installation options
    parser.add_argument('--install-db-libs', action='store_true',
                       help='Install optional database libraries (psycopg2-binary, SQLAlchemy)')

    # Test options
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests')

    # General options
    parser.add_argument('--force', action='store_true',
                       help='Continue setup even if errors occur')

    args = parser.parse_args()

    setup = EnvironmentSetup(args)
    return setup.run()


if __name__ == '__main__':
    sys.exit(main())
