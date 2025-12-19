"""
Test version alignment between package and pyproject.toml.

Phase 9.1 TDD: Verify that version numbers are synchronized across:
- finance_ml/__init__.py (__version__)
- pyproject.toml (version)

Run with pytest:
    pytest tests/unit/test_version_alignment.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


class TestVersionAlignment:
    """Test version consistency across package files."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        # Navigate from tests/unit/ to project root
        return Path(__file__).parent.parent.parent

    def test_init_version_exists(self):
        """Verify finance_ml.__version__ is defined."""
        import finance_ml

        assert hasattr(finance_ml, "__version__")
        assert finance_ml.__version__ is not None
        assert isinstance(finance_ml.__version__, str)

    def test_init_version_format(self):
        """Verify __version__ follows semantic versioning pattern."""
        import finance_ml

        version = finance_ml.__version__
        # Accept formats: 0.8.3, v0.8.3, 0.8.3-beta, v9_10
        pattern = r"^v?\d+[._]\d+([._]\d+)?(-\w+)?$"
        assert re.match(pattern, version), f"Version '{version}' doesn't match expected pattern"

    def test_pyproject_version_exists(self, project_root):
        """Verify pyproject.toml has version defined."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text(encoding="utf-8")
        assert "version" in content, "version not found in pyproject.toml"

    def test_version_alignment(self, project_root):
        """Verify __version__ matches pyproject.toml version."""
        import finance_ml

        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")

        # Extract version from pyproject.toml
        # Match: version = "0.8.3"
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        assert match, "Could not extract version from pyproject.toml"

        pyproject_version = match.group(1)
        init_version = finance_ml.__version__

        assert init_version == pyproject_version, (
            f"Version mismatch: __init__.py has '{init_version}', "
            f"pyproject.toml has '{pyproject_version}'"
        )

    def test_version_is_not_placeholder(self):
        """Verify version is not a placeholder value."""
        import finance_ml

        placeholder_values = ["0.0.0", "0.0.1", "0.1.0", "VERSION", "TBD", ""]
        assert (
            finance_ml.__version__ not in placeholder_values
        ), f"Version '{finance_ml.__version__}' appears to be a placeholder"


class TestVersionExport:
    """Test that version is properly exported."""

    def test_version_in_all(self):
        """Verify __version__ is accessible from package."""
        from finance_ml import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_version_accessible_as_attribute(self):
        """Verify __version__ accessible as module attribute."""
        import finance_ml

        version = getattr(finance_ml, "__version__", None)
        assert version is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
