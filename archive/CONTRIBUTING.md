# Contributing to Finance ML Analytics Platform

Thank you for your interest in contributing to the Finance ML Analytics Platform! This document provides comprehensive
guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect everyone to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discriminatory comments, or personal attacks
- Trolling, insulting/derogatory comments
- Publishing others' private information without permission
- Other conduct deemed inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- PostgreSQL 15+ (for database features)
- Git for version control
- Familiarity with machine learning and financial analytics concepts

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
    - Visit https://github.com/Kabenge42/Finance_ML_Analytics_Platform
    - Click the "Fork" button in the top right

2. **Clone your fork locally**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Finance_ML_Analytics_Platform.git
   cd Finance_ML_Analytics_Platform
   ```

3. **Add the upstream repository**
   ```bash
   git remote add upstream https://github.com/Kabenge42/Finance_ML_Analytics_Platform.git
   ```

4. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS/Linux (bash):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. **Upgrade packaging tools**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

6. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

7. **Verify installation**
   ```bash
   python -m unittest discover -s tests -v
   ```

## Development Workflow

### Branch Naming Conventions

Use descriptive branch names that reflect the work being done:

- **Feature branches**: `feature/description-of-feature`
- **Bug fixes**: `fix/description-of-bug`
- **Documentation**: `docs/description-of-changes`
- **Refactoring**: `refactor/description-of-changes`
- **Tests**: `test/description-of-test`

Examples:

```bash
git checkout -b feature/add-momentum-indicators
git checkout -b fix/database-connection-timeout
git checkout -b docs/update-installation-guide
```

### Making Changes

1. **Create a new branch** from `main`
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style guidelines

3. **Write or update tests** for your changes

4. **Run tests locally** to ensure everything passes
   ```bash
   python -m unittest discover -s tests -v
   ```

5. **Run code quality checks**
   ```bash
   black finance_ml tests
   isort finance_ml tests
   flake8 finance_ml
   mypy finance_ml --ignore-missing-imports
   ```

6. **Commit your changes** with descriptive commit messages
   ```bash
   git add .
   git commit -m "Add momentum indicators to feature engineering"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

Write clear, concise commit messages that explain what and why:

**Good commit messages:**

```
Add basic ratio and volatility features

- Enhance engineer_basic_ratios() and engineer_volatility_features() in features.py
- Add unit tests for feature engineering functions
- Update documentation with usage examples
```

**Bad commit messages:**

```
Fixed stuff
Update code
Changes
```

**Format:**

- Use the imperative mood ("Add feature" not "Added feature")
- First line is a brief summary (50 characters or less)
- Blank line separating summary from body
- Detailed explanation in the body (if needed)
- Reference issue numbers when applicable (#123)

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Required tests**: All new functions and classes must have unit tests
- **Test types**: Unit tests, integration tests where applicable

### Writing Tests

1. **Location**: Place tests in the `tests/` directory
2. **Naming**: Test files should be named `test_*.py`
3. **Structure**: Use Python's `unittest` framework

Example test structure:

```python
import unittest
import pandas as pd
from finance_ml.data import validate_schema


class TestDataValidation(unittest.TestCase):
    def test_validate_schema_missing_columns_raises(self):
        df = pd.DataFrame({"ticker": ["AAPL"], "last_price": [150.0]})
        with self.assertRaises(ValueError):
            validate_schema(df, require_target=False)

    def test_validate_schema_passes_with_required_columns(self):
        df = pd.DataFrame({"ticker": ["AAPL"], "sector": ["Tech"], "last_price": [150.0]})
        # Should not raise
        self.assertIsNone(validate_schema(df, require_target=False))


if __name__ == '__main__':
    unittest.main()
```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_features

# Run specific test class
python -m unittest tests.test_features.TestMomentumIndicators

# Run specific test method
python -m unittest tests.test_features.TestMomentumIndicators.test_calculate_rsi_returns_valid_range

# Run with pytest (if installed)
pytest tests/ -v --cov=finance_ml
```

## Code Style Guidelines

### Python Style Guide

We follow **PEP 8** conventions with some modifications:

- **Line length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized using `isort`

### Code Formatting Tools

We use automated tools to enforce consistent style:

1. **Black** - Code formatter
   ```bash
   black finance_ml tests
   ```

2. **isort** - Import sorter
   ```bash
   isort finance_ml tests
   ```

3. **flake8** - Linter
   ```bash
   flake8 finance_ml
   ```

4. **mypy** - Type checker
   ```bash
   mypy finance_ml --ignore-missing-imports
   ```

### Docstring Format

Use **Google-style docstrings**:

```python
def calculate_moving_average(prices: list[float], window: int = 3) -> float:
    """Calculate a simple moving average for a price series.

    This is a general example to demonstrate Google-style docstrings.

    Args:
        prices: List of historical prices in chronological order
        window: Window length for the moving average (default: 3)

    Returns:
        The moving average as a float.

    Raises:
        ValueError: If prices list is empty or window <= 0

    Example:
        >>> prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83]
        >>> ma = calculate_moving_average(prices, window=3)
        >>> print(f"MA: {ma:.2f}")
        MA: 44.08
    """
    if not prices or window <= 0:
        raise ValueError("Invalid inputs")

    return sum(prices[-window:]) / min(window, len(prices))
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Optional, Union
import pandas as pd


def load_from_csv(
        path: str,
        limit: Optional[int] = None,
        region: Optional[str] = None
        ) -> pd.DataFrame:
    """Load data from CSV file."""
    pass
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest changes from upstream
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests** and ensure they pass
   ```bash
   python -m unittest discover -s tests -v
   ```

3. **Run code quality checks**
   ```bash
   black --check finance_ml tests
   isort --check-only finance_ml tests
   flake8 finance_ml
   mypy finance_ml --ignore-missing-imports
   ```

4. **Update documentation** if needed

5. **Update CHANGELOG.md** if your changes are user-facing

### Submitting a Pull Request

1. **Push your changes** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a pull request** on GitHub
    - Go to your fork on GitHub
    - Click "New Pull Request"
    - Select your branch
    - Fill out the PR template

### Pull Request Template

```markdown
## Description

Brief description of what this PR does and why.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Related Issues

Closes #123
Related to #456

## Changes Made

- Change 1
- Change 2
- Change 3

## Testing

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added that prove fix/feature works
- [ ] Dependent changes merged and published
```

### Review Process

1. **Automated checks** will run on your PR
2. **Code review** by maintainers (usually within 1-3 business days)
3. **Address feedback** by pushing new commits to your branch
4. **Approval** required from at least one maintainer
5. **Merge** by maintainers once approved

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** to ensure it's not a known limitation
3. **Test with latest version** to see if issue still exists

### Bug Report Template

```markdown
**Description**
Clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Run '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**

- OS: [e.g., Windows 11, macOS 13, Ubuntu 22.04]
- Python version: [e.g., 3.10.9]
- Package version: [e.g., 0.3.0]
- Database: [e.g., PostgreSQL 15.2]

**Additional Context**

- Error messages
- Stack traces
- Screenshots (if applicable)
- Sample data (if applicable)
```

### Feature Request Template

```markdown
**Problem Statement**
Describe the problem or limitation you're experiencing.

**Proposed Solution**
Describe your proposed solution or feature.

**Alternatives Considered**
Describe any alternative solutions you've considered.

**Use Case**
Provide a specific use case for this feature.

**Additional Context**
Any other context, mockups, or examples.
```

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - This will not be worked on
- `duplicate` - Duplicate of existing issue

## Documentation

### When to Update Documentation

Update documentation when you:

- Add new features or functions
- Change existing functionality
- Fix bugs that affect documented behavior
- Improve code clarity or organization

### Documentation Locations

1. **README.md** - Project overview, setup, usage
2. **CHANGELOG.md** - Version history and changes
3. **Docstrings** - In-code documentation for functions/classes
4. **Type hints** - Function signatures
5. **Comments** - Complex logic explanations

### Writing Good Documentation

- Use clear, concise language
- Include practical examples
- Explain the "why" not just the "what"
- Keep examples up-to-date
- Use proper formatting and structure

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code review and collaboration

### Contributing Beyond Code

You can contribute by:

- Reporting bugs
- Suggesting features
- Improving documentation
- Reviewing pull requests
- Helping others in discussions
- Sharing the project

### Recognition

Contributors are recognized in:

- GitHub contributors list
- CHANGELOG.md for significant contributions
- Project documentation

## Questions?

If you have questions about contributing, please:

1. Check this guide thoroughly
2. Search existing issues and discussions
3. Open a new discussion on GitHub
4. Tag maintainers if needed

Thank you for contributing to Finance ML Analytics Platform!
