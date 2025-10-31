"""Verify that all required package dependencies are installed and compatible.

This module checks for missing or incompatible package versions against
the requirements specified in the project.
"""

import sys

import pkg_resources

required = {
    'numpy': '>=1.24.0,<2.1.0',
    'pandas': '>=2.0.0,<3.0.0',
    'imbalanced-learn': '>=0.11.0,<1.0.0',
    'lightgbm': '>=4.0.0,<5.0.0',
    'catboost': '>=1.2.0,<2.0.0',
    # Add all required packages
}

missing = []
incompatible = []

for package, version in required.items():
    try:
        pkg = pkg_resources.get_distribution(package)
        if not pkg_resources.require(f"{package}{version}"):
            incompatible.append((package, pkg.version, version))
    except pkg_resources.DistributionNotFound:
        missing.append((package, version))

if missing or incompatible:
    print("❌ Dependency issues found:")
    for pkg, ver in missing:
        print(f"  Missing: {pkg} {ver}")
    for pkg, installed, required in incompatible:
        print(f"  Incompatible: {pkg} (installed: {installed}, required: {required})")
    sys.exit(1)
else:
    print("✅ All dependencies satisfied")
