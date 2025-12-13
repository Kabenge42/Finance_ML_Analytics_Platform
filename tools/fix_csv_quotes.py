#!/usr/bin/env python3
"""
Fix CSV escaping issue in screening_apac.csv for KazTransOil JSC record.

The Description field contains nested quotes that are not properly escaped:
  "... Joint Stock Company "National Company "KazMunayGas"."

Should be:
  "... Joint Stock Company ""National Company ""KazMunayGas""."""

import re
from pathlib import Path


def fix_kaztransoil_record(csv_path: Path, output_path: Path):
    """
    Fix the CSV escaping issue in the KazTransOil JSC record.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the problematic line with KazTransOil
    # The issue is in the Description field which contains:
    # Joint Stock Company "National Company "KazMunayGas".

    # Pattern to find and fix nested quotes in the description
    # We need to escape internal quotes by doubling them

    # Search for the specific problematic text
    old_pattern = 'Joint Stock Company "National Company "KazMunayGas".'
    new_pattern = 'Joint Stock Company ""National Company ""KazMunayGas"".'

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print(f"[OK] Fixed KazTransOil JSC description field")
        fixed = True
    else:
        print(f"[WARNING] Could not find the expected pattern")
        # Try to detect if it's already fixed
        if new_pattern in content:
            print(f"[INFO] Pattern appears to already be correctly escaped")
            fixed = False
        else:
            print(f"[ERROR] Neither old nor new pattern found")
            fixed = False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return fixed


def main():
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "screening_apac.csv"
    output_path = base_dir / "data" / "screening_apac.csv"

    print("Fixing CSV escaping issues in screening_apac.csv...")
    fixed = fix_kaztransoil_record(csv_path, output_path)

    if fixed:
        print(f"\n[SUCCESS] Fixed CSV saved to: {output_path}")
        print("\nTo apply the fix:")
        print(
            f"  1. Backup original: copy data\\screening_apac.csv data\\screening_apac_backup.csv"
        )
        print(f"  2. Replace original: copy data\\screening_apac.csv data\\screening_apac.csv")
    else:
        print(f"\nNo changes needed or fix could not be applied")


if __name__ == "__main__":
    main()
