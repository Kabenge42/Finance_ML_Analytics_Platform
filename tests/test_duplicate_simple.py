#!/usr/bin/env python
"""
Simple test to verify no duplicate function definitions exist
"""

import re
from pathlib import Path


def main():
    print("=" * 80)
    print("SIMPLE DUPLICATE FUNCTION TEST")
    print("=" * 80)

    file_path = Path(__file__).parent / "finance_ml" / "advanced_models.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all function definitions
    all_functions = re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content, re.MULTILINE)

    # Find duplicates
    from collections import Counter

    counts = Counter(all_functions)
    duplicates = {name: count for name, count in counts.items() if count > 1}

    print(f"\nTotal functions defined: {len(all_functions)}")
    print(f"Unique functions: {len(set(all_functions))}")

    if duplicates:
        print(f"\n[FAIL] Found {len(duplicates)} duplicate function(s):")
        for name, count in duplicates.items():
            print(f"  - {name}: defined {count} times")
            # Find line numbers
            pattern = rf"^def {name}\s*\("
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            for i, match in enumerate(matches, 1):
                line_num = content[: match.start()].count("\n") + 1
                print(f"    Definition {i} at line {line_num}")
        return 1
    else:
        print("\n[SUCCESS] No duplicate function definitions found!")

        # Specifically check validate_training_data
        pattern = r"^def validate_training_data\s*\("
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        print(f"\nvalidate_training_data: {len(matches)} definition(s)")
        for i, match in enumerate(matches, 1):
            line_num = content[: match.start()].count("\n") + 1
            print(f"  Definition at line {line_num}")

        if len(matches) == 1:
            print("\n[SUCCESS] validate_training_data is defined exactly once (as expected)")
            return 0
        else:
            print(f"\n[FAIL] Expected 1 definition of validate_training_data, found {len(matches)}")
            return 1


if __name__ == "__main__":
    exit(main())
