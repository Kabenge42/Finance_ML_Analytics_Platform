#!/usr/bin/env python3
"""Auto-generated cleanup script for duplicate virtual environments."""

import shutil
from pathlib import Path

# Directories to remove:
# - C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\.venv1
# - C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\anaconda_projects


def cleanup():
    paths_to_remove = [
        Path(r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\.venv1"),
        Path(r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\anaconda_projects"),
    ]

    for path in paths_to_remove:
        if path.exists():
            print(f"Removing {path}...")
            shutil.rmtree(path)
            print(f"[+] Removed {path}")
        else:
            print(f"[!] Path not found: {path}")


if __name__ == "__main__":
    response = input("This will remove the directories listed above. Continue? (yes/no): ")
    if response.lower() == "yes":
        cleanup()
        print("Cleanup complete!")
    else:
        print("Cleanup cancelled.")
