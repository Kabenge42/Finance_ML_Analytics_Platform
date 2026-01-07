import shutil
from pathlib import Path
import os

def sync_assets():
    """DEPRECATED: HTML artifacts are now served dynamically via /app_assets/ route.

    This script previously copied all HTML artifacts to the dashboard assets folder.
    It is now kept for legacy compatibility but is no longer required for
    the Equities Dashboard.
    """
    # Define project root relative to this script
    project_root = Path(__file__).parent.parent
    outputs = project_root / "outputs"
    assets_dir = project_root / "finance_ml" / "dashboards" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(
        "NOTE: HTML artifacts are now served dynamically from 'outputs/' via '/app_assets/' route."
    )
    print("Manual syncing to 'assets/' is no longer required for recent dashboard versions.")

    # Optional: Keep syncing if someone still uses legacy /assets/ paths in other apps
    # but we've updated dash_app.py to use dynamic paths.

if __name__ == "__main__":
    sync_assets()
