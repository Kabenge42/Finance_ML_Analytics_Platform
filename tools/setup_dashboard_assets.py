
import shutil
from pathlib import Path
import os

def sync_assets():
    """Copy all HTML artifacts to dashboard assets folder."""
    # Define project root relative to this script
    project_root = Path(__file__).parent.parent
    outputs = project_root / "outputs"
    assets_dir = project_root / "finance_ml" / "dashboards" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Syncing assets from {outputs} to {assets_dir}...")
    
    count = 0
    for html_file in outputs.rglob("*.html"):
        dest = assets_dir / html_file.name
        # Only copy if source is newer than destination or destination doesn't exist
        if not dest.exists() or html_file.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(html_file, dest)
            print(f"✓ Copied {html_file.name}")
            count += 1
        else:
            # print(f"  Skipped {html_file.name} (up to date)")
            pass
            
    print(f"Sync complete. Copied/Updated {count} files.")

if __name__ == "__main__":
    sync_assets()
