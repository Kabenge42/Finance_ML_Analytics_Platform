"""
Integrate NOTEBOOK_INTEGRATION_CELLS.md into ml_finance_model_main_backup.ipynb

This script:
1. Reads the existing notebook
2. Parses the 5 cells from NOTEBOOK_INTEGRATION_CELLS.md
3. Inserts them after Phase 9.5 (around cell 140)
4. Saves the updated notebook with proper JSON structure
"""

import json
import re
from pathlib import Path


def parse_integration_cells(md_path: Path):
    """Parse NOTEBOOK_INTEGRATION_CELLS.md and extract cell definitions.

    Returns:
        List of dicts with 'cell_type' (markdown/code) and 'source' (list of lines)
    """
    content = md_path.read_text(encoding="utf-8")
    cells = []

    # Pattern to identify cells: ## Cell N: Title (Type)
    cell_pattern = r"## (Cell \d+: .+?) \((\w+)\)"

    # Split by cell headers
    parts = re.split(cell_pattern, content)

    # parts[0] is the preamble, then groups of (title, type, content)
    for i in range(1, len(parts), 3):
        if i + 2 > len(parts):
            break

        title = parts[i].strip()
        cell_type_raw = parts[i + 1].strip().lower()
        cell_content = parts[i + 2].strip()

        # Map type
        cell_type = "markdown" if cell_type_raw in ["markdown", "md"] else "code"

        # Extract content from code fence if present
        if cell_type == "code":
            # Look for ```python ... ``` or ```markdown ... ```
            code_match = re.search(
                r"```(?:python|markdown)?\s*\n(.*?)\n```", cell_content, re.DOTALL
            )
            if code_match:
                source = code_match.group(1)
            else:
                source = cell_content
        else:
            # For markdown, look for ```markdown ... ```
            md_match = re.search(r"```markdown\s*\n(.*?)\n```", cell_content, re.DOTALL)
            if md_match:
                source = md_match.group(1)
            else:
                source = cell_content

        # Convert to list of lines (Jupyter format)
        source_lines = source.split("\n")
        # Add newline to each line except the last
        source_lines = (
            [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
        )

        cells.append(
            {
                "cell_type": cell_type,
                "source": source_lines,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            }
        )

        print(f"✓ Parsed {title} ({cell_type}, {len(source_lines)} lines)")

    return cells


def find_insertion_point(notebook, search_text="Phase 9.5"):
    """Find the cell index after Phase 9.5 section.

    Returns:
        Index where new cells should be inserted
    """
    for i, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if search_text in source and "Phase 9.5" in source:
            # Check if this is the main Phase 9.5 header (not 9.5.1, 9.5.2, etc.)
            if "Phase 9.5 —" in source or "Phase 9.5:" in source:
                if "Phase 9.5.1" not in source:
                    print(f"✓ Found Phase 9.5 at cell {i}")
                    # Insert after this cell and its code implementation
                    # Look ahead to find the end of Phase 9.5 content
                    for j in range(i + 1, min(i + 10, len(notebook["cells"]))):
                        next_source = "".join(notebook["cells"][j].get("source", []))
                        # Stop if we hit Phase 9.6 or another major phase
                        if "Phase 9.6" in next_source or "Phase 10" in next_source:
                            print(f"✓ Insertion point: cell {j} (before Phase 9.6)")
                            return j
                        # If we find a section marker or significant markdown, insert here
                        if (
                            notebook["cells"][j]["cell_type"] == "markdown"
                            and len(next_source) > 50
                        ):
                            if "##" in next_source[:10]:  # Looks like a header
                                print(f"✓ Insertion point: cell {j} (before next section)")
                                return j
                    # Default: insert 2 cells after Phase 9.5 header
                    return i + 2

    # Fallback: insert at cell 141 if Phase 9.5 not found
    print("⚠ Phase 9.5 not found, using default position (cell 141)")
    return 141


def integrate_cells(notebook_path: Path, cells_md_path: Path, output_path: Path):
    """Main integration function."""

    print(f"\n{'='*80}")
    print("NOTEBOOK INTEGRATION: Phase 9.5.1 and 9.6.1")
    print(f"{'='*80}\n")

    # Load notebook
    print(f"📖 Loading notebook: {notebook_path.name}")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    original_cell_count = len(notebook["cells"])
    print(f"   Original cell count: {original_cell_count}")

    # Parse integration cells
    print(f"\n📋 Parsing integration cells: {cells_md_path.name}")
    new_cells = parse_integration_cells(cells_md_path)
    print(f"   Extracted {len(new_cells)} cells")

    # Find insertion point
    print(f"\n🔍 Finding insertion point...")
    insertion_index = find_insertion_point(notebook)

    # Insert cells
    print(f"\n✏️  Inserting {len(new_cells)} cells at position {insertion_index}...")
    for i, cell in enumerate(new_cells):
        notebook["cells"].insert(insertion_index + i, cell)
        print(f"   ✓ Inserted cell {insertion_index + i}: {cell['cell_type']}")

    # Save updated notebook
    print(f"\n💾 Saving updated notebook: {output_path.name}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    new_cell_count = len(notebook["cells"])
    print(f"   New cell count: {new_cell_count} (+{new_cell_count - original_cell_count})")

    print(f"\n{'='*80}")
    print("✅ INTEGRATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"📝 Summary:")
    print(f"   - Original notebook: {original_cell_count} cells")
    print(f"   - Added: {len(new_cells)} cells (Phase 9.5.1 + 9.6.1)")
    print(f"   - Updated notebook: {new_cell_count} cells")
    print(f"   - Insertion point: cell {insertion_index}")
    print(f"\n✓ Notebook saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent

    notebook_path = project_root / "ml_finance_model_main_backup.ipynb"
    cells_md_path = project_root / "docs" / "NOTEBOOK_INTEGRATION_CELLS.md"
    output_path = notebook_path  # Overwrite in place

    # Safety check: create backup first
    backup_path = project_root / "ml_finance_model_main_backup.ipynb.backup_before_integration"
    if not backup_path.exists():
        print(f"🔒 Creating backup: {backup_path.name}")
        import shutil

        shutil.copy2(notebook_path, backup_path)

    integrate_cells(notebook_path, cells_md_path, output_path)

    print(f"\n💡 Next steps:")
    print(f"   1. Open notebook in Jupyter: ml_finance_model_main_backup.ipynb")
    print(f"   2. Navigate to Phase 9.5.1 (new section)")
    print(f"   3. Run cells to validate integration")
    print(f"   4. Check outputs in outputs/models/ directory")
