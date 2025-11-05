"""Fix phase ordering in restructured notebook."""

import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def main():
    path = Path("ml_finance_model_main_backup.ipynb")

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # Find all phase sections
    phases = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            if source.startswith("## Phase 9."):
                # Extract phase number
                parts = source.split()[2]  # "Phase 9.X"
                phase_num = parts.strip()
                phases[phase_num] = i

    print("\nCurrent Phase Positions:")
    for phase, idx in sorted(phases.items()):
        print(f"  {phase}: Cell {idx}")

    # Define correct ordering
    correct_order = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.5.1", "9.6", "9.6.1", "9.7", "9.8"]

    print("\nCorrect Order Should Be:")
    for i, phase in enumerate(correct_order):
        print(f"  {i+1}. Phase {phase}")

    # Check if reordering is needed
    current_order = [p for p in correct_order if p in phases]
    current_indices = [phases[p] for p in current_order]

    if current_indices == sorted(current_indices):
        print("\nPhases already in correct order!")
        return

    print("\nPhases need reordering...")

    # Extract phase sections
    phase_sections = {}
    for phase in phases:
        start_idx = phases[phase]

        # Find end of this phase (next phase header or end of notebook)
        end_idx = len(nb["cells"])
        for i in range(start_idx + 1, len(nb["cells"])):
            if nb["cells"][i]["cell_type"] == "markdown" and nb["cells"][i]["source"]:
                source = "".join(nb["cells"][i]["source"])
                if source.startswith("## Phase 9."):
                    end_idx = i
                    break

        phase_sections[phase] = nb["cells"][start_idx:end_idx]
        print(f"  Phase {phase}: Cells {start_idx}-{end_idx-1} ({end_idx-start_idx} cells)")

    # Find where Phase 9 sections start
    phase9_start = min(phases.values())

    # Remove all Phase 9 sections
    # Work backwards to preserve indices
    for phase in sorted(phases.items(), key=lambda x: x[1], reverse=True):
        phase_num, start_idx = phase
        end_idx = start_idx + len(phase_sections[phase_num])
        del nb["cells"][start_idx:end_idx]

    # Insert phases in correct order
    insert_pos = phase9_start
    for phase in correct_order:
        if phase in phase_sections:
            cells = phase_sections[phase]
            for cell in cells:
                nb["cells"].insert(insert_pos, cell)
                insert_pos += 1
            print(f"Inserted Phase {phase} at position {insert_pos - len(cells)}")

    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\n✓ Notebook saved with corrected phase ordering")


if __name__ == "__main__":
    main()
