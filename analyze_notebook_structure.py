#!/usr/bin/env python3
"""
Analyze ml_finance_model_main.ipynb structure to identify phase organization and gaps.
"""
import json
from pathlib import Path


def analyze_notebook():
    nb_path = Path("ml_finance_model_main.ipynb")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"Total cells: {len(cells)}")
    print(f"\n{'='*80}")
    print("NOTEBOOK STRUCTURE ANALYSIS")
    print(f"{'='*80}\n")

    # Track phases
    current_phase = None
    phase_sections = []

    for i, cell in enumerate(cells):
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))

        # Look for phase markers
        if "##" in source or "###" in source or "Phase 9." in source:
            # Extract header
            lines = source.split("\n")
            for line in lines:
                if line.strip().startswith("#"):
                    header = line.strip()
                    # Check for phase markers
                    if "Phase 9." in header or "PHASE 9." in header.upper():
                        if "Phase 9.1" in header or "PHASE 9.1" in header.upper():
                            current_phase = "9.1"
                        elif "Phase 9.2" in header or "PHASE 9.2" in header.upper():
                            current_phase = "9.2"
                        elif "Phase 9.3" in header or "PHASE 9.3" in header.upper():
                            current_phase = "9.3"
                        elif "Phase 9.4" in header or "PHASE 9.4" in header.upper():
                            current_phase = "9.4"
                        elif "Phase 9.5" in header or "PHASE 9.5" in header.upper():
                            current_phase = "9.5"
                        elif "Phase 9.6" in header or "PHASE 9.6" in header.upper():
                            current_phase = "9.6"
                        elif "Phase 9.7" in header or "PHASE 9.7" in header.upper():
                            current_phase = "9.7"
                        elif "Phase 9.8" in header or "PHASE 9.8" in header.upper():
                            current_phase = "9.8"

                        phase_sections.append({"cell": i, "phase": current_phase, "header": header})
                        print(f"Cell {i:4d} | Phase {current_phase} | {header[:70]}")
                    elif line.strip().startswith("##"):
                        # Major section marker
                        print(f"Cell {i:4d} | Section   | {header[:70]}")

    # Summary
    print(f"\n{'='*80}")
    print("PHASE SUMMARY")
    print(f"{'='*80}\n")

    phases_found = set(s["phase"] for s in phase_sections if s["phase"])
    phases_expected = ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"]

    print(f"Phases found: {sorted(phases_found)}")
    print(f"Phases expected: {phases_expected}")

    missing_phases = set(phases_expected) - phases_found
    if missing_phases:
        print(f"\n⚠️  Missing phases: {sorted(missing_phases)}")
    else:
        print(f"\n✅ All 8 phases are present")

    # Count sections per phase
    print(f"\n{'='*80}")
    print("SECTIONS PER PHASE")
    print(f"{'='*80}\n")

    for phase in phases_expected:
        count = sum(1 for s in phase_sections if s["phase"] == phase)
        status = "✓" if count > 0 else "✗"
        print(f"{status} Phase {phase}: {count} section(s)")


if __name__ == "__main__":
    analyze_notebook()
