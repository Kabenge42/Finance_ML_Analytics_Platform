import json

# Load notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Phase 9.3 section
in_phase93 = False
phase93_cells = []
cell_index = 0

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))

    # Start of Phase 9.3 (markdown header with ##)
    if "## Phase 9.3" in source or "##Phase 9.3" in source:
        in_phase93 = True
        cell_index = 0
        print(f"Found Phase 9.3 start at cell {i}")

    # End of Phase 9.3 (when we hit Phase 9.4)
    if in_phase93 and ("## Phase 9.4" in source or "##Phase 9.4" in source):
        print(f"Found Phase 9.4 (end of 9.3) at cell {i}")
        break

    if in_phase93:
        phase93_cells.append((i, cell_index, cell))
        cell_index += 1

print(f"Found {len(phase93_cells)} cells in Phase 9.3 section")
print("\n" + "=" * 80)

for global_idx, local_idx, cell in phase93_cells[:10]:  # Show first 10
    cell_type = cell["cell_type"]
    source = "".join(cell.get("source", []))
    preview = source[:150].replace("\n", " ")
    print(f"\nCell {local_idx} (global {global_idx}) - {cell_type}:")
    print(f"  {preview}...")

print(f"\n... and {max(0, len(phase93_cells) - 10)} more cells")
