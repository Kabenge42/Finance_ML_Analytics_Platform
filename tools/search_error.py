import json

# Load the notebook
with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Search for the specific text mentioned in the error
search_text = "Model comparison failed"
line_count = 0

print(f"Searching for: '{search_text}'\n")

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        # Check if search text is in this cell
        if search_text in source_text:
            lines = source_text.split("\n")
            print(f"{'='*80}")
            print(f"Found in Cell {i}")
            print(f"{'='*80}\n")

            # Find the specific line and show context
            for j, line in enumerate(lines):
                if search_text in line:
                    start = max(0, j - 10)
                    end = min(len(lines), j + 10)

                    print(f"Context around line {j} in cell {i}:")
                    print(f"Global line number approximately: {line_count + j}\n")

                    for k in range(start, end):
                        marker = " >>> " if k == j else "     "
                        # Handle encoding issues
                        line_content = lines[k].encode("ascii", "replace").decode("ascii")
                        print(f"{marker}Line {k}: {repr(line_content)}")

                    print(
                        f"\nProblem line indentation: {len(lines[j]) - len(lines[j].lstrip())} spaces"
                    )

                    # Check previous except line
                    for m in range(j - 1, -1, -1):
                        if "except" in lines[m]:
                            print(
                                f"Previous 'except' line indentation: {len(lines[m]) - len(lines[m].lstrip())} spaces"
                            )
                            print(f"Previous 'except' line: {repr(lines[m])}")
                            break

                    print()

        num_lines = len(source_text.split("\n"))
        line_count += num_lines

print("\nDone!")
