import json

# Load the notebook
with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the problematic code around line 2686
line_count = 0
target_line = 2686
context_lines = 20

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        num_lines = len(source_text.split("\n"))

        # Check if target line is in this cell
        if line_count <= target_line <= line_count + num_lines:
            print(f"\n{'='*80}")
            print(f"Found problematic code in Cell {i}")
            print(f"Cell line range: {line_count} - {line_count + num_lines}")
            print(
                f"Target line {target_line} is at position {target_line - line_count} in this cell"
            )
            print(f"{'='*80}\n")

            # Show context around the error
            lines = source_text.split("\n")
            target_pos = target_line - line_count
            start = max(0, target_pos - context_lines)
            end = min(len(lines), target_pos + context_lines)

            print("Context (line numbers relative to cell start):")
            for j in range(start, end):
                marker = " >>> " if j == target_pos else "     "
                # Handle encoding issues by replacing problematic characters
                line_content = lines[j].encode("ascii", "replace").decode("ascii")
                print(f"{marker}Line {j}: {repr(line_content)}")

            print(f"\n{'='*80}")
            print("Analysis:")
            if target_pos < len(lines):
                problem_line = lines[target_pos]
                print(f"Problem line content: {repr(problem_line)}")
                print(f"Leading spaces: {len(problem_line) - len(problem_line.lstrip())}")

                # Check previous non-empty line
                for k in range(target_pos - 1, -1, -1):
                    if lines[k].strip():
                        prev_line = lines[k]
                        print(f"Previous non-empty line: {repr(prev_line)}")
                        print(
                            f"Previous line leading spaces: {len(prev_line) - len(prev_line.lstrip())}"
                        )
                        break

            break

        line_count += num_lines

print("\nDone!")
