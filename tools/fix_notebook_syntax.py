import json
import sys

# Read the notebook
try:
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

# Find the cell with the syntax error
cell_found = False
cell_index = -1

for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "Category Performance Heatmaps" in source and "category_mapping.items()" in source:
            cell_found = True
            cell_index = i
            print(f"Found cell at index {i}")

            # The corrected code with proper indentation
            corrected_code = """# Visualization 1: Category Heatmaps (Sector × Category)
print("\\n📊 Category Performance Heatmaps:")

# Compute category scores by averaging z-scores of metrics within each category
from scipy.stats import zscore

category_sector_scores = {}

for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in all_stocks_scaled.columns]
    
    if len(available_in_category) == 0:
        print(f"  ⚠️ Skipping {category_name}: No available metrics")
        continue
    
    # Compute z-scores for available metrics and average by sector
    category_data = all_stocks_scaled[available_in_category + ['sector']].copy()
    
    # Convert to numeric and compute z-scores
    for col in available_in_category:
        category_data[col] = pd.to_numeric(category_data[col], errors='coerce')
    
    # Compute z-scores (handle NaNs)
    z_scored_data = category_data[available_in_category].apply(lambda x: zscore(x, nan_policy='omit'))
    category_data['category_score'] = z_scored_data.mean(axis=1)
    
    # Aggregate by sector
    sector_scores = category_data.groupby('sector')['category_score'].mean().sort_values(ascending=False)
    category_sector_scores[category_name] = sector_scores

# Create heatmap matrix
if category_sector_scores:
    heatmap_df = pd.DataFrame(category_sector_scores).T
    
    # Create interactive heatmap
    fig_category_heatmap = px.imshow(
        heatmap_df,
        labels=dict(x="Sector", y="Category", color="Avg Z-Score"),
        title="Sector Performance Across 11 Feature Categories (Phase 9.3)",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    fig_category_heatmap.update_layout(
        height=600,
        xaxis_tickangle=-45,
        font=dict(size=10)
    )
    
    fig_category_heatmap.show()
    output_path = eda_output_dir / "phase93_category_sector_heatmap.html"
    fig_category_heatmap.write_html(output_path)
    
    print(f"\\n✓ Category heatmap visualization complete")
    print(f"  Categories visualized: {len(category_sector_scores)}")
    print(f"  Sectors analyzed: {len(heatmap_df.columns)}")
    print(f"  Output: {output_path}")
    
    # Display top performing sector per category
    print(f"\\n  🏆 Top Performing Sectors by Category:")
    for category, scores in list(category_sector_scores.items())[:5]:
        top_sector = scores.idxmax()
        top_score = scores.max()
        print(f"    {category}: {top_sector} (z-score: {top_score:.2f})")
else:
    print("  ⚠️ No category data available for visualization")
"""

            # Update the cell with corrected code
            notebook["cells"][i]["source"] = corrected_code.split("\n")
            break

if not cell_found:
    print("Cell with 'Category Performance Heatmaps' not found!")
    sys.exit(1)

# Write the corrected notebook
try:
    with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"\n✓ Successfully fixed syntax errors in cell {cell_index}")
    print("\nFixed issues:")
    print("  1. Indented 'available_in_category' assignment inside the for loop")
    print("  2. Properly indented the if/continue block")
    print("  3. Indented all subsequent code blocks within the for loop")
    print("  4. Fixed indentation for the remaining visualization code")
except Exception as e:
    print(f"Error writing notebook: {e}")
    sys.exit(1)
