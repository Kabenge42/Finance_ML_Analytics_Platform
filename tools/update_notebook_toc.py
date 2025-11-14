#!/usr/bin/env python3
"""
Update Quick Reference Navigation in ml_finance_model_main.ipynb to use Phase 9.1-9.8 nomenclature.
"""
import json
from pathlib import Path


def update_toc(notebook_path):
    """Update the Quick Reference Navigation section with Phase 9.1-9.8 links."""
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook["cells"]

    # Find cell containing Quick Reference Navigation
    nav_cell_idx = None
    for i, cell in enumerate(cells[:10]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            if "Quick Reference Navigation" in source or "Workflow Overview" in source:
                nav_cell_idx = i
                break

    if nav_cell_idx is None:
        print("Error: Could not find navigation cell in first 10 cells")
        return False

    print(f"Found navigation in cell {nav_cell_idx}")

    # Get current content
    cell = cells[nav_cell_idx]
    source = "".join(cell.get("source", []))

    # Replace old section references with phase references
    replacements = [
        (
            "- [Section 2](#2-loading-and-preprocessing): Data Loading and Preprocessing (6-step imputation)",
            "- [Phase 9.1](#phase-91-loading-and-preprocessing-with-6-step-imputation-strategy): Loading and Preprocessing with 6-Step Imputation",
        ),
        (
            "- [Section 3](#3-exploratory-data-analysis): Exploratory Data Analysis (EDA)",
            "- [Phase 9.2](#phase-92-enhanced-exploratory-data-analysis-with-statistical-testing): Enhanced Exploratory Data Analysis",
        ),
        (
            "- [Section 4](#4-feature-engineering): Advanced Feature Engineering",
            "- [Phase 9.3](#phase-93-advanced-feature-engineering-with-sector-specific-optimizations): Advanced Feature Engineering",
        ),
        (
            "- [Section 5](#5-classification): Multi-Class Event Classification",
            "- [Phase 9.4](#phase-94-multi-class-event-classification): Multi-Class Event Classification",
        ),
        (
            "- [Section 6](#6-regression): Sector-Optimized Regression Models",
            "- [Phase 9.5](#phase-95-sector-optimized-regression-models-with-quantile-predictions): Sector-Optimized Regression Models",
        ),
        (
            "- [Section 7](#7-evaluation): Model Evaluation and Error Analysis",
            "- [Phase 9.6](#phase-96-model-evaluation-and-comprehensive-error-analysis): Model Evaluation and Error Analysis",
        ),
        (
            "- [Section 8](#8-valuation): Stock Valuation Analysis",
            "- [Phase 9.7](#phase-97-stock-ranking-analytics-and-analyst-comparison): Stock Ranking, Analytics, and Analyst Comparison",
        ),
        (
            "- [Section 9](#9-analytics): Predicted vs. Analyst Analytics",
            "",
        ),  # Remove - now part of 9.7
        (
            "- [Section 10](#10-portfolio): Portfolio Optimization",
            "- [Phase 9.8](#phase-98-comprehensive-reporting-and-dashboard-data): Comprehensive Reporting and Dashboard Data",
        ),
    ]

    new_source = source
    for old, new in replacements:
        if old in new_source:
            if new:
                new_source = new_source.replace(old, new)
                print(f"✓ Replaced: {old[:50]}...")
            else:
                # Remove the line
                lines = new_source.split("\n")
                lines = [line for line in lines if old not in line]
                new_source = "\n".join(lines)
                print(f"✓ Removed: {old[:50]}...")

    # Also update the Workflow Overview section numbers to phases
    workflow_replacements = [
        ("2. **Loading and Preprocessing**", "Phase 9.1: **Loading and Preprocessing**"),
        ("3. **Exploratory Data Analysis**", "Phase 9.2: **Exploratory Data Analysis**"),
        ("4. **Feature Engineering**", "Phase 9.3: **Feature Engineering**"),
        ("5. **Multi-Class Classification**", "Phase 9.4: **Multi-Class Classification**"),
        ("6. **Sector-Optimized Regression**", "Phase 9.5: **Sector-Optimized Regression**"),
        ("7. **Model Evaluation**", "Phase 9.6: **Model Evaluation**"),
        ("8. **Stock Valuation**", "Phase 9.7: **Stock Valuation**"),
        ("9. **Predicted vs. Analyst Analytics**", ""),  # Remove
        ("10. **Portfolio Optimization**", "Phase 9.8: **Portfolio Optimization**"),
    ]

    for old, new in workflow_replacements:
        if old in new_source:
            if new:
                new_source = new_source.replace(old, new)
                print(f"✓ Updated workflow: {old[:40]}...")
            else:
                lines = new_source.split("\n")
                lines = [line for line in lines if old not in line]
                new_source = "\n".join(lines)
                print(f"✓ Removed workflow: {old[:40]}...")

    # Update the cell source
    source_lines = new_source.split("\n")
    cells[nav_cell_idx]["source"] = [
        line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)
    ]

    # Save notebook
    print(f"\nSaving updated notebook...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("✓ Quick Reference Navigation updated successfully")
    return True


if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_main.ipynb")
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        exit(1)

    update_toc(notebook_path)
    print("\nNext: Run tests to verify: python -m unittest tests.test_notebook_refactoring -v")
