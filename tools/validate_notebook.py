"""Validate the updated notebook."""
import json
from pathlib import Path

notebook_path = Path("ml_finance_model_main.ipynb")

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("✅ Notebook is valid JSON")
    print(f"✅ Total cells: {len(notebook['cells'])}")
    print(f"✅ Notebook format: {notebook.get('nbformat', 'N/A')}.{notebook.get('nbformat_minor', 'N/A')}")
    
    # Count sections
    phase94_cells = sum(1 for c in notebook['cells'] if 'PHASE 9.4' in ''.join(c.get('source', [])))
    phase95_cells = sum(1 for c in notebook['cells'] if 'PHASE 9.5' in ''.join(c.get('source', [])))
    phase96_cells = sum(1 for c in notebook['cells'] if 'PHASE 9.6' in ''.join(c.get('source', [])))
    phase97_cells = sum(1 for c in notebook['cells'] if 'PHASE 9.7' in ''.join(c.get('source', [])))
    phase98_cells = sum(1 for c in notebook['cells'] if 'PHASE 9.8' in ''.join(c.get('source', [])))
    
    print(f"\n✅ Phase 9.4 cells: {phase94_cells}")
    print(f"✅ Phase 9.5 cells: {phase95_cells}")
    print(f"✅ Phase 9.6 cells: {phase96_cells}")
    print(f"✅ Phase 9.7 cells: {phase97_cells}")
    print(f"✅ Phase 9.8 cells: {phase98_cells}")
    print(f"✅ Total Phase 9.4-9.8 cells: {phase94_cells + phase95_cells + phase96_cells + phase97_cells + phase98_cells}")
    
    print("\n✅ Notebook validation successful!")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON validation failed: {e}")
except Exception as e:
    print(f"❌ Validation error: {e}")
