#!/usr/bin/env python3
"""
Analyze ml_finance_model_main.ipynb to identify malformed Phase 9.4-6 cells.
"""
import json
import sys
from pathlib import Path

def analyze_notebook_phases(notebook_path):
    """Analyze notebook and identify Phase 9.4-6 cells."""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return None
    
    cells = nb.get('cells', [])
    print(f"Total cells: {len(cells)}")
    print("\n" + "="*80)
    
    # Find Phase 9.4-6 cells
    phase94_cells = []
    phase95_cells = []
    phase96_cells = []
    misplaced_cells = []
    
    # Track current phase
    current_phase = None
    last_major_phase = None
    
    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        # Join source lines
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        # Check for phase markers
        if 'PHASE 9.4' in source_text or 'Phase 9.4' in source_text or 'Section 9.4' in source_text:
            phase94_cells.append({'idx': idx, 'type': cell_type, 'preview': source_text[:200]})
            current_phase = '9.4'
        elif 'PHASE 9.5' in source_text or 'Phase 9.5' in source_text or 'Section 9.5' in source_text:
            phase95_cells.append({'idx': idx, 'type': cell_type, 'preview': source_text[:200]})
            current_phase = '9.5'
        elif 'PHASE 9.6' in source_text or 'Phase 9.6' in source_text or 'Section 9.6' in source_text:
            phase96_cells.append({'idx': idx, 'type': cell_type, 'preview': source_text[:200]})
            current_phase = '9.6'
        elif 'PHASE 9.7' in source_text or 'Phase 9.7' in source_text:
            last_major_phase = '9.7'
        elif 'PHASE 9.8' in source_text or 'Phase 9.8' in source_text:
            last_major_phase = '9.8'
        
        # Check if Phase 9.4-6 cells appear after Phase 9.7+
        if current_phase in ['9.4', '9.5', '9.6'] and last_major_phase in ['9.7', '9.8']:
            misplaced_cells.append({
                'idx': idx,
                'phase': current_phase,
                'type': cell_type,
                'preview': source_text[:200]
            })
    
    # Report findings
    print("\nPHASE 9.4 CELLS:")
    print("-" * 80)
    for cell_info in phase94_cells:
        print(f"Cell {cell_info['idx']} ({cell_info['type']}):")
        print(f"  {cell_info['preview']}")
        print()
    
    print("\nPHASE 9.5 CELLS:")
    print("-" * 80)
    for cell_info in phase95_cells:
        print(f"Cell {cell_info['idx']} ({cell_info['type']}):")
        print(f"  {cell_info['preview']}")
        print()
    
    print("\nPHASE 9.6 CELLS:")
    print("-" * 80)
    for cell_info in phase96_cells:
        print(f"Cell {cell_info['idx']} ({cell_info['type']}):")
        print(f"  {cell_info['preview']}")
        print()
    
    print("\nMISPLACED CELLS (appearing after Phase 9.7+):")
    print("-" * 80)
    for cell_info in misplaced_cells:
        print(f"Cell {cell_info['idx']} (Phase {cell_info['phase']}, {cell_info['type']}):")
        print(f"  {cell_info['preview']}")
        print()
    
    # Look for cells at the end that might be malformed
    print("\nLAST 10 CELLS:")
    print("-" * 80)
    for idx in range(max(0, len(cells) - 10), len(cells)):
        cell = cells[idx]
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        print(f"Cell {idx} ({cell_type}):")
        preview = source_text[:300] if len(source_text) > 300 else source_text
        print(f"  {preview}")
        print()
    
    return {
        'phase94': phase94_cells,
        'phase95': phase95_cells,
        'phase96': phase96_cells,
        'misplaced': misplaced_cells,
        'total_cells': len(cells)
    }

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main.ipynb')
    
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        sys.exit(1)
    
    result = analyze_notebook_phases(notebook_path)
    
    if result:
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        print(f"Total cells: {result['total_cells']}")
        print(f"Phase 9.4 cells found: {len(result['phase94'])}")
        print(f"Phase 9.5 cells found: {len(result['phase95'])}")
        print(f"Phase 9.6 cells found: {len(result['phase96'])}")

