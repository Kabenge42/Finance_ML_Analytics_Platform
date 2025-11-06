#!/usr/bin/env python3
"""
Analyze notebook structure to identify Phase 9 sections
"""
import json
from pathlib import Path

def analyze_notebook_phases(notebook_path):
    """Analyze notebook for Phase 9 sections"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"=" * 80)
    print(f"NOTEBOOK PHASE ANALYSIS: {notebook_path}")
    print(f"=" * 80)
    print(f"Total cells: {len(nb['cells'])}\n")
    
    phase_sections = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and cell['source']:
            # Join source lines for analysis
            source_text = ''.join(cell['source'])
            
            # Check for Phase 9 headers
            if 'Phase 9' in source_text or '## Phase 9' in source_text or '### Phase 9' in source_text:
                # Extract first line
                first_line = cell['source'][0].strip()
                phase_sections.append({
                    'cell_index': i,
                    'header': first_line[:120],
                    'full_text': source_text[:300]
                })
    
    print(f"Phase 9 sections found: {len(phase_sections)}\n")
    
    for section in phase_sections:
        print(f"Cell {section['cell_index']:3d}: {section['header']}")
    
    print("\n" + "=" * 80)
    print("DETAILED SECTION CONTENT")
    print("=" * 80)
    
    # Check for specific phases mentioned in issue
    target_phases = {
        '9.5': 'Sector-Optimized Regression Models',
        '9.5.1': 'Model Optimization Enhancements',
        '9.6': 'Model Evaluation and Error Analysis',
        '9.6.1': 'Enhanced Error Analysis',
        '9.7': 'Identification of Under/Overvalued Stocks',
        '9.8': 'Comprehensive Analytics'
    }
    
    print("\nTarget Phase Sections:")
    for phase, description in target_phases.items():
        found = False
        for section in phase_sections:
            if f'Phase {phase}' in section['header'] or f'9.{phase.split(".")[-1]}' in section['header']:
                print(f"  ✓ Phase {phase}: FOUND at Cell {section['cell_index']}")
                print(f"    {section['header']}")
                found = True
                break
        if not found:
            print(f"  ✗ Phase {phase}: NOT FOUND - {description}")
    
    return phase_sections

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main_backup.ipynb')
    
    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        exit(1)
    
    analyze_notebook_phases(notebook_path)
