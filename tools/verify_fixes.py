#!/usr/bin/env python3
"""Verify that the notebook fixes were applied correctly."""

import json


def verify_fixes(notebook_path):
    """Verify both fixes in the notebook."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    valuation_fixed = False
    analytics_dir_fixed = False
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check Fix 1: VALUATION_CATEGORIES
        if "VALUATION_CATEGORIES = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']" in source:
            print(f"[OK] Cell {cell_idx}: VALUATION_CATEGORIES fixed (no duplicate)")
            valuation_fixed = True
        elif "VALUATION_CATEGORIES = ['Strong Buy', 'Buy', 'Hold', 'Hold', 'Sell', 'Strong Sell']" in source:
            print(f"[ISSUE] Cell {cell_idx}: VALUATION_CATEGORIES still has duplicate 'Hold'")
        
        # Check Fix 2: analytics_dir handling
        if 'def setup_output_directory():' in source:
            if 'if hasattr(config, \'analytics_dir\'):' in source:
                print(f"[OK] Cell {cell_idx}: setup_output_directory() has analytics_dir fallback")
                analytics_dir_fixed = True
            elif 'output_dir = config.analytics_dir' in source and 'if hasattr(config' not in source:
                print(f"[ISSUE] Cell {cell_idx}: setup_output_directory() missing analytics_dir fallback")
    
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print(f"  VALUATION_CATEGORIES fix: {'[OK]' if valuation_fixed else '[NOT FOUND]'}")
    print(f"  analytics_dir fallback fix: {'[OK]' if analytics_dir_fixed else '[NOT FOUND]'}")
    print("=" * 60)
    
    return valuation_fixed and analytics_dir_fixed


if __name__ == '__main__':
    notebook_path = 'ml_finance_model_main_v10.ipynb'
    print(f"Verifying fixes in: {notebook_path}\n")
    
    all_fixed = verify_fixes(notebook_path)
    
    if all_fixed:
        print("\n[SUCCESS] All fixes verified!")
    else:
        print("\n[WARNING] Some fixes may not have been applied")
