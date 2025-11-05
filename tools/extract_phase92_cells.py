#!/usr/bin/env python3
"""Extract Phase 9.2 cells from notebook for analysis."""

import json

def extract_phase92_cells():
    """Extract and analyze Phase 9.2 cells (59, 62, 65)."""
    with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cell_indices = [59, 62, 65]
    all_content = []
    
    print("Phase 9.2 Cells Analysis:")
    print("=" * 80)
    
    for idx in cell_indices:
        cell = notebook['cells'][idx]
        source = ''.join(cell['source'])
        all_content.append(source)
        
        lines = source.split('\n')
        func_defs = [line for line in lines if line.strip().startswith('def ')]
        
        print(f"\nCell {idx}:")
        print(f"  Type: {cell['cell_type']}")
        print(f"  Lines: {len(cell['source'])}")
        print(f"  Characters: {len(source)}")
        print(f"  Function definitions: {len(func_defs)}")
        
        if func_defs:
            print(f"  Functions:")
            for func_def in func_defs[:5]:
                print(f"    {func_def.strip()}")
            if len(func_defs) > 5:
                print(f"    ... and {len(func_defs) - 5} more")
        
        print(f"\n  First 300 characters:")
        print(f"  {source[:300]}")
        
        if len(source) > 300:
            print(f"  ...[truncated]...")
    
    # Save all content to file
    with open('phase92_cells_content.txt', 'w', encoding='utf-8') as f:
        for idx, content in zip(cell_indices, all_content):
            f.write(f"{'=' * 80}\n")
            f.write(f"CELL {idx}\n")
            f.write(f"{'=' * 80}\n")
            f.write(content)
            f.write(f"\n\n")
    
    print(f"\n\n✓ Full content saved to phase92_cells_content.txt")
    
    # Analyze for similar patterns
    print(f"\n📊 Pattern Analysis:")
    print(f"  Total cells: {len(cell_indices)}")
    print(f"  Total lines: {sum(len(c.split('\\n')) for c in all_content)}")
    print(f"  Total characters: {sum(len(c) for c in all_content)}")
    
    # Look for common function names across cells
    all_funcs = []
    for content in all_content:
        lines = content.split('\n')
        funcs = [line.strip().split('(')[0].replace('def ', '') 
                for line in lines if line.strip().startswith('def ')]
        all_funcs.extend(funcs)
    
    print(f"  Total function definitions: {len(all_funcs)}")
    
    # Check for duplicates
    from collections import Counter
    func_counts = Counter(all_funcs)
    duplicates = {name: count for name, count in func_counts.items() if count > 1}
    
    if duplicates:
        print(f"\n⚠️  Duplicate function names found (consolidation opportunity):")
        for name, count in duplicates.items():
            print(f"    '{name}' appears {count} times")
    else:
        print(f"\n✓ No duplicate function names found")
    
    return all_content

if __name__ == '__main__':
    extract_phase92_cells()
