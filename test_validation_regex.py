"""Test what the validation script's regex actually extracts."""
import json
import re
from collections import defaultdict

with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 4 (the main imports cell)
cell = nb['cells'][4]
source = ''.join(cell.get('source', []))

print("Testing validation script regex patterns on cell 4:")
print("=" * 80)

# Pattern 1: from module import (...)
pattern1 = r'from\s+(finance_ml[\w.]*)\s+import\s+\((.*?)\)'
matches1 = re.findall(pattern1, source, re.DOTALL)
print(f"\nPattern 1 (multi-line with parens): Found {len(matches1)} matches")
for i, (module, funcs) in enumerate(matches1[:3]):
    func_list = [f.strip() for f in funcs.split(',')]
    func_list = [f for f in func_list if f and not f.startswith('#')]
    print(f"  Match {i+1}: {module}")
    print(f"    Functions: {len(func_list)}")
    if func_list:
        print(f"    Sample: {func_list[:5]}")

# Pattern 2: from module import ...
pattern2 = r'from\s+(finance_ml[\w.]*)\s+import\s+([^\n(]+)'
matches2 = re.findall(pattern2, source)
print(f"\nPattern 2 (single line, no parens): Found {len(matches2)} matches")
for i, (module, funcs) in enumerate(matches2[:3]):
    func_list = [f.strip() for f in funcs.split(',')]
    func_list = [f.split(' as ')[0].strip() for f in func_list]
    func_list = [f for f in func_list if f and not f.startswith('#')]
    print(f"  Match {i+1}: {module}")
    print(f"    Raw: {funcs[:80]}...")
    print(f"    Functions: {func_list}")

# Simulate the full extraction logic
imports = defaultdict(set)
for module, funcs in matches1:
    func_list = [f.strip() for f in funcs.split(',')]
    func_list = [f for f in func_list if f and not f.startswith('#')]
    imports[module].update(func_list)

for module, funcs in matches2:
    func_list = [f.strip() for f in funcs.split(',')]
    func_list = [f.split(' as ')[0].strip() for f in func_list]
    func_list = [f for f in func_list if f and not f.startswith('#')]
    imports[module].update(func_list)

all_imported = set()
for module_funcs in imports.values():
    all_imported.update(module_funcs)

print("\n" + "=" * 80)
print(f"Total unique functions extracted: {len(all_imported)}")
print("\nChecking for specific functions:")
test_funcs = ['assign_valuation_category', 'calculate_mispricing_score', 
              'simple_eda', 'calculate_sector_zscores']
for func in test_funcs:
    status = "✓ FOUND" if func in all_imported else "✗ NOT FOUND"
    print(f"  {func}: {status}")

if len(all_imported) > 0:
    print(f"\nSample of imported functions: {list(all_imported)[:10]}")
