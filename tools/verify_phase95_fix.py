"""
Verify that the Phase 9.5 fix was applied correctly to ml_finance_model_main_v10.ipynb
"""
import json

print("=" * 80)
print("VERIFICATION: Phase 9.5 Fix for KeyError: 'Model'")
print("=" * 80)

# Read the fixed notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the train_and_compare_models function
found = False
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_and_compare_models' in source:
            print(f"\n[1] Found train_and_compare_models function in cell {i}")
            found = True
            
            # Check for the fixed code patterns
            checks = {
                'pd.DataFrame.from_dict(comparison_results, orient=\'index\')': False,
                '.reset_index().rename(columns={\'index\': \'Model\'})': False,
                '.sort_values(\'r2\', ascending=False)': False,
                '\'mae\': \'MAE\'': False,
                '\'r2\': \'R2\'': False
            }
            
            for pattern, _ in checks.items():
                if pattern in source:
                    checks[pattern] = True
            
            print("\n[2] Verification checks:")
            all_passed = True
            for pattern, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                print(f"    [{status}] {pattern[:60]}...")
                if not passed:
                    all_passed = False
            
            # Check that the old problematic pattern is NOT present
            print("\n[3] Checking old problematic pattern was removed:")
            if 'pd.DataFrame([comparison_results])' in source:
                print("    [FAIL] Old pattern 'pd.DataFrame([comparison_results])' still present!")
                all_passed = False
            else:
                print("    [PASS] Old pattern successfully removed")
            
            # Summary
            print("\n" + "=" * 80)
            if all_passed:
                print("SUCCESS: All verification checks passed!")
                print("\nThe fix correctly:")
                print("  1. Converts dict to DataFrame using from_dict(orient='index')")
                print("  2. Resets index and renames to 'Model' column")
                print("  3. Sorts by R2 score (descending)")
                print("  4. Renames columns to uppercase (MAE, RMSE, R2, etc.)")
                print("  5. Removed the problematic pd.DataFrame([dict]) pattern")
                print("\nThe KeyError: 'Model' should now be resolved.")
            else:
                print("WARNING: Some verification checks failed!")
                print("Manual review recommended.")
            print("=" * 80)
            
            break

if not found:
    print("\n[ERROR] Could not find train_and_compare_models function!")
    print("The notebook may be corrupted or the function was moved.")
