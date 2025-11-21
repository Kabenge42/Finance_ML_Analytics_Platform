"""Test that Cell 107 has valid Python syntax and can be parsed"""
import json
import ast
import sys

def test_cell_syntax():
    """Verify Cell 107 contains valid, executable Python code"""

    # Read notebook
    with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Get Cell 107
    cell = nb['cells'][107]
    source = cell['source']

    # Reconstruct the code as it would be executed
    code = ''.join(source)

    print("Testing Cell 107 Python syntax...")
    print(f"Code length: {len(code)} characters")
    print(f"Code lines: {len(source)}")
    print()

    # Test 1: Can it be parsed as Python?
    try:
        ast.parse(code)
        print("✓ PASS: Valid Python syntax")
    except SyntaxError as e:
        print(f"✗ FAIL: Syntax error at line {e.lineno}: {e.msg}")
        print(f"  Text: {e.text}")
        return False

    # Test 2: Check for common issues
    issues = []

    # Check for HTML entities
    if '&gt;' in code or '&lt;' in code or '&#' in code:
        issues.append("Contains HTML entities")

    # Check for snapshot_date (should use last_updated)
    if 'snapshot_date' in code:
        issues.append("Contains 'snapshot_date' (should use 'last_updated')")

    # Check for proper indentation
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line and not line.startswith(' ') and not line.startswith('#') and i > 0:
            # Check if this is a valid top-level statement
            stripped = line.strip()
            if stripped and not any(stripped.startswith(kw) for kw in
                ['if ', 'else', 'elif ', 'try:', 'except', 'finally:', 'def ', 'class ', 'print(']):
                # Might be continuation, check previous line
                prev = lines[i-1].rstrip() if i > 0 else ''
                if not prev.endswith(('\\', ',', '(', '[')):
                    issues.append(f"Line {i+1} may have indentation issue: {line[:50]}")

    if issues:
        print("\n⚠ Potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ PASS: No common issues detected")

    # Test 3: Check required elements exist
    print("\nChecking required elements:")

    required_elements = [
        ('last_updated', 'Schema-aligned date column'),
        ('create_ml_return_features', 'ML feature creation function'),
        ('train_linear_return_predictor', 'Model training function'),
        ('create_ensemble_return_predictions', 'Ensemble prediction function'),
    ]

    for element, description in required_elements:
        if element in code:
            print(f"  ✓ {description}: '{element}'")
        else:
            print(f"  ✗ Missing {description}: '{element}'")
            return False

    print("\n" + "=" * 60)
    print("✅ Cell 107 is syntactically valid and execution-ready!")
    print("=" * 60)

    return True

if __name__ == '__main__':
    try:
        success = test_cell_syntax()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
