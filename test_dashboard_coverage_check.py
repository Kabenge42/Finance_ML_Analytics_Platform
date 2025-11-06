"""
Quick script to verify dashboard helper function coverage
"""
import sys
import subprocess

# Run coverage on just the dashboard helper tests
result = subprocess.run([
    sys.executable, '-m', 'coverage', 'run', 
    '--source=finance_ml.eval',
    '-m', 'unittest', 
    'tests.test_dashboard_helpers',
    '-v'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Generate report focusing on specific functions
print("\n" + "="*60)
print("Coverage Report for Dashboard Helper Functions")
print("="*60)

result2 = subprocess.run([
    sys.executable, '-m', 'coverage', 'report',
    '--include=finance_ml/eval.py',
], capture_output=True, text=True)

print(result2.stdout)

# Show which lines are covered/missed for the helper functions
print("\n" + "="*60)
print("Detailed Analysis - Looking for helper functions")
print("="*60)

# Get annotated source
result3 = subprocess.run([
    sys.executable, '-m', 'coverage', 'annotate',
    '--include=finance_ml/eval.py',
], capture_output=True, text=True)

print("Generated annotated file: finance_ml/eval.py,cover")
print("\nTo see detailed coverage, check the .cover file or HTML report")
