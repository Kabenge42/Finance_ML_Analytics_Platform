# Quick script to reload the analyst_comparison module
# Run this in your notebook to reload the updated code without restarting kernel

import importlib
import sys

# Remove the cached module
if 'finance_ml.ml_workflow.analytics.analyst_comparison' in sys.modules:
    del sys.modules['finance_ml.ml_workflow.analytics.analyst_comparison']
if 'finance_ml' in sys.modules:
    del sys.modules['finance_ml']

# Re-import
from finance_ml.ml_workflow.analytics.analyst_comparison import PredictionAnalystAnalytics

print("✓ Module reloaded successfully")
print("  You can now use PredictionAnalystAnalytics with the fix applied")
