"""Quick validation test for Phase 9.5 training functions."""
import numpy as np
import pandas as pd
from finance_ml.advanced_models import (
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor
)

print("Testing Phase 9.5 Training Functions with ensure_nonnegative parameter...\n")

# Create synthetic data
np.random.seed(42)
X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
y = pd.Series(np.abs(np.random.randn(100)) * 20 + 10)

# Test 1: train_ridge_regressor
print("Test 1: train_ridge_regressor with ensure_nonnegative=True")
results = train_ridge_regressor(X, y, alpha=1.0, cv=3, random_state=42, ensure_nonnegative=True)
model = results["model"]
preds = model.predict(X)
print(f"  Model type: {results['model_type']}")
print(f"  Non-negative constraint: {results['nonnegative_constraint']}")
print(f"  All predictions >= 0: {(preds >= 0).all()}")
print(f"  R2 score: {results['train_score']:.3f}")
assert (preds >= 0).all(), "FAIL: Found negative predictions"
print("  PASS")

# Test 2: train_lasso_regressor
print("\nTest 2: train_lasso_regressor with ensure_nonnegative=True")
results = train_lasso_regressor(X, y, alpha=0.1, cv=3, random_state=42, ensure_nonnegative=True)
model = results["model"]
preds = model.predict(X)
print(f"  Model type: {results['model_type']}")
print(f"  Non-negative constraint: {results['nonnegative_constraint']}")
print(f"  All predictions >= 0: {(preds >= 0).all()}")
print(f"  Non-zero coefficients: {results['n_nonzero_coefs']}")
assert (preds >= 0).all(), "FAIL: Found negative predictions"
print("  PASS")

# Test 3: train_elastic_net_regressor
print("\nTest 3: train_elastic_net_regressor with ensure_nonnegative=True")
results = train_elastic_net_regressor(X, y, alpha=0.1, l1_ratio=0.5, cv=3, random_state=42, ensure_nonnegative=True)
model = results["model"]
preds = model.predict(X)
print(f"  Model type: {results['model_type']}")
print(f"  Non-negative constraint: {results['nonnegative_constraint']}")
print(f"  All predictions >= 0: {(preds >= 0).all()}")
print(f"  Best L1 ratio: {results['best_l1_ratio']}")
assert (preds >= 0).all(), "FAIL: Found negative predictions"
print("  PASS")

print("\n" + "="*80)
print("SUCCESS: All Phase 9.5 training functions work correctly!")
print("="*80)
