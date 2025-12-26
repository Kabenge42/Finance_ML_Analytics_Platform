"""
Test script to verify the quantile regression fix.
"""

import numpy as np
import pandas as pd
from finance_ml.advanced_models import train_quantile_regressor

# Create sample data
np.random.seed(42)
n_samples = 100
X = pd.DataFrame(
    {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    }
)
y = pd.Series(
    2 * X["feature1"] - X["feature2"] + 0.5 * X["feature3"] + np.random.randn(n_samples) * 0.5
)

# Test the fixed function
print("Testing train_quantile_regressor() fix...")
print("=" * 60)

quantiles = [0.1, 0.5, 0.9]
models, results = train_quantile_regressor(X, y, quantiles=quantiles, random_state=42)

print(f"\n? Function executed successfully")
print(f"  Number of models: {len(models)}")
print(f"  Quantiles: {results['quantiles']}")
print(f"  Model type: {results['model_type']}")

# Check that quantile_results exists
if "quantile_results" not in results:
    print("\n? ERROR: 'quantile_results' key not found in results!")
    exit(1)

print(f"\n? 'quantile_results' key found in results")
print(f"  Number of per-quantile results: {len(results['quantile_results'])}")

# Verify per-quantile results structure
print("\n?? Per-Quantile Results:")
for i, qr in enumerate(results["quantile_results"]):
    expected_keys = ["quantile", "train_score", "model_type"]
    missing_keys = [k for k in expected_keys if k not in qr]

    if missing_keys:
        print(f"\n? ERROR: Quantile {i} missing keys: {missing_keys}")
        exit(1)

    print(
        f"  Q{qr['quantile']}: train_score={qr['train_score']:.4f}, model_type={qr['model_type']}"
    )

# Test accessing results the way the notebook does
print("\n? Testing notebook access pattern...")
for i, (q, model) in enumerate(zip(quantiles, models)):
    predictions = model.predict(X)
    if "quantile_results" in results:
        train_score = results["quantile_results"][i]["train_score"]
        print(f"  Q{q}: {train_score:.4f} (train R²) - accessed from results")
    else:
        train_score = model.score(X, y)
        print(f"  Q{q}: {train_score:.4f} (train R²) - computed fallback")

print("\n" + "=" * 60)
print("? All tests passed! The fix is working correctly.")
print("=" * 60)
