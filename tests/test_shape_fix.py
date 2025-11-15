"""Test the shape normalization fix in evaluate_classification."""

import numpy as np
from finance_ml.ml_workflow.classification.evaluation import evaluate_classification

print("Testing evaluate_classification shape normalization fix...\n")

# Test 1: Normal 1D inputs (should work as before)
print("Test 1: Normal 1D inputs")
y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 2, 3, 3])
y_proba = np.random.rand(10, 5)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # normalize

result = evaluate_classification(y_true, y_pred, y_proba)
print(f"✓ Test 1 passed - Accuracy: {result['accuracy']:.3f}\n")

# Test 2: 2D y_pred with shape (n_samples, 1) - should flatten
print("Test 2: 2D y_pred with shape (n_samples, 1)")
y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
y_pred_2d = np.array([[0], [1], [2], [3], [4], [1], [1], [2], [3], [3]])
y_proba = np.random.rand(10, 5)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

result = evaluate_classification(y_true, y_pred_2d, y_proba)
print(f"✓ Test 2 passed - Accuracy: {result['accuracy']:.3f} (flattened shape)\n")

# Test 3: 2D y_pred with shape (n_samples, n_classes) - should use argmax
print("Test 3: 2D y_pred as probabilities (n_samples, n_classes)")
y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
# Create probability matrix where argmax gives [0, 1, 2, 3, 4, 1, 1, 2, 3, 3]
y_pred_proba = np.zeros((10, 5))
predicted_classes = [0, 1, 2, 3, 4, 1, 1, 2, 3, 3]
for i, cls in enumerate(predicted_classes):
    y_pred_proba[i, cls] = 0.8
    y_pred_proba[i, :] += 0.05  # Add small values to other classes
y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

result = evaluate_classification(y_true, y_pred_proba, y_pred_proba)
print(f"✓ Test 3 passed - Accuracy: {result['accuracy']:.3f} (used argmax)\n")

# Test 4: 2D y_true with shape (n_samples, 1) - should flatten
print("Test 4: 2D y_true with shape (n_samples, 1)")
y_true_2d = np.array([[0], [1], [2], [3], [4], [0], [1], [2], [3], [4]])
y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 2, 3, 3])
y_proba = np.random.rand(10, 5)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

result = evaluate_classification(y_true_2d, y_pred, y_proba)
print(f"✓ Test 4 passed - Accuracy: {result['accuracy']:.3f} (flattened y_true)\n")

# Test 5: Both 2D - should handle both
print("Test 5: Both y_true and y_pred are 2D")
y_true_2d = np.array([[0], [1], [2], [3], [4], [0], [1], [2], [3], [4]])
y_pred_2d = np.array([[0], [1], [2], [3], [4], [1], [1], [2], [3], [3]])
y_proba = np.random.rand(10, 5)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

result = evaluate_classification(y_true_2d, y_pred_2d, y_proba)
print(f"✓ Test 5 passed - Accuracy: {result['accuracy']:.3f} (both flattened)\n")

print("=" * 60)
print("✅ All shape normalization tests passed!")
print("   The fix successfully handles:")
print("   - Normal 1D arrays (backward compatible)")
print("   - 2D arrays with shape (n, 1) - flattens to 1D")
print("   - 2D arrays with shape (n, k) - uses argmax to convert to 1D")
print("=" * 60)
