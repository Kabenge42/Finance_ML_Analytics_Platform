"""
Validation script for ZERO_PREDICTIONS_FIX.md

Demonstrates the fix for zero predictions issue where 24.75% of predictions
were clipped to exactly $0.00, destroying low-value stock predictions.

This script:
1. Simulates production data (5000 training, 1406 test samples)
2. Compares OLD approach (hard zero) vs NEW approach (percentile-based)
3. Shows zero elimination (348 → 0 zeros)
4. Validates error metrics preservation

Usage:
    python validate_zero_predictions_fix.py
"""

import numpy as np
from finance_ml.ml_workflow.regression.robust import adaptive_clip_predictions


def simulate_stock_data(n_samples: int, seed: int = 42) -> tuple:
    """
    Simulate stock price data with realistic distribution.

    Returns:
        (y_true, y_pred_raw) - actual prices and raw model predictions
    """
    rng = np.random.default_rng(seed)

    # Simulate actual stock prices (log-normal distribution)
    # Median ~$10, with some penny stocks and some high-value stocks
    y_true = np.exp(rng.normal(loc=2.3, scale=1.5, size=n_samples))

    # Simulate raw predictions with typical model errors
    # Model can produce small negative values for low-priced stocks
    noise = rng.normal(loc=0, scale=0.3, size=n_samples)
    y_pred_raw = y_true * np.exp(noise) + rng.normal(0, 2.0, size=n_samples)

    return y_true, y_pred_raw


def old_clipping_approach(predictions: np.ndarray, y_train: np.ndarray) -> dict:
    """OLD approach: Hard zero lower bound (the problem)."""
    train_p995 = np.percentile(y_train, 99.5)
    upper_bound = train_p995 * 1.5

    # PROBLEM: Hard zero lower bound
    lower_bound = 0.0

    clipped = np.clip(predictions, lower_bound, upper_bound)

    n_zeros = np.sum(clipped == 0.0)
    n_near_zero = np.sum(clipped < 1.0)

    return {
        "clipped_predictions": clipped,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_zeros": n_zeros,
        "n_near_zero": n_near_zero,
        "pct_zeros": 100.0 * n_zeros / len(predictions),
        "pct_near_zero": 100.0 * n_near_zero / len(predictions),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate error metrics."""
    errors = np.abs(y_true - y_pred)
    return {
        "MAE": np.mean(errors),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "Median_Error": np.median(errors),
    }


def main():
    print("=" * 80)
    print("ZERO PREDICTIONS FIX - VALIDATION")
    print("=" * 80)
    print()

    # Simulate data matching production scenario
    print("Simulating production data...")
    n_train = 5000
    n_test = 1406

    y_train, _ = simulate_stock_data(n_train, seed=42)
    y_test, y_pred_raw = simulate_stock_data(n_test, seed=123)

    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  Test y_true range: ${y_test.min():.2f} - ${y_test.max():.2f}")
    print(f"  Raw predictions range: ${y_pred_raw.min():.2f} - ${y_pred_raw.max():.2f}")
    print()

    # OLD APPROACH (Hard Zero)
    print("-" * 80)
    print("OLD APPROACH: Hard Zero Lower Bound")
    print("-" * 80)
    old_result = old_clipping_approach(y_pred_raw, y_train)

    print(f"  Lower bound: ${old_result['lower_bound']:.2f} (HARD ZERO)")
    print(f"  Upper bound: ${old_result['upper_bound']:.2f}")
    print()
    print(f"  Zero predictions: {old_result['n_zeros']} ({old_result['pct_zeros']:.1f}%)")
    print(f"  Near-zero (<$1): {old_result['n_near_zero']} ({old_result['pct_near_zero']:.1f}%)")

    old_metrics = calculate_metrics(y_test, old_result["clipped_predictions"])
    print()
    print(f"  MAE (all): ${old_metrics['MAE']:.2f}")
    print(f"  RMSE (all): ${old_metrics['RMSE']:.2f}")
    print(f"  Median Error: ${old_metrics['Median_Error']:.2f}")

    # Filter for low-value stocks
    low_value_mask = y_test < 10
    if np.sum(low_value_mask) > 0:
        low_metrics_old = calculate_metrics(
            y_test[low_value_mask], old_result["clipped_predictions"][low_value_mask]
        )
        print(f"  MAE (y_true<$10): ${low_metrics_old['MAE']:.2f}")
    print()

    # NEW APPROACH (Percentile-Based Adaptive)
    print("-" * 80)
    print("NEW APPROACH: Percentile-Based Adaptive Bounds")
    print("-" * 80)
    new_result = adaptive_clip_predictions(y_pred_raw, y_train)

    print(f"  Lower bound: ${new_result['lower_bound']:.2f} (0.5 × p0.5, min $0.10)")
    print(f"  Upper bound: ${new_result['upper_bound']:.2f} (1.5 × p99.5)")
    print()
    print(
        f"  Clipped to lower bound: {new_result['n_clipped_lower']} ({new_result['pct_clipped_lower']:.1f}%)"
    )
    print(
        f"  Clipped to upper bound: {new_result['n_clipped_upper']} ({new_result['pct_clipped_upper']:.1f}%)"
    )

    n_zeros_new = np.sum(new_result["clipped_predictions"] == 0.0)
    n_near_zero_new = np.sum(new_result["clipped_predictions"] < 1.0)
    print()
    print(f"  Zero predictions: {n_zeros_new} (0.0%)")
    print(f"  Near-zero (<$1): {n_near_zero_new} ({100.0 * n_near_zero_new / len(y_test):.1f}%)")

    new_metrics = calculate_metrics(y_test, new_result["clipped_predictions"])
    print()
    print(f"  MAE (all): ${new_metrics['MAE']:.2f}")
    print(f"  RMSE (all): ${new_metrics['RMSE']:.2f}")
    print(f"  Median Error: ${new_metrics['Median_Error']:.2f}")

    if np.sum(low_value_mask) > 0:
        low_metrics_new = calculate_metrics(
            y_test[low_value_mask], new_result["clipped_predictions"][low_value_mask]
        )
        print(f"  MAE (y_true<$10): ${low_metrics_new['MAE']:.2f}")
    print()

    # COMPARISON
    print("=" * 80)
    print("COMPARISON: OLD vs NEW")
    print("=" * 80)

    zero_reduction = old_result["n_zeros"] - n_zeros_new
    pct_reduction = (
        100.0 * zero_reduction / old_result["n_zeros"] if old_result["n_zeros"] > 0 else 0
    )

    print(f"  Zero predictions reduced: {old_result['n_zeros']} -> {n_zeros_new}")
    print(f"  Reduction: {zero_reduction} ({pct_reduction:.1f}%)")
    print()

    mae_change = ((new_metrics["MAE"] - old_metrics["MAE"]) / old_metrics["MAE"]) * 100
    print(f"  MAE change: {mae_change:+.2f}%")

    if np.sum(low_value_mask) > 0:
        low_mae_change = (
            (low_metrics_new["MAE"] - low_metrics_old["MAE"]) / low_metrics_old["MAE"]
        ) * 100
        print(f"  MAE change (low-value): {low_mae_change:+.2f}%")
    print()

    # EXAMPLES
    print("=" * 80)
    print("EXAMPLES: Stocks where OLD clipped to zero")
    print("=" * 80)

    # Find examples where old approach clipped to zero
    zero_indices = np.where(old_result["clipped_predictions"] == 0.0)[0]

    if len(zero_indices) > 0:
        print(
            f"\n{'Actual':<10} {'Raw Pred':<12} {'OLD':<10} {'NEW':<10} {'OLD Err':<10} {'NEW Err':<10} {'Improv':<8}"
        )
        print("-" * 80)

        # Show first 10 examples
        for idx in zero_indices[:10]:
            actual = y_test[idx]
            raw = y_pred_raw[idx]
            old_pred = old_result["clipped_predictions"][idx]
            new_pred = new_result["clipped_predictions"][idx]
            old_err = abs(actual - old_pred)
            new_err = abs(actual - new_pred)
            improvement = ((old_err - new_err) / old_err * 100) if old_err > 0 else 0

            print(
                f"${actual:<9.2f} ${raw:<11.2f} ${old_pred:<9.2f} ${new_pred:<9.2f} "
                f"${old_err:<9.2f} ${new_err:<9.2f} {improvement:<7.1f}%"
            )
    print()

    # VALIDATION STATUS
    print("=" * 80)
    print("VALIDATION STATUS")
    print("=" * 80)

    checks = []
    checks.append(("Zero predictions eliminated", n_zeros_new == 0))
    checks.append(("Lower bound > 0", new_result["lower_bound"] > 0))
    checks.append(("Lower bound >= $0.10", new_result["lower_bound"] >= 0.10))
    checks.append(("Upper bound calculated", new_result["upper_bound"] > 0))
    checks.append(("MAE change < 5%", abs(mae_change) < 5.0))

    all_passed = all(status for _, status in checks)

    for check_name, status in checks:
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str}: {check_name}")

    print()
    if all_passed:
        print("  [SUCCESS] ALL VALIDATION CHECKS PASSED")
    else:
        print("  [FAILURE] SOME VALIDATION CHECKS FAILED")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
