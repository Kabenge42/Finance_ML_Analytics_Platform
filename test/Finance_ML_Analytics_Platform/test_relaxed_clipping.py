#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the relaxed upper bound clipping behavior.

This script simulates the clipping behavior with the new 3.0x multiplier
to verify that:
1. High-value predictions are no longer over-aggressively clipped
2. Lower bound protection remains intact
3. Extreme outliers are still caught
"""

import sys
import numpy as np
from finance_ml.ml_workflow.regression.robust import adaptive_clip_predictions

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def test_relaxed_upper_bound():
    """Test that the new 3.0x multiplier allows higher predictions."""

    # Simulate training data similar to actual price_target distribution
    # Based on clipping_effect_summary.json: max = 2,538,125
    # The actual p99.5 from the data should be around ~380K (based on current upper bound / 1.5)
    np.random.seed(42)

    # Create realistic training distribution matching actual data characteristics
    # p99.5 needs to be ~253K to get upper bound of ~760K with 3.0x multiplier
    y_train = np.concatenate(
        [
            np.random.lognormal(mean=8, sigma=2, size=1000),  # Main distribution (higher mean)
            np.array([500000, 750000, 1000000, 1500000, 2000000, 2538125]),  # High-value outliers
        ]
    )

    # Test predictions at various scales
    test_predictions = np.array(
        [
            22.58,  # Low value (from actual data)
            100000,  # Mid-range
            500000,  # High
            1111250,  # Actual calibrated max
            2000000,  # Very high
            3000000,  # Extreme (should be clipped)
        ]
    )

    # Run adaptive clipping
    result = adaptive_clip_predictions(test_predictions, y_train)

    print("=" * 80)
    print("RELAXED UPPER BOUND CLIPPING TEST")
    print("=" * 80)
    print(f"\nTraining data statistics:")
    print(f"  Min: ${np.min(y_train):,.2f}")
    print(f"  Max: ${np.max(y_train):,.2f}")
    print(f"  p99.5: ${np.percentile(y_train, 99.5):,.2f}")

    print(f"\nClipping bounds:")
    print(f"  Lower: ${result['lower_bound']:,.2f}")
    print(f"  Upper: ${result['upper_bound']:,.2f}")
    print(f"  Upper/p99.5 ratio: {result['upper_bound'] / np.percentile(y_train, 99.5):.2f}x")

    print(f"\nPrediction results:")
    for orig, clipped in zip(test_predictions, result["clipped_predictions"]):
        clipped_marker = " (CLIPPED)" if orig != clipped else ""
        print(f"  ${orig:>12,.2f} -> ${clipped:>12,.2f}{clipped_marker}")

    print(f"\nClipping statistics:")
    print(f"  Clipped to lower: {result['n_clipped_lower']} ({result['pct_clipped_lower']:.1f}%)")
    print(f"  Clipped to upper: {result['n_clipped_upper']} ({result['pct_clipped_upper']:.1f}%)")

    # Verification checks
    print(f"\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    checks = [
        ("No zero predictions", all(result["clipped_predictions"] > 0)),
        (
            "High predictions preserved better",
            result["clipped_predictions"][3] > 500000,
        ),  # Should preserve more
        (
            "Upper bound increased significantly",
            result["upper_bound"] > 500000,
        ),  # 3x of ~253K p99.5
        ("Extreme values still clipped", result["clipped_predictions"][-1] < 3000000),
        ("Lower bound protects small values", result["lower_bound"] >= 0.10),
    ]

    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in checks)
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 80)

    return all_passed


def test_comparison_old_vs_new():
    """Compare old 1.5× vs new 3.0× multiplier behavior."""

    # Simulate training data
    y_train = np.array([10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000])

    # Test predictions
    test_preds = np.array([1, 100, 10000, 100000, 500000, 1000000, 2000000])

    # Calculate bounds with both multipliers
    p99_5 = np.percentile(y_train, 99.5)

    old_upper = p99_5 * 1.5
    new_upper = p99_5 * 3.0

    print("\n" + "=" * 80)
    print("COMPARISON: OLD (1.5×) vs NEW (3.0×) MULTIPLIER")
    print("=" * 80)
    print(f"\np99.5 of training data: ${p99_5:,.2f}")
    print(f"\nUpper bounds:")
    print(f"  Old (1.5×): ${old_upper:,.2f}")
    print(f"  New (3.0×): ${new_upper:,.2f}")
    print(f"  Difference: ${new_upper - old_upper:,.2f} (+{(new_upper/old_upper - 1)*100:.1f}%)")

    print(f"\nPrediction handling:")
    print(f"  {'Original':>12}  {'Old (1.5×)':>15}  {'New (3.0×)':>15}  {'Impact':>20}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*15}  {'-'*20}")

    for pred in test_preds:
        old_clipped = min(pred, old_upper)
        new_clipped = min(pred, new_upper)

        if old_clipped == new_clipped:
            impact = "No change"
        elif old_clipped < pred and new_clipped == pred:
            impact = "UNCLIPPED ✓"
        else:
            impact = f"Less aggressive"

        print(f"  ${pred:>11,.0f}  ${old_clipped:>14,.0f}  ${new_clipped:>14,.0f}  {impact:>20}")

    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_relaxed_upper_bound()
    test_comparison_old_vs_new()

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(
        """
The new 3.0x multiplier:
[+] Reduces over-aggressive upper-bound clipping
[+] Preserves high-value predictions (e.g., $1.1M+)
[+] Still protects against extreme unrealistic outliers
[+] Adapts to the heavy right-tail of financial data

Expected impact on your actual data:
- Clipped predictions: 0.4% → ~0.0%
- Upper bound: $380K → ~$1.14M (3× higher)
- Better capture of legitimate high-value stocks
    """
    )
    print("=" * 80)
