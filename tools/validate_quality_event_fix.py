"""
Validation snippet for quality_event red flag penalty fix.

This script validates that the reduced penalty (-0.2 instead of -0.5) 
produces balanced class distributions closer to the intended quantile-based 
design (15%, 20%, 30%, 20%, 15%).

Usage in notebook:
    # After creating event_labels with quality_event method
    %run validate_quality_event_fix.py
    validate_class_distribution(event_labels)
"""

import numpy as np
import pandas as pd


def validate_class_distribution(labels, method_name="quality_event"):
    """
    Validate that class distribution matches intended quantile design.
    
    Expected distribution (from quantile thresholds in labels.py):
    - Class 0 (Strong Negative): ≤15th percentile → ~15%
    - Class 1 (Negative): 15th-35th percentile → ~20%
    - Class 2 (Neutral): 35th-65th percentile → ~30%
    - Class 3 (Positive): 65th-85th percentile → ~20%
    - Class 4 (Strong Positive): ≥85th percentile → ~15%
    
    Args:
        labels: numpy array of class labels (0-4)
        method_name: name of the labeling method being validated
    
    Returns:
        dict with validation results and metrics
    """
    total = len(labels)
    
    # Calculate actual distribution
    actual_dist = {
        0: (labels == 0).sum(),
        1: (labels == 1).sum(),
        2: (labels == 2).sum(),
        3: (labels == 3).sum(),
        4: (labels == 4).sum(),
    }
    
    actual_pct = {k: v / total * 100 for k, v in actual_dist.items()}
    
    # Expected distribution based on quantile thresholds
    expected_pct = {
        0: 15.0,  # Bottom 15%
        1: 20.0,  # 15th-35th percentile
        2: 30.0,  # 35th-65th percentile
        3: 20.0,  # 65th-85th percentile
        4: 15.0,  # Top 15%
    }
    
    # Calculate deviations
    deviations = {k: abs(actual_pct[k] - expected_pct[k]) for k in range(5)}
    max_deviation = max(deviations.values())
    
    # Validation thresholds
    # ✓ Excellent: max deviation ≤5%
    # ⚠ Acceptable: max deviation ≤10%
    # ✗ Poor: max deviation >10%
    
    if max_deviation <= 5.0:
        status = "✓ EXCELLENT"
        status_color = "green"
    elif max_deviation <= 10.0:
        status = "⚠ ACCEPTABLE"
        status_color = "yellow"
    else:
        status = "✗ POOR"
        status_color = "red"
    
    # Print validation report
    print(f"\n{'='*70}")
    print(f"CLASS DISTRIBUTION VALIDATION - {method_name}")
    print(f"{'='*70}")
    print(f"\nTotal samples: {total:,}")
    print(f"\nStatus: {status} (max deviation: {max_deviation:.1f}%)")
    print(f"\n{'Class':<20} {'Expected':<12} {'Actual':<12} {'Deviation':<12} {'Status'}")
    print("-" * 70)
    
    for class_id in range(5):
        class_names = {
            0: "Strong Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Strong Positive"
        }
        
        exp = expected_pct[class_id]
        act = actual_pct[class_id]
        dev = deviations[class_id]
        
        # Status indicator per class
        if dev <= 5.0:
            class_status = "✓"
        elif dev <= 10.0:
            class_status = "⚠"
        else:
            class_status = "✗"
        
        print(f"{class_names[class_id]:<20} "
              f"{exp:>5.1f}% ({total*exp/100:>5.0f})  "
              f"{act:>5.1f}% ({actual_dist[class_id]:>5})  "
              f"{dev:>6.1f}%       "
              f"{class_status}")
    
    print("-" * 70)
    
    # Summary
    print(f"\nValidation Summary:")
    print(f"  - Expected distribution follows quantile design (15-20-30-20-15)")
    print(f"  - Max deviation: {max_deviation:.1f}%")
    print(f"  - Overall status: {status}")
    
    if max_deviation > 10.0:
        print(f"\n⚠ WARNING: Class distribution significantly deviates from expected.")
        print(f"  Most likely cause: Aggressive negative penalties in scoring logic.")
        print(f"  Recommendation: Review penalty weights in labels.py")
    elif max_deviation <= 5.0:
        print(f"\n✓ Class distribution is well-balanced and matches quantile design.")
    
    print(f"{'='*70}\n")
    
    return {
        "status": status,
        "max_deviation": max_deviation,
        "actual_distribution": actual_dist,
        "actual_percentages": actual_pct,
        "expected_percentages": expected_pct,
        "deviations": deviations,
    }


def compare_before_after():
    """
    Display comparison of class distributions before and after the fix.
    """
    print("\n" + "="*70)
    print("QUALITY_EVENT RED FLAG PENALTY FIX - BEFORE/AFTER COMPARISON")
    print("="*70)
    
    print("\nBEFORE (penalty = -0.5):")
    print("  Strong Negative (0): 4463 (63.4%) ← SEVERELY IMBALANCED")
    print("  Negative (1):         605 ( 8.6%)")
    print("  Neutral (2):          908 (12.9%)")
    print("  Positive (3):         605 ( 8.6%)")
    print("  Strong Positive (4):  455 ( 6.5%)")
    print("  Max deviation: 48.4% from expected 15%")
    
    print("\nEXPECTED (quantile design):")
    print("  Strong Negative (0): ~1055 (15.0%)")
    print("  Negative (1):        ~1407 (20.0%)")
    print("  Neutral (2):         ~2111 (30.0%)")
    print("  Positive (3):        ~1407 (20.0%)")
    print("  Strong Positive (4): ~1055 (15.0%)")
    
    print("\nFIX APPLIED:")
    print("  - Reduced red flag penalty from -0.5 to -0.2")
    print("  - Location: labels.py, line 1025")
    print("  - Rationale: Most stocks have ≥1 red flag; cumulative penalties")
    print("               created strong negative bias preventing quantile")
    print("               thresholds from working as designed")
    
    print("\nAFTER (penalty = -0.2):")
    print("  Run validation after recreating labels with updated code")
    print("  Expected: Max deviation ≤5% from quantile design")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print(__doc__)
    compare_before_after()
