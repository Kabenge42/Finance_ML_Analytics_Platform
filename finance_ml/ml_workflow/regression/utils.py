"""Utility helpers for regression workflows."""

from __future__ import annotations

from typing import Any


def get_r2_score(item: Any) -> float:
    """Safely extract R² score from various data structures.

    Supports the standardized training return format (code_guidelines.md §7.1)
    and legacy structures produced before metrics normalization.

    Args:
        item: Result from a training function or (key, value) tuple.

    Returns:
        Extracted R² score. Defaults to 0.0 if not found.
    """

    # Handle tuple from dict.items() → (name, value)
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        value = item[1]
    else:
        value = item

    # Standard format: dict with metrics sub-dict
    if isinstance(value, dict):
        if "metrics" in value and isinstance(value["metrics"], dict):
            metrics = value["metrics"]
            return float(metrics.get("r2_score", metrics.get("r2", 0.0)))

        if "r2_score" in value:
            return float(value["r2_score"])

        if "r2" in value:
            return float(value["r2"])

    # Tuple/list format (assume first element is score)
    elif isinstance(value, (tuple, list)) and len(value) > 0:
        first_elem = value[0]
        if isinstance(first_elem, (int, float)):
            return float(first_elem)

    # Direct numeric value
    elif isinstance(value, (int, float)):
        return float(value)

    return 0.0


__all__ = ["get_r2_score"]
