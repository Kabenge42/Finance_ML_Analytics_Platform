"""PDF reporting utilities (facade).

Target-architecture module exposing PDF report generation helpers. Re-exports
implementations from ``finance_ml.ml_workflow.analytics.eval`` for stability
during decomposition of the monster eval module.
"""

from __future__ import annotations

from finance_ml.ml_workflow.analytics.eval import (  # noqa: E402
    generate_pdf_report,
    generate_enhanced_pdf_report,
)

__all__ = [
    "generate_pdf_report",
    "generate_enhanced_pdf_report",
]
