"""Interactive Dashboards for Finance ML Analytics Platform.

This package provides interactive dashboard applications and helper
widgets/figures used in Section 10 of the main notebook.

Dash applications:
    - :mod:`dash_app` – Plotly Dash-based dashboard
    - :mod:`streamlit_app` – Streamlit-based dashboard

Phase 6 of the portfolio optimisation enhancement plan introduces
additional helpers exposed at the package level:

    - :class:`PortfolioRebalanceWidget`
    - :func:`create_multi_period_comparison`
    - :func:`create_factor_exposure_dashboard`
"""

from .portfolio_widgets import (  # noqa: F401
    PortfolioRebalanceWidget,
    create_multi_period_comparison,
    create_factor_exposure_dashboard,
)

__all__ = [
    "PortfolioRebalanceWidget",
    "create_multi_period_comparison",
    "create_factor_exposure_dashboard",
]

__version__ = "1.1.0"
