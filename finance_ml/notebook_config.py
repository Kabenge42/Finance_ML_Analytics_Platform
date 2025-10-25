"""
Notebook Configuration Management

Provides feature flag management for Jupyter notebook workflows.
Supports runtime configuration overrides and feature status checks.
"""


class NotebookConfig:
    """Configuration class for notebook feature flags and runtime settings.

    Attributes:
        have_finance_prediction: Enable financial prediction models
        have_database_connection: Enable database connectivity
        have_advanced_analytics: Enable advanced analytics features
        have_dim_reduction: Enable dimensionality reduction
        debug_mode: Enable debug mode with verbose output
        enable_sector_analysis: Enable sector-level analysis
        enable_region_analysis: Enable region-level analysis
        enable_interactive_plots: Enable interactive visualizations
        enable_excel_export: Enable Excel export functionality
    """

    def __init__(
        self,
        have_finance_prediction: bool = True,
        have_database_connection: bool = False,
        have_advanced_analytics: bool = True,
        have_dim_reduction: bool = False,
        debug_mode: bool = False,
        enable_sector_analysis: bool = True,
        enable_region_analysis: bool = True,
        enable_interactive_plots: bool = True,
        enable_excel_export: bool = True,
    ):
        """Initialize notebook configuration with feature flags.

        Args:
            have_finance_prediction: Enable financial prediction models (default: True)
            have_database_connection: Enable database connectivity (default: False)
            have_advanced_analytics: Enable advanced analytics (default: True)
            have_dim_reduction: Enable dimensionality reduction (default: False)
            debug_mode: Enable debug mode (default: False)
            enable_sector_analysis: Enable sector analysis (default: True)
            enable_region_analysis: Enable region analysis (default: True)
            enable_interactive_plots: Enable interactive plots (default: True)
            enable_excel_export: Enable Excel export (default: True)
        """
        self.have_finance_prediction = have_finance_prediction
        self.have_database_connection = have_database_connection
        self.have_advanced_analytics = have_advanced_analytics
        self.have_dim_reduction = have_dim_reduction
        self.debug_mode = debug_mode
        self.enable_sector_analysis = enable_sector_analysis
        self.enable_region_analysis = enable_region_analysis
        self.enable_interactive_plots = enable_interactive_plots
        self.enable_excel_export = enable_excel_export

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled.

        Args:
            feature_name: Name of the feature attribute to check

        Returns:
            bool: True if feature is enabled, False otherwise (including unknown features)
        """
        return getattr(self, feature_name, False)

    def display_summary(self) -> None:
        """Display formatted configuration summary to stdout."""
        print("=" * 60)
        print("FEATURE FLAGS CONFIGURATION")
        print("=" * 60)
        print(
            f"Financial Prediction:        {'✓ Enabled' if self.have_finance_prediction else '✗ Disabled'}"
        )
        print(
            f"Database Connection:         {'✓ Enabled' if self.have_database_connection else '✗ Disabled'}"
        )
        print(
            f"Advanced Analytics:          {'✓ Enabled' if self.have_advanced_analytics else '✗ Disabled'}"
        )
        print(
            f"Dimensionality Reduction:    {'✓ Enabled' if self.have_dim_reduction else '✗ Disabled'}"
        )
        print(f"Debug Mode:                  {'✓ Enabled' if self.debug_mode else '✗ Disabled'}")
        print(
            f"Sector Analysis:             {'✓ Enabled' if self.enable_sector_analysis else '✗ Disabled'}"
        )
        print(
            f"Region Analysis:             {'✓ Enabled' if self.enable_region_analysis else '✗ Disabled'}"
        )
        print(
            f"Interactive Plots:           {'✓ Enabled' if self.enable_interactive_plots else '✗ Disabled'}"
        )
        print(
            f"Excel Export:                {'✓ Enabled' if self.enable_excel_export else '✗ Disabled'}"
        )
        print("=" * 60)
