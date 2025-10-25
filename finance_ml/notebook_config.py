"""
Notebook Configuration Management

Encapsulates notebook feature flags and consolidated configuration handling.
This refactoring provides a dataclass with clear naming, formatted display,
and a feature query API. It also supports attaching the package config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NotebookConfig:
    """Configuration for notebook features and capabilities.

    This class centralizes notebook feature flags and allows attaching the
    finance_ml package configuration for consolidated access.

    Supports both `enable_*` and `have_*` naming conventions for compatibility.
    """

    # Prediction and modeling capabilities
    have_finance_prediction: bool = True

    # Database connectivity
    have_database_connection: bool = False

    # Advanced analytics and visualizations
    have_advanced_analytics: bool = True

    # Dimensionality reduction visualizations
    have_dim_reduction: bool = False

    # Debug and development features
    debug_mode: bool = False

    # Optional feature toggles
    enable_sector_analysis: bool = True
    enable_region_analysis: bool = True
    enable_interactive_plots: bool = True
    enable_excel_export: bool = True

    # Package configuration (loaded later)
    finance_ml_config: Optional[object] = None

    def display_summary(self) -> None:
        """Display configuration summary in formatted output."""
        print("=" * 80)
        print("FEATURE FLAGS CONFIGURATION")
        print("=" * 80)
        print("\nCore Features:")
        print(f"  Financial Prediction:        {self._format_status(self.have_finance_prediction)}")
        print(
            f"  Database Connection:         {self._format_status(self.have_database_connection)}"
        )
        print(f"  Advanced Analytics:          {self._format_status(self.have_advanced_analytics)}")
        print(f"  Dimensionality Reduction:    {self._format_status(self.have_dim_reduction)}")

        print("\nAnalysis Features:")
        print(f"  Sector Analysis:             {self._format_status(self.enable_sector_analysis)}")
        print(f"  Region Analysis:             {self._format_status(self.enable_region_analysis)}")

        print("\nOutput Features:")
        print(
            f"  Interactive Plots:           {self._format_status(self.enable_interactive_plots)}"
        )
        print(f"  Excel Export:                {self._format_status(self.enable_excel_export)}")

        print("\nDevelopment:")
        print(f"  Debug Mode:                  {self._format_status(self.debug_mode)}")
        print("=" * 80)

    @staticmethod
    def _format_status(enabled: bool) -> str:
        """Format boolean status with visual indicator."""
        return "✓ Enabled" if enabled else "✗ Disabled"

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific feature is enabled.

        Supports both short names and full attribute names.

        Args:
            feature_name: Feature name to check (case-insensitive).
                         Examples: 'prediction', 'have_finance_prediction',
                                   'enable_sector_analysis', etc.

        Returns:
            True if feature is enabled, False otherwise.
        """
        # Normalize feature name
        normalized = feature_name.lower().replace("_", "")

        # Build feature map from actual attributes
        feature_map = {
            # Short names
            "prediction": self.have_finance_prediction,
            "financeprediction": self.have_finance_prediction,
            "database": self.have_database_connection,
            "databaseconnection": self.have_database_connection,
            "analytics": self.have_advanced_analytics,
            "advancedanalytics": self.have_advanced_analytics,
            "dimreduction": self.have_dim_reduction,
            "sector": self.enable_sector_analysis,
            "sectoranalysis": self.enable_sector_analysis,
            "region": self.enable_region_analysis,
            "regionanalysis": self.enable_region_analysis,
            "interactive": self.enable_interactive_plots,
            "interactiveplots": self.enable_interactive_plots,
            "excel": self.enable_excel_export,
            "excelexport": self.enable_excel_export,
            "debug": self.debug_mode,
            "debugmode": self.debug_mode,
            # Full attribute names (normalized)
            "havefinanceprediction": self.have_finance_prediction,
            "havedatabaseconnection": self.have_database_connection,
            "haveadvancedanalytics": self.have_advanced_analytics,
            "havedimreduction": self.have_dim_reduction,
            "enablesectoranalysis": self.enable_sector_analysis,
            "enableregionanalysis": self.enable_region_analysis,
            "enableinteractiveplots": self.enable_interactive_plots,
            "enableexcelexport": self.enable_excel_export,
        }
        return feature_map.get(normalized, False)
