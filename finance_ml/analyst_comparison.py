"""
Phase 9.8: Prediction vs. Analyst Price Target Analytics

Compare ML model predictions against analyst consensus targets to identify
opportunities where the model has a different view than analysts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

__all__ = [
    "compare_prediction_vs_analyst_targets",
    "calculate_agreement_rate",
    "calculate_directional_accuracy",
    "analyze_systematic_bias",
    "identify_disagreement_opportunities",
    "segment_comparison_by_attribute",
    "generate_prediction_analyst_excel_report",
    "PredictionAnalystAnalytics",
]


def compare_prediction_vs_analyst_targets(stocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model predictions vs analyst consensus price targets.

    Args:
        stocks_df: DataFrame with columns: predicted_price_target, price_target, last_price

    Returns:
        DataFrame with comparison metrics added
    """
    df = stocks_df.copy()

    # Validate required columns
    required = ["predicted_price_target", "price_target", "last_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate differences
    df["model_analyst_diff"] = df["predicted_price_target"] - df["price_target"]
    df["model_analyst_diff_pct"] = df["model_analyst_diff"] / df["price_target"] * 100

    # Direction agreement
    df["model_direction"] = np.where(df["predicted_price_target"] > df["last_price"], "up", "down")
    df["analyst_direction"] = np.where(df["price_target"] > df["last_price"], "up", "down")
    df["agreement_direction"] = df["model_direction"] == df["analyst_direction"]

    return df


def calculate_agreement_rate(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate agreement rate between model and analysts."""
    same_direction = comparison_df["agreement_direction"].sum()
    total = len(comparison_df)

    return {
        "agreement_rate": same_direction / total if total > 0 else 0,
        "same_direction_count": int(same_direction),
        "total_count": int(total),
    }


def calculate_directional_accuracy(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate model directional accuracy vs current price."""
    correct = (comparison_df["model_direction"] == comparison_df["analyst_direction"]).sum()
    total = len(comparison_df)

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct_predictions": int(correct),
        "total_predictions": int(total),
    }


def analyze_systematic_bias(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze systematic bias between model and analyst predictions."""
    diff = comparison_df["model_analyst_diff"].dropna()

    mean_bias = diff.mean()
    median_bias = diff.median()

    return {
        "mean_model_bias": float(mean_bias),
        "median_model_bias": float(median_bias),
        "bias_direction": "bullish" if mean_bias > 0 else "bearish" if mean_bias < 0 else "neutral",
    }


def identify_disagreement_opportunities(
    comparison_df: pd.DataFrame, threshold_pct: float = 10.0
) -> pd.DataFrame:
    """
    Identify stocks where model significantly disagrees with analysts.

    Args:
        comparison_df: Comparison DataFrame
        threshold_pct: Minimum percentage difference (default: 10%)

    Returns:
        DataFrame with high-conviction disagreement opportunities
    """
    mask = comparison_df["model_analyst_diff_pct"].abs() > threshold_pct
    opportunities = comparison_df[mask].copy()

    # Sort by absolute difference
    opportunities["abs_diff_pct"] = opportunities["model_analyst_diff_pct"].abs()
    opportunities = opportunities.sort_values("abs_diff_pct", ascending=False)
    opportunities.drop("abs_diff_pct", axis=1, inplace=True)

    return opportunities


def segment_comparison_by_attribute(
    comparison_df: pd.DataFrame, segment_col: str
) -> Dict[str, Dict[str, Any]]:
    """
    Segment comparison metrics by attribute (sector, region, etc.).

    Args:
        comparison_df: Comparison DataFrame
        segment_col: Column to segment by

    Returns:
        Dictionary with metrics per segment
    """
    if segment_col not in comparison_df.columns:
        return {}

    results = {}
    for segment_name, segment_df in comparison_df.groupby(segment_col):
        agreement = calculate_agreement_rate(segment_df)

        results[segment_name] = {
            "count": len(segment_df),
            "agreement_rate": agreement["agreement_rate"],
            "avg_model_analyst_diff": float(segment_df["model_analyst_diff"].mean()),
        }

    return results


def generate_prediction_analyst_excel_report(
    comparison_df: pd.DataFrame, output_path: Path, top_n_opportunities: int = 50
) -> None:
    """
    Generate comprehensive Excel report comparing predictions vs analyst targets.

    Args:
        comparison_df: Comparison DataFrame
        output_path: Path to save Excel file
        top_n_opportunities: Number of top opportunities to include
    """
    try:
        import xlsxwriter
    except ImportError:
        raise ImportError(
            "xlsxwriter required for Excel export. Install with: pip install xlsxwriter"
        )

    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

    # Sheet 1: Executive Summary
    summary_data = {
        "Metric": [
            "Total Stocks",
            "Avg Predicted Target",
            "Avg Analyst Target",
            "Agreement Rate",
            "Mean Model-Analyst Diff",
        ],
        "Value": [
            len(comparison_df),
            comparison_df["predicted_price_target"].mean(),
            comparison_df["price_target"].mean(),
            calculate_agreement_rate(comparison_df)["agreement_rate"],
            comparison_df["model_analyst_diff"].mean(),
        ],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Executive_Summary", index=False)

    # Sheet 2: Detailed Stock List
    detail_cols = [
        "ticker",
        "sector",
        "region",
        "last_price",
        "predicted_price_target",
        "price_target",
        "model_analyst_diff",
        "model_analyst_diff_pct",
        "agreement_direction",
    ]
    available_cols = [c for c in detail_cols if c in comparison_df.columns]
    comparison_df[available_cols].to_excel(writer, sheet_name="Detailed_Stock_List", index=False)

    # Sheet 3: Top Opportunities (Undervalued)
    undervalued = comparison_df[comparison_df["model_analyst_diff_pct"] > 0].nlargest(
        top_n_opportunities, "model_analyst_diff_pct"
    )
    undervalued[available_cols].to_excel(writer, sheet_name="Top_Opportunities", index=False)

    # Sheet 4: Risk Analysis (Overvalued)
    overvalued = comparison_df[comparison_df["model_analyst_diff_pct"] < 0].nsmallest(
        top_n_opportunities, "model_analyst_diff_pct"
    )
    overvalued[available_cols].to_excel(writer, sheet_name="Risk_Analysis", index=False)

    # Sheet 5: Prediction Accuracy
    accuracy_data = {
        "Agreement Rate": [calculate_agreement_rate(comparison_df)["agreement_rate"]],
        "Directional Accuracy": [calculate_directional_accuracy(comparison_df)["accuracy"]],
        "Mean Bias": [analyze_systematic_bias(comparison_df)["mean_model_bias"]],
        "Median Bias": [analyze_systematic_bias(comparison_df)["median_model_bias"]],
    }
    pd.DataFrame(accuracy_data).to_excel(writer, sheet_name="Prediction_Accuracy", index=False)

    # Sheet 6: Sector Analysis
    if "sector" in comparison_df.columns:
        sector_metrics = segment_comparison_by_attribute(comparison_df, "sector")
        sector_df = pd.DataFrame(sector_metrics).T.reset_index()
        sector_df.columns = ["Sector", "Count", "Agreement Rate", "Avg Model-Analyst Diff"]
        sector_df.to_excel(writer, sheet_name="Sector_Analysis", index=False)

    writer.close()


class PredictionAnalystAnalytics:
    """
    Phase 9.8 Analytics: Compare model predictions vs analyst consensus targets.

    This class encapsulates all functionality for analyzing and reporting on the
    differences between ML model predictions and analyst consensus price targets.
    """

    def __init__(self, stocks_df: pd.DataFrame, config=None):
        """
        Initialize analytics with stock data.

        Args:
            stocks_df: DataFrame with stock data including predictions
            config: Configuration object (optional)
        """
        self.stocks_df = stocks_df
        self.config = config
        self.comparison_df = None
        self.agreement_metrics = None
        self.directional_metrics = None
        self.bias_metrics = None
        self.disagreements = None

    def print_header(self):
        """Print Phase 9.8 section header."""
        print("\n" + "=" * 80)
        print("PHASE 9.8 — PREDICTION VS. ANALYST PRICE TARGET ANALYTICS")
        print("=" * 80)

    def prepare_analyst_data(self):
        """Prepare and validate analyst data for comparison."""
        print("\n📊 Preparing Analyst Data for Comparison...")

        # Check required columns
        required_cols = ["predicted_price_target", "price_target", "last_price"]
        missing = [col for col in required_cols if col not in self.stocks_df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter to rows with valid data
        self.stocks_df = self.stocks_df.dropna(subset=required_cols)
        print(f"  ✓ {len(self.stocks_df)} stocks with valid prediction and analyst data")

    def perform_comparison(self):
        """Perform prediction vs analyst comparison."""
        print("\n🔍 Comparing Model Predictions vs Analyst Targets...")

        self.comparison_df = compare_prediction_vs_analyst_targets(self.stocks_df)

        # Display summary statistics
        print(
            f"\n  Avg Model Prediction: ${self.comparison_df['predicted_price_target'].mean():.2f}"
        )
        print(f"  Avg Analyst Target:   ${self.comparison_df['price_target'].mean():.2f}")
        print(f"  Avg Difference:       ${self.comparison_df['model_analyst_diff'].mean():+.2f}")
        print(
            f"  Avg % Difference:     {self.comparison_df['model_analyst_diff_pct'].mean():+.2f}%"
        )

    def analyze_agreement(self):
        """Analyze agreement and directional accuracy."""
        print("\n📈 Analyzing Model-Analyst Agreement...")

        # Calculate agreement rate
        self.agreement_metrics = calculate_agreement_rate(self.comparison_df)
        print(f"\n  Directional Agreement Rate: {self.agreement_metrics['agreement_rate']:.1%}")
        print(
            f"  Same Direction: {self.agreement_metrics['same_direction_count']} of {self.agreement_metrics['total_count']} stocks"
        )

        # Calculate directional accuracy
        self.directional_metrics = calculate_directional_accuracy(self.comparison_df)
        print(f"\n  Directional Accuracy: {self.directional_metrics['accuracy']:.1%}")

        # Analyze systematic bias
        self.bias_metrics = analyze_systematic_bias(self.comparison_df)
        print(f"\n  Systematic Bias:")
        print(f"    Mean Bias:     ${self.bias_metrics['mean_model_bias']:+.2f}")
        print(f"    Median Bias:   ${self.bias_metrics['median_model_bias']:+.2f}")
        print(f"    Direction:     {self.bias_metrics['bias_direction']}")

    def identify_opportunities(self, threshold_pct: float = 10.0):
        """
        Identify high-conviction disagreement opportunities.

        Args:
            threshold_pct: Minimum percentage difference to flag as opportunity

        Returns:
            DataFrame with high-conviction disagreement opportunities
        """
        print(f"\n🎯 Identifying High-Conviction Disagreement Opportunities (>{threshold_pct}%)...")

        self.disagreements = identify_disagreement_opportunities(
            self.comparison_df, threshold_pct=threshold_pct
        )
        print(
            f"\n  Found {len(self.disagreements)} stocks with >{threshold_pct}% model-analyst difference"
        )

        if len(self.disagreements) > 0:
            print("\n  Top 10 Disagreement Opportunities:")
            opp_cols = [
                "ticker",
                "sector",
                "last_price",
                "predicted_price_target",
                "price_target",
                "model_analyst_diff_pct",
            ]
            available_opp_cols = [col for col in opp_cols if col in self.disagreements.columns]
            print(self.disagreements[available_opp_cols].head(10).to_string(index=False))

        return self.disagreements

    def segment_analysis(self):
        """Perform segmented analysis by sector and region."""
        print("\n📊 Segmented Analysis — By Sector & Region...")

        # By Sector
        if "sector" in self.comparison_df.columns:
            by_sector = segment_comparison_by_attribute(self.comparison_df, segment_col="sector")
            print("\n  Agreement Rate by Sector:")
            for sector, metrics in sorted(
                by_sector.items(), key=lambda x: x[1]["agreement_rate"], reverse=True
            ):
                print(
                    f"    {sector:20s}: {metrics['agreement_rate']:6.1%} "
                    f"({metrics['count']} stocks, "
                    f"avg diff: ${metrics['avg_model_analyst_diff']:+.2f})"
                )

        # By Region
        if "region" in self.comparison_df.columns:
            by_region = segment_comparison_by_attribute(self.comparison_df, segment_col="region")
            print("\n  Agreement Rate by Region:")
            for region, metrics in sorted(
                by_region.items(), key=lambda x: x[1]["agreement_rate"], reverse=True
            ):
                print(
                    f"    {region:20s}: {metrics['agreement_rate']:6.1%} "
                    f"({metrics['count']} stocks, "
                    f"avg diff: ${metrics['avg_model_analyst_diff']:+.2f})"
                )

    def generate_report(self, top_n_opportunities: int = 50):
        """
        Generate comprehensive Excel report.

        Args:
            top_n_opportunities: Number of top opportunities to include (default: 50)
        """
        print("\n📊 Generating Comprehensive Excel Report...")
        try:
            report_path = self.config.output_dir / "prediction_analyst_comparison_report.xlsx"
            generate_prediction_analyst_excel_report(
                self.comparison_df, report_path, top_n_opportunities=top_n_opportunities
            )
            print(f"  ✓ Comprehensive report saved: {report_path}")
            print("\n  Report includes 6 sheets:")
            print("    1. Executive_Summary - Overall statistics and performance")
            print("    2. Detailed_Stock_List - All stocks with predictions and targets")
            print("    3. Top_Opportunities - Top 50 undervalued stocks")
            print("    4. Risk_Analysis - Top 50 overvalued stocks")
            print("    5. Prediction_Accuracy - Model vs analyst comparison metrics")
            print("    6. Sector_Analysis - Performance breakdown by sector")
        except Exception as e:
            print(f"  ⚠ Failed to generate report: {str(e)}")

    def run_full_analysis(
        self, disagreement_threshold: float = 10.0, top_n: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Execute complete Phase 9.8 analytics pipeline.

        Args:
            disagreement_threshold: Percentage threshold for disagreement opportunities
            top_n: Number of top opportunities for report

        Returns:
            Dictionary with all analytics results, or None if error
        """
        self.print_header()

        try:
            # Prepare data
            self.prepare_analyst_data()

            # Perform comparison
            self.perform_comparison()

            # Analyze agreement and accuracy
            self.analyze_agreement()

            # Identify opportunities
            self.identify_opportunities(threshold_pct=disagreement_threshold)

            # Segmented analysis
            self.segment_analysis()

            # Generate report
            self.generate_report(top_n_opportunities=top_n)

            print("\n✓ PHASE 9.8 COMPLETE — PREDICTION VS. ANALYST ANALYTICS")
            print(
                "✓ ALL PHASES COMPLETE — END-TO-END ML ANALYTICS PLATFORM WITH ANALYST COMPARISON"
            )

            return {
                "comparison_df": self.comparison_df,
                "agreement_metrics": self.agreement_metrics,
                "directional_metrics": self.directional_metrics,
                "bias_metrics": self.bias_metrics,
                "disagreements": self.disagreements,
            }

        except Exception as e:
            print(f"\n❌ Error in Phase 9.8: {str(e)}")
            import traceback

            traceback.print_exc()
            return None
