# Configure matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments

"""
Finance ML Evaluation Module

Evaluation, analytics, and visualization functions for model results.

Phase 7 TDD refactoring: Extracted from ml_finance_model_v8_2.py with
comprehensive test coverage.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

# Optional imports for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


def calculate_mispricing_score(df: pd.DataFrame) -> pd.Series:
    """Calculate mispricing score for each stock.

    Formula: (predicted_target - last_price) / last_price

    Positive score = undervalued (predicted > current)
    Negative score = overvalued (predicted < current)

    Args:
        df: DataFrame with 'predicted_target' and 'last_price' columns

    Returns:
        Series with mispricing scores
    """
    score = (df["predicted_target"] - df["last_price"]) / df["last_price"]
    return score


def rank_undervalued_stocks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank and return top N most undervalued stocks.

    Args:
        df: DataFrame with 'mispricing_score' column
        top_n: Number of top stocks to return

    Returns:
        DataFrame sorted by mispricing_score descending (most undervalued first)
    """
    sorted_df = df.sort_values("mispricing_score", ascending=False)
    return sorted_df.head(top_n)


def rank_overvalued_stocks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank and return top N most overvalued stocks.

    Args:
        df: DataFrame with 'mispricing_score' column
        top_n: Number of top stocks to return

    Returns:
        DataFrame sorted by mispricing_score ascending (most overvalued first)
    """
    sorted_df = df.sort_values("mispricing_score", ascending=True)
    return sorted_df.head(top_n)


def rank_stocks_by_sector(
    df: pd.DataFrame, top_n: int = 5, order: str = "undervalued"
) -> Dict[str, pd.DataFrame]:
    """Rank stocks within each sector.

    Args:
        df: DataFrame with 'sector' and 'mispricing_score' columns
        top_n: Number of top stocks per sector
        order: 'undervalued' (descending score) or 'overvalued' (ascending score)

    Returns:
        Dict with sector names as keys and ranked DataFrames as values
    """
    result = {}
    ascending = order == "overvalued"

    for sector, group in df.groupby("sector"):
        sorted_group = group.sort_values("mispricing_score", ascending=ascending)
        result[sector] = sorted_group.head(top_n)

    return result


def simple_eda(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    save_plots: bool = False,
) -> dict:
    """Perform exploratory data analysis.

    When out_dir is provided, write eda_summary.json (and optional plots) to disk.
    Always return the computed summary as a dictionary for programmatic use.

    Args:
        df: DataFrame to analyze
        out_dir: Optional directory to save output files. If None, files are not written.
        save_plots: If True and out_dir is provided, generate and save matplotlib visualizations

    Returns:
        A dictionary with EDA summary statistics.
    """
    logging.info("Rows: %d, Columns: %d", df.shape[0], df.shape[1])
    # Robust dtype handling: some objects may raise AttributeError on dtype access
    try:
        numeric_cols = [c for c in df.columns if getattr(df[c], "dtype", object) != object]
    except AttributeError:
        # Fallback: treat no columns as numeric if dtype access fails
        logging.warning(
            "simple_eda: dtype inspection failed due to AttributeError; skipping numeric stats"
        )
        numeric_cols = []
    except Exception as e:
        logging.warning("simple_eda: dtype inspection failed: %s", e)
        numeric_cols = []

    try:
        basic_stats = df[numeric_cols].describe().to_dict() if numeric_cols else {}
        numeric_count = int((df.dtypes != object).sum())
        categorical_count = int((df.dtypes == object).sum())
    except AttributeError:
        basic_stats = {}
        numeric_count = 0
        categorical_count = 0
    except Exception as e:
        logging.warning("simple_eda: basic stats computation failed: %s", e)
        basic_stats = {}
        numeric_count = 0
        categorical_count = 0

    # Safe value_counts extraction
    def _safe_counts(series_name: str):
        try:
            return df[series_name].value_counts().to_dict()
        except Exception:
            return {}

    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols_count": numeric_count,
        "categorical_cols_count": categorical_count,
        "null_counts": df.isnull().sum().to_dict(),
        "region_counts": _safe_counts("region") if "region" in df.columns else {},
        "sector_counts": _safe_counts("sector") if "sector" in df.columns else {},
        "basic_stats": basic_stats,
    }

    # Enhanced statistical analysis (Phase 9.2)
    # Only compute if we have sufficient numeric data
    if numeric_cols and len(df) >= 3:
        try:
            # Distribution analysis: skewness and kurtosis
            skew_kurt_df = calculate_skewness_kurtosis(df, numeric_cols)
            # Convert DataFrame to dict for JSON serialization
            summary["distribution_analysis"] = (
                skew_kurt_df.to_dict(orient="index") if not skew_kurt_df.empty else {}
            )
        except Exception as e:
            logging.warning("Distribution analysis failed: %s", e)
            summary["distribution_analysis"] = {}

        try:
            # Outlier detection (using IQR method as default for robustness)
            outliers = {}
            for col in numeric_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) >= 3:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                        outliers[col] = {
                            "count": int(outlier_mask.sum()),
                            "percentage": float(outlier_mask.sum() / len(col_data) * 100),
                        }
                except Exception:
                    outliers[col] = {"count": 0, "percentage": 0.0}
            summary["outlier_detection"] = outliers
        except Exception as e:
            logging.warning("Outlier detection failed: %s", e)
            summary["outlier_detection"] = {}

        try:
            # Normality tests (only if enough samples)
            if len(df) >= 8:
                normality = test_normality(df, numeric_cols, alpha=0.05)
                # Convert numpy bool to Python bool for JSON serialization
                for col, result in normality.items():
                    if isinstance(result, dict) and "is_normal" in result:
                        if result["is_normal"] is not None:
                            result["is_normal"] = bool(result["is_normal"])
                summary["normality_tests"] = normality
            else:
                summary["normality_tests"] = {}
        except Exception as e:
            logging.warning("Normality tests failed: %s", e)
            summary["normality_tests"] = {}

        try:
            # Correlation analysis (Pearson and Spearman)
            if len(numeric_cols) >= 2:
                corr_analysis = {}
                pearson_corr = calculate_correlation_matrix(df, numeric_cols, method="pearson")
                spearman_corr = calculate_correlation_matrix(df, numeric_cols, method="spearman")
                # Convert DataFrames to dicts for JSON serialization
                corr_analysis["pearson"] = pearson_corr.to_dict() if not pearson_corr.empty else {}
                corr_analysis["spearman"] = (
                    spearman_corr.to_dict() if not spearman_corr.empty else {}
                )
                summary["correlation_analysis"] = corr_analysis
            else:
                summary["correlation_analysis"] = {}
        except Exception as e:
            logging.warning("Correlation analysis failed: %s", e)
            summary["correlation_analysis"] = {}

        try:
            # Sector-wise statistics
            if "sector" in df.columns and df["sector"].notna().any():
                sector_stats = {}
                for sector in df["sector"].dropna().unique():
                    sector_df = df[df["sector"] == sector]
                    sector_numeric = [c for c in numeric_cols if c in sector_df.columns]
                    if sector_numeric and len(sector_df) > 0:
                        sector_stats[sector] = {
                            "count": int(len(sector_df)),
                            "means": sector_df[sector_numeric].mean().to_dict(),
                            "medians": sector_df[sector_numeric].median().to_dict(),
                            "stds": sector_df[sector_numeric].std().to_dict(),
                        }
                summary["sector_statistics"] = sector_stats
            else:
                summary["sector_statistics"] = {}
        except Exception as e:
            logging.warning("Sector statistics failed: %s", e)
            summary["sector_statistics"] = {}
    else:
        # Insufficient data for advanced analysis
        summary["distribution_analysis"] = {}
        summary["outlier_detection"] = {}
        summary["normality_tests"] = {}
        summary["correlation_analysis"] = {}
        summary["sector_statistics"] = {}

    # Persist summary and plots only if an output directory is provided
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If out_dir cannot be created, continue without writing files
            logging.warning("Could not create out_dir=%s; skipping file outputs", out_dir)
        else:
            out_path = out_dir / "eda_summary.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f)
            logging.info("Wrote EDA summary to %s", out_path)

            # Generate visualizations if requested
            if save_plots and numeric_cols:
                if plt is None or sns is None:
                    logging.warning("Matplotlib/seaborn not available for plots")
                else:
                    try:
                        # Create distribution plots for key numeric features
                        n_cols = min(len(numeric_cols), 6)
                        if n_cols > 0:
                            fig, axes = plt.subplots(
                                nrows=(n_cols + 2) // 3,
                                ncols=3,
                                figsize=(15, 5 * ((n_cols + 2) // 3)),
                            )
                            if n_cols == 1:
                                axes = [axes]
                            else:
                                axes = axes.flatten() if hasattr(axes, "flatten") else axes

                            for idx, col in enumerate(numeric_cols[:n_cols]):
                                ax = axes[idx] if n_cols > 1 else axes[0]
                                df[col].hist(bins=30, ax=ax, edgecolor="black")
                                ax.set_title(f"Distribution of {col}")
                                ax.set_xlabel(col)
                                ax.set_ylabel("Frequency")

                            # Hide unused subplots
                            for idx in range(n_cols, len(axes)):
                                axes[idx].set_visible(False)

                            plt.tight_layout()
                            dist_plot_path = out_dir / "eda_distributions.png"
                            plt.savefig(dist_plot_path, dpi=100, bbox_inches="tight")
                            plt.close()
                            logging.info("Saved distribution plots to %s", dist_plot_path)

                        # Create correlation heatmap
                        if len(numeric_cols) >= 2:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            corr_matrix = df[numeric_cols].corr()
                            sns.heatmap(
                                corr_matrix,
                                annot=True,
                                fmt=".2f",
                                cmap="coolwarm",
                                center=0,
                                ax=ax,
                                square=True,
                                linewidths=1,
                            )
                            ax.set_title("Correlation Heatmap")
                            plt.tight_layout()
                            corr_plot_path = out_dir / "eda_correlation.png"
                            plt.savefig(corr_plot_path, dpi=100, bbox_inches="tight")
                            plt.close()
                            logging.info("Saved correlation heatmap to %s", corr_plot_path)

                    except Exception as e:
                        logging.warning("Error generating visualizations: %s", e)

    return summary


def export_predictions_to_excel(
    df: pd.DataFrame, excel_path: Path, include_summary: bool = False
) -> None:
    """Export predictions and analytics to Excel format.

    Args:
        df: DataFrame with predictions, scores, and stock information
        excel_path: Path to save Excel file
        include_summary: If True, create multiple sheets with summary statistics
    """
    # Try multiple Excel engines
    engines_to_try = ["openpyxl", "xlsxwriter"]
    writer_created = False

    for engine in engines_to_try:
        try:
            with pd.ExcelWriter(excel_path, engine=engine) as writer:
                # Main predictions sheet
                df.to_excel(writer, sheet_name="Predictions", index=False)

                if include_summary:
                    # Summary statistics sheet
                    summary_data = {
                        "Metric": [
                            "Total Stocks",
                            "Average Mispricing Score",
                            "Stocks Undervalued (score > 0)",
                            "Stocks Overvalued (score < 0)",
                            "Stocks Fairly Valued (score ≈ 0)",
                        ],
                        "Value": [
                            len(df),
                            (
                                df["mispricing_score"].mean()
                                if "mispricing_score" in df.columns
                                else "N/A"
                            ),
                            (
                                (df["mispricing_score"] > 0.05).sum()
                                if "mispricing_score" in df.columns
                                else "N/A"
                            ),
                            (
                                (df["mispricing_score"] < -0.05).sum()
                                if "mispricing_score" in df.columns
                                else "N/A"
                            ),
                            (
                                (df["mispricing_score"].abs() <= 0.05).sum()
                                if "mispricing_score" in df.columns
                                else "N/A"
                            ),
                        ],
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)

                    # Per-sector breakdown
                    if "sector" in df.columns:
                        sector_summary = (
                            df.groupby("sector")
                            .agg(
                                {
                                    "ticker": "count",
                                    "mispricing_score": (
                                        ["mean", "min", "max"]
                                        if "mispricing_score" in df.columns
                                        else "count"
                                    ),
                                }
                            )
                            .reset_index()
                        )
                        sector_summary.columns = [
                            "_".join(col).strip("_") for col in sector_summary.columns.values
                        ]
                        sector_summary.to_excel(writer, sheet_name="By_Sector", index=False)

            logging.info("Exported predictions to Excel using %s engine: %s", engine, excel_path)
            writer_created = True
            break

        except (ImportError, ModuleNotFoundError):
            logging.debug("%s engine not available, trying next...", engine)
            continue
        except Exception as e:
            logging.error("Error exporting to Excel with %s engine: %s", engine, e)
            raise

    if not writer_created:
        logging.warning("No Excel engine available (openpyxl or xlsxwriter).")
        raise ImportError(
            "No Excel engine available. Install openpyxl or xlsxwriter: pip install openpyxl"
        )


def create_sector_heatmap(
    df: pd.DataFrame, out_path: Optional[Path] = None, metric: str = "mispricing_score"
):
    """Create a heatmap visualization showing metrics by sector.

    Args:
        df: DataFrame with sector and metric data
        out_path: Path to save the figure (PNG format)
        metric: Column name to visualize (default: mispricing_score)

    Returns:
        Matplotlib figure object, or None if required columns are missing
    """
    # Check if required columns exist FIRST (before checking library availability)
    if "sector" not in df.columns or metric not in df.columns:
        logging.error(
            f"Error creating sector heatmap: 'Column not found: {metric if metric not in df.columns else 'sector'}'"
        )
        return None

    # Filter out rows with null sector or metric values
    df_clean = df[df["sector"].notna() & df[metric].notna()].copy()
    if df_clean.empty:
        logging.warning("Sector heatmap skipped: no non-null data for sector and %s", metric)
        return None

    # Now check if visualization libraries are available
    if plt is None or sns is None:
        raise ImportError("Matplotlib and seaborn required for heatmap visualization")

    try:

        # Aggregate metric by sector
        sector_stats = (
            df_clean.groupby("sector")[metric].agg(["mean", "median", "std", "count"]).reset_index()
        )

        if sector_stats.empty:
            logging.warning("Sector heatmap skipped: no data after aggregation")
            return None

        # Create pivot for heatmap
        pivot_data = sector_stats.set_index("sector")[["mean", "median", "std"]]
        if pivot_data.empty:
            logging.warning("Sector heatmap skipped: pivot is empty")
            return None

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(sector_stats) * 0.5)))
        sns.heatmap(
            pivot_data.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"{metric} by Sector")
        ax.set_xlabel("Sector")
        ax.set_ylabel("Statistic")
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            logging.info("Saved sector heatmap to %s", out_path)

        return fig

    except Exception as e:
        logging.error("Error creating sector heatmap: %s", e)
        raise


def create_interactive_prediction_plot(df: pd.DataFrame, out_path: Optional[Path] = None):
    """Create an interactive scatter plot of predictions vs actual prices.

    Args:
        df: DataFrame with predicted_target, last_price, and other columns
        out_path: Path to save the HTML file

    Returns:
        Plotly figure object, or None if required columns are missing
    """
    if px is None:
        raise ImportError("Plotly required for interactive plots")

    try:
        # Check if required columns exist
        if "last_price" not in df.columns or "predicted_target" not in df.columns:
            logging.warning(
                "Missing required columns (last_price, predicted_target) for interactive plot"
            )
            return None

        # Drop rows with NaNs in required columns to avoid zero-size issues
        df_plot = df.dropna(subset=["last_price", "predicted_target"]).copy()
        if df_plot.empty:
            logging.warning("Interactive plot skipped: no valid rows after dropping NaNs")
            return None

        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x="last_price",
            y="predicted_target",
            color="sector" if "sector" in df_plot.columns else None,
            hover_data=["ticker"] if "ticker" in df_plot.columns else None,
            title="Predicted Target vs Current Price",
            labels={"last_price": "Current Price", "predicted_target": "Predicted Target Price"},
        )

        # Add diagonal line (y=x) for reference
        min_val = float(min(df_plot["last_price"].min(), df_plot["predicted_target"].min()))
        max_val = float(max(df_plot["last_price"].max(), df_plot["predicted_target"].max()))
        if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(dash="dash", color="gray"),
                )
            )

        if out_path:
            fig.write_html(str(out_path))
            logging.info("Saved interactive plot to %s", out_path)

        return fig

    except Exception as e:
        logging.error("Error creating interactive plot: %s", e)
        raise


def create_region_sector_heatmap(
    df: pd.DataFrame, metric: str = "mispricing_score", out_path: Optional[Path] = None
):
    """Create a heatmap showing metrics by region and sector.

    Args:
        df: DataFrame with region, sector, and metric columns
        metric: Column name to visualize (default: mispricing_score)
        out_path: Path to save the figure (PNG format)

    Returns:
        Matplotlib figure object, or None if required columns are missing
    """
    if plt is None or sns is None:
        raise ImportError("Matplotlib and seaborn required for heatmap visualization")

    try:
        # Check if required columns exist
        if "region" not in df.columns or "sector" not in df.columns or metric not in df.columns:
            logging.warning("Missing region, sector, or metric columns for heatmap")
            return None

        # Filter out nulls to avoid zero-size or treemap-like errors
        df_clean = df[df["region"].notna() & df["sector"].notna() & df[metric].notna()].copy()
        if df_clean.empty:
            logging.warning("Region-sector heatmap skipped: no non-null data for required fields")
            return None

        # Create pivot table
        pivot_data = df_clean.pivot_table(
            values=metric, index="sector", columns="region", aggfunc="mean"
        )
        if pivot_data.empty or pivot_data.shape[0] == 0 or pivot_data.shape[1] == 0:
            logging.warning("Region-sector heatmap skipped: pivot is empty")
            return None

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.6)))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"{metric} by Region and Sector")
        ax.set_xlabel("Region")
        ax.set_ylabel("Sector")
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            logging.info("Saved region-sector heatmap to %s", out_path)

        return fig

    except Exception as e:
        logging.error("Error creating region-sector heatmap: %s", e)
        raise


# ============================================================================
# Phase 9.2: Advanced EDA and Statistical Analysis Functions
# ============================================================================


def calculate_correlation_matrix(
    df: pd.DataFrame, columns: list, method: str = "pearson"
) -> pd.DataFrame:
    """Calculate correlation matrix using specified method.

    Phase 9.2 enhancement for comprehensive correlation analysis.

    Args:
        df: DataFrame containing the data
        columns: List of columns to include in correlation matrix
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix as DataFrame

    Raises:
        ValueError: If method is not supported
    """
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            f"Method '{method}' not supported. Use 'pearson', 'spearman', or 'kendall'."
        )

    # Select numeric columns
    data = df[columns].select_dtypes(include=[np.number])

    if data.empty:
        raise ValueError("No numeric columns found in specified columns")

    # Calculate correlation
    corr_matrix = data.corr(method=method)

    return corr_matrix


def find_top_correlations(
    df: pd.DataFrame, columns: list, n_top: int = 10, method: str = "pearson"
) -> list:
    """Find top correlated variable pairs.

    Phase 9.2 enhancement for identifying strongest correlations.

    Args:
        df: DataFrame containing the data
        columns: List of columns to analyze
        n_top: Number of top correlations to return
        method: Correlation method

    Returns:
        List of tuples (var1, var2, correlation) sorted by absolute correlation
    """
    corr_matrix = calculate_correlation_matrix(df, columns, method)

    # Get upper triangle (avoid duplicates and self-correlations)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)

    # Convert to list of tuples
    correlations = []
    for i in range(len(upper_tri)):
        for j in range(i + 1, len(upper_tri)):
            var1 = upper_tri.index[i]
            var2 = upper_tri.columns[j]
            corr_value = upper_tri.iloc[i, j]

            if not pd.isna(corr_value):
                correlations.append((var1, var2, corr_value))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    return correlations[:n_top]


def test_normality(df: pd.DataFrame, columns: list, alpha: float = 0.05) -> dict:
    """Test normality of distributions using Shapiro-Wilk test.

    Phase 9.2 enhancement for distribution analysis.

    Args:
        df: DataFrame containing the data
        columns: List of columns to test
        alpha: Significance level for the test

    Returns:
        Dictionary with test results for each column
    """
    from scipy import stats

    results = {}

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if len(data) < 3:
            results[col] = {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "message": "Insufficient data for normality test",
            }
            continue

        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(data)

        results[col] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > alpha,
            "sample_size": len(data),
        }

    return results


def calculate_skewness_kurtosis(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Calculate skewness and kurtosis for specified columns.

    Phase 9.2 enhancement for distribution shape analysis.

    Args:
        df: DataFrame containing the data
        columns: List of columns to analyze

    Returns:
        DataFrame with skewness and kurtosis for each column
    """
    from scipy import stats

    results = []

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if len(data) < 3:
            continue

        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        results.append(
            {"column": col, "skewness": skewness, "kurtosis": kurtosis, "n_samples": len(data)}
        )

    return pd.DataFrame(results).set_index("column")


def detect_outliers_statistical(
    df: pd.DataFrame, columns: list, method: str = "grubbs", alpha: float = 0.05
) -> pd.DataFrame:
    """Detect outliers using statistical methods.

    Phase 9.2 enhancement for advanced outlier detection.

    Args:
        df: DataFrame containing the data
        columns: List of columns to analyze
        method: Detection method ('grubbs', 'modified_z')
        alpha: Significance level

    Returns:
        DataFrame with boolean outlier indicators
    """
    outliers = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if col not in df.columns:
            continue

        data = pd.to_numeric(df[col], errors="coerce")

        if method == "modified_z":
            # Modified Z-score using median absolute deviation
            median = data.median()
            mad = np.median(np.abs(data - median))

            if mad == 0:
                continue

            modified_z_scores = 0.6745 * (data - median) / mad
            outliers[col] = np.abs(modified_z_scores) > 3.5

        elif method == "grubbs":
            # Simplified Grubbs test (flag extreme values)
            mean = data.mean()
            std = data.std()

            if std == 0:
                continue

            z_scores = np.abs((data - mean) / std)
            # Grubbs critical value approximation for large samples
            outliers[col] = z_scores > 3

    return outliers


def calculate_mutual_information(
    X: pd.DataFrame, y: pd.Series, discrete_features: bool = False
) -> pd.Series:
    """Calculate mutual information between features and target.

    Phase 9.2 enhancement for feature importance analysis.

    Args:
        X: Feature DataFrame
        y: Target variable
        discrete_features: Whether features are discrete

    Returns:
        Series with MI scores for each feature
    """
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

    # Handle missing values
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median()) if pd.api.types.is_numeric_dtype(y) else y.fillna("missing")

    # Determine if regression or classification
    if pd.api.types.is_numeric_dtype(y_clean):
        mi_scores = mutual_info_regression(
            X_clean, y_clean, discrete_features=discrete_features, random_state=42
        )
    else:
        mi_scores = mutual_info_classif(
            X_clean, y_clean, discrete_features=discrete_features, random_state=42
        )

    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


def calculate_feature_importance_rf(
    X: pd.DataFrame, y: pd.Series, n_estimators: int = 100
) -> pd.Series:
    """Calculate feature importance using Random Forest.

    Phase 9.2 enhancement for feature importance analysis.

    Args:
        X: Feature DataFrame
        y: Target variable
        n_estimators: Number of trees in the forest

    Returns:
        Series with feature importances
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Handle missing values
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median()) if pd.api.types.is_numeric_dtype(y) else y.fillna("missing")

    # Choose appropriate model
    if pd.api.types.is_numeric_dtype(y_clean):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    # Fit model
    model.fit(X_clean, y_clean)

    # Get importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    return importances


def perform_pca(X: pd.DataFrame, n_components: int = 3) -> dict:
    """Perform PCA dimensionality reduction.

    Phase 9.2 enhancement for multivariate analysis.

    Args:
        X: Feature DataFrame
        n_components: Number of principal components

    Returns:
        Dictionary with PCA results
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Handle missing values
    X_clean = X.fillna(X.median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    # Create result DataFrame
    component_df = pd.DataFrame(
        components, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index
    )

    return {
        "components": component_df,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "feature_loadings": pd.DataFrame(
            pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=X.columns
        ),
    }


def calculate_optimal_pca_components(X: pd.DataFrame, variance_threshold: float = 0.95) -> int:
    """Calculate optimal number of PCA components.

    Phase 9.2 enhancement for determining dimensionality.

    Args:
        X: Feature DataFrame
        variance_threshold: Minimum cumulative variance to retain

    Returns:
        Optimal number of components
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Handle missing values
    X_clean = X.fillna(X.median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Perform PCA with all components
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    # Find number of components for threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    return int(n_components)


def compare_sector_means(
    df: pd.DataFrame,
    column: str,
    group_column: str = "sector",
    method: str = "anova",
    alpha: float = 0.05,
) -> dict:
    """Compare means across sectors using statistical tests.

    Phase 9.2 enhancement for hypothesis testing.

    Args:
        df: DataFrame containing the data
        column: Column to compare
        group_column: Grouping column (e.g., 'sector')
        method: Test method ('anova', 'kruskal')
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    # Get groups
    groups = []
    for group_name in df[group_column].unique():
        if pd.isna(group_name):
            continue

        group_data = df[df[group_column] == group_name][column].dropna()

        if len(group_data) > 0:
            groups.append(group_data)

    if len(groups) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "message": "Insufficient groups for comparison",
        }

    # Perform test
    if method == "anova":
        statistic, p_value = stats.f_oneway(*groups)
    elif method == "kruskal":
        statistic, p_value = stats.kruskal(*groups)
    else:
        raise ValueError(f"Method '{method}' not supported")

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "n_groups": len(groups),
        "method": method,
    }


def compare_two_groups(
    group1: pd.Series, group2: pd.Series, method: str = "ttest", alpha: float = 0.05
) -> dict:
    """Compare two groups using statistical tests.

    Phase 9.2 enhancement for pairwise comparisons.

    Args:
        group1: First group data
        group2: Second group data
        method: Test method ('ttest', 'mannwhitney')
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    # Clean data
    g1 = group1.dropna()
    g2 = group2.dropna()

    if len(g1) < 2 or len(g2) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "message": "Insufficient data for comparison",
        }

    # Perform test
    if method == "ttest":
        statistic, p_value = stats.ttest_ind(g1, g2)
    elif method == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(g1, g2)
    else:
        raise ValueError(f"Method '{method}' not supported")

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "n1": len(g1),
        "n2": len(g2),
        "method": method,
    }


def generate_eda_report(
    df: pd.DataFrame,
    output_path: Path = None,
    include_correlations: bool = True,
    include_distributions: bool = True,
    include_statistical_tests: bool = True,
) -> dict:
    """Generate comprehensive EDA report.

    Phase 9.2 enhancement for automated EDA.

    Args:
        df: DataFrame to analyze
        output_path: Optional path to save HTML report
        include_correlations: Include correlation analysis
        include_distributions: Include distribution analysis
        include_statistical_tests: Include hypothesis tests

    Returns:
        Dictionary with report sections
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    # Summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["summary_stats"] = df[numeric_cols].describe().to_dict()

    # Correlations
    if include_correlations and len(numeric_cols) > 1:
        corr_matrix = calculate_correlation_matrix(df, numeric_cols[:20])  # Limit to 20 cols
        report["correlations"] = {
            "matrix": corr_matrix.to_dict(),
            "top_pairs": find_top_correlations(df, numeric_cols[:20], n_top=10),
        }

    # Distributions
    if include_distributions and len(numeric_cols) > 0:
        dist_cols = numeric_cols[:10]  # Analyze first 10 numeric columns
        report["distributions"] = {
            "normality_tests": test_normality(df, dist_cols),
            "skew_kurtosis": calculate_skewness_kurtosis(df, dist_cols).to_dict(),
        }

    # Statistical tests (if sector column exists)
    if include_statistical_tests and "sector" in df.columns and len(numeric_cols) > 0:
        test_col = numeric_cols[0]
        report["statistical_tests"] = {
            "sector_comparison": compare_sector_means(df, test_col, "sector")
        }

    # Save HTML report if requested
    if output_path:
        # Simple HTML generation (can be enhanced)
        html_content = f"""
        <html>
        <head><title>EDA Report</title></head>
        <body>
        <h1>Exploratory Data Analysis Report</h1>
        <p>Generated: {report['timestamp']}</p>
        <p>Rows: {report['n_rows']}, Columns: {report['n_columns']}</p>
        </body>
        </html>
        """
        output_path.write_text(html_content)

    return report


def generate_sector_comparison_report(
    df: pd.DataFrame, sector_column: str = "sector", metrics: list = None
) -> dict:
    """Generate sector comparison report.

    Phase 9.2 enhancement for sector analysis.

    Args:
        df: DataFrame containing the data
        sector_column: Name of sector column
        metrics: List of metrics to compare

    Returns:
        Dictionary with sector comparison results
    """
    if metrics is None:
        # Default metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = numeric_cols[:5]  # First 5 numeric columns

    report = {"sectors": df[sector_column].unique().tolist(), "n_metrics": len(metrics)}

    # Sector statistics
    sector_stats = {}
    for sector in report["sectors"]:
        if pd.isna(sector):
            continue

        sector_data = df[df[sector_column] == sector]
        sector_stats[sector] = sector_data[metrics].describe().to_dict()

    report["sector_stats"] = sector_stats

    # Statistical tests
    statistical_tests = {}
    for metric in metrics:
        test_result = compare_sector_means(df, metric, sector_column, method="anova")
        statistical_tests[metric] = test_result

    report["statistical_tests"] = statistical_tests

    return report


# ============================================================================
# Phase 9.6: Model Evaluation and Error Analysis
# ============================================================================


def comprehensive_regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.

    Computes MAE, RMSE, MAPE, R², Median Absolute Error, and Max Error.

    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values

    Returns:
        dict: Dictionary containing all metrics

    Metrics:
        - mae: Mean Absolute Error (interpretable dollar error)
        - rmse: Root Mean Squared Error (penalizes large errors)
        - mape: Mean Absolute Percentage Error (relative error)
        - r2: R² coefficient of determination (variance explained)
        - median_ae: Median Absolute Error (robust to outliers)
        - max_error: Maximum absolute error (worst-case performance)
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        median_absolute_error,
        max_error as sklearn_max_error,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = sklearn_max_error(y_true, y_pred)

    # MAPE - handle zeros by excluding them
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "median_ae": float(median_ae),
        "max_error": float(max_err),
    }


def compute_metrics_by_segment(df, y_true_col, y_pred_col, segment_col):
    """
    Compute regression metrics for each segment (sector, region, market cap, etc.).

    Args:
        df: DataFrame containing predictions and segment information
        y_true_col: Name of column with true values
        y_pred_col: Name of column with predicted values
        segment_col: Name of column to segment by

    Returns:
        pd.DataFrame: Metrics for each segment
    """
    results = []

    for segment_value in df[segment_col].dropna().unique():
        segment_data = df[df[segment_col] == segment_value]
        y_true = segment_data[y_true_col].values
        y_pred = segment_data[y_pred_col].values

        if len(y_true) > 0:
            metrics = comprehensive_regression_metrics(y_true, y_pred)
            metrics["segment"] = segment_value
            metrics["n_samples"] = len(y_true)
            results.append(metrics)

    result_df = pd.DataFrame(results)

    # Reorder columns to have segment first
    cols = ["segment", "n_samples"] + [
        c for c in result_df.columns if c not in ["segment", "n_samples"]
    ]
    return result_df[cols]


def residual_analysis_suite(y_true, y_pred, output_dir=None):
    """
    Perform comprehensive residual analysis.

    Computes residual statistics, normality tests, and optionally creates plots.

    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        output_dir: Optional Path to save plots (PNG files)

    Returns:
        dict: Residual analysis results including statistics and test results
    """
    from scipy import stats

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Compute statistics
    results = {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
    }

    # Normality test (Shapiro-Wilk for n < 5000, otherwise Anderson-Darling)
    if len(residuals) < 5000:
        stat, p_value = stats.shapiro(residuals)
        test_name = "shapiro"
    else:
        # Use Kolmogorov-Smirnov test for large samples
        stat, p_value = stats.kstest(
            residuals, "norm", args=(np.mean(residuals), np.std(residuals))
        )
        test_name = "ks"

    results["normality_test"] = {
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }

    # Create plots if output_dir is provided
    if output_dir and plt:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Residuals vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "residuals_vs_predicted.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 2. Q-Q plot
        plt.figure(figsize=(8, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot (Normal Distribution)")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "qq_plot.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 3. Histogram with normal distribution overlay
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black")

        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal PDF")

        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title(f"Residual Distribution (μ={mu:.2f}, σ={sigma:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "residual_histogram.png", dpi=100, bbox_inches="tight")
        plt.close()

    return results


def error_bucketing_analysis(df, y_true_col, y_pred_col, bucket_cols):
    """
    Analyze prediction errors by various buckets (market cap, volatility, sector).

    Args:
        df: DataFrame with predictions and bucketing columns
        y_true_col: Name of column with true values
        y_pred_col: Name of column with predicted values
        bucket_cols: List of column names to bucket by

    Returns:
        dict: Error analysis results for each bucket type
    """
    results = {}

    # Compute errors
    df = df.copy()
    df["error"] = df[y_true_col] - df[y_pred_col]
    df["abs_error"] = np.abs(df["error"])

    # Analyze each bucket type
    for bucket_col in bucket_cols:
        if bucket_col in df.columns:
            bucket_metrics = compute_metrics_by_segment(df, y_true_col, y_pred_col, bucket_col)
            results[bucket_col] = bucket_metrics

    # Identify outliers (errors > 3 std dev)
    error_mean = df["error"].mean()
    error_std = df["error"].std()
    outlier_threshold = 3 * error_std

    outliers = df[np.abs(df["error"] - error_mean) > outlier_threshold]

    results["outliers"] = {
        "n_outliers": len(outliers),
        "outlier_threshold": float(outlier_threshold),
        "outlier_percentage": float(len(outliers) / len(df) * 100),
        "mean_error": float(error_mean),
        "std_error": float(error_std),
    }

    return results


def create_stratified_sector_cv(n_splits=5):
    """
    Create a stratified cross-validation splitter by sector.

    Maintains sector balance across folds.

    Args:
        n_splits: Number of CV splits

    Returns:
        Cross-validation splitter object
    """
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def create_grouped_ticker_cv(n_splits=5):
    """
    Create a grouped cross-validation splitter by ticker.

    Ensures no ticker leakage between train and test sets.

    Args:
        n_splits: Number of CV splits

    Returns:
        Cross-validation splitter object
    """
    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=n_splits)


def evaluate_with_cross_validation(model, X, y, cv_strategy="simple", groups=None, n_splits=5):
    """
    Evaluate model using cross-validation with various strategies.

    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        cv_strategy: 'simple', 'stratified', or 'grouped'
        groups: Group labels for stratified or grouped CV
        n_splits: Number of CV splits

    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import r2_score

    # Select CV strategy
    if cv_strategy == "simple":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    elif cv_strategy == "stratified":
        # For stratified CV with groups (e.g., sectors), we need to manually iterate
        # because StratifiedKFold stratifies on y, not groups
        cv = create_stratified_sector_cv(n_splits=n_splits)
        scores = []

        if groups is not None:
            # Use groups for stratification
            for train_idx, test_idx in cv.split(X, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                scores.append(score)
        else:
            # No groups provided, fall back to simple CV
            scores = cross_val_score(
                model,
                X,
                y,
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring="r2",
            )

        scores = np.array(scores)
    elif cv_strategy == "grouped":
        cv = create_grouped_ticker_cv(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2")
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    return {
        "cv_scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "n_splits": n_splits,
        "cv_strategy": cv_strategy,
    }


# ==============================================================================
# Phase 9.7: Identification of Under/Overvalued Stocks with Visualization
# ==============================================================================


def assign_valuation_category(
    mispricing_scores: pd.Series, thresholds: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Assign valuation categories based on mispricing scores.

    Categories:
    - Strong Buy: mispricing > strong_buy threshold (default 20%)
    - Buy: mispricing between buy and strong_buy (default 10-20%)
    - Hold: mispricing between -sell and +buy (default -10% to +10%)
    - Sell: mispricing between -strong_sell and -sell (default -20% to -10%)
    - Strong Sell: mispricing < -strong_sell threshold (default -20%)

    Args:
        mispricing_scores: Series of mispricing scores (percentage)
        thresholds: Optional dict with keys 'strong_buy', 'buy', 'sell', 'strong_sell'
                   Default: {'strong_buy': 20, 'buy': 10, 'sell': 10, 'strong_sell': 20}

    Returns:
        Series with valuation categories

    Example:
        >>> scores = pd.Series([25, 15, 5, -5, -15, -25])
        >>> categories = assign_valuation_category(scores)
        >>> print(categories.tolist())
        ['Strong Buy', 'Buy', 'Hold', 'Hold', 'Sell', 'Strong Sell']
    """
    if thresholds is None:
        thresholds = {"strong_buy": 20.0, "buy": 10.0, "sell": 10.0, "strong_sell": 20.0}

    def categorize(score):
        if pd.isna(score):
            return "Unknown"
        elif score > thresholds["strong_buy"]:
            return "Strong Buy"
        elif score > thresholds["buy"]:
            return "Buy"
        elif score >= -thresholds["sell"]:
            return "Hold"
        elif score >= -thresholds["strong_sell"]:
            return "Sell"
        else:
            return "Strong Sell"

    return mispricing_scores.apply(categorize)


def calculate_sector_zscores(
    df: pd.DataFrame, metrics: list, sector_col: str = "sector"
) -> pd.DataFrame:
    """
    Calculate z-scores for metrics within each sector.

    Z-score = (value - sector_mean) / sector_std

    Useful for identifying stocks trading at premium/discount relative to sector peers.

    Args:
        df: DataFrame with stock data
        metrics: List of column names to calculate z-scores for
        sector_col: Name of sector column (default 'sector')

    Returns:
        DataFrame with original columns plus '{metric}_zscore' columns

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B', 'C'],
        ...     'sector': ['Tech', 'Tech', 'Finance'],
        ...     'pe_ratio': [20, 30, 15]
        ... })
        >>> result = calculate_sector_zscores(df, metrics=['pe_ratio'])
        >>> 'pe_ratio_zscore' in result.columns
        True
    """
    result = df.copy()

    for metric in metrics:
        if metric not in df.columns:
            continue

        zscore_col = f"{metric}_zscore"

        # Calculate z-scores within each sector
        result[zscore_col] = df.groupby(sector_col)[metric].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return result


def calculate_percentile_ranks(
    df: pd.DataFrame, metrics: list, sector_col: str = "sector"
) -> pd.DataFrame:
    """
    Calculate percentile ranks for metrics within each sector.

    Percentile rank indicates the percentage of sector peers a stock outperforms.

    Args:
        df: DataFrame with stock data
        metrics: List of column names to calculate percentiles for
        sector_col: Name of sector column (default 'sector')

    Returns:
        DataFrame with original columns plus '{metric}_percentile' columns (0-100 scale)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B'],
        ...     'sector': ['Tech', 'Tech'],
        ...     'pe_ratio': [10, 20]
        ... })
        >>> result = calculate_percentile_ranks(df, metrics=['pe_ratio'])
        >>> 'pe_ratio_percentile' in result.columns
        True
    """
    result = df.copy()

    for metric in metrics:
        if metric not in df.columns:
            continue

        percentile_col = f"{metric}_percentile"

        # Calculate percentile ranks within each sector (0-100 scale)
        result[percentile_col] = df.groupby(sector_col)[metric].rank(pct=True) * 100

    return result


def calculate_multi_factor_score(
    df: pd.DataFrame,
    valuation_col: str = "mispricing_score",
    quality_cols: Optional[list] = None,
    growth_cols: Optional[list] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Calculate a composite multi-factor score combining valuation, quality, and growth.

    Score = weighted_valuation + weighted_quality + weighted_growth

    Each component is normalized to z-scores before weighting.

    Args:
        df: DataFrame with stock metrics
        valuation_col: Column name for valuation metric (e.g., 'mispricing_score')
        quality_cols: List of quality metric columns (e.g., ['roe', 'ebitda_margin'])
        growth_cols: List of growth metric columns (e.g., ['revenue_growth'])
        weights: Optional dict with keys 'valuation', 'quality', 'growth'
                Default: {'valuation': 0.4, 'quality': 0.3, 'growth': 0.3}

    Returns:
        Series with multi-factor scores (higher = better)

    Example:
        >>> df = pd.DataFrame({
        ...     'mispricing_score': [20, 10],
        ...     'roe': [0.15, 0.10],
        ...     'revenue_growth': [0.20, 0.15]
        ... })
        >>> scores = calculate_multi_factor_score(
        ...     df,
        ...     quality_cols=['roe'],
        ...     growth_cols=['revenue_growth']
        ... )
        >>> isinstance(scores, pd.Series)
        True
    """
    if quality_cols is None:
        quality_cols = []
    if growth_cols is None:
        growth_cols = []
    if weights is None:
        weights = {"valuation": 0.4, "quality": 0.3, "growth": 0.3}

    scores = pd.Series(0.0, index=df.index)

    # Valuation component
    if valuation_col in df.columns:
        val_zscore = (df[valuation_col] - df[valuation_col].mean()) / df[valuation_col].std()
        scores += weights.get("valuation", 0.4) * val_zscore.fillna(0)

    # Quality component (average of quality metrics)
    if quality_cols:
        quality_zscores = []
        for col in quality_cols:
            if col in df.columns:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:
                    zscore = (df[col] - col_mean) / col_std
                    quality_zscores.append(zscore.fillna(0))

        if quality_zscores:
            avg_quality = pd.concat(quality_zscores, axis=1).mean(axis=1)
            scores += weights.get("quality", 0.3) * avg_quality

    # Growth component (average of growth metrics)
    if growth_cols:
        growth_zscores = []
        for col in growth_cols:
            if col in df.columns:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:
                    zscore = (df[col] - col_mean) / col_std
                    growth_zscores.append(zscore.fillna(0))

        if growth_zscores:
            avg_growth = pd.concat(growth_zscores, axis=1).mean(axis=1)
            scores += weights.get("growth", 0.3) * avg_growth

    return scores


def filter_stocks_by_criteria(
    df: pd.DataFrame,
    sectors: Optional[list] = None,
    regions: Optional[list] = None,
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    min_mispricing: Optional[float] = None,
    max_mispricing: Optional[float] = None,
    valuation_categories: Optional[list] = None,
) -> pd.DataFrame:
    """
    Filter stocks based on multiple criteria.

    Args:
        df: DataFrame with stock data
        sectors: List of sectors to include (e.g., ['Tech', 'Finance'])
        regions: List of regions to include (e.g., ['US', 'EU'])
        min_market_cap: Minimum market cap threshold
        max_market_cap: Maximum market cap threshold
        min_mispricing: Minimum mispricing score threshold
        max_mispricing: Maximum mispricing score threshold
        valuation_categories: List of valuation categories to include

    Returns:
        Filtered DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B', 'C'],
        ...     'sector': ['Tech', 'Finance', 'Energy'],
        ...     'market_cap': [100e9, 50e9, 10e9]
        ... })
        >>> filtered = filter_stocks_by_criteria(df, sectors=['Tech'], min_market_cap=50e9)
        >>> len(filtered)
        1
    """
    result = df.copy()

    # Filter by sector
    if sectors is not None:
        if "sector" in result.columns:
            result = result[result["sector"].isin(sectors)]

    # Filter by region
    if regions is not None:
        if "region" in result.columns:
            result = result[result["region"].isin(regions)]

    # Filter by market cap
    if min_market_cap is not None:
        if "market_cap" in result.columns:
            result = result[result["market_cap"] >= min_market_cap]

    if max_market_cap is not None:
        if "market_cap" in result.columns:
            result = result[result["market_cap"] <= max_market_cap]

    # Filter by mispricing score
    if min_mispricing is not None:
        if "mispricing_score" in result.columns:
            result = result[result["mispricing_score"] >= min_mispricing]

    if max_mispricing is not None:
        if "mispricing_score" in result.columns:
            result = result[result["mispricing_score"] <= max_mispricing]

    # Filter by valuation category
    if valuation_categories is not None:
        if "valuation_category" in result.columns:
            result = result[result["valuation_category"].isin(valuation_categories)]

    return result


def create_valuation_scatter_plot(
    df: pd.DataFrame, out_path: Optional[Path] = None, color_by: str = "sector"
):
    """
    Create an interactive scatter plot of current price vs. predicted target.

    Args:
        df: DataFrame with columns 'last_price', 'predicted_target', and color_by column
        out_path: Optional path to save HTML file
        color_by: Column to color points by (default 'sector')

    Returns:
        Plotly figure object (or None if plotly not available)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B'],
        ...     'last_price': [100, 50],
        ...     'predicted_target': [120, 55],
        ...     'sector': ['Tech', 'Finance']
        ... })
        >>> fig = create_valuation_scatter_plot(df)
        >>> fig is not None
        True
    """
    if px is None:
        logging.warning("plotly not available; skipping scatter plot")
        return None

    # Ensure required columns exist
    required = ["last_price", "predicted_target"]
    for col in required:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found; skipping scatter plot")
            return None

    # Create hover data
    hover_data_cols = ["ticker"] if "ticker" in df.columns else []
    if "mispricing_score" in df.columns:
        hover_data_cols.append("mispricing_score")
    if "valuation_category" in df.columns:
        hover_data_cols.append("valuation_category")

    # Create scatter plot
    fig = px.scatter(
        df,
        x="last_price",
        y="predicted_target",
        color=color_by if color_by in df.columns else None,
        hover_data=hover_data_cols,
        title="Current Price vs. Predicted Target",
        labels={"last_price": "Current Price ($)", "predicted_target": "Predicted Target ($)"},
    )

    # Add diagonal line (y=x) for reference
    max_val = max(df["last_price"].max(), df["predicted_target"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Fair Value (y=x)",
            showlegend=True,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Current Price ($)", yaxis_title="Predicted Target ($)", hovermode="closest"
    )

    # Save if path provided
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logging.info(f"Saved valuation scatter plot to {out_path}")

    return fig
