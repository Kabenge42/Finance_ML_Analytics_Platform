# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

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
    numeric_cols = [c for c in df.columns if df[c].dtype != object]
    basic_stats = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols_count": int((df.dtypes != object).sum()),
        "categorical_cols_count": int((df.dtypes == object).sum()),
        "null_counts": df.isnull().sum().to_dict(),
        "region_counts": df["region"].value_counts().to_dict() if "region" in df.columns else {},
        "sector_counts": df["sector"].value_counts().to_dict() if "sector" in df.columns else {},
        "basic_stats": basic_stats,
    }

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
    
    # Now check if visualization libraries are available
    if plt is None or sns is None:
        raise ImportError("Matplotlib and seaborn required for heatmap visualization")

    try:

        # Aggregate metric by sector
        sector_stats = (
            df.groupby("sector")[metric].agg(["mean", "median", "std", "count"]).reset_index()
        )

        # Create pivot for heatmap
        pivot_data = sector_stats.set_index("sector")[["mean", "median", "std"]]

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

        # Create scatter plot
        fig = px.scatter(
            df,
            x="last_price",
            y="predicted_target",
            color="sector" if "sector" in df.columns else None,
            hover_data=["ticker"] if "ticker" in df.columns else None,
            title="Predicted Target vs Current Price",
            labels={"last_price": "Current Price", "predicted_target": "Predicted Target Price"},
        )

        # Add diagonal line (y=x) for reference
        min_val = min(df["last_price"].min(), df["predicted_target"].min())
        max_val = max(df["last_price"].max(), df["predicted_target"].max())
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

        # Create pivot table
        pivot_data = df.pivot_table(values=metric, index="sector", columns="region", aggfunc="mean")

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
