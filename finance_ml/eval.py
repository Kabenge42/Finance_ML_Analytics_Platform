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
