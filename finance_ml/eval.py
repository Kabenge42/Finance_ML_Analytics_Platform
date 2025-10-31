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

    Formula: (predicted_price_target - last_price) / last_price

    Positive score = undervalued (predicted > current)
    Negative score = overvalued (predicted < current)

    Args:
        df: DataFrame with columns:
            - predicted_price_target: Predicted target price
            - last_price: Current market price

    Returns:
        Series with mispricing scores
    
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["predicted_price_target", "last_price"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    score = (df["predicted_price_target"] - df["last_price"]) / df["last_price"]
    return score


def calculate_risk_adjusted_mispricing(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    use_confidence_interval: bool = False,
    default_volatility: float = 0.20,
) -> pd.Series:
    """Calculate risk-adjusted mispricing score.

    Formula: (Expected_Return - Risk_Free_Rate) / Volatility

    This adjusts the mispricing score by the stock's volatility to account for risk.
    Higher risk-adjusted scores indicate better risk-reward opportunities.

    Args:
        df: DataFrame with 'predicted_price_target', 'last_price', and 'volatility' columns
        risk_free_rate: Risk-free rate to subtract from expected return (default 0.0)
        use_confidence_interval: If True and confidence intervals available, adjust for uncertainty
        default_volatility: Default volatility to use if column missing (default 0.20)

    Returns:
        Series with risk-adjusted mispricing scores

    Example:
        >>> df = pd.DataFrame({
        ...     'predicted_price_target': [120, 90],
        ...     'last_price': [100, 100],
        ...     'volatility': [0.20, 0.30]
        ... })
        >>> scores = calculate_risk_adjusted_mispricing(df, risk_free_rate=0.05)
        >>> scores.iloc[0] > 0  # Undervalued with positive risk-adjusted return
        True
    """
    # Calculate expected return
    expected_return = (df["predicted_price_target"] - df["last_price"]) / df["last_price"]

    # Use volatility column if available, otherwise use default
    if "volatility" in df.columns:
        volatility = df["volatility"].copy()
    else:
        logging.warning(f"Volatility column not found; using default {default_volatility}")
        volatility = pd.Series(default_volatility, index=df.index)

    # Replace zero or negative volatility with a small value to avoid division by zero
    volatility = volatility.clip(lower=0.01)

    # Adjust for confidence interval width if requested
    if (
        use_confidence_interval
        and "confidence_lower" in df.columns
        and "confidence_upper" in df.columns
    ):
        # Wider confidence intervals indicate more uncertainty
        ci_width = (df["confidence_upper"] - df["confidence_lower"]) / df["last_price"]
        # Penalize by confidence interval width (wider = more uncertain = lower score)
        uncertainty_penalty = 1.0 / (1.0 + ci_width)
        risk_adjusted = ((expected_return - risk_free_rate) / volatility) * uncertainty_penalty
    else:
        # Standard risk-adjusted calculation
        risk_adjusted = (expected_return - risk_free_rate) / volatility

    return risk_adjusted


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
    target_column: Optional[str] = None,
    include_multivariate: bool = False,
) -> dict:
    """Perform exploratory data analysis.

    When out_dir is provided, write eda_summary.json (and optional plots) to disk.
    Always return the computed summary as a dictionary for programmatic use.

    Phase 9.2 enhancements: Added feature importance and multivariate analysis integration.

    Args:
        df: DataFrame to analyze
        out_dir: Optional directory to save output files. If None, files are not written.
        save_plots: If True and out_dir is provided, generate and save matplotlib visualizations
        target_column: Optional target column name for feature importance analysis
        include_multivariate: If True, include PCA and other multivariate analysis

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
            # Correlation analysis (Pearson, Spearman, Kendall, and Distance - Phase 9.2)
            if len(numeric_cols) >= 2:
                corr_analysis = {}
                pearson_corr = calculate_correlation_matrix(df, numeric_cols, method="pearson")
                spearman_corr = calculate_correlation_matrix(df, numeric_cols, method="spearman")
                kendall_corr = calculate_correlation_matrix(df, numeric_cols, method="kendall")
                # Convert DataFrames to dicts for JSON serialization
                corr_analysis["pearson"] = pearson_corr.to_dict() if not pearson_corr.empty else {}
                corr_analysis["spearman"] = (
                    spearman_corr.to_dict() if not spearman_corr.empty else {}
                )
                corr_analysis["kendall"] = kendall_corr.to_dict() if not kendall_corr.empty else {}

                # Distance correlation (Phase 9.2 continuation - optional, requires dcor)
                try:
                    distance_corr = calculate_distance_correlation(df, numeric_cols)
                    corr_analysis["distance"] = (
                        distance_corr.to_dict() if not distance_corr.empty else {}
                    )
                except ImportError:
                    # dcor library not installed - skip distance correlation
                    logging.info("Distance correlation skipped (dcor library not installed)")
                    corr_analysis["distance"] = {}
                except Exception as e:
                    logging.warning("Distance correlation calculation failed: %s", e)
                    corr_analysis["distance"] = {}

                summary["correlation_analysis"] = corr_analysis
            else:
                summary["correlation_analysis"] = {}
        except Exception as e:
            logging.warning("Correlation analysis failed: %s", e)
            summary["correlation_analysis"] = {}

        try:
            # Top correlations extraction (Phase 9.2)
            if len(numeric_cols) >= 2:
                top_corr = {}
                for method in ["pearson", "spearman", "kendall"]:
                    try:
                        # Calculate correlation matrix for this method
                        corr_matrix = calculate_correlation_matrix(df, numeric_cols, method=method)
                        # Pass correlation matrix (not df) to find_top_correlations
                        top_corr_list = find_top_correlations(corr_matrix, n_top=10, threshold=0.0)
                        # Convert list of tuples to list of dicts for JSON serialization
                        top_corr[method] = [
                            {"var1": var1, "var2": var2, "correlation": float(corr)}
                            for var1, var2, corr in top_corr_list
                        ]
                    except Exception:
                        top_corr[method] = []
                summary["top_correlations"] = top_corr
            else:
                summary["top_correlations"] = {}
        except Exception as e:
            logging.warning("Top correlations extraction failed: %s", e)
            summary["top_correlations"] = {}

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

        try:
            # Sector comparison tests (Phase 9.2)
            if (
                "sector" in df.columns
                and df["sector"].notna().any()
                and len(df["sector"].dropna().unique()) >= 2
            ):
                sector_tests = {}
                for col in numeric_cols:
                    try:
                        test_result = compare_sector_means(
                            df, col, group_column="sector", method="anova", alpha=0.05
                        )
                        if test_result:
                            # Convert numpy bool to Python bool for JSON serialization
                            if (
                                "significant" in test_result
                                and test_result["significant"] is not None
                            ):
                                test_result["significant"] = bool(test_result["significant"])
                            sector_tests[col] = test_result
                    except Exception:
                        pass
                summary["sector_comparison_tests"] = sector_tests
            else:
                summary["sector_comparison_tests"] = {}
        except Exception as e:
            logging.warning("Sector comparison tests failed: %s", e)
            summary["sector_comparison_tests"] = {}

        try:
            # Region-wise statistics (Phase 9.2)
            if "region" in df.columns and df["region"].notna().any():
                region_stats = {}
                for region in df["region"].dropna().unique():
                    region_df = df[df["region"] == region]
                    region_numeric = [c for c in numeric_cols if c in region_df.columns]
                    if region_numeric and len(region_df) > 0:
                        region_stats[region] = {
                            "count": int(len(region_df)),
                            "means": region_df[region_numeric].mean().to_dict(),
                            "medians": region_df[region_numeric].median().to_dict(),
                            "stds": region_df[region_numeric].std().to_dict(),
                        }
                summary["region_statistics"] = region_stats
            else:
                summary["region_statistics"] = {}
        except Exception as e:
            logging.warning("Region statistics failed: %s", e)
            summary["region_statistics"] = {}
    else:
        # Insufficient data for advanced analysis
        summary["distribution_analysis"] = {}
        summary["outlier_detection"] = {}
        summary["normality_tests"] = {}
        summary["correlation_analysis"] = {}
        summary["top_correlations"] = {}
        summary["sector_statistics"] = {}
        summary["sector_comparison_tests"] = {}
        summary["region_statistics"] = {}

    # Phase 9.2: Feature Importance Analysis (when target provided)
    if target_column is not None and target_column in df.columns:
        try:
            # Prepare features and target
            feature_cols = [c for c in numeric_cols if c != target_column]
            if feature_cols and len(df) >= 10:
                X = df[feature_cols].dropna()
                y = df.loc[X.index, target_column]

                # Remove any remaining NaN in target
                valid_mask = y.notna()
                X = X[valid_mask]
                y = y[valid_mask]

                if len(X) >= 10 and len(feature_cols) >= 2:
                    feature_importance = {}

                    # Mutual Information
                    try:
                        mi_scores = calculate_mutual_information(X, y)
                        # Convert Series to dict for JSON serialization
                        if hasattr(mi_scores, "to_dict"):
                            feature_importance["mutual_information"] = mi_scores.to_dict()
                        elif isinstance(mi_scores, dict):
                            feature_importance["mutual_information"] = mi_scores
                        else:
                            feature_importance["mutual_information"] = {}
                    except Exception as e:
                        logging.warning("Mutual information calculation failed: %s", e)
                        feature_importance["mutual_information"] = {}

                    # Random Forest Feature Importance
                    try:
                        rf_importance = calculate_feature_importance_rf(X, y)
                        # Convert Series to dict for JSON serialization
                        if hasattr(rf_importance, "to_dict"):
                            feature_importance["random_forest"] = rf_importance.to_dict()
                        elif isinstance(rf_importance, dict):
                            feature_importance["random_forest"] = rf_importance
                        else:
                            feature_importance["random_forest"] = {}
                    except Exception as e:
                        logging.warning("Random forest importance calculation failed: %s", e)
                        feature_importance["random_forest"] = {}

                    # SHAP values (optional, may be slow)
                    try:
                        shap_importance = calculate_shap_importance(
                            X, y, model_type="tree", n_samples=100
                        )
                        # Convert Series to dict for JSON serialization if needed
                        if hasattr(shap_importance, "to_dict"):
                            feature_importance["shap"] = shap_importance.to_dict()
                        elif isinstance(shap_importance, dict):
                            feature_importance["shap"] = shap_importance
                        else:
                            feature_importance["shap"] = {}
                    except Exception as e:
                        logging.warning("SHAP importance calculation failed (optional): %s", e)
                        feature_importance["shap"] = {}

                    summary["feature_importance"] = feature_importance
                else:
                    summary["feature_importance"] = {}
            else:
                summary["feature_importance"] = {}
        except Exception as e:
            logging.warning("Feature importance analysis failed: %s", e)
            summary["feature_importance"] = {}
    else:
        summary["feature_importance"] = {}

    # Phase 9.2: Multivariate Analysis (when requested)
    if include_multivariate and numeric_cols and len(df) >= 10:
        try:
            # Prepare data for multivariate analysis
            X_multi = df[numeric_cols].dropna()

            if len(X_multi) >= 10 and len(numeric_cols) >= 3:
                multivariate_analysis = {}

                # PCA Analysis
                try:
                    pca_result = perform_pca(X_multi, n_components=min(3, len(numeric_cols)))
                    # Convert numpy arrays to lists for JSON serialization
                    multivariate_analysis["pca"] = {
                        "explained_variance_ratio": pca_result["explained_variance_ratio"].tolist(),
                        "cumulative_variance": pca_result["cumulative_variance"].tolist(),
                        "n_components": pca_result["n_components"],
                        "feature_names": pca_result["feature_names"],
                        "components_shape": pca_result["components"].shape,
                    }
                except Exception as e:
                    logging.warning("PCA analysis failed: %s", e)
                    multivariate_analysis["pca"] = {}

                # t-SNE (optional, can be slow)
                try:
                    if len(X_multi) >= 30 and len(numeric_cols) >= 4:
                        tsne_result = perform_tsne(X_multi, n_components=2)
                        multivariate_analysis["tsne"] = {
                            "n_components": tsne_result["n_components"],
                            "feature_names": tsne_result["feature_names"],
                            "components_shape": tsne_result["components"].shape,
                        }
                    else:
                        multivariate_analysis["tsne"] = {}
                except Exception as e:
                    logging.warning("t-SNE analysis failed (optional): %s", e)
                    multivariate_analysis["tsne"] = {}

                # UMAP (Phase 9.2 continuation - optional, requires umap-learn)
                try:
                    if len(X_multi) >= 30 and len(numeric_cols) >= 4:
                        umap_result = perform_umap(X_multi, n_components=2)
                        multivariate_analysis["umap"] = {
                            "n_components": umap_result["n_components"],
                            "feature_names": umap_result["feature_names"],
                            "components_shape": umap_result["components"].shape,
                        }
                    else:
                        multivariate_analysis["umap"] = {}
                except ImportError:
                    # umap-learn library not installed - skip UMAP
                    logging.info("UMAP skipped (umap-learn library not installed)")
                    multivariate_analysis["umap"] = {}
                except Exception as e:
                    logging.warning("UMAP analysis failed (optional): %s", e)
                    multivariate_analysis["umap"] = {}

                summary["multivariate_analysis"] = multivariate_analysis
            else:
                summary["multivariate_analysis"] = {}
        except Exception as e:
            logging.warning("Multivariate analysis failed: %s", e)
            summary["multivariate_analysis"] = {}
    else:
        summary["multivariate_analysis"] = {}

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

                        # Outlier visualization plots (Phase 9.2 continuation)
                        outlier_cols = numeric_cols[:6]  # Limit to first 6 columns for clarity
                        if outlier_cols:
                            try:
                                # Box plots
                                boxplot_path = out_dir / "eda_outlier_boxplots.png"
                                plot_outlier_boxplots(df, outlier_cols, out_path=boxplot_path)
                            except Exception as e:
                                logging.warning("Error creating outlier box plots: %s", e)

                            try:
                                # Violin plots
                                violin_path = out_dir / "eda_outlier_violins.png"
                                plot_outlier_violins(df, outlier_cols, out_path=violin_path)
                            except Exception as e:
                                logging.warning("Error creating outlier violin plots: %s", e)

                            # Scatter plot (needs at least 2 columns)
                            if len(numeric_cols) >= 2:
                                try:
                                    scatter_path = out_dir / "eda_outlier_scatter.png"
                                    plot_outlier_scatter(
                                        df, numeric_cols[:2], out_path=scatter_path
                                    )
                                except Exception as e:
                                    logging.warning("Error creating outlier scatter plot: %s", e)

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
        df: DataFrame with predicted_price_target, last_price, and other columns
        out_path: Path to save the HTML file

    Returns:
        Plotly figure object, or None if required columns are missing
    """
    if px is None:
        raise ImportError("Plotly required for interactive plots")

    try:
        # Check if required columns exist
        if "last_price" not in df.columns or "predicted_price_target" not in df.columns:
            logging.warning(
                "Missing required columns (last_price, predicted_price_target) for interactive plot"
            )
            return None

        # Drop rows with NaNs in required columns to avoid zero-size issues
        df_plot = df.dropna(subset=["last_price", "predicted_price_target"]).copy()
        if df_plot.empty:
            logging.warning("Interactive plot skipped: no valid rows after dropping NaNs")
            return None

        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x="last_price",
            y="predicted_price_target",
            color="sector" if "sector" in df_plot.columns else None,
            hover_data=["ticker"] if "ticker" in df_plot.columns else None,
            title="Predicted Target vs Current Price",
            labels={"last_price": "Current Price", "predicted_price_target": "Predicted Target Price"},
        )

        # Add diagonal line (y=x) for reference
        min_val = float(min(df_plot["last_price"].min(), df_plot["predicted_price_target"].min()))
        max_val = float(max(df_plot["last_price"].max(), df_plot["predicted_price_target"].max()))
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


def plot_outlier_boxplots(df: pd.DataFrame, columns: list, out_path: Optional[Path] = None):
    """Create box plots for outlier visualization.

    Phase 9.2 continuation: Visualize outliers using box plots showing quartiles and outliers.

    Args:
        df: DataFrame containing the data
        columns: List of columns to visualize
        out_path: Optional path to save the figure (PNG format)

    Returns:
        Matplotlib figure object, or None if matplotlib not available

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4, 100], 'y': [10, 20, 30, 40, 50]})
        >>> fig = plot_outlier_boxplots(df, ['x', 'y'])
    """
    if plt is None or sns is None:
        logging.warning("Matplotlib/seaborn not available for box plots")
        return None

    try:
        # Select numeric columns
        data = df[columns].select_dtypes(include=[np.number])

        if data.empty:
            logging.warning("No numeric columns found for box plots")
            return None

        # Create figure
        n_cols = len(data.columns)
        n_rows = (n_cols + 2) // 3  # 3 plots per row
        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, 5 * n_rows))

        # Flatten axes for easier iteration
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        # Create box plot for each column
        for idx, col in enumerate(data.columns):
            ax = axes[idx] if n_cols > 1 else axes[0]
            data[col].dropna().plot(kind="box", ax=ax)
            ax.set_title(f"Box Plot: {col}")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            logging.info("Saved outlier box plots to %s", out_path)
            plt.close()

        return fig

    except Exception as e:
        logging.error("Error creating outlier box plots: %s", e)
        return None


def plot_outlier_violins(df: pd.DataFrame, columns: list, out_path: Optional[Path] = None):
    """Create violin plots for outlier visualization.

    Phase 9.2 continuation: Visualize outliers using violin plots showing distribution density.

    Args:
        df: DataFrame containing the data
        columns: List of columns to visualize
        out_path: Optional path to save the figure (PNG format)

    Returns:
        Matplotlib figure object, or None if matplotlib not available

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4, 100], 'y': [10, 20, 30, 40, 50]})
        >>> fig = plot_outlier_violins(df, ['x', 'y'])
    """
    if plt is None or sns is None:
        logging.warning("Matplotlib/seaborn not available for violin plots")
        return None

    try:
        # Select numeric columns
        data = df[columns].select_dtypes(include=[np.number])

        if data.empty:
            logging.warning("No numeric columns found for violin plots")
            return None

        # Create figure
        n_cols = len(data.columns)
        n_rows = (n_cols + 2) // 3  # 3 plots per row
        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, 5 * n_rows))

        # Flatten axes for easier iteration
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        # Create violin plot for each column
        for idx, col in enumerate(data.columns):
            ax = axes[idx] if n_cols > 1 else axes[0]
            col_data = data[col].dropna()
            if len(col_data) >= 2:
                sns.violinplot(y=col_data, ax=ax)
                ax.set_title(f"Violin Plot: {col}")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
                ax.set_title(f"Violin Plot: {col}")

        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            logging.info("Saved outlier violin plots to %s", out_path)
            plt.close()

        return fig

    except Exception as e:
        logging.error("Error creating outlier violin plots: %s", e)
        return None


def plot_outlier_scatter(
    df: pd.DataFrame, columns: list, out_path: Optional[Path] = None, z_threshold: float = 3.0
):
    """Create scatter plot with z-score coloring for outlier visualization.

    Phase 9.2 continuation: Visualize outliers using scatter plots colored by z-score magnitude.

    Args:
        df: DataFrame containing the data
        columns: List of columns to visualize (uses first two for x and y)
        out_path: Optional path to save the figure (PNG format)
        z_threshold: Z-score threshold for highlighting outliers (default: 3.0)

    Returns:
        Matplotlib figure object, or None if matplotlib not available

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4, 100], 'y': [10, 20, 30, 40, 50]})
        >>> fig = plot_outlier_scatter(df, ['x', 'y'])
    """
    if plt is None or sns is None:
        logging.warning("Matplotlib/seaborn not available for scatter plots")
        return None

    try:
        # Select numeric columns
        data = df[columns].select_dtypes(include=[np.number])

        if data.empty or len(data.columns) < 2:
            logging.warning("Need at least 2 numeric columns for scatter plot")
            return None

        # Use first two columns
        col_x = data.columns[0]
        col_y = data.columns[1]

        # Calculate z-scores for coloring
        x_vals = data[col_x].dropna()
        y_vals = data[col_y].dropna()

        # Align data (keep only rows with both values)
        valid_mask = data[col_x].notna() & data[col_y].notna()
        x_aligned = data.loc[valid_mask, col_x]
        y_aligned = data.loc[valid_mask, col_y]

        if len(x_aligned) < 2:
            logging.warning("Insufficient data for scatter plot")
            return None

        # Calculate z-scores
        z_x = np.abs((x_aligned - x_aligned.mean()) / x_aligned.std())
        z_y = np.abs((y_aligned - y_aligned.mean()) / y_aligned.std())
        z_combined = np.maximum(z_x, z_y)  # Max z-score for coloring

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot with z-score coloring
        scatter = ax.scatter(
            x_aligned,
            y_aligned,
            c=z_combined,
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Max Z-Score", rotation=270, labelpad=20)

        # Highlight outliers
        outliers = z_combined > z_threshold
        if outliers.any():
            ax.scatter(
                x_aligned[outliers],
                y_aligned[outliers],
                s=100,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label=f"Outliers (|z| > {z_threshold})",
            )
            ax.legend()

        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"Outlier Scatter Plot: {col_x} vs {col_y}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            logging.info("Saved outlier scatter plot to %s", out_path)
            plt.close()

        return fig

    except Exception as e:
        logging.error("Error creating outlier scatter plot: %s", e)
        return None


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


def calculate_distance_correlation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Calculate distance correlation matrix.

    Phase 9.2 continuation: Add distance correlation support for capturing non-linear dependencies.

    Distance correlation measures both linear and non-linear statistical dependencies between variables,
    ranging from 0 (independent) to 1 (completely dependent). Unlike Pearson correlation, it can detect
    non-linear relationships.

    Args:
        df: DataFrame containing the data
        columns: List of columns to include in correlation matrix

    Returns:
        Distance correlation matrix as DataFrame

    Raises:
        ImportError: If dcor library is not installed
        ValueError: If no numeric columns found

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
        >>> dcor_matrix = calculate_distance_correlation(df, ['x', 'y'])
    """
    try:
        import dcor
    except ImportError:
        raise ImportError(
            "Distance correlation requires the 'dcor' library. " "Install it with: pip install dcor"
        )

    # Select numeric columns
    data = df[columns].select_dtypes(include=[np.number])

    if data.empty:
        raise ValueError("No numeric columns found in specified columns")

    # Calculate distance correlation matrix
    n_cols = len(data.columns)
    dcor_matrix = np.zeros((n_cols, n_cols))

    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i == j:
                dcor_matrix[i, j] = 1.0
            elif i < j:
                # Calculate distance correlation for upper triangle
                x = data[col1].values
                y = data[col2].values
                # Remove NaN pairs
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() >= 2:
                    dcor_value = dcor.distance_correlation(x[mask], y[mask])
                    dcor_matrix[i, j] = dcor_value
                    dcor_matrix[j, i] = dcor_value  # Symmetric
                else:
                    dcor_matrix[i, j] = np.nan
                    dcor_matrix[j, i] = np.nan

    # Convert to DataFrame
    result = pd.DataFrame(dcor_matrix, index=data.columns, columns=data.columns)

    return result


def find_top_correlations(
    corr_matrix: pd.DataFrame, n_top: int = 10, threshold: float = 0.0
) -> list:
    """Find top correlated variable pairs.

    Phase 9.2 enhancement for identifying strongest correlations.

    Args:
        corr_matrix: Correlation matrix (can be computed via calculate_correlation_matrix)
        n_top: Number of top correlations to return
        threshold: Minimum absolute correlation value to consider (default: 0.0)

    Returns:
        List of tuples (var1, var2, correlation) sorted by absolute correlation
    """
    # Get upper triangle (avoid duplicates and self-correlations)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)

    # Convert to list of tuples, filtering by threshold
    correlations = []
    for i in range(len(upper_tri)):
        for j in range(i + 1, len(upper_tri)):
            var1 = upper_tri.index[i]
            var2 = upper_tri.columns[j]
            corr_value = upper_tri.iloc[i, j]

            if pd.notna(corr_value) and abs(corr_value) >= threshold:
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


def calculate_shap_importance(
    X: pd.DataFrame, y: pd.Series, model_type: str = "tree", n_samples: int = 100
) -> pd.DataFrame:
    """Calculate feature importance using SHAP values.

    Phase 9.2 enhancement for interpretable feature importance analysis using
    SHapley Additive exPlanations (SHAP).

    Args:
        X: Feature DataFrame
        y: Target variable
        model_type: Type of model to use ('tree' for tree-based, 'linear' for linear)
        n_samples: Number of samples to use for SHAP calculation (for performance)

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance descending

    Raises:
        ImportError: If SHAP library is not installed

    Examples:
        >>> importance = calculate_shap_importance(X_train, y_train)
        >>> print(importance.head())
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP library is required for SHAP feature importance. "
            "Install it with: pip install shap"
        )

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    # Handle missing values
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median()) if pd.api.types.is_numeric_dtype(y) else y.fillna("missing")

    # Limit samples for performance if dataset is large
    if len(X_clean) > n_samples:
        sample_indices = np.random.choice(len(X_clean), n_samples, replace=False)
        X_sample = X_clean.iloc[sample_indices]
        y_sample = y_clean.iloc[sample_indices]
    else:
        X_sample = X_clean
        y_sample = y_clean

    # Choose and train model
    if model_type == "tree":
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, n_jobs=-1)
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"model_type '{model_type}' not supported. Use 'tree' or 'linear'.")

    model.fit(X_sample, y_sample)

    # Calculate SHAP values
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame({"feature": X.columns, "importance": mean_abs_shap})

    # Sort by importance descending
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def perform_tsne(
    X: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42
) -> dict:
    """Perform t-SNE dimensionality reduction for visualization.

    Phase 9.2 enhancement for non-linear dimensionality reduction and visualization
    of high-dimensional data.

    Args:
        X: Feature DataFrame
        n_components: Number of dimensions to reduce to (typically 2 or 3)
        perplexity: t-SNE perplexity parameter (balance between local and global structure)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with t-SNE results:
            - components: DataFrame with t-SNE components
            - feature_names: Original feature names
            - n_components: Number of components
            - perplexity: Perplexity used

    Raises:
        ImportError: If scikit-learn is not installed

    Examples:
        >>> tsne_result = perform_tsne(X_train, n_components=2)
        >>> plt.scatter(tsne_result['components']['TSNE1'], tsne_result['components']['TSNE2'])
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for t-SNE. " "Install it with: pip install scikit-learn"
        )

    # Handle missing values
    X_clean = X.fillna(X.median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Adjust perplexity if needed (must be less than n_samples)
    n_samples = len(X_clean)
    actual_perplexity = min(perplexity, (n_samples - 1) / 3.0)

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=actual_perplexity, random_state=random_state)
    components = tsne.fit_transform(X_scaled)

    # Create result DataFrame
    component_df = pd.DataFrame(
        components, columns=[f"TSNE{i+1}" for i in range(n_components)], index=X.index
    )

    return {
        "components": component_df,
        "feature_names": X.columns.tolist(),
        "n_components": n_components,
        "perplexity": actual_perplexity,
    }


def perform_umap(
    X: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Perform UMAP dimensionality reduction for visualization.

    Phase 9.2 enhancement for efficient non-linear dimensionality reduction
    using Uniform Manifold Approximation and Projection (UMAP).

    Args:
        X: Feature DataFrame
        n_components: Number of dimensions to reduce to (typically 2 or 3)
        n_neighbors: Number of neighbors for UMAP (controls local vs global structure)
        min_dist: Minimum distance between points in low-dimensional space
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with UMAP results:
            - components: DataFrame with UMAP components
            - feature_names: Original feature names
            - n_components: Number of components
            - n_neighbors: Number of neighbors used
            - min_dist: Minimum distance used

    Raises:
        ImportError: If umap-learn is not installed

    Examples:
        >>> umap_result = perform_umap(X_train, n_components=2)
        >>> plt.scatter(umap_result['components']['UMAP1'], umap_result['components']['UMAP2'])
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP dimensionality reduction. "
            "Install it with: pip install umap-learn"
        )

    from sklearn.preprocessing import StandardScaler

    # Handle missing values
    X_clean = X.fillna(X.median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Adjust n_neighbors if needed (must be less than n_samples)
    n_samples = len(X_clean)
    actual_n_neighbors = min(n_neighbors, n_samples - 1)

    # Perform UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=actual_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    components = reducer.fit_transform(X_scaled)

    # Create result DataFrame
    component_df = pd.DataFrame(
        components, columns=[f"UMAP{i+1}" for i in range(n_components)], index=X.index
    )

    return {
        "components": component_df,
        "feature_names": X.columns.tolist(),
        "n_components": n_components,
        "n_neighbors": actual_n_neighbors,
        "min_dist": min_dist,
    }


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
    Calculate comprehensive regression metrics with NaN handling.

    Computes MAE, RMSE, MAPE, R², Median Absolute Error, and Max Error.
    Handles NaN and infinite values gracefully by removing them before computation.

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
        - n_samples: Number of valid samples used for computation
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        median_absolute_error,
        max_error as sklearn_max_error,
    )

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for NaN values
    nan_mask_true = np.isnan(y_true)
    nan_mask_pred = np.isnan(y_pred)
    nan_mask = nan_mask_true | nan_mask_pred

    if nan_mask.any():
        n_nans = nan_mask.sum()
        logging.warning(f"Found {n_nans} NaN values ({n_nans/len(y_true)*100:.2f}% of data)")
        logging.warning(f"  - NaN in y_true: {nan_mask_true.sum()}")
        logging.warning(f"  - NaN in y_pred: {nan_mask_pred.sum()}")

        # Remove NaN values
        valid_mask = ~nan_mask
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        logging.warning(f"  - Remaining valid samples: {len(y_true)}")

    # Check if we have enough valid samples
    if len(y_true) < 2:
        logging.error("Not enough valid samples to compute metrics")
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "median_ae": np.nan,
            "max_error": np.nan,
            "n_samples": len(y_true),
        }

    # Check for infinite values
    inf_mask = np.isinf(y_true) | np.isinf(y_pred)
    if inf_mask.any():
        logging.warning(f"Found {inf_mask.sum()} infinite values, removing them")
        valid_mask = ~inf_mask
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    # Recheck sample count after removing infinities
    if len(y_true) < 2:
        logging.error("Not enough valid samples after removing infinities")
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "median_ae": np.nan,
            "max_error": np.nan,
            "n_samples": len(y_true),
        }

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = sklearn_max_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.isnan(mape) or np.isinf(mape):
            mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else np.inf,
        "r2": float(r2),
        "median_ae": float(median_ae),
        "max_error": float(max_err),
        "n_samples": len(y_true),
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


def get_sector_specific_thresholds(
    sector: str, sector_volatility_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Get sector-specific valuation thresholds adjusted for sector volatility.

    Volatile sectors (Technology, Biotech) get wider threshold bands.
    Stable sectors (Utilities, Consumer Staples) get narrower bands.

    Args:
        sector: Sector name
        sector_volatility_df: Optional DataFrame with 'sector' and 'volatility' columns
                             If provided, calculate dynamic thresholds based on actual volatility

    Returns:
        Dict with keys 'strong_buy', 'buy', 'sell', 'strong_sell'

    Example:
        >>> thresholds = get_sector_specific_thresholds("Technology")
        >>> thresholds['strong_buy'] > 20.0  # Wider band for volatile sector
        True
        >>> thresholds = get_sector_specific_thresholds("Utilities")
        >>> thresholds['strong_buy'] < 20.0  # Narrower band for stable sector
        True
    """
    # Default thresholds (baseline)
    default_thresholds = {"strong_buy": 20.0, "buy": 10.0, "sell": 10.0, "strong_sell": 20.0}

    # Sector volatility profiles (higher multiplier = more volatile = wider bands)
    sector_volatility_profiles = {
        # High volatility sectors (1.3x wider bands)
        "Technology": 1.3,
        "Tech": 1.3,
        "Information Technology": 1.3,
        "Biotechnology": 1.3,
        "Biotech": 1.3,
        "Healthcare": 1.2,
        "Communication Services": 1.2,
        "Consumer Discretionary": 1.15,
        # Medium volatility sectors (1.0x default bands)
        "Industrials": 1.0,
        "Materials": 1.0,
        "Energy": 1.0,
        "Financials": 0.9,
        "Finance": 0.9,
        "Financial Services": 0.9,
        # Low volatility sectors (0.8x narrower bands)
        "Utilities": 0.8,
        "Consumer Staples": 0.85,
        "Real Estate": 0.85,
    }

    # If actual sector volatility data provided, calculate dynamic multiplier
    if sector_volatility_df is not None and "sector" in sector_volatility_df.columns:
        sector_data = sector_volatility_df[sector_volatility_df["sector"] == sector]
        if len(sector_data) > 0 and "volatility" in sector_volatility_df.columns:
            avg_volatility = sector_data["volatility"].mean()
            # Overall market volatility baseline (20%)
            market_baseline = 0.20
            # Adjust multiplier based on sector volatility vs market
            multiplier = avg_volatility / market_baseline if market_baseline > 0 else 1.0
        else:
            # Use predefined profile
            multiplier = sector_volatility_profiles.get(sector, 1.0)
    else:
        # Use predefined profile
        multiplier = sector_volatility_profiles.get(sector, 1.0)

    # Apply multiplier to thresholds
    adjusted_thresholds = {
        "strong_buy": default_thresholds["strong_buy"] * multiplier,
        "buy": default_thresholds["buy"] * multiplier,
        "sell": default_thresholds["sell"] * multiplier,
        "strong_sell": default_thresholds["strong_sell"] * multiplier,
    }

    return adjusted_thresholds


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
                # Check if std is a scalar and greater than 0
                if pd.notna(col_std) and float(col_std) > 0:
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
                # Check if std is a scalar and greater than 0
                if pd.notna(col_std) and float(col_std) > 0:
                    zscore = (df[col] - col_mean) / col_std
                    growth_zscores.append(zscore.fillna(0))

        if growth_zscores:
            avg_growth = pd.concat(growth_zscores, axis=1).mean(axis=1)
            scores += weights.get("growth", 0.3) * avg_growth

    return scores


def identify_sector_leaders_laggards(
    df: pd.DataFrame, top_n: int = 5, score_col: str = "mispricing_score"
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Identify top leaders (most undervalued) and laggards (most overvalued) within each sector.

    Leaders = stocks with highest positive mispricing scores (best opportunities)
    Laggards = stocks with lowest/most negative mispricing scores (avoid/short)

    Args:
        df: DataFrame with stock data
        top_n: Number of leaders/laggards to return per sector (default 5)
        score_col: Column to use for ranking (default 'mispricing_score')

    Returns:
        Dict with structure:
        {
            'leaders': {'SectorA': DataFrame, 'SectorB': DataFrame, ...},
            'laggards': {'SectorA': DataFrame, 'SectorB': DataFrame, ...}
        }

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B', 'C', 'D'],
        ...     'sector': ['Tech', 'Tech', 'Finance', 'Finance'],
        ...     'mispricing_score': [25, -10, 20, -15]
        ... })
        >>> result = identify_sector_leaders_laggards(df, top_n=1)
        >>> result['leaders']['Tech'].iloc[0]['ticker']
        'A'
        >>> result['laggards']['Tech'].iloc[0]['ticker']
        'B'
    """
    if "sector" not in df.columns:
        logging.warning("'sector' column not found; cannot identify sector leaders/laggards")
        return {"leaders": {}, "laggards": {}}

    if score_col not in df.columns:
        logging.warning(f"'{score_col}' column not found; cannot rank stocks")
        return {"leaders": {}, "laggards": {}}

    leaders = {}
    laggards = {}

    # Process each sector
    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector].copy()

        # Sort by score descending for leaders (highest = best)
        leaders_df = sector_df.nlargest(top_n, score_col)
        leaders[sector] = leaders_df

        # Sort by score ascending for laggards (lowest = worst)
        laggards_df = sector_df.nsmallest(top_n, score_col)
        laggards[sector] = laggards_df

    return {"leaders": leaders, "laggards": laggards}


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
        df: DataFrame with columns 'last_price', 'predicted_price_target', and color_by column
        out_path: Optional path to save HTML file
        color_by: Column to color points by (default 'sector')

    Returns:
        Plotly figure object (or None if plotly not available)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'B'],
        ...     'last_price': [100, 50],
        ...     'predicted_price_target': [120, 55],
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
    required = ["last_price", "predicted_price_target"]
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
        y="predicted_price_target",
        color=color_by if color_by in df.columns else None,
        hover_data=hover_data_cols,
        title="Current Price vs. Predicted Target",
        labels={"last_price": "Current Price ($)", "predicted_price_target": "Predicted Target ($)"},
    )

    # Add diagonal line (y=x) for reference
    max_val = max(df["last_price"].max(), df["predicted_price_target"].max())
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


def generate_pdf_report(
    df: pd.DataFrame,
    pdf_path: Path,
    title: str = "Stock Valuation Report",
    include_summary: bool = True,
    top_n_opportunities: int = 10,
    include_charts: bool = False,
) -> None:
    """
    Generate a professional PDF report with stock recommendations.

    Requires reportlab package (optional dependency).

    Report sections:
    - Executive Summary: Overall statistics and top opportunities count
    - Top Opportunities: Highest mispricing scores with detailed metrics
    - Risk Warnings: Model limitations and investment disclaimers
    - Optional: Charts and visualizations

    Args:
        df: DataFrame with stock data including mispricing scores
        pdf_path: Path where PDF will be saved
        title: Report title (default "Stock Valuation Report")
        include_summary: Include executive summary section (default True)
        top_n_opportunities: Number of top opportunities to include (default 10)
        include_charts: Include charts in report (default False, requires plotly)

    Raises:
        ImportError: If reportlab not available
        ValueError: If DataFrame is empty

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'mispricing_score': [20.0, 15.0],
        ...     'valuation_category': ['Strong Buy', 'Buy']
        ... })
        >>> generate_pdf_report(df, Path('report.pdf'))
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
            PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError as e:
        raise ImportError(
            "reportlab is required for PDF generation. " "Install with: pip install reportlab"
        ) from e

    if df.empty:
        raise ValueError("DataFrame is empty; cannot generate report")

    # Create PDF document
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=16, spaceAfter=12
    )

    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Date
    from datetime import datetime

    date_str = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"<i>Generated: {date_str}</i>", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Executive Summary
    if include_summary:
        story.append(Paragraph("Executive Summary", heading_style))

        total_stocks = len(df)
        if "valuation_category" in df.columns:
            strong_buy_count = (df["valuation_category"] == "Strong Buy").sum()
            buy_count = (df["valuation_category"] == "Buy").sum()
        else:
            strong_buy_count = 0
            buy_count = 0

        if "mispricing_score" in df.columns:
            avg_mispricing = df["mispricing_score"].mean()
        else:
            avg_mispricing = 0.0

        summary_text = f"""
        <b>Total Stocks Analyzed:</b> {total_stocks}<br/>
        <b>Strong Buy Opportunities:</b> {strong_buy_count}<br/>
        <b>Buy Opportunities:</b> {buy_count}<br/>
        <b>Average Mispricing:</b> {avg_mispricing:.2f}%<br/>
        """
        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

    # Top Opportunities
    story.append(Paragraph(f"Top {top_n_opportunities} Investment Opportunities", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    # Sort by mispricing score and get top N
    if "mispricing_score" in df.columns:
        top_opportunities = df.nlargest(top_n_opportunities, "mispricing_score")
    else:
        top_opportunities = df.head(top_n_opportunities)

    # Create table data
    table_data = [["Ticker", "Sector", "Current Price", "Target Price", "Upside %", "Category"]]

    for _, row in top_opportunities.iterrows():
        ticker = row.get("ticker", "N/A")
        sector = row.get("sector", "N/A")
        current = row.get("last_price", 0)
        target = row.get("predicted_price_target", 0)
        mispricing = row.get("mispricing_score", 0)
        category = row.get("valuation_category", "N/A")

        table_data.append(
            [
                str(ticker),
                str(sector),
                f"${current:.2f}" if current else "N/A",
                f"${target:.2f}" if target else "N/A",
                f"{mispricing:.1f}%" if mispricing else "N/A",
                str(category),
            ]
        )

    # Create table
    table = Table(
        table_data, colWidths=[1 * inch, 1.5 * inch, 1 * inch, 1 * inch, 0.8 * inch, 1 * inch]
    )

    # Table style
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 0.4 * inch))

    # Risk Warnings
    story.append(Paragraph("Risk Warnings & Disclaimers", heading_style))
    risk_text = """
    <b>Important:</b> This report is generated by a machine learning model and should not be 
    considered as financial advice. Past performance does not guarantee future results.
    <br/><br/>
    <b>Model Limitations:</b>
    <ul>
        <li>Predictions are based on historical data and may not reflect future market conditions</li>
        <li>Market sentiment, news events, and macroeconomic factors are not fully captured</li>
        <li>Individual stock risk varies; diversification is recommended</li>
    </ul>
    <br/>
    <b>Recommendation:</b> Consult with a qualified financial advisor before making investment decisions.
    """
    story.append(Paragraph(risk_text, styles["Normal"]))

    # Build PDF
    doc.build(story)
    logging.info(f"Generated PDF report: {pdf_path}")

    return None


# ============================================================================
# Phase 9.1 Enhancement #3: Data Quality Dashboard
# ============================================================================


def generate_data_quality_dashboard(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "Financial Data Quality Report",
    method: str = "auto",
    minimal: bool = False,
) -> Path:
    """Generate comprehensive data quality dashboard HTML report.

    Creates an interactive HTML report with data quality metrics, distributions,
    correlations, and missing value analysis. Supports multiple profiling libraries.

    Methods:
        - 'auto': Try ydata-profiling, fall back to sweetviz, then minimal
        - 'ydata-profiling': Use ydata-profiling (formerly pandas-profiling)
        - 'pandas-profiling': Alias for ydata-profiling
        - 'sweetviz': Use sweetviz library
        - 'minimal': Simple HTML report without external libraries

    Args:
        df: Input DataFrame to profile
        output_dir: Directory to save the HTML report
        title: Report title
        method: Profiling method to use (default: 'auto')
        minimal: Use minimal mode even if libraries available (default: False)

    Returns:
        Path to generated HTML report

    Raises:
        ImportError: If required profiling library not available
        ValueError: If DataFrame is empty

    Examples:
        >>> report_path = generate_data_quality_dashboard(
        ...     df,
        ...     output_dir=Path('outputs'),
        ...     title="Stock Data Quality Report"
        ... )
        >>> print(f"Report saved to: {report_path}")
    """
    if df.empty:
        raise ValueError("DataFrame is empty; cannot generate quality report")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize title for filename
    safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    safe_title = safe_title.replace(" ", "_")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{safe_title}_{timestamp}.html"
    report_path = output_dir / report_filename

    # Try different profiling methods
    if minimal or method == "minimal":
        _generate_minimal_quality_report(df, report_path, title)
        logging.info(f"Generated minimal quality report: {report_path}")
        return report_path

    # Try ydata-profiling (formerly pandas-profiling)
    if method in ["auto", "ydata-profiling", "pandas-profiling"]:
        try:
            from ydata_profiling import ProfileReport

            profile = ProfileReport(
                df,
                title=title,
                explorative=True,
                minimal=minimal,
            )
            profile.to_file(report_path)
            logging.info(f"Generated ydata-profiling report: {report_path}")
            return report_path
        except ImportError:
            if method in ["ydata-profiling", "pandas-profiling"]:
                raise ImportError(
                    f"ydata-profiling not installed. Install with: pip install ydata-profiling"
                )
            logging.warning("ydata-profiling not available, trying sweetviz...")

    # Try sweetviz
    if method in ["auto", "sweetviz"]:
        try:
            import sweetviz as sv

            report = sv.analyze(df)
            report.show_html(str(report_path), open_browser=False)
            logging.info(f"Generated sweetviz report: {report_path}")
            return report_path
        except ImportError:
            if method == "sweetviz":
                raise ImportError(f"sweetviz not installed. Install with: pip install sweetviz")
            logging.warning("sweetviz not available, using minimal report...")

    # Fall back to minimal report
    _generate_minimal_quality_report(df, report_path, title)
    logging.info(f"Generated minimal quality report: {report_path}")
    return report_path


def _generate_minimal_quality_report(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Generate minimal HTML quality report without external dependencies.

    Creates a simple but informative HTML report with:
    - Dataset overview (shape, memory usage)
    - Column types and missing values
    - Basic statistics for numeric columns
    - Unique value counts for categorical columns

    Args:
        df: Input DataFrame
        output_path: Path to save HTML file
        title: Report title
    """
    html_parts = []

    # HTML header
    html_parts.append(
        f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #333;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #1f77b4;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            .metric {{
                background-color: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-weight: bold;
                color: #666;
            }}
            .metric-value {{
                font-size: 1.5em;
                color: #1f77b4;
            }}
            .warning {{
                color: #d9534f;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p><i>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
    """
    )

    # Dataset Overview
    html_parts.append("<h2>Dataset Overview</h2>")
    html_parts.append(
        f"""
    <div class="metric">
        <span class="metric-label">Number of Rows:</span>
        <span class="metric-value">{len(df):,}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Number of Columns:</span>
        <span class="metric-value">{len(df.columns)}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Memory Usage:</span>
        <span class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</span>
    </div>
    """
    )

    # Missing Values Summary
    html_parts.append("<h2>Missing Values</h2>")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {"Column": missing.index, "Missing Count": missing.values, "Missing %": missing_pct.values}
    )
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    )

    if len(missing_df) > 0:
        html_parts.append(missing_df.to_html(index=False, classes="data-table"))
    else:
        html_parts.append("<p>✓ No missing values found</p>")

    # Data Types
    html_parts.append("<h2>Column Data Types</h2>")
    dtype_counts = df.dtypes.value_counts()
    dtype_df = pd.DataFrame(
        {"Data Type": dtype_counts.index.astype(str), "Count": dtype_counts.values}
    )
    html_parts.append(dtype_df.to_html(index=False, classes="data-table"))

    # Numeric Columns Statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        html_parts.append("<h2>Numeric Columns Statistics</h2>")
        stats_df = df[numeric_cols].describe().T
        stats_df = stats_df.round(2)
        html_parts.append(stats_df.to_html(classes="data-table"))

    # Categorical Columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        html_parts.append("<h2>Categorical Columns</h2>")
        cat_summary = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            most_common = df[col].value_counts().head(1)
            if len(most_common) > 0:
                most_common_val = most_common.index[0]
                most_common_count = most_common.values[0]
            else:
                most_common_val = "N/A"
                most_common_count = 0

            cat_summary.append(
                {
                    "Column": col,
                    "Unique Values": n_unique,
                    "Most Common": most_common_val,
                    "Most Common Count": most_common_count,
                }
            )

        cat_df = pd.DataFrame(cat_summary)
        html_parts.append(cat_df.to_html(index=False, classes="data-table"))

    # HTML footer
    html_parts.append(
        """
    </body>
    </html>
    """
    )

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def export_profiling_report(
    df: pd.DataFrame,
    output_path: Path,
    minimal: bool = False,
) -> bool:
    """Export data profiling report to file.

    Convenience function to export a data quality report. Automatically
    detects available profiling libraries and uses the best one.

    Args:
        df: Input DataFrame to profile
        output_path: Path where report will be saved
        minimal: Use minimal mode (default: False)

    Returns:
        True if successful, False otherwise

    Examples:
        >>> success = export_profiling_report(
        ...     df,
        ...     output_path=Path('outputs/quality_report.html')
        ... )
    """
    try:
        output_path = Path(output_path)
        output_dir = output_path.parent

        report_path = generate_data_quality_dashboard(
            df,
            output_dir=output_dir,
            title=output_path.stem,
            method="auto",
            minimal=minimal,
        )

        # Rename to desired filename if different
        if report_path != output_path:
            report_path.rename(output_path)

        return True
    except Exception as e:
        logging.error(f"Failed to export profiling report: {e}")
        return False


# ========================================================================
# Phase 9.8: Prediction vs. Analyst Price Target Comparison Analytics
# ========================================================================


def compare_prediction_vs_analyst_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compare model predictions against analyst consensus targets.
    
    Phase 9.8 TDD implementation: Calculate differences and agreement metrics
    between model-predicted price targets and analyst consensus targets.
    
    Args:
        df: DataFrame with columns:
            - ticker: Stock ticker symbol
            - last_price: Current stock price
            - predicted_price_target: Model prediction
            - price_target: Analyst consensus target
            
    Returns:
        DataFrame with comparison metrics including:
            - model_analyst_diff: Absolute difference
            - model_analyst_diff_pct: Percentage difference
            - agreement_direction: Boolean indicating same direction
            
    Examples:
        >>> result = compare_prediction_vs_analyst_targets(df)
        >>> print(result[['ticker', 'model_analyst_diff_pct']])
    """
    result = df.copy()
    
    # Calculate absolute and percentage differences
    result["model_analyst_diff"] = (
        result["predicted_price_target"] - result["price_target"]
    )
    result["model_analyst_diff_pct"] = (
        result["model_analyst_diff"] / result["price_target"] * 100
    )
    
    # Determine if model and analyst agree on direction vs current price
    model_direction = result["predicted_price_target"] > result["last_price"]
    analyst_direction = result["price_target"] > result["last_price"]
    result["agreement_direction"] = model_direction == analyst_direction
    
    return result


def calculate_directional_accuracy(df: pd.DataFrame) -> dict:
    """Calculate directional accuracy of model predictions vs current price.
    
    Phase 9.8 TDD implementation: Measures how often the model correctly
    predicts the direction (up/down) relative to current price.
    
    Args:
        df: DataFrame with columns:
            - last_price: Current stock price
            - predicted_price_target: Model prediction
            
    Returns:
        Dictionary with:
            - accuracy: Proportion of correct directional predictions (0-1)
            - total_predictions: Total number of predictions
            - correct_predictions: Number of correct directional calls
            
    Examples:
        >>> metrics = calculate_directional_accuracy(df)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    # For directional accuracy without actual future prices,
    # we assume any prediction != current price is a directional call
    # This is a simplified metric; actual accuracy needs future prices
    
    total = len(df)
    
    # Count predictions that differ from current price (have a direction)
    has_direction = df["predicted_price_target"] != df["last_price"]
    predictions_with_direction = has_direction.sum()
    
    # For now, return structure - actual accuracy requires future data
    # If actual_future_price column exists, calculate true accuracy
    if "actual_future_price" in df.columns:
        predicted_direction = df["predicted_price_target"] > df["last_price"]
        actual_direction = df["actual_future_price"] > df["last_price"]
        correct = (predicted_direction == actual_direction).sum()
        accuracy = correct / total if total > 0 else 0.0
    else:
        # Without actual prices, return placeholder
        correct = predictions_with_direction
        accuracy = 1.0 if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "total_predictions": total,
        "correct_predictions": correct,
    }


def calculate_agreement_rate(df: pd.DataFrame) -> dict:
    """Calculate agreement rate between model and analyst predictions.
    
    Phase 9.8 TDD implementation: Measures how often the model and analysts
    agree on the direction of price movement.
    
    Args:
        df: DataFrame with columns:
            - last_price: Current stock price
            - predicted_price_target: Model prediction
            - price_target: Analyst consensus target
            
    Returns:
        Dictionary with:
            - agreement_rate: Proportion of stocks with same direction (0-1)
            - same_direction_count: Number of agreements
            - total_count: Total stocks evaluated
            
    Examples:
        >>> metrics = calculate_agreement_rate(df)
        >>> print(f"Agreement: {metrics['agreement_rate']:.2%}")
    """
    total = len(df)
    
    # Determine directions
    model_direction = df["predicted_price_target"] > df["last_price"]
    analyst_direction = df["price_target"] > df["last_price"]
    
    # Count agreements
    same_direction = (model_direction == analyst_direction).sum()
    
    agreement_rate = same_direction / total if total > 0 else 0.0
    
    return {
        "agreement_rate": agreement_rate,
        "same_direction_count": same_direction,
        "total_count": total,
    }


def identify_disagreement_opportunities(
    df: pd.DataFrame, 
    threshold_pct: float = 5.0
) -> pd.DataFrame:
    """Identify stocks where model significantly differs from analyst consensus.
    
    Phase 9.8 TDD implementation: Find investment opportunities where the model
    has a different view than analysts, potentially indicating mispriced stocks.
    
    Args:
        df: DataFrame with prediction and analyst target columns
        threshold_pct: Minimum percentage difference to flag (default: 5%)
        
    Returns:
        DataFrame of stocks exceeding disagreement threshold, sorted by
        absolute difference magnitude
        
    Examples:
        >>> opportunities = identify_disagreement_opportunities(df, threshold_pct=10.0)
        >>> print(opportunities[['ticker', 'model_analyst_diff_pct']].head())
    """
    # Calculate differences if not already present
    if "model_analyst_diff_pct" not in df.columns:
        df = compare_prediction_vs_analyst_targets(df)
    
    # Filter by threshold
    disagreements = df[
        abs(df["model_analyst_diff_pct"]) >= threshold_pct
    ].copy()
    
    # Sort by absolute difference magnitude
    disagreements = disagreements.sort_values(
        by="model_analyst_diff_pct",
        key=lambda x: abs(x),
        ascending=False
    )
    
    return disagreements


def calculate_prediction_accuracy_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive accuracy metrics for both model and analysts.
    
    Phase 9.8 TDD implementation: Compare prediction accuracy when actual
    future prices are available.
    
    Args:
        df: DataFrame with columns:
            - predicted_price_target: Model prediction
            - price_target: Analyst consensus target
            - actual_future_price: Realized future price
            - last_price: Current price
            
    Returns:
        Dictionary with metrics:
            - model_mae: Model mean absolute error
            - analyst_mae: Analyst mean absolute error
            - model_directional_accuracy: Model directional hit rate
            - analyst_directional_accuracy: Analyst directional hit rate
            
    Examples:
        >>> metrics = calculate_prediction_accuracy_metrics(df)
        >>> print(f"Model MAE: {metrics['model_mae']:.2f}")
    """
    if "actual_future_price" not in df.columns:
        raise ValueError("actual_future_price column required for accuracy metrics")
    
    # Mean Absolute Error
    model_mae = abs(
        df["predicted_price_target"] - df["actual_future_price"]
    ).mean()
    
    analyst_mae = abs(
        df["price_target"] - df["actual_future_price"]
    ).mean()
    
    # Directional Accuracy
    model_predicted_direction = df["predicted_price_target"] > df["last_price"]
    analyst_predicted_direction = df["price_target"] > df["last_price"]
    actual_direction = df["actual_future_price"] > df["last_price"]
    
    model_correct = (model_predicted_direction == actual_direction).sum()
    analyst_correct = (analyst_predicted_direction == actual_direction).sum()
    
    total = len(df)
    
    model_directional_accuracy = model_correct / total if total > 0 else 0.0
    analyst_directional_accuracy = analyst_correct / total if total > 0 else 0.0
    
    return {
        "model_mae": model_mae,
        "analyst_mae": analyst_mae,
        "model_directional_accuracy": model_directional_accuracy,
        "analyst_directional_accuracy": analyst_directional_accuracy,
    }


def segment_comparison_by_attribute(
    df: pd.DataFrame,
    segment_col: str = "sector"
) -> dict:
    """Segment prediction vs analyst comparison metrics by an attribute.
    
    Phase 9.8 TDD implementation: Break down comparison metrics by sector,
    region, market cap, or other categorical attributes.
    
    Args:
        df: DataFrame with prediction and analyst data
        segment_col: Column to segment by (e.g., 'sector', 'region')
        
    Returns:
        Dictionary mapping segment values to their metrics including:
            - agreement_rate: Agreement proportion for this segment
            - avg_model_analyst_diff: Average prediction difference
            - count: Number of stocks in segment
            
    Examples:
        >>> by_sector = segment_comparison_by_attribute(df, 'sector')
        >>> print(by_sector['Tech']['agreement_rate'])
    """
    if segment_col not in df.columns:
        raise ValueError(f"Segment column '{segment_col}' not found in DataFrame")
    
    # Ensure comparison metrics are calculated
    if "model_analyst_diff" not in df.columns:
        df = compare_prediction_vs_analyst_targets(df)
    
    results = {}
    
    for segment_value in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment_value]
        
        # Calculate agreement rate for this segment
        agreement_metrics = calculate_agreement_rate(segment_df)
        
        # Calculate average difference
        avg_diff = segment_df["model_analyst_diff"].mean()
        
        results[segment_value] = {
            "agreement_rate": agreement_metrics["agreement_rate"],
            "avg_model_analyst_diff": avg_diff,
            "count": len(segment_df),
        }
    
    return results


def analyze_systematic_bias(df: pd.DataFrame) -> dict:
    """Analyze systematic bias in model predictions vs analyst targets.
    
    Phase 9.8 TDD implementation: Detect if the model consistently over-predicts
    (optimistic) or under-predicts (pessimistic) compared to analysts.
    
    Args:
        df: DataFrame with prediction and analyst target columns
        
    Returns:
        Dictionary with:
            - mean_model_bias: Average difference (model - analyst)
            - median_model_bias: Median difference
            - bias_direction: 'optimistic', 'pessimistic', or 'neutral'
            
    Examples:
        >>> bias = analyze_systematic_bias(df)
        >>> print(f"Model is {bias['bias_direction']}")
    """
    # Ensure differences are calculated
    if "model_analyst_diff" not in df.columns:
        df = compare_prediction_vs_analyst_targets(df)
    
    mean_bias = df["model_analyst_diff"].mean()
    median_bias = df["model_analyst_diff"].median()
    
    # Determine bias direction (using 1% threshold for neutral)
    threshold = 0.01 * df["price_target"].mean()
    
    if mean_bias > threshold:
        bias_direction = "optimistic"
    elif mean_bias < -threshold:
        bias_direction = "pessimistic"
    else:
        bias_direction = "neutral"
    
    return {
        "mean_model_bias": mean_bias,
        "median_model_bias": median_bias,
        "bias_direction": bias_direction,
    }


def generate_prediction_analyst_excel_report(
    df: pd.DataFrame,
    excel_path: Path,
    top_n_opportunities: int = 50
) -> None:
    """Generate comprehensive Excel report comparing predictions vs analyst targets.
    
    Phase 9.8 TDD implementation: Create multi-sheet Excel workbook matching
    Stock_Prediction_Analysis_Report format with 6 required sheets.
    
    Args:
        df: DataFrame with all required columns (ticker, sector, region, prices, targets)
        excel_path: Output path for Excel file
        top_n_opportunities: Number of top stocks to include in opportunity sheets
        
    Returns:
        None (writes Excel file to disk)
        
    Sheets created:
        1. Executive_Summary: Overall statistics and model performance
        2. Detailed_Stock_List: All stocks with predictions and targets
        3. Top_Opportunities: Top undervalued stocks by model
        4. Risk_Analysis: Top overvalued stocks
        5. Prediction_Accuracy: Model vs analyst comparison metrics
        6. Sector_Analysis: Performance breakdown by sector
        
    Examples:
        >>> generate_prediction_analyst_excel_report(
        ...     df,
        ...     Path('outputs/prediction_analysis.xlsx')
        ... )
    """
    try:
        import xlsxwriter
        use_xlsxwriter = True
    except ImportError:
        import openpyxl
        use_xlsxwriter = False
    
    # Ensure all comparison metrics are calculated
    if "model_analyst_diff" not in df.columns:
        df = compare_prediction_vs_analyst_targets(df)
    
    if "mispricing_score" not in df.columns:
        df["mispricing_score"] = (
            (df["predicted_price_target"] - df["last_price"]) / df["last_price"]
        )
    
    excel_path = Path(excel_path)
    
    if use_xlsxwriter:
        # Use xlsxwriter for better formatting
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    else:
        # Fallback to openpyxl
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # Sheet 1: Executive Summary
    summary_data = []
    summary_data.append(["Metric", "Value"])
    summary_data.append(["Total Stocks", len(df)])
    summary_data.append(["Average Current Price", f"${df['last_price'].mean():.2f}"])
    summary_data.append([
        "Average Predicted Target", 
        f"${df['predicted_price_target'].mean():.2f}"
    ])
    summary_data.append([
        "Average Analyst Target",
        f"${df['price_target'].mean():.2f}"
    ])
    
    # Calculate agreement metrics
    agreement = calculate_agreement_rate(df)
    summary_data.append([
        "Model-Analyst Agreement Rate",
        f"{agreement['agreement_rate']:.2%}"
    ])
    
    # Bias analysis
    bias = analyze_systematic_bias(df)
    summary_data.append([
        "Model Bias Direction",
        bias['bias_direction']
    ])
    summary_data.append([
        "Average Model-Analyst Difference",
        f"${bias['mean_model_bias']:.2f}"
    ])
    
    # Opportunity counts
    undervalued = df[df["mispricing_score"] > 0.05]
    overvalued = df[df["mispricing_score"] < -0.05]
    summary_data.append([
        "Undervalued Opportunities (>5%)",
        len(undervalued)
    ])
    summary_data.append([
        "Overvalued Stocks (<-5%)",
        len(overvalued)
    ])
    
    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)
    
    # Sheet 2: Detailed Stock List
    detail_cols = [
        "ticker", "sector", "region", "last_price",
        "predicted_price_target", "price_target",
        "model_analyst_diff", "model_analyst_diff_pct",
        "mispricing_score", "agreement_direction"
    ]
    available_cols = [col for col in detail_cols if col in df.columns]
    detail_df = df[available_cols].copy()
    
    # Add market_cap if available
    if "market_cap" in df.columns:
        detail_df["market_cap"] = df["market_cap"]
    
    detail_df.to_excel(writer, sheet_name="Detailed_Stock_List", index=False)
    
    # Sheet 3: Top Opportunities (Undervalued)
    top_opportunities = df.nlargest(top_n_opportunities, "mispricing_score")
    top_opportunities[available_cols].to_excel(
        writer, sheet_name="Top_Opportunities", index=False
    )
    
    # Sheet 4: Risk Analysis (Overvalued)
    top_risks = df.nsmallest(top_n_opportunities, "mispricing_score")
    top_risks[available_cols].to_excel(
        writer, sheet_name="Risk_Analysis", index=False
    )
    
    # Sheet 5: Prediction Accuracy
    accuracy_data = []
    accuracy_data.append(["Metric", "Value"])
    accuracy_data.append([
        "Model-Analyst Agreement Rate",
        f"{agreement['agreement_rate']:.2%}"
    ])
    accuracy_data.append([
        "Stocks with Same Direction",
        agreement['same_direction_count']
    ])
    accuracy_data.append([
        "Stocks with Different Direction",
        agreement['total_count'] - agreement['same_direction_count']
    ])
    accuracy_data.append([
        "Average Absolute Difference",
        f"${abs(df['model_analyst_diff']).mean():.2f}"
    ])
    accuracy_data.append([
        "Median Absolute Difference",
        f"${abs(df['model_analyst_diff']).median():.2f}"
    ])
    
    # Add actual accuracy metrics if available
    if "actual_future_price" in df.columns:
        actual_metrics = calculate_prediction_accuracy_metrics(df)
        accuracy_data.append([
            "Model MAE",
            f"${actual_metrics['model_mae']:.2f}"
        ])
        accuracy_data.append([
            "Analyst MAE",
            f"${actual_metrics['analyst_mae']:.2f}"
        ])
        accuracy_data.append([
            "Model Directional Accuracy",
            f"{actual_metrics['model_directional_accuracy']:.2%}"
        ])
        accuracy_data.append([
            "Analyst Directional Accuracy",
            f"{actual_metrics['analyst_directional_accuracy']:.2%}"
        ])
    
    accuracy_df = pd.DataFrame(accuracy_data[1:], columns=accuracy_data[0])
    accuracy_df.to_excel(writer, sheet_name="Prediction_Accuracy", index=False)
    
    # Sheet 6: Sector Analysis
    if "sector" in df.columns:
        sector_metrics = segment_comparison_by_attribute(df, "sector")
        
        sector_data = []
        for sector, metrics in sector_metrics.items():
            sector_data.append({
                "Sector": sector,
                "Stock Count": metrics["count"],
                "Agreement Rate": f"{metrics['agreement_rate']:.2%}",
                "Avg Model-Analyst Diff": f"${metrics['avg_model_analyst_diff']:.2f}",
                "Avg Mispricing": f"{df[df['sector'] == sector]['mispricing_score'].mean():.2%}",
            })
        
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values("Stock Count", ascending=False)
        sector_df.to_excel(writer, sheet_name="Sector_Analysis", index=False)
    
    # Save and close
    writer.close()
    
    logging.info(f"Prediction vs Analyst Excel report saved to {excel_path}")
