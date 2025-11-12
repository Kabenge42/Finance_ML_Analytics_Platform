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
from datetime import datetime
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


def calculate_mispricing_score(
    df: pd.DataFrame, predicted_col: str = "predicted_price_target", current_col: str = "last_price"
) -> pd.DataFrame:
    """Calculate mispricing score for each stock.

    Formula: (predicted_price_target - last_price) / last_price

    Positive score = undervalued (predicted > current)
    Negative score = overvalued (predicted < current)

    Args:
        df: DataFrame with stock data
        predicted_col: Name of predicted price column (default: "predicted_price_target")
        current_col: Name of current price column (default: "last_price")

    Returns:
        DataFrame with added 'mispricing_pct' column

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = [predicted_col, current_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    result_df = df.copy()
    mispricing = (df[predicted_col] - df[current_col]) / df[current_col]
    result_df["mispricing_pct"] = mispricing * 100
    result_df["mispricing_score"] = (
        mispricing  # Alias for backward compatibility with rank functions
    )
    return result_df


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
    # Exclude datetime columns to prevent errors in statistical analysis (scipy.stats operations)
    try:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    except AttributeError:
        # Fallback: treat no columns as numeric if dtype access fails
        logging.warning(
            "simple_eda: dtype inspection failed due to AttributeError; skipping numeric stats"
        )
        numeric_cols = []
    except (KeyError, ValueError, TypeError) as e:
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
    except (KeyError, ValueError, TypeError) as e:
        logging.warning("simple_eda: basic stats computation failed: %s", e)
        basic_stats = {}
        numeric_count = 0
        categorical_count = 0

    # Safe value_counts extraction
    def _safe_counts(series_name: str):
        try:
            return df[series_name].value_counts().to_dict()
        except (KeyError, AttributeError, ValueError, TypeError):
            return {}

    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols_count": numeric_count,
        "categorical_cols_count": categorical_count,
        "numeric_columns": numeric_cols,  # Add for test compatibility
        "categorical_columns": [
            c for c in df.columns if c not in numeric_cols
        ],  # Add for test compatibility
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                except (KeyError, ValueError, TypeError, AttributeError):
                    outliers[col] = {"count": 0, "percentage": 0.0}
            summary["outlier_detection"] = outliers
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logging.warning("Distance correlation calculation failed: %s", e)
                    corr_analysis["distance"] = {}

                summary["correlation_analysis"] = corr_analysis
            else:
                summary["correlation_analysis"] = {}
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                    except (KeyError, ValueError, TypeError, AttributeError):
                        top_corr[method] = []
                summary["top_correlations"] = top_corr
            else:
                summary["top_correlations"] = {}
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                    except (KeyError, ValueError, TypeError, AttributeError):
                        pass
                summary["sector_comparison_tests"] = sector_tests
            else:
                summary["sector_comparison_tests"] = {}
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                        "explained_variance_ratio": pca_result["explained_variance_ratio"],
                        "cumulative_variance": pca_result["cumulative_variance"],
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
        except (KeyError, ValueError, TypeError, AttributeError, ImportError) as e:
            logging.warning("Multivariate analysis failed: %s", e)
            summary["multivariate_analysis"] = {}
    else:
        summary["multivariate_analysis"] = {}

    # Persist summary and plots only if an output directory is provided
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # If out_dir cannot be created, continue without writing files
            logging.warning("Could not create out_dir=%s; skipping file outputs", out_dir)
        else:
            # Convert summary to JSON-serializable format
            def convert_to_serializable(obj):
                """
                Convert non-serializable objects to JSON-compatible types.

                Handles:
                - NumPy types (int, float, bool, ndarray)
                - Pandas types (Timestamp, Timedelta, Series, Index)
                - NaN/Infinity values
                - Nested structures (dict, list)
                """
                # Handle numpy scalar types
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                # Handle pandas types
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                if isinstance(obj, pd.Timedelta):
                    # Convert to string for human readability
                    # Alternative: return obj.total_seconds() for numeric representation
                    return str(obj)
                if isinstance(obj, (pd.Series, pd.Index)):
                    return obj.tolist()

                # Handle dictionaries recursively
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}

                # Handle lists/tuples recursively
                if isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]

                # Handle NaN values
                if pd.isna(obj):
                    return None

                # Return as-is for JSON-compatible types
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj

                # Fallback: convert to string representation
                return str(obj)

            out_path = out_dir / "eda_summary.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(convert_to_serializable(summary), f, indent=2)
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
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
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


def export_predictions_to_csv(
    df: pd.DataFrame,
    csv_path: Path,
    required_columns: Optional[list] = None,
    compute_mispricing: bool = True,
    export_all_columns: bool = False,
) -> Path:
    """Export standardized predictions CSV for dashboards and downstream tools.

    Expected columns in output CSV (when export_all_columns=False):
    ticker, name, exchange, sector, region, last_price, price_target,
    predicted_price_target, market_cap, mispricing_score

    Args:
        df: DataFrame containing predictions and related fields
        csv_path: Destination path
        required_columns: Optional explicit list of columns to include/order
        compute_mispricing: If True and mispricing_score missing, compute it as
            (predicted_price_target - last_price) / last_price
        export_all_columns: If True, export all columns from the dataframe (overrides required_columns)

    Returns:
        The path to the written CSV file.
    """
    # Normalize column names to lower case for matching
    df_cols_lower = {c.lower(): c for c in df.columns}

    # Compute mispricing_score if requested and possible
    if compute_mispricing and "mispricing_score" not in df_cols_lower:
        if "predicted_price_target" in df_cols_lower and "last_price" in df_cols_lower:
            try:
                pred_col = df_cols_lower["predicted_price_target"]
                last_col = df_cols_lower["last_price"]
                # Avoid division by zero
                denom = df[last_col].replace({0: np.nan}).astype(float)
                df["mispricing_score"] = (
                    df[pred_col].astype(float) - df[last_col].astype(float)
                ) / denom
            except Exception as e:
                logging.warning("Could not compute mispricing_score: %s", e)
        # refresh map after potential addition
        df_cols_lower = {c.lower(): c for c in df.columns}

    # If export_all_columns is True, export everything
    if export_all_columns:
        out_df = df.copy()
        # Ensure numeric types where applicable for common columns
        numeric_cols = [
            "last_price",
            "price_target",
            "predicted_price_target",
            "market_cap",
            "mispricing_score",
            "mispricing_pct",
            "prediction_error",
            "prediction_error_pct",
            "model_analyst_diff_pct",
            "p_e",
            "p_b",
            "roe",
            "roa",
            "ev_ebitda",
            "operating_margin",
            "net_margin",
            "debt_to_equity",
            "current_ratio",
        ]
        for num_col in numeric_cols:
            if num_col in out_df.columns:
                with np.errstate(all="ignore"):
                    out_df[num_col] = pd.to_numeric(out_df[num_col], errors="coerce")
    else:
        # Original behavior: use default or specified columns
        default_required = [
            "ticker",
            "name",
            "exchange",
            "sector",
            "region",
            "last_price",
            "price_target",
            "predicted_price_target",
            "market_cap",
            "mispricing_score",
        ]
        use_columns = required_columns or default_required

        # Build column mapping from desired lowercase to actual columns present
        available = {}
        missing = []
        for col in use_columns:
            if col in df_cols_lower:
                available[col] = df_cols_lower[col]
            else:
                missing.append(col)

        if missing:
            logging.info(
                "Some required columns are missing and will be filled with NA: %s", missing
            )
            # Create placeholders for missing columns with NA
            for col in missing:
                df[col] = np.nan
                available[col] = col

        # Reorder and select
        ordered_cols = [available[c] for c in use_columns]
        out_df = df[ordered_cols].copy()

        # Ensure numeric types where applicable
        for num_col in [
            "last_price",
            "price_target",
            "predicted_price_target",
            "market_cap",
            "mispricing_score",
        ]:
            if num_col in out_df.columns:
                with np.errstate(all="ignore"):
                    out_df[num_col] = pd.to_numeric(out_df[num_col], errors="coerce")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    logging.info("Exported predictions to CSV: %s (%d columns)", csv_path, len(out_df.columns))
    return csv_path


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
            labels={
                "last_price": "Current Price",
                "predicted_price_target": "Predicted Target Price",
            },
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
    X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, top_k: int = 20, random_state: int = 42
) -> pd.DataFrame:
    """Calculate feature importance using Random Forest.

    Phase 9.2 enhancement for feature importance analysis.
    Updated to return DataFrame format for notebook compatibility.

    Args:
        X: Feature DataFrame
        y: Target variable
        n_estimators: Number of trees in the forest
        top_k: Number of top features to return
        random_state: Random state for reproducibility

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance descending
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Handle missing values
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median()) if pd.api.types.is_numeric_dtype(y) else y.fillna("missing")

    # Choose appropriate model with optimized parameters
    if pd.api.types.is_numeric_dtype(y_clean):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    # Fit model
    model.fit(X_clean, y_clean)

    # Extract feature importances
    importances = model.feature_importances_

    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({"feature": X.columns, "importance": importances})

    # Sort by importance and return top_k
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_k)
    importance_df = importance_df.reset_index(drop=True)

    return importance_df


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

    from sklearn.ensemble import RandomForestRegressor
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

    # Safely convert to list (handle case where it's already a list or array)
    explained_var = pca.explained_variance_ratio_
    if isinstance(explained_var, list):
        explained_var_list = explained_var
    else:
        explained_var_list = explained_var.tolist()

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    if isinstance(cumulative_var, list):
        cumulative_var_list = cumulative_var
    else:
        cumulative_var_list = cumulative_var.tolist()

    return {
        "components": component_df,
        "explained_variance_ratio": explained_var_list,
        "cumulative_variance": cumulative_var_list,
        "n_components": n_components,
        "feature_names": X.columns.tolist(),
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
    include_financial_dashboard: bool = False,
    include_quality_alerts: bool = False,
) -> dict:
    """Generate comprehensive EDA report.

    Phase 9.2 enhancement for automated EDA with financial metrics dashboard,
    data quality alerts, and comprehensive hypothesis testing.

    Args:
        df: DataFrame to analyze
        output_path: Optional path to save JSON report
        include_correlations: Include correlation analysis
        include_distributions: Include distribution analysis
        include_statistical_tests: Include comprehensive hypothesis tests
        include_financial_dashboard: Include financial metrics dashboard (Phase 9.2)
        include_quality_alerts: Include data quality alerts (Phase 9.2)

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

    # Phase 9.2: Financial Metrics Dashboard
    if include_financial_dashboard:
        try:
            report["financial_dashboard"] = calculate_financial_metrics_dashboard(df)
        except Exception as e:
            logging.warning(f"Failed to generate financial dashboard: {e}")
            report["financial_dashboard"] = {"error": str(e)}

    # Phase 9.2: Data Quality Alerts
    if include_quality_alerts:
        try:
            report["quality_alerts"] = generate_data_quality_alerts(df)
        except Exception as e:
            logging.warning(f"Failed to generate quality alerts: {e}")
            report["quality_alerts"] = []

    # Correlations
    if include_correlations and len(numeric_cols) > 1:
        corr_matrix = calculate_correlation_matrix(df, numeric_cols[:20])  # Limit to 20 cols
        report["correlations"] = {
            "matrix": corr_matrix.to_dict(),
            "top_pairs": find_top_correlations(corr_matrix, n_top=10),
        }

    # Distributions
    if include_distributions and len(numeric_cols) > 0:
        dist_cols = numeric_cols[:10]  # Analyze first 10 numeric columns
        report["distributions"] = {
            "normality_tests": test_normality(df, dist_cols),
            "skew_kurtosis": calculate_skewness_kurtosis(df, dist_cols).to_dict(),
        }

    # Phase 9.2: Comprehensive Statistical Hypothesis Tests
    if include_statistical_tests:
        try:
            # Sector-based hypothesis tests
            if "sector" in df.columns and len(numeric_cols) > 0:
                metrics_to_test = [
                    m for m in numeric_cols[:5] if m in df.columns
                ]  # Test top 5 metrics
                sector_tests = perform_comprehensive_hypothesis_tests(
                    df, group_column="sector", metrics=metrics_to_test
                )

                # Region-based hypothesis tests
                region_tests = {}
                if "region" in df.columns:
                    region_tests = perform_comprehensive_hypothesis_tests(
                        df, group_column="region", metrics=metrics_to_test
                    )

                report["hypothesis_tests"] = {
                    "sector_tests": sector_tests.get("sector_tests", {}),
                    "region_tests": region_tests.get("region_tests", {}),
                }

                # Market efficiency test (if price data available)
                if "last_price" in df.columns and "price_target" in df.columns:
                    report["hypothesis_tests"]["market_efficiency"] = (
                        test_market_efficiency_hypothesis(df)
                    )
            else:
                # Fallback to basic statistical tests
                test_col = numeric_cols[0]
                report["hypothesis_tests"] = {
                    "sector_comparison": (
                        compare_sector_means(df, test_col, "sector")
                        if "sector" in df.columns
                        else {}
                    )
                }
        except Exception as e:
            logging.warning(f"Failed to perform hypothesis tests: {e}")
            report["hypothesis_tests"] = {"error": str(e)}

    # Save JSON report if requested
    if output_path:
        try:
            import json

            output_path.write_text(json.dumps(report, indent=2, default=str))
            logging.info(f"EDA report saved to {output_path}")
        except Exception as e:
            logging.warning(f"Failed to save report: {e}")

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


# ============================================================================
# SHAP Detailed Analysis
# ============================================================================


def compute_shap_values(model, X, model_type="auto", n_samples=100):
    """
    Compute SHAP values for a given model and dataset.

    Args:
        model: Trained model (scikit-learn compatible)
        X: Feature data (DataFrame or array)
        model_type: Type of explainer to use
            - "auto": Automatically detect best explainer
            - "tree": Use TreeExplainer (for tree-based regression)
            - "kernel": Use KernelExplainer (model-agnostic, slower)
            - "linear": Use LinearExplainer (for linear regression only)
        n_samples: Number of background samples for KernelExplainer

    Returns:
        dict with keys:
            - "shap_values": SHAP values array
            - "expected_value": Base value
            - "feature_names": List of feature names
    """
    import shap

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Get base model if this is a stacking ensemble
    base_model = getattr(model, "final_estimator_", model)

    # Auto-detect model type if requested
    if model_type == "auto":
        model_type = _detect_model_type(base_model)

    # Compute SHAP values based on model type
    if model_type == "tree":
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
            model_type = "kernel"

    if model_type == "linear":
        try:
            # For stacking regression, we need to check if final estimator is truly linear
            if hasattr(base_model, "coef_"):
                n_coef = len(base_model.coef_)
                n_features = X.shape[1]

                if n_coef != n_features:
                    print(
                        f"Warning: Coefficient count ({n_coef}) doesn't match feature count ({n_features}). Using KernelExplainer instead."
                    )
                    model_type = "kernel"
                else:
                    explainer = shap.LinearExplainer(base_model, X)
                    shap_values = explainer.shap_values(X)
            else:
                print("Model doesn't have coef_ attribute. Using KernelExplainer instead.")
                model_type = "kernel"
        except Exception as e:
            print(f"LinearExplainer failed: {e}. Falling back to KernelExplainer.")
            model_type = "kernel"

    if model_type == "kernel":
        # Use a subset for background distribution
        background = shap.sample(X, min(n_samples, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    # Handle multi-output case (for multi-class classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Use first class for visualization

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": list(X.columns) if hasattr(X, "columns") else None,
    }


def _detect_model_type(model):
    """Detect appropriate SHAP explainer type for a model."""
    model_class = type(model).__name__

    # Tree-based regression
    tree_models = [
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "ExtraTrees",
        "DecisionTree",
    ]
    if any(tree in model_class for tree in tree_models):
        return "tree"

    # Linear regression
    linear_models = ["Linear", "Ridge", "Lasso", "ElasticNet", "SGD"]
    if any(linear in model_class for linear in linear_models):
        return "linear"

    # Default to kernel for everything else
    return "kernel"


def create_shap_summary_plot(model, X, output_path=None, model_type="auto", n_samples=100):
    """
    Create SHAP summary plot showing feature importance.

    Args:
        model: Trained model
        X: Feature data
        output_path: Path to save plot (optional)
        model_type: Type of explainer ("auto", "tree", "kernel", "linear")
        n_samples: Number of samples for KernelExplainer
    """
    import shap
    import matplotlib.pyplot as plt

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values with automatic fallback
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]

    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"SHAP summary plot saved to {output_path}")

    plt.close()

    print("✓ SHAP analysis complete")


def create_shap_waterfall_plot(
    model, X, sample_idx=0, output_path=None, model_type="tree", n_samples=100
):
    """
    Create SHAP waterfall plot for individual prediction explanation.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        output_path: Path to save plot
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        None (saves plot to file)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]
    expected_value = result["expected_value"]

    # Create waterfall plot for specific sample
    if isinstance(shap_values, list):
        # Multi-output model, use first output
        shap_values_sample = shap_values[0][sample_idx]
    else:
        shap_values_sample = shap_values[sample_idx]

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values_sample,
        base_values=expected_value,
        data=X.iloc[sample_idx].values,
        feature_names=list(X.columns),
    )

    shap.plots.waterfall(explanation, show=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def create_shap_dependence_plot(
    model, X, feature, output_path=None, model_type="tree", n_samples=100
):
    """
    Create SHAP dependence plot showing feature interactions.

    Args:
        model: Trained model
        X: Feature matrix
        feature: Feature name or index for dependence plot
        output_path: Path to save plot
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        None (saves plot to file)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]

    # Create dependence plot
    shap.dependence_plot(feature, shap_values, X, show=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def analyze_shap_by_sector(model, X, sectors, model_type="tree", n_samples=100):
    """
    Analyze SHAP values separately by sector.

    Args:
        model: Trained model
        X: Feature matrix
        sectors: Series with sector labels
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        dict: SHAP analysis for each sector
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    results = {}

    for sector in sectors.unique():
        sector_mask = sectors == sector
        X_sector = X[sector_mask]

        if len(X_sector) > 0:
            result = compute_shap_values(
                model, X_sector, model_type=model_type, n_samples=n_samples
            )

            # Compute mean absolute SHAP values for feature importance
            shap_values = result["shap_values"]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            results[sector] = {
                "shap_values": shap_values,
                "expected_value": result["expected_value"],
                "feature_importance": dict(zip(result["feature_names"], mean_abs_shap)),
                "n_samples": len(X_sector),
            }

    return results


# ============================================================================
# LIME Integration
# ============================================================================


def explain_with_lime(model, X, sample_idx=0, output_path=None, n_features=10):
    """
    Generate LIME explanation for a single prediction.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        output_path: Optional path to save HTML explanation
        n_features: Number of features to show in explanation

    Returns:
        dict: LIME explanation with feature weights
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise ImportError("LIME library is required. Install with: pip install lime")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X.values, feature_names=list(X.columns), mode="regression", random_state=42
    )

    # Generate explanation for sample
    explanation = explainer.explain_instance(
        X.iloc[sample_idx].values, model.predict, num_features=n_features
    )

    # Extract feature weights
    feature_weights = dict(explanation.as_list())

    # Save HTML if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation.save_to_file(str(output_path))

    return {
        "feature_weights": feature_weights,
        "prediction": float(model.predict(X.iloc[[sample_idx]])[0]),
        "intercept": explanation.intercept[0] if hasattr(explanation, "intercept") else 0.0,
        "score": explanation.score if hasattr(explanation, "score") else None,
    }


def compare_lime_shap_consistency(model, X, sample_idx=0, model_type="tree", n_features=10):
    """
    Compare LIME and SHAP explanations for consistency.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        model_type: Type of model for SHAP
        n_features: Number of features to compare

    Returns:
        dict: Comparison results with correlation metric
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Get LIME explanation
    lime_result = explain_with_lime(model, X, sample_idx=sample_idx, n_features=n_features)
    lime_weights = lime_result["feature_weights"]

    # Get SHAP explanation
    shap_result = compute_shap_values(model, X.iloc[[sample_idx]], model_type=model_type)
    shap_values = shap_result["shap_values"]

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_dict = dict(zip(X.columns, shap_values[0] if len(shap_values.shape) > 1 else shap_values))

    # Compare feature importances
    common_features = set(lime_weights.keys()) & set(shap_dict.keys())

    if len(common_features) > 0:
        lime_vals = [lime_weights[f] for f in common_features]
        shap_vals = [shap_dict[f] for f in common_features]

        # Calculate correlation
        correlation = np.corrcoef(lime_vals, shap_vals)[0, 1]
    else:
        correlation = np.nan

    return {
        "lime_weights": lime_weights,
        "shap_values": shap_dict,
        "correlation": float(correlation) if not np.isnan(correlation) else None,
        "common_features": list(common_features),
    }


# ============================================================================
# Model Comparison Framework
# ============================================================================


def create_model_comparison_table(models, X, y):
    """
    Create comparison table for multiple regression.

    Args:
        models: Dictionary of model_name -> model
        X: Feature matrix
        y: Target vector

    Returns:
        pd.DataFrame: Comparison table with metrics for each model
    """
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X)
        metrics = comprehensive_regression_metrics(y, y_pred)
        metrics["model_name"] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ["model_name"] + [c for c in df.columns if c != "model_name"]
    return df[cols].sort_values("rmse")


def statistical_model_comparison(y_true, y_pred1, y_pred2, test_type="paired_ttest"):
    """
    Statistical significance test between two regression' predictions.

    Args:
        y_true: True target values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        test_type: 'paired_ttest' or 'wilcoxon'

    Returns:
        dict: Test results with statistic, p_value, and significance
    """
    from scipy import stats

    # Calculate errors
    errors1 = np.abs(y_true - y_pred1)
    errors2 = np.abs(y_true - y_pred2)

    # Perform test
    if test_type == "paired_ttest":
        statistic, p_value = stats.ttest_rel(errors1, errors2)
        test_name = "Paired t-test"
    elif test_type == "wilcoxon":
        statistic, p_value = stats.wilcoxon(errors1, errors2)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return {
        "test_name": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def automated_model_selection(models, X, y, metric="rmse", cross_validate=True, n_splits=5):
    """
    Automated model selection based on validation metrics.

    Args:
        models: Dictionary of model_name -> model
        X: Feature matrix
        y: Target vector
        metric: Metric to optimize ('rmse', 'mae', 'r2')
        cross_validate: Whether to use cross-validation
        n_splits: Number of CV splits

    Returns:
        dict: Best model name, score, and all scores
    """
    scores = {}

    for model_name, model in models.items():
        if cross_validate:
            from sklearn.model_selection import cross_val_score

            if metric == "r2":
                cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="r2")
                score = np.mean(cv_scores)
            elif metric == "mae":
                cv_scores = cross_val_score(
                    model, X, y, cv=n_splits, scoring="neg_mean_absolute_error"
                )
                score = -np.mean(cv_scores)
            elif metric == "rmse":
                cv_scores = cross_val_score(
                    model, X, y, cv=n_splits, scoring="neg_root_mean_squared_error"
                )
                score = -np.mean(cv_scores)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            y_pred = model.predict(X)
            metrics = comprehensive_regression_metrics(y, y_pred)
            score = metrics[metric]

        scores[model_name] = score

    # Select best model
    if metric == "r2":
        best_model_name = max(scores.items(), key=lambda x: x[1])[0]
    else:
        best_model_name = min(scores.items(), key=lambda x: x[1])[0]

    return {
        "best_model_name": best_model_name,
        "best_score": scores[best_model_name],
        "all_scores": scores,
        "metric": metric,
    }


# ============================================================================
# Learning Curves and Validation Curves
# ============================================================================


def generate_learning_curve(model, X, y, train_sizes=None, cv=5, scoring="r2"):
    """
    Generate learning curve data showing performance vs training size.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target vector
        train_sizes: List of training sizes (fractions or absolute numbers)
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        dict: Learning curve data with train/val scores
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1, random_state=42
    )

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores": train_scores.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores": val_scores.tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }


def plot_learning_curve(model, X, y, output_path=None, train_sizes=None, cv=5):
    """
    Plot learning curve showing training and validation performance.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target vector
        output_path: Path to save plot
        train_sizes: Training sizes to evaluate
        cv: Number of CV folds

    Returns:
        None (saves plot to file)
    """
    if not plt:
        raise ImportError("Matplotlib is required for plotting")

    result = generate_learning_curve(model, X, y, train_sizes=train_sizes, cv=cv)

    train_sizes_abs = result["train_sizes"]
    train_mean = result["train_scores_mean"]
    train_std = result["train_scores_std"]
    val_mean = result["val_scores_mean"]
    val_std = result["val_scores_std"]

    plt.figure(figsize=(10, 6))

    # Plot training scores
    plt.plot(train_sizes_abs, train_mean, "o-", color="r", label="Training score")
    plt.fill_between(
        train_sizes_abs,
        np.array(train_mean) - np.array(train_std),
        np.array(train_mean) + np.array(train_std),
        alpha=0.1,
        color="r",
    )

    # Plot validation scores
    plt.plot(train_sizes_abs, val_mean, "o-", color="g", label="Validation score")
    plt.fill_between(
        train_sizes_abs,
        np.array(val_mean) - np.array(val_std),
        np.array(val_mean) + np.array(val_std),
        alpha=0.1,
        color="g",
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("Score (R²)")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def generate_validation_curve(model, X, y, param_name, param_range, cv=5, scoring="r2"):
    """
    Generate validation curve for hyperparameter tuning.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target vector
        param_name: Name of parameter to vary
        param_range: Range of parameter values
        cv: Number of CV folds
        scoring: Scoring metric

    Returns:
        dict: Validation curve data
    """
    from sklearn.model_selection import validation_curve

    train_scores, val_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "param_name": param_name,
        "param_range": [float(p) if isinstance(p, (int, float)) else str(p) for p in param_range],
        "train_scores": train_scores.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores": val_scores.tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }


def plot_validation_curve(model, X, y, param_name, param_range, output_path=None, cv=5):
    """
    Plot validation curve for hyperparameter analysis.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target vector
        param_name: Name of parameter to vary
        param_range: Range of parameter values
        output_path: Path to save plot
        cv: Number of CV folds

    Returns:
        None (saves plot to file)
    """
    if not plt:
        raise ImportError("Matplotlib is required for plotting")

    result = generate_validation_curve(model, X, y, param_name, param_range, cv=cv)

    param_vals = result["param_range"]
    train_mean = result["train_scores_mean"]
    train_std = result["train_scores_std"]
    val_mean = result["val_scores_mean"]
    val_std = result["val_scores_std"]

    plt.figure(figsize=(10, 6))

    # Plot training scores
    plt.plot(param_vals, train_mean, "o-", color="r", label="Training score")
    plt.fill_between(
        param_vals,
        np.array(train_mean) - np.array(train_std),
        np.array(train_mean) + np.array(train_std),
        alpha=0.1,
        color="r",
    )

    # Plot validation scores
    plt.plot(param_vals, val_mean, "o-", color="g", label="Validation score")
    plt.fill_between(
        param_vals,
        np.array(val_mean) - np.array(val_std),
        np.array(val_mean) + np.array(val_std),
        alpha=0.1,
        color="g",
    )

    plt.xlabel(param_name)
    plt.ylabel("Score (R²)")
    plt.title(f"Validation Curve ({param_name})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


# ============================================================================
# Bias-Variance Diagnosis
# ============================================================================


def diagnose_bias_variance(model, X_train, y_train, X_val, y_val):
    """
    Diagnose bias-variance issues by comparing train vs validation performance.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        dict: Diagnosis with train/val scores and interpretation
    """
    from sklearn.metrics import r2_score

    # Compute scores
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_score = r2_score(y_train, y_train_pred)
    val_score = r2_score(y_val, y_val_pred)

    # Diagnose
    if train_score > 0.9 and val_score < 0.7:
        diagnosis = "High variance (overfitting)"
    elif train_score < 0.7 and val_score < 0.7:
        diagnosis = "High bias (underfitting)"
    elif train_score > 0.8 and val_score > 0.75:
        diagnosis = "Good fit"
    else:
        diagnosis = "Moderate fit"

    return {
        "train_score": float(train_score),
        "val_score": float(val_score),
        "score_gap": float(train_score - val_score),
        "diagnosis": diagnosis,
    }


def bias_variance_decomposition(model, X_train, y_train, X_val, y_val, n_bootstraps=50):
    """
    Decompose prediction error into bias and variance components.

    Args:
        model: Model class (not fitted)
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_bootstraps: Number of bootstrap iterations

    Returns:
        dict: Bias squared, variance, and total MSE
    """
    from sklearn.utils import resample
    from sklearn.metrics import mean_squared_error

    predictions = []

    # Bootstrap and collect predictions
    for _ in range(n_bootstraps):
        # Resample training data
        X_boot, y_boot = resample(X_train, y_train, random_state=_)

        # Clone and fit model
        from sklearn.base import clone

        model_clone = clone(model)
        model_clone.fit(X_boot, y_boot)

        # Predict on validation set
        y_pred = model_clone.predict(X_val)
        predictions.append(y_pred)

    predictions = np.array(predictions)

    # Calculate bias and variance
    mean_predictions = predictions.mean(axis=0)
    bias_squared = np.mean((mean_predictions - y_val) ** 2)
    variance = np.mean(predictions.var(axis=0))
    mse = mean_squared_error(y_val, mean_predictions)

    return {
        "bias_squared": float(bias_squared),
        "variance": float(variance),
        "mse": float(mse),
        "n_bootstraps": n_bootstraps,
    }


def plot_bias_variance(model, X_train, y_train, X_val, y_val, output_path=None):
    """
    Plot bias-variance diagnosis visualization.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        output_path: Path to save plot

    Returns:
        None (saves plot to file)
    """
    if not plt:
        raise ImportError("Matplotlib is required for plotting")

    diagnosis = diagnose_bias_variance(model, X_train, y_train, X_val, y_val)

    fig, ax = plt.subplots(figsize=(8, 6))

    scores = [diagnosis["train_score"], diagnosis["val_score"]]
    labels = ["Training", "Validation"]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, scores, color=colors, alpha=0.7)
    ax.set_ylabel("R² Score")
    ax.set_title(f'Bias-Variance Diagnosis: {diagnosis["diagnosis"]}')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{score:.3f}", ha="center", va="bottom"
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def identify_optimal_complexity(
    X_train,
    y_train,
    X_val,
    y_val,
    model_type="RandomForest",
    complexity_param="max_depth",
    complexity_range=None,
):
    """
    Identify optimal model complexity to balance bias and variance.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: Type of model ('RandomForest', 'GradientBoosting')
        complexity_param: Parameter controlling complexity
        complexity_range: Range of values to test

    Returns:
        dict: Optimal parameter value and performance curves
    """
    from sklearn.metrics import r2_score

    if complexity_range is None:
        if complexity_param == "max_depth":
            complexity_range = [3, 5, 10, 15, 20, None]
        elif complexity_param == "n_estimators":
            complexity_range = [10, 50, 100, 200, 500]
        else:
            raise ValueError(f"Please specify complexity_range for {complexity_param}")

    train_scores = []
    val_scores = []

    for value in complexity_range:
        # Create model with specified complexity
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(**{complexity_param: value}, random_state=42)
        elif model_type == "GradientBoosting":
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(**{complexity_param: value}, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Fit and evaluate
        model.fit(X_train, y_train)
        train_score = r2_score(y_train, model.predict(X_train))
        val_score = r2_score(y_val, model.predict(X_val))

        train_scores.append(train_score)
        val_scores.append(val_score)

    # Find optimal value (max validation score)
    optimal_idx = np.argmax(val_scores)
    optimal_value = complexity_range[optimal_idx]

    return {
        "optimal_value": optimal_value,
        "optimal_val_score": val_scores[optimal_idx],
        "complexity_param": complexity_param,
        "complexity_range": [str(v) for v in complexity_range],
        "train_scores": train_scores,
        "val_scores": val_scores,
    }


# ============================================================================
# Time-Series Cross-Validation
# ============================================================================


def create_expanding_window_cv(n_splits=5, min_train_size=None):
    """
    Create expanding window cross-validation splitter for time-series.

    In expanding window CV, training set grows while test set is fixed size.
    Maintains temporal ordering: train on past, test on future.

    Args:
        n_splits: Number of splits
        min_train_size: Minimum training set size (optional)

    Returns:
        TimeSeriesSplit object
    """
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(n_splits=n_splits, max_train_size=None)  # Expanding window (no max)


def create_rolling_window_cv(n_splits=5, max_train_size=None):
    """
    Create rolling window cross-validation splitter for time-series.

    In rolling window CV, both training and test sets are fixed size.
    Maintains temporal ordering.

    Args:
        n_splits: Number of splits
        max_train_size: Maximum training set size (fixed window)

    Returns:
        TimeSeriesSplit object
    """
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)


def evaluate_with_time_series_cv(model, X, y, cv_type="expanding", n_splits=5, max_train_size=None):
    """
    Evaluate model using time-series aware cross-validation.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target vector
        cv_type: 'expanding' or 'rolling'
        n_splits: Number of splits
        max_train_size: Maximum training size for rolling window

    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import cross_val_score

    if cv_type == "expanding":
        cv = create_expanding_window_cv(n_splits=n_splits)
    elif cv_type == "rolling":
        cv = create_rolling_window_cv(n_splits=n_splits, max_train_size=max_train_size)
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return {
        "cv_type": cv_type,
        "cv_scores": scores.tolist(),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "n_splits": n_splits,
    }


# ============================================================================
# Performance Heatmaps (Sector × Region)
# ============================================================================


def compute_sector_region_metrics(
    df, y_true_col, y_pred_col, sector_col="sector", region_col="region"
):
    """
    Compute metrics for each sector-region combination.

    Args:
        df: DataFrame with predictions and grouping columns
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        sector_col: Column name for sector
        region_col: Column name for region

    Returns:
        pd.DataFrame: Metrics for each sector-region combination
    """
    results = []

    for sector in df[sector_col].dropna().unique():
        for region in df[region_col].dropna().unique():
            mask = (df[sector_col] == sector) & (df[region_col] == region)
            subset = df[mask]

            if len(subset) > 0:
                y_true = subset[y_true_col].values
                y_pred = subset[y_pred_col].values

                metrics = comprehensive_regression_metrics(y_true, y_pred)
                metrics["sector"] = sector
                metrics["region"] = region
                results.append(metrics)

    return pd.DataFrame(results)


def create_sector_region_performance_heatmap(
    df,
    y_true_col,
    y_pred_col,
    sector_col="sector",
    region_col="region",
    metric="mae",
    output_path=None,
):
    """
    Create heatmap showing performance across sector-region combinations.

    Args:
        df: DataFrame with predictions
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        sector_col: Column name for sector
        region_col: Column name for region
        metric: Metric to display ('mae', 'rmse', 'r2', 'mape')
        output_path: Path to save heatmap

    Returns:
        None (saves plot to file)
    """
    if not plt or not sns:
        raise ImportError("Matplotlib and seaborn are required for heatmaps")

    # Compute metrics
    metrics_df = compute_sector_region_metrics(df, y_true_col, y_pred_col, sector_col, region_col)

    # Pivot for heatmap
    heatmap_data = metrics_df.pivot(index=sector_col, columns=region_col, values=metric)

    # Create heatmap
    plt.figure(figsize=(10, 8))

    # Choose colormap based on metric (lower is better for mae/rmse/mape, higher is better for r2)
    if metric == "r2":
        cmap = "RdYlGn"
        fmt = ".3f"
    else:
        cmap = "RdYlGn_r"
        fmt = ".2f"

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": metric.upper()},
        linewidths=0.5,
    )

    plt.title(f"Performance Heatmap: {metric.upper()} by Sector × Region")
    plt.xlabel("Region")
    plt.ylabel("Sector")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


# ============================================================================
# Enhanced Residual Analysis
# ============================================================================


def plot_residuals_vs_features(df, y_true_col, y_pred_col, feature_cols, output_dir=None):
    """
    Plot residuals vs individual features to detect non-linearities.

    Args:
        df: DataFrame with predictions and features
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        feature_cols: List of feature columns to plot
        output_dir: Directory to save plots

    Returns:
        None (saves plots to directory)
    """
    if not plt:
        raise ImportError("Matplotlib is required for plotting")

    # Calculate residuals
    residuals = df[y_true_col] - df[y_pred_col]

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for feature in feature_cols:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[feature], residuals, alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel(feature)
            plt.ylabel("Residuals")
            plt.title(f"Residuals vs {feature}")
            plt.grid(True, alpha=0.3)

            if output_dir:
                plt.savefig(
                    output_dir / f"residuals_vs_{feature}.png", dpi=100, bbox_inches="tight"
                )
                plt.close()


def identify_systematic_bias_patterns(df, y_true_col, y_pred_col, segment_cols):
    """
    Identify systematic bias patterns in predictions by segment.

    Args:
        df: DataFrame with predictions and segment columns
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        segment_cols: List of columns to segment by

    Returns:
        dict: Bias analysis for each segment
    """
    results = {}

    for segment_col in segment_cols:
        if segment_col in df.columns:
            segment_biases = []

            for segment_value in df[segment_col].dropna().unique():
                mask = df[segment_col] == segment_value
                subset = df[mask]

                if len(subset) > 0:
                    residuals = subset[y_true_col] - subset[y_pred_col]
                    mean_bias = float(residuals.mean())
                    std_bias = float(residuals.std())

                    segment_biases.append(
                        {
                            "segment_value": segment_value,
                            "mean_bias": mean_bias,
                            "std_bias": std_bias,
                            "n_samples": len(subset),
                        }
                    )

            results[segment_col] = segment_biases

    return results


def analyze_residual_homoscedasticity(y_true, y_pred):
    """
    Test residuals for homoscedasticity (constant variance).

    Uses Breusch-Pagan test to detect heteroscedasticity.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        dict: Test results with statistic and p-value
    """
    from scipy import stats

    residuals = np.array(y_true) - np.array(y_pred)
    y_pred = np.array(y_pred)

    # Breusch-Pagan test approximation
    # Regress squared residuals on predictions
    residuals_squared = residuals**2

    # Calculate correlation between squared residuals and predictions
    if len(residuals) > 3:
        correlation, p_value = stats.spearmanr(y_pred, residuals_squared)
        test_statistic = correlation
        is_homoscedastic = p_value > 0.05
    else:
        test_statistic = np.nan
        p_value = np.nan
        is_homoscedastic = None

    return {
        "test_name": "Spearman correlation (residuals² vs predictions)",
        "test_statistic": float(test_statistic) if not np.isnan(test_statistic) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "is_homoscedastic": is_homoscedastic,
    }


# ============================================================================
# Feature Importance Ranking
# ============================================================================


def compute_permutation_importance(model, X, y, n_repeats=10):
    """
    Compute permutation importance for features.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        n_repeats: Number of permutation repeats

    Returns:
        pd.DataFrame: Feature importance with mean and std
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance_df.sort_values("importance_mean", ascending=False)


def rank_features_by_importance(model, X, y, method="all"):
    """
    Rank features by importance using multiple methods.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        method: 'tree', 'permutation', 'all'

    Returns:
        pd.DataFrame: Feature rankings
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    results = pd.DataFrame({"feature": feature_names})

    # Tree-based importance
    if method in ["tree", "all"]:
        if hasattr(model, "feature_importances_"):
            results["tree_importance"] = model.feature_importances_

    # Permutation importance
    if method in ["permutation", "all"]:
        perm_importance = compute_permutation_importance(model, X, y, n_repeats=5)
        results = results.merge(
            perm_importance[["feature", "importance_mean"]].rename(
                columns={"importance_mean": "permutation_importance"}
            ),
            on="feature",
            how="left",
        )

    return results.sort_values(
        by=[c for c in results.columns if "importance" in c][0], ascending=False
    )


def feature_importance_stability_across_folds(model, X, y, n_splits=5):
    """
    Assess feature importance stability across CV folds.

    Args:
        model: Model class (will be cloned and fitted)
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits

    Returns:
        pd.DataFrame: Feature importance stability metrics
    """
    from sklearn.model_selection import KFold
    from sklearn.base import clone

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    importance_matrix = []

    for train_idx, _ in kf.split(X):
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        if hasattr(model_clone, "feature_importances_"):
            importance_matrix.append(model_clone.feature_importances_)

    if len(importance_matrix) > 0:
        importance_matrix = np.array(importance_matrix)

        results = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": importance_matrix.mean(axis=0),
                "importance_std": importance_matrix.std(axis=0),
            }
        )

        # Calculate stability score (inverse of coefficient of variation)
        results["stability_score"] = 1.0 / (
            1.0 + results["importance_std"] / (results["importance_mean"] + 1e-10)
        )

        return results.sort_values("importance_mean", ascending=False)
    else:
        return pd.DataFrame({"feature": feature_names})


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
        """Helper function to categorize a single mispricing score.

        Args:
            score: Mispricing score percentage

        Returns:
            Valuation category string
        """
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

    # Filter by mispricing score (use percentage column for threshold comparison)
    if min_mispricing is not None:
        if "mispricing_pct" in result.columns:
            result = result[result["mispricing_pct"] >= min_mispricing]

    if max_mispricing is not None:
        if "mispricing_pct" in result.columns:
            result = result[result["mispricing_pct"] <= max_mispricing]

    # Filter by valuation category
    if valuation_categories is not None:
        if "valuation_category" in result.columns:
            result = result[result["valuation_category"].isin(valuation_categories)]

    return result


def create_valuation_scatter_plot(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    color_by: str = "sector",
    size_by: Optional[str] = None,
    opacity: float = 0.7,
    show_diagonal: bool = True,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 900,
    log_scale: bool = True,
):
    """
    Create an interactive scatter plot of current price vs. predicted target.

    This function creates a comprehensive visualization comparing current stock prices
    with predicted price targets, highlighting potential investment opportunities through
    color coding and optional sizing by market cap or other metrics.

    Args:
        df: DataFrame with required columns 'last_price' and 'predicted_price_target'
        out_path: Optional path to save HTML file
        color_by: Column to color points by (default 'sector')
        size_by: Optional column to size points by (e.g., 'market_cap'). If None, uniform size.
        opacity: Marker opacity between 0 and 1 (default 0.7)
        show_diagonal: Whether to show diagonal reference line (default True)
        title: Custom plot title. If None, uses default title.
        height: Plot height in pixels (default 600)
        width: Plot width in pixels (default 900)
        log_scale: If True, use logarithmic scale for both axes (default False).
                  Useful for visualizing stocks with widely varying price ranges.

    Returns:
        Plotly figure object (or None if plotly not available or data invalid)

    Raises:
        None (handles errors gracefully with logging)

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        ...     'name': ['Apple', 'Microsoft', 'Alphabet'],
        ...     'last_price': [150, 300, 2800],
        ...     'predicted_price_target': [180, 290, 3000],
        ...     'sector': ['Tech', 'Tech', 'Tech'],
        ...     'market_cap': [2.5e12, 2.3e12, 1.8e12],
        ...     'mispricing_pct': [20, -3.3, 7.1]
        ... })
        >>> # Standard linear scale
        >>> fig = create_valuation_scatter_plot(df, color_by='sector')
        >>>
        >>> # Log scale for wide price ranges
        >>> fig_log = create_valuation_scatter_plot(df, log_scale=True)
        >>> fig_log is not None
        True

    Notes:
        - Points above the diagonal line represent undervalued stocks (predicted > current)
        - Points below the diagonal line represent overvalued stocks (predicted < current)
        - Hover over points to see detailed information including ticker, prices, and metrics
        - Log scale is recommended when prices span multiple orders of magnitude (e.g., $1 to $1000+)
        - With log scale, equal distances represent equal percentage changes
    """
    # Check plotly availability
    if px is None or go is None:
        logging.warning("Plotly not available; cannot create scatter plot")
        return None

    # Validate DataFrame
    if df is None or df.empty:
        logging.warning("Empty DataFrame provided; cannot create scatter plot")
        return None

    # Ensure required columns exist
    required_cols = ["last_price", "predicted_price_target"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Required columns missing: {missing_cols}; cannot create scatter plot")
        return None

    # Clean data: remove rows with NaN in required columns
    df_clean = df.dropna(subset=required_cols).copy()

    if df_clean.empty:
        logging.warning(
            "No valid data after removing NaN values in required columns; cannot create scatter plot"
        )
        return None

    # Filter out non-positive prices (if any)
    valid_mask = (df_clean["last_price"] > 0) & (df_clean["predicted_price_target"] > 0)
    df_clean = df_clean[valid_mask]

    if df_clean.empty:
        logging.warning("No valid positive price data; cannot create scatter plot")
        return None

    # Prepare hover data with comprehensive information
    hover_data_dict = {}

    # Always include ticker if available
    if "ticker" in df_clean.columns:
        hover_data_dict["ticker"] = True

    # Add name if available
    if "name" in df_clean.columns:
        hover_data_dict["name"] = True

    # Add mispricing percentage with custom formatting
    if "mispricing_pct" in df_clean.columns:
        hover_data_dict["mispricing_pct"] = ":.2f"

    # Add valuation category
    if "valuation_category" in df_clean.columns:
        hover_data_dict["valuation_category"] = True

    # Add market cap with custom formatting if it's the size variable
    if "market_cap" in df_clean.columns and size_by != "market_cap":
        hover_data_dict["market_cap"] = ":,.0f"

    # Add exchange if available
    if "exchange" in df_clean.columns:
        hover_data_dict["exchange"] = True

    # Add region if available
    if "region" in df_clean.columns:
        hover_data_dict["region"] = True

    # Validate color_by column
    color_col = color_by if color_by in df_clean.columns else None
    if color_by not in df_clean.columns and color_by != "sector":
        logging.info(f"Color column '{color_by}' not found; using no color grouping")

    # Validate size_by column
    size_col = None
    if size_by is not None:
        if size_by in df_clean.columns:
            # Ensure size column has valid positive values
            if (df_clean[size_by] > 0).all():
                size_col = size_by
            else:
                logging.warning(f"Size column '{size_by}' contains non-positive values; ignoring")
        else:
            logging.warning(f"Size column '{size_by}' not found; ignoring")

    # Determine plot title
    plot_title = title if title else "Current Price vs. Predicted Target"
    n_stocks = len(df_clean)
    plot_title += f" ({n_stocks} stocks)"

    # Create scatter plot with enhanced configuration
    try:
        fig = px.scatter(
            df_clean,
            x="last_price",
            y="predicted_price_target",
            color=color_col,
            size=size_col,
            hover_data=hover_data_dict if hover_data_dict else None,
            title=plot_title,
            labels={
                "last_price": "Current Price ($)",
                "predicted_price_target": "Predicted Target ($)",
                color_by: color_by.replace("_", " ").title() if color_col else "",
                size_by: size_by.replace("_", " ").title() if size_col else "",
            },
            opacity=opacity,
            height=height,
            width=width,
        )

        # Customize marker appearance
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color="DarkSlateGray"),
                sizemode="diameter",
                sizemin=4,
            )
        )

        # Add diagonal reference line (y=x) for fair value comparison
        if show_diagonal:
            # Calculate range for diagonal line with some padding
            min_val = min(df_clean["last_price"].min(), df_clean["predicted_price_target"].min())
            max_val = max(df_clean["last_price"].max(), df_clean["predicted_price_target"].max())

            # Add 5% padding to the range
            padding = (max_val - min_val) * 0.05
            line_start = max(0, min_val - padding)
            line_end = max_val + padding

            fig.add_trace(
                go.Scatter(
                    x=[line_start, line_end],
                    y=[line_start, line_end],
                    mode="lines",
                    line=dict(dash="dash", color="gray", width=2),
                    name="Fair Value (y=x)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Update layout with enhanced styling
        fig.update_layout(
            xaxis_title="Current Price ($)",
            yaxis_title="Predicted Target ($)",
            hovermode="closest",
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="LightGray",
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="LightGray",
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="Gray",
                borderwidth=1,
            ),
        )

        # Save to file if path provided
        if out_path is not None:
            try:
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(
                    str(out_path),
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    },
                )
                logging.info(f"Saved valuation scatter plot to {out_path}")
            except Exception as e:
                logging.error(f"Failed to save scatter plot to {out_path}: {e}")

        return fig

    except Exception as e:
        logging.error(f"Error creating valuation scatter plot: {e}")
        return None


def generate_pdf_report(
    df: pd.DataFrame,
    pdf_path: Path,
    title: str = "Stock Valuation Report",
    include_summary: bool = True,
    top_n_opportunities: int = 100,
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

        if "mispricing_pct" in df.columns:
            avg_mispricing = df["mispricing_pct"].mean()
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
    if "mispricing_pct" in df.columns:
        top_opportunities = df.nlargest(top_n_opportunities, "mispricing_pct")
    else:
        top_opportunities = df.head(top_n_opportunities)

    # Create table data
    table_data = [
        [
            "Ticker",
            "Name",
            "Exchange",
            "Sector",
            "Current Price",
            "Target Price",
            "Upside %",
            "Category",
        ]
    ]

    for _, row in top_opportunities.iterrows():
        ticker = row.get("ticker", "N/A")
        name = row.get("name", "N/A")
        exchange = row.get("exchange", "N/A")
        sector = row.get("sector", "N/A")
        current = row.get("last_price", 0)
        target = row.get("predicted_price_target", 0)
        mispricing = row.get("mispricing_score", 0)
        category = row.get("valuation_category", "N/A")

        table_data.append(
            [
                str(ticker),
                str(sector),
                str(exchange),
                str(sector),
                f"${current:.2f}" if current else "N/A",
                f"${target:.2f}" if target else "N/A",
                f"{mispricing:.1f}%" if mispricing else "N/A",
                str(category),
            ]
        )

    # Create table
    table = Table(
        table_data,
        colWidths=[
            1 * inch,
            3 * inch,
            1.5 * inch,
            1.5 * inch,
            1 * inch,
            1 * inch,
            0.8 * inch,
            1 * inch,
        ],
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


def compare_prediction_vs_analyst_targets(
    df: pd.DataFrame,
    predicted_col: str = "predicted_price_target",
    analyst_col: str = "price_target",
    current_price_col: str = "last_price",
) -> dict:
    """Compare model predictions against analyst consensus targets.

    Phase 9.8 TDD implementation: Calculate differences and agreement metrics
    between model-predicted price targets and analyst consensus targets.

    Args:
        df: DataFrame with stock data
        predicted_col: Name of model prediction column (default: "predicted_price_target")
        analyst_col: Name of analyst target column (default: "price_target")
        current_price_col: Name of current price column (default: "last_price")

    Returns:
        Dictionary with comparison metrics

    Examples:
        >>> result = compare_prediction_vs_analyst_targets(df)
        >>> print(result)
    """
    result = df.copy()

    # Calculate absolute and percentage differences
    result["model_analyst_diff"] = result[predicted_col] - result[analyst_col]
    result["model_analyst_diff_pct"] = result["model_analyst_diff"] / result[analyst_col] * 100

    # Determine if model and analyst agree on direction vs current price
    model_direction = result[predicted_col] > result[current_price_col]
    analyst_direction = result[analyst_col] > result[current_price_col]
    result["agreement_direction"] = model_direction == analyst_direction

    # Calculate summary metrics
    agreement_count = result["agreement_direction"].sum()
    total_count = len(result)
    agreement_rate = agreement_count / total_count if total_count > 0 else 0.0

    return {
        "comparison_df": result,
        "agreement_rate": agreement_rate,
        "agreement_count": int(agreement_count),
        "total_count": total_count,
    }


def calculate_directional_accuracy(
    df: pd.DataFrame,
    predicted_col: str = "predicted_price_target",
    analyst_col: str = "price_target",
    current_price_col: str = "last_price",
) -> float:
    """Calculate directional accuracy of model predictions vs analyst targets.

    Phase 9.8 TDD implementation: Measures how often the model and analysts
    agree on the direction (up/down) relative to current price.

    Args:
        df: DataFrame with stock data
        predicted_col: Name of model prediction column (default: "predicted_price_target")
        analyst_col: Name of analyst target column (default: "price_target")
        current_price_col: Name of current price column (default: "last_price")

    Returns:
        Float: Proportion of directional agreement (0.0 to 1.0)

    Examples:
        >>> accuracy = calculate_directional_accuracy(df)
        >>> print(f"Accuracy: {accuracy:.2%}")
    """
    total = len(df)

    if total == 0:
        return 0.0

    # Calculate directions
    model_direction = df[predicted_col] > df[current_price_col]
    analyst_direction = df[analyst_col] > df[current_price_col]

    # Count agreements
    agreements = (model_direction == analyst_direction).sum()
    accuracy = agreements / total

    return float(accuracy)


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
    df: pd.DataFrame, threshold_pct: float = 5.0
) -> pd.DataFrame | None:
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
    disagreements = df[abs(df["model_analyst_diff_pct"]) >= threshold_pct].copy()

    # Sort by absolute difference magnitude
    disagreements = disagreements.sort_values(
        by="model_analyst_diff_pct", key=lambda x: abs(x), ascending=False
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
    model_mae = abs(df["predicted_price_target"] - df["actual_future_price"]).mean()

    analyst_mae = abs(df["price_target"] - df["actual_future_price"]).mean()

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


def segment_comparison_by_attribute(df: pd.DataFrame, segment_col: str = "sector") -> dict:
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


def calculate_hit_rate_by_confidence_level(df: pd.DataFrame) -> dict:
    """Calculate prediction hit rate segmented by confidence level.

    Phase 9.8 TDD implementation: Analyze if high-confidence predictions
    are more accurate than low-confidence predictions.

    Args:
        df: DataFrame with columns:
            - predicted_price_target: Model prediction
            - actual_future_price: Realized future price
            - last_price: Current price
            - prediction_lower: Lower bound of prediction interval (optional)
            - prediction_upper: Upper bound of prediction interval (optional)

    Returns:
        Dictionary mapping confidence levels to metrics:
            - high_confidence: Narrow prediction intervals
            - medium_confidence: Moderate prediction intervals
            - low_confidence: Wide prediction intervals
        Each containing:
            - hit_rate: Proportion of correct directional predictions
            - count: Number of predictions in this bucket
            - correct_predictions: Number of correct predictions

    Examples:
        >>> hit_rates = calculate_hit_rate_by_confidence_level(df)
        >>> print(f"High confidence hit rate: {hit_rates['high_confidence']['hit_rate']:.2%}")
    """
    if "actual_future_price" not in df.columns:
        raise ValueError("actual_future_price column required for hit rate calculation")

    # Calculate confidence level based on prediction interval width
    # If no intervals provided, use prediction magnitude as proxy
    if "prediction_lower" in df.columns and "prediction_upper" in df.columns:
        # Calculate interval width as percentage of prediction
        df = df.copy()
        df["confidence_width"] = (df["prediction_upper"] - df["prediction_lower"]) / df[
            "predicted_price_target"
        ]
    else:
        # Use prediction magnitude as proxy for confidence
        df = df.copy()
        df["confidence_width"] = abs(
            (df["predicted_price_target"] - df["last_price"]) / df["last_price"]
        )

    # Define confidence buckets based on interval width (tertiles)
    # Narrow intervals = high confidence, wide intervals = low confidence
    tertiles = df["confidence_width"].quantile([0.33, 0.67])

    results = {}

    # High confidence: narrowest intervals (bottom tertile)
    high_conf_df = df[df["confidence_width"] <= tertiles.iloc[0]]
    if len(high_conf_df) > 0:
        predicted_dir = high_conf_df["predicted_price_target"] > high_conf_df["last_price"]
        actual_dir = high_conf_df["actual_future_price"] > high_conf_df["last_price"]
        correct = (predicted_dir == actual_dir).sum()
        results["high_confidence"] = {
            "hit_rate": correct / len(high_conf_df),
            "count": len(high_conf_df),
            "correct_predictions": int(correct),
        }
    else:
        results["high_confidence"] = {"hit_rate": 0.0, "count": 0, "correct_predictions": 0}

    # Medium confidence: middle tertile
    medium_conf_df = df[
        (df["confidence_width"] > tertiles.iloc[0]) & (df["confidence_width"] <= tertiles.iloc[1])
    ]
    if len(medium_conf_df) > 0:
        predicted_dir = medium_conf_df["predicted_price_target"] > medium_conf_df["last_price"]
        actual_dir = medium_conf_df["actual_future_price"] > medium_conf_df["last_price"]
        correct = (predicted_dir == actual_dir).sum()
        results["medium_confidence"] = {
            "hit_rate": correct / len(medium_conf_df),
            "count": len(medium_conf_df),
            "correct_predictions": int(correct),
        }
    else:
        results["medium_confidence"] = {"hit_rate": 0.0, "count": 0, "correct_predictions": 0}

    # Low confidence: widest intervals (top tertile)
    low_conf_df = df[df["confidence_width"] > tertiles.iloc[1]]
    if len(low_conf_df) > 0:
        predicted_dir = low_conf_df["predicted_price_target"] > low_conf_df["last_price"]
        actual_dir = low_conf_df["actual_future_price"] > low_conf_df["last_price"]
        correct = (predicted_dir == actual_dir).sum()
        results["low_confidence"] = {
            "hit_rate": correct / len(low_conf_df),
            "count": len(low_conf_df),
            "correct_predictions": int(correct),
        }
    else:
        results["low_confidence"] = {"hit_rate": 0.0, "count": 0, "correct_predictions": 0}

    return results


def calculate_calibration_metrics(df: pd.DataFrame) -> dict:
    """Calculate calibration metrics comparing predicted vs. realized upside.

    Phase 9.8 TDD implementation: Analyze if the model's predicted price changes
    match the realized price changes (calibration analysis).

    Args:
        df: DataFrame with columns:
            - last_price: Current stock price
            - predicted_price_target: Model prediction
            - actual_future_price: Realized future price

    Returns:
        Dictionary with calibration metrics:
            - predicted_upside_mean: Average predicted price change (%)
            - realized_upside_mean: Average realized price change (%)
            - calibration_error: Absolute difference between predicted and realized
            - calibration_slope: Regression slope (realized ~ predicted)
            - calibration_r2: R-squared of calibration regression

    Examples:
        >>> calibration = calculate_calibration_metrics(df)
        >>> print(f"Calibration error: {calibration['calibration_error']:.2%}")
    """
    if "actual_future_price" not in df.columns:
        raise ValueError("actual_future_price column required for calibration metrics")

    # Calculate predicted and realized upside/downside
    predicted_upside = (df["predicted_price_target"] - df["last_price"]) / df["last_price"] * 100
    realized_upside = (df["actual_future_price"] - df["last_price"]) / df["last_price"] * 100

    # Calculate calibration metrics
    predicted_upside_mean = predicted_upside.mean()
    realized_upside_mean = realized_upside.mean()
    calibration_error = abs(predicted_upside_mean - realized_upside_mean)

    # Calculate calibration slope using linear regression
    # Perfect calibration would have slope = 1.0
    try:
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            predicted_upside, realized_upside
        )
        calibration_slope = slope
        calibration_r2 = r_value**2
    except ImportError:
        # Fallback without scipy
        correlation = predicted_upside.corr(realized_upside)
        calibration_slope = correlation * (realized_upside.std() / predicted_upside.std())
        calibration_r2 = correlation**2

    return {
        "predicted_upside_mean": predicted_upside_mean,
        "realized_upside_mean": realized_upside_mean,
        "calibration_error": calibration_error,
        "calibration_slope": calibration_slope,
        "calibration_r2": calibration_r2,
    }


def generate_prediction_analyst_excel_report(
    df: pd.DataFrame, excel_path: Path, top_n_opportunities: int = 50
) -> None:
    """Generate comprehensive Excel report comparing predictions vs analyst targets.

    Phase 9.8 TDD implementation: Create multi-sheet Excel workbook matching
    Stock_Prediction_Analysis_Report format with 7 required sheets.

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
        7. Model_Interpretation: Model methodology and feature importance

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
        df["mispricing_score"] = (df["predicted_price_target"] - df["last_price"]) / df[
            "last_price"
        ]

    excel_path = Path(excel_path)

    if use_xlsxwriter:
        # Use xlsxwriter for better formatting
        writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    else:
        # Fallback to openpyxl
        writer = pd.ExcelWriter(excel_path, engine="openpyxl")

    # Sheet 1: Executive Summary
    summary_data = []
    summary_data.append(["Metric", "Value"])
    summary_data.append(["Total Stocks", len(df)])
    summary_data.append(["Average Current Price", f"${df['last_price'].mean():.2f}"])
    summary_data.append(["Average Predicted Target", f"${df['predicted_price_target'].mean():.2f}"])
    summary_data.append(["Average Analyst Target", f"${df['price_target'].mean():.2f}"])

    # Calculate agreement metrics
    agreement = calculate_agreement_rate(df)
    summary_data.append(["Model-Analyst Agreement Rate", f"{agreement['agreement_rate']:.2%}"])

    # Bias analysis
    bias = analyze_systematic_bias(df)
    summary_data.append(["Model Bias Direction", bias["bias_direction"]])
    summary_data.append(["Average Model-Analyst Difference", f"${bias['mean_model_bias']:.2f}"])

    # Opportunity counts
    undervalued = df[df["mispricing_score"] > 0.05]
    overvalued = df[df["mispricing_score"] < -0.05]
    summary_data.append(["Undervalued Opportunities (>5%)", len(undervalued)])
    summary_data.append(["Overvalued Stocks (<-5%)", len(overvalued)])

    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

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
        "mispricing_score",
        "agreement_direction",
    ]
    available_cols = [col for col in detail_cols if col in df.columns]
    detail_df = df[available_cols].copy()

    # Add market_cap if available
    if "market_cap" in df.columns:
        detail_df["market_cap"] = df["market_cap"]

    detail_df.to_excel(writer, sheet_name="Detailed_Stock_List", index=False)

    # Sheet 3: Top Opportunities (Undervalued)
    top_opportunities = df.nlargest(top_n_opportunities, "mispricing_score")
    top_opportunities[available_cols].to_excel(writer, sheet_name="Top_Opportunities", index=False)

    # Sheet 4: Risk Analysis (Overvalued)
    top_risks = df.nsmallest(top_n_opportunities, "mispricing_score")
    top_risks[available_cols].to_excel(writer, sheet_name="Risk_Analysis", index=False)

    # Sheet 5: Prediction Accuracy
    accuracy_data = []
    accuracy_data.append(["Metric", "Value"])
    accuracy_data.append(["Model-Analyst Agreement Rate", f"{agreement['agreement_rate']:.2%}"])
    accuracy_data.append(["Stocks with Same Direction", agreement["same_direction_count"]])
    accuracy_data.append(
        [
            "Stocks with Different Direction",
            agreement["total_count"] - agreement["same_direction_count"],
        ]
    )
    accuracy_data.append(
        ["Average Absolute Difference", f"${abs(df['model_analyst_diff']).mean():.2f}"]
    )
    accuracy_data.append(
        ["Median Absolute Difference", f"${abs(df['model_analyst_diff']).median():.2f}"]
    )

    # Add actual accuracy metrics if available
    if "actual_future_price" in df.columns:
        actual_metrics = calculate_prediction_accuracy_metrics(df)
        accuracy_data.append(["Model MAE", f"${actual_metrics['model_mae']:.2f}"])
        accuracy_data.append(["Analyst MAE", f"${actual_metrics['analyst_mae']:.2f}"])
        accuracy_data.append(
            ["Model Directional Accuracy", f"{actual_metrics['model_directional_accuracy']:.2%}"]
        )
        accuracy_data.append(
            [
                "Analyst Directional Accuracy",
                f"{actual_metrics['analyst_directional_accuracy']:.2%}",
            ]
        )

    accuracy_df = pd.DataFrame(accuracy_data[1:], columns=accuracy_data[0])
    accuracy_df.to_excel(writer, sheet_name="Prediction_Accuracy", index=False)

    # Sheet 6: Sector Analysis
    if "sector" in df.columns:
        sector_metrics = segment_comparison_by_attribute(df, "sector")

        sector_data = []
        for sector, metrics in sector_metrics.items():
            sector_data.append(
                {
                    "Sector": sector,
                    "Stock Count": metrics["count"],
                    "Agreement Rate": f"{metrics['agreement_rate']:.2%}",
                    "Avg Model-Analyst Diff": f"${metrics['avg_model_analyst_diff']:.2f}",
                    "Avg Mispricing": f"{df[df['sector'] == sector]['mispricing_score'].mean():.2%}",
                }
            )

        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values("Stock Count", ascending=False)
        sector_df.to_excel(writer, sheet_name="Sector_Analysis", index=False)

    # Sheet 7: Model_Interpretation
    interpretation_data = []
    interpretation_data.append(["Section", "Description"])

    # Model Methodology
    interpretation_data.append(["Model Type", "Sector-Optimized Gradient Boosting Regression"])
    interpretation_data.append(
        [
            "Methodology",
            "Phase 9.8: Comprehensive analytics comparing model predictions vs. analyst consensus targets",
        ]
    )
    interpretation_data.append(
        [
            "Key Features",
            "Financial ratios, valuation metrics, growth indicators, sector-specific features",
        ]
    )
    interpretation_data.append(
        [
            "Training Approach",
            "Sector-specific regression with cross-validation and ensemble stacking",
        ]
    )
    interpretation_data.append(["", ""])  # Blank row

    # Feature Importance (if available in DataFrame)
    interpretation_data.append(["Feature Importance", ""])
    if "sector" in df.columns:
        interpretation_data.append(
            [
                "Sector-Specific Features",
                "Model uses sector-optimized features for improved accuracy",
            ]
        )

    # Add calibration and hit rate info if actual prices available
    if "actual_future_price" in df.columns:
        try:
            calibration = calculate_calibration_metrics(df)
            interpretation_data.append(["", ""])  # Blank row
            interpretation_data.append(["Calibration Analysis", ""])
            interpretation_data.append(
                [
                    "Predicted Upside (Mean)",
                    f"{calibration['predicted_upside_mean']:.2f}%",
                ]
            )
            interpretation_data.append(
                [
                    "Realized Upside (Mean)",
                    f"{calibration['realized_upside_mean']:.2f}%",
                ]
            )
            interpretation_data.append(
                [
                    "Calibration Error",
                    f"{calibration['calibration_error']:.2f}%",
                ]
            )
            interpretation_data.append(
                [
                    "Calibration Slope",
                    f"{calibration['calibration_slope']:.3f} (1.0 = perfect)",
                ]
            )
            interpretation_data.append(
                [
                    "Calibration R²",
                    f"{calibration['calibration_r2']:.3f}",
                ]
            )

            # Add hit rate by confidence if we have prediction intervals
            if "prediction_lower" in df.columns and "prediction_upper" in df.columns:
                hit_rates = calculate_hit_rate_by_confidence_level(df)
                interpretation_data.append(["", ""])  # Blank row
                interpretation_data.append(["Hit Rate by Confidence Level", ""])
                interpretation_data.append(
                    [
                        "High Confidence Hit Rate",
                        f"{hit_rates['high_confidence']['hit_rate']:.2%} (n={hit_rates['high_confidence']['count']})",
                    ]
                )
                interpretation_data.append(
                    [
                        "Medium Confidence Hit Rate",
                        f"{hit_rates['medium_confidence']['hit_rate']:.2%} (n={hit_rates['medium_confidence']['count']})",
                    ]
                )
                interpretation_data.append(
                    [
                        "Low Confidence Hit Rate",
                        f"{hit_rates['low_confidence']['hit_rate']:.2%} (n={hit_rates['low_confidence']['count']})",
                    ]
                )
        except Exception as e:
            logging.warning(f"Could not calculate advanced metrics for interpretation sheet: {e}")

    # Model performance summary
    interpretation_data.append(["", ""])  # Blank row
    interpretation_data.append(["Model Performance Summary", ""])
    interpretation_data.append(
        [
            "Agreement with Analysts",
            f"{agreement['agreement_rate']:.2%}",
        ]
    )
    interpretation_data.append(
        [
            "Systematic Bias",
            f"{bias['bias_direction'].capitalize()}",
        ]
    )

    # Usage recommendations
    interpretation_data.append(["", ""])  # Blank row
    interpretation_data.append(["Usage Recommendations", ""])
    interpretation_data.append(
        [
            "Best Use Cases",
            "Investment screening, portfolio construction, valuation cross-checks",
        ]
    )
    interpretation_data.append(
        [
            "Limitations",
            "Model predictions reflect historical patterns; market conditions may change",
        ]
    )
    interpretation_data.append(
        [
            "Update Frequency",
            "Retrain model quarterly or when significant market regime changes occur",
        ]
    )

    interpretation_df = pd.DataFrame(interpretation_data[1:], columns=interpretation_data[0])
    interpretation_df.to_excel(writer, sheet_name="Model_Interpretation", index=False)

    # Save and close
    writer.close()

    logging.info(f"Prediction vs Analyst Excel report saved to {excel_path}")


# ============================================================================
# Phase 9.2: Enhanced Exploratory Data Analysis of Financial Metrics
# ============================================================================


def calculate_financial_metrics_dashboard(df: pd.DataFrame, group_by: Optional[str] = None) -> Dict:
    """
    Calculate comprehensive financial metrics dashboard.

    Computes statistics for four categories of financial metrics:
    - Valuation: P/E, P/B, EV/EBITDA
    - Profitability: Margins (gross, operating, net), ROE, ROA
    - Growth: Revenue growth, earnings growth
    - Leverage: Debt-to-equity, debt ratios

    Args:
        df: DataFrame with financial data
        group_by: Optional column name to group by (e.g., 'sector', 'region')

    Returns:
        Dictionary with metrics organized by category, each containing
        mean, median, std, min, max for available metrics
    """
    dashboard = {
        "valuation": {},
        "profitability": {},
        "growth": {},
        "leverage": {},
    }

    # Define metric mappings
    valuation_metrics = ["p_e", "p_b", "ev_ebitda"]
    profitability_metrics = ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]
    growth_metrics = ["revenue_growth", "earnings_growth", "ebitda_growth"]
    leverage_metrics = ["debt_to_equity", "debt_to_assets", "net_debt_to_ebitda"]

    def calculate_stats(series: pd.Series) -> Dict:
        """Calculate statistics for a series, handling NaN values."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        return {
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "std": float(clean_series.std()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "count": int(len(clean_series)),
        }

    # Calculate valuation metrics
    for metric in valuation_metrics:
        if metric in df.columns:
            dashboard["valuation"][metric] = calculate_stats(df[metric])

    # Calculate profitability metrics
    for metric in profitability_metrics:
        if metric in df.columns:
            dashboard["profitability"][metric] = calculate_stats(df[metric])

    # Calculate growth metrics
    for metric in growth_metrics:
        if metric in df.columns:
            dashboard["growth"][metric] = calculate_stats(df[metric])

    # Calculate leverage metrics
    for metric in leverage_metrics:
        if metric in df.columns:
            dashboard["leverage"][metric] = calculate_stats(df[metric])

    # If group_by is specified, add grouped statistics
    if group_by and group_by in df.columns:
        dashboard["by_group"] = {}
        for group_val in df[group_by].dropna().unique():
            group_df = df[df[group_by] == group_val]
            dashboard["by_group"][str(group_val)] = {
                "valuation": {},
                "profitability": {},
                "growth": {},
                "leverage": {},
            }

            # Valuation by group
            for metric in valuation_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["valuation"][metric] = calculate_stats(
                        group_df[metric]
                    )

            # Profitability by group
            for metric in profitability_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["profitability"][metric] = (
                        calculate_stats(group_df[metric])
                    )

            # Growth by group
            for metric in growth_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["growth"][metric] = calculate_stats(
                        group_df[metric]
                    )

            # Leverage by group
            for metric in leverage_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["leverage"][metric] = calculate_stats(
                        group_df[metric]
                    )

    return dashboard


def generate_data_quality_alerts(df: pd.DataFrame, outlier_threshold: float = 3.0) -> list:
    """
    Generate data quality alerts for financial data.

    Detects:
    - Missing values (NaN, null)
    - Statistical outliers (using Z-score method)
    - Negative values in metrics that should be positive
    - Extreme values that may indicate data errors

    Args:
        df: DataFrame with financial data
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        List of alert dictionaries with keys:
        - severity: 'low', 'medium', 'high', 'critical'
        - message: Human-readable alert message
        - column: Column name with the issue
        - count: Number of rows affected (optional)
    """
    alerts = []

    # Financial columns that should not be negative
    positive_only_columns = [
        "market_cap",
        "revenue",
        "total_assets",
        "total_equity",
        "ebitda",
        "last_price",
        "price_target",
    ]

    # Check for missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct_missing = (count / len(df)) * 100
            if pct_missing > 50:
                severity = "critical"
            elif pct_missing > 20:
                severity = "high"
            elif pct_missing > 5:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(
                {
                    "severity": severity,
                    "message": f"Column '{col}' has {count} missing values ({pct_missing:.1f}%)",
                    "column": col,
                    "count": int(count),
                }
            )

    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        clean_data = df[col].dropna()
        if len(clean_data) > 10:  # Need sufficient data for outlier detection
            mean = clean_data.mean()
            std = clean_data.std()

            if std > 0:  # Avoid division by zero
                z_scores = np.abs((clean_data - mean) / std)
                outlier_count = (z_scores > outlier_threshold).sum()

                if outlier_count > 0:
                    pct_outliers = (outlier_count / len(clean_data)) * 100

                    if pct_outliers > 10:
                        severity = "high"
                    elif pct_outliers > 5:
                        severity = "medium"
                    else:
                        severity = "low"

                    alerts.append(
                        {
                            "severity": severity,
                            "message": f"Column '{col}' has {outlier_count} outliers ({pct_outliers:.1f}%) beyond {outlier_threshold} standard deviations",
                            "column": col,
                            "count": int(outlier_count),
                        }
                    )

    # Check for negative values in columns that should be positive
    for col in positive_only_columns:
        if col in df.columns:
            clean_data = df[col].dropna()
            negative_count = (clean_data < 0).sum()

            if negative_count > 0:
                alerts.append(
                    {
                        "severity": "high",
                        "message": f"Column '{col}' has {negative_count} negative values (should be positive)",
                        "column": col,
                        "count": int(negative_count),
                    }
                )

    # Check for zero or near-zero values in key financial metrics
    critical_columns = ["market_cap", "revenue", "last_price"]
    for col in critical_columns:
        if col in df.columns:
            clean_data = df[col].dropna()
            zero_count = (clean_data == 0).sum()
            near_zero_count = ((clean_data > 0) & (clean_data < 0.01)).sum()

            if zero_count > 0:
                alerts.append(
                    {
                        "severity": "medium",
                        "message": f"Column '{col}' has {zero_count} zero values",
                        "column": col,
                        "count": int(zero_count),
                    }
                )

            if near_zero_count > 0:
                alerts.append(
                    {
                        "severity": "low",
                        "message": f"Column '{col}' has {near_zero_count} near-zero values (< 0.01)",
                        "column": col,
                        "count": int(near_zero_count),
                    }
                )

    return alerts


def perform_comprehensive_hypothesis_tests(
    df: pd.DataFrame,
    group_column: str = "sector",
    metrics: Optional[list] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform comprehensive statistical hypothesis tests on financial data.

    Conducts multiple hypothesis tests to compare groups:
    - ANOVA: Parametric test for comparing means across multiple groups
    - Kruskal-Wallis: Non-parametric alternative to ANOVA
    - Pairwise t-tests: For comparing pairs of groups
    - Mann-Whitney U: Non-parametric test for two groups

    Args:
        df: DataFrame with financial data
        group_column: Column to group by (e.g., 'sector', 'region')
        metrics: List of metric columns to test (if None, uses numeric columns)
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results including p-values, statistics, and interpretations
    """
    from scipy import stats

    results = {}

    # Determine metrics to test
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude non-metric columns
        exclude_cols = ["ticker", "year", "quarter"]
        metrics = [m for m in metrics if m not in exclude_cols]

    if group_column not in df.columns:
        return {"error": f"Group column '{group_column}' not found in DataFrame"}

    # Get unique groups
    groups = df[group_column].dropna().unique()
    if len(groups) < 2:
        return {"error": f"Need at least 2 groups for comparison, found {len(groups)}"}

    # Determine test type based on group column
    if group_column == "sector":
        test_type = "sector_tests"
    elif group_column == "region":
        test_type = "region_tests"
    else:
        test_type = f"{group_column}_tests"

    results[test_type] = {}

    # For each metric, perform hypothesis tests
    for metric in metrics:
        if metric not in df.columns:
            continue

        # Prepare data: get metric values for each group
        group_data = []
        for group in groups:
            group_values = df[df[group_column] == group][metric].dropna()
            if len(group_values) >= 2:  # Need at least 2 values per group
                group_data.append(group_values.values)

        if len(group_data) < 2:
            continue  # Skip if insufficient groups

        metric_results = {}

        # ANOVA test (parametric)
        try:
            f_stat, p_value = stats.f_oneway(*group_data)
            metric_results["anova"] = {
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "interpretation": (
                    f"Groups have significantly different means (p={p_value:.4f})"
                    if p_value < alpha
                    else f"No significant difference in means (p={p_value:.4f})"
                ),
            }
        except Exception as e:
            metric_results["anova"] = {"error": str(e)}

        # Kruskal-Wallis test (non-parametric)
        try:
            h_stat, p_value = stats.kruskal(*group_data)
            metric_results["kruskal_wallis"] = {
                "statistic": float(h_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "interpretation": (
                    f"Groups have significantly different distributions (p={p_value:.4f})"
                    if p_value < alpha
                    else f"No significant difference in distributions (p={p_value:.4f})"
                ),
            }
        except Exception as e:
            metric_results["kruskal_wallis"] = {"error": str(e)}

        # Pairwise comparisons (if 2 groups, or store for later analysis)
        if len(group_data) == 2:
            # Perform t-test
            try:
                t_stat, p_value = stats.ttest_ind(group_data[0], group_data[1])
                metric_results["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < alpha,
                    "groups": [str(groups[0]), str(groups[1])],
                }
            except Exception as e:
                metric_results["t_test"] = {"error": str(e)}

            # Perform Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(
                    group_data[0], group_data[1], alternative="two-sided"
                )
                metric_results["mann_whitney_u"] = {
                    "statistic": float(u_stat),
                    "p_value": float(p_value),
                    "significant": p_value < alpha,
                    "groups": [str(groups[0]), str(groups[1])],
                }
            except Exception as e:
                metric_results["mann_whitney_u"] = {"error": str(e)}

        # Store results for this metric
        if metric_results:
            results[test_type][metric] = metric_results

    # Add overall summary
    if test_type in results and results[test_type]:
        results[test_type]["summary"] = {
            "total_metrics_tested": len(results[test_type]) - 1,  # -1 for summary itself
            "groups_compared": list(map(str, groups)),
            "alpha": alpha,
        }

    return results


def test_market_efficiency_hypothesis(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test market efficiency hypothesis using price/target relationships.

    Tests whether analyst price targets are significantly different from
    current prices, which can indicate market inefficiency or information
    asymmetry.

    Performs:
    - Paired t-test: Tests if price targets differ from current prices
    - Directional bias test: Tests if targets are systematically higher/lower
    - Correlation test: Tests relationship between price and target

    Args:
        df: DataFrame with 'last_price' and 'price_target' columns
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results and interpretations
    """
    from scipy import stats

    results = {}

    # Check if required columns exist
    if "last_price" not in df.columns or "price_target" not in df.columns:
        return {
            "error": "Required columns 'last_price' and 'price_target' not found",
            "price_target_test": {"error": "Missing required columns"},
        }

    # Get clean data
    clean_df = df[["last_price", "price_target"]].dropna()

    if len(clean_df) < 2:
        return {
            "error": "Insufficient data for hypothesis testing",
            "price_target_test": {"error": "Insufficient data"},
        }

    prices = clean_df["last_price"].values
    targets = clean_df["price_target"].values

    # Paired t-test: Are targets significantly different from prices?
    try:
        t_stat, p_value = stats.ttest_rel(targets, prices)
        mean_diff = np.mean(targets - prices)
        pct_diff = (mean_diff / np.mean(prices)) * 100

        results["price_target_test"] = {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "mean_difference": float(mean_diff),
            "mean_difference_pct": float(pct_diff),
            "interpretation": (
                f"Price targets are significantly different from current prices "
                f"(p={p_value:.4f}, {pct_diff:+.2f}% difference)"
                if p_value < alpha
                else f"No significant difference between targets and prices (p={p_value:.4f})"
            ),
        }
    except Exception as e:
        results["price_target_test"] = {"error": str(e)}

    # Directional bias test: Are targets systematically higher or lower?
    try:
        upside_count = (targets > prices).sum()
        downside_count = (targets < prices).sum()
        total_count = len(targets)

        # Binomial test: is the proportion of upside significantly different from 50%?
        binomial_p = stats.binom_test(upside_count, total_count, 0.5, alternative="two-sided")

        results["directional_bias_test"] = {
            "upside_count": int(upside_count),
            "downside_count": int(downside_count),
            "upside_pct": float((upside_count / total_count) * 100),
            "p_value": float(binomial_p),
            "significant": binomial_p < alpha,
            "interpretation": (
                f"Analyst targets show significant directional bias "
                f"({upside_count/total_count:.1%} upside vs {downside_count/total_count:.1%} downside, p={binomial_p:.4f})"
                if binomial_p < alpha
                else f"No significant directional bias in analyst targets (p={binomial_p:.4f})"
            ),
        }
    except Exception as e:
        results["directional_bias_test"] = {"error": str(e)}

    # Correlation test: How correlated are prices and targets?
    try:
        corr, p_value = stats.pearsonr(prices, targets)

        results["correlation_test"] = {
            "correlation": float(corr),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "interpretation": (
                f"Strong correlation between prices and targets (r={corr:.3f}, p={p_value:.4f})"
                if corr > 0.7 and p_value < alpha
                else (
                    f"Moderate correlation (r={corr:.3f}, p={p_value:.4f})"
                    if corr > 0.4 and p_value < alpha
                    else f"Weak correlation (r={corr:.3f}, p={p_value:.4f})"
                )
            ),
        }
    except Exception as e:
        results["correlation_test"] = {"error": str(e)}

    # Market efficiency interpretation
    if "price_target_test" in results and not results["price_target_test"].get("error"):
        if results["price_target_test"]["significant"]:
            efficiency = "INEFFICIENT"
            explanation = (
                "Significant difference between prices and targets suggests "
                "market inefficiency or information asymmetry"
            )
        else:
            efficiency = "EFFICIENT"
            explanation = "Prices align with analyst targets, suggesting market efficiency"

        results["market_efficiency"] = {
            "assessment": efficiency,
            "explanation": explanation,
        }

    return results


def prepare_interactive_dashboard_data(df: pd.DataFrame) -> Dict:
    """
    Prepare structured data for interactive dashboards.

    Organizes data into sections suitable for visualization:
    - Summary statistics for key metrics
    - Breakdown by sector
    - Breakdown by region
    - Top performers and laggards

    Args:
        df: DataFrame with financial data

    Returns:
        Dictionary with structured dashboard data
    """
    dashboard_data = {
        "summary_stats": {},
        "by_sector": {},
        "by_region": {},
        "top_performers": {},
        "data_quality": {},
    }

    # Summary statistics for key metrics
    key_metrics = [
        "market_cap",
        "last_price",
        "price_target",
        "p_e",
        "p_b",
        "revenue",
        "net_income",
        "roe",
        "revenue_growth",
    ]

    for metric in key_metrics:
        if metric in df.columns:
            clean_data = df[metric].dropna()
            if len(clean_data) > 0:
                dashboard_data["summary_stats"][metric] = {
                    "mean": float(clean_data.mean()),
                    "median": float(clean_data.median()),
                    "min": float(clean_data.min()),
                    "max": float(clean_data.max()),
                    "std": float(clean_data.std()),
                    "count": int(len(clean_data)),
                }

    # Breakdown by sector
    if "sector" in df.columns:
        for sector in df["sector"].dropna().unique():
            sector_df = df[df["sector"] == sector]
            dashboard_data["by_sector"][str(sector)] = {
                "count": int(len(sector_df)),
                "avg_market_cap": (
                    float(sector_df["market_cap"].mean())
                    if "market_cap" in sector_df.columns
                    else None
                ),
                "avg_p_e": (float(sector_df["p_e"].mean()) if "p_e" in sector_df.columns else None),
                "avg_roe": (float(sector_df["roe"].mean()) if "roe" in sector_df.columns else None),
            }

            # Add mispricing score if available
            if "mispricing_score" in sector_df.columns:
                dashboard_data["by_sector"][str(sector)]["avg_mispricing"] = float(
                    sector_df["mispricing_score"].mean()
                )

    # Breakdown by region
    if "region" in df.columns:
        for region in df["region"].dropna().unique():
            region_df = df[df["region"] == region]
            dashboard_data["by_region"][str(region)] = {
                "count": int(len(region_df)),
                "avg_market_cap": (
                    float(region_df["market_cap"].mean())
                    if "market_cap" in region_df.columns
                    else None
                ),
                "avg_p_e": (float(region_df["p_e"].mean()) if "p_e" in region_df.columns else None),
                "avg_roe": (float(region_df["roe"].mean()) if "roe" in region_df.columns else None),
            }

            # Add mispricing score if available
            if "mispricing_score" in region_df.columns:
                dashboard_data["by_region"][str(region)]["avg_mispricing"] = float(
                    region_df["mispricing_score"].mean()
                )

    # Top performers (if mispricing_score available)
    if "mispricing_score" in df.columns and "ticker" in df.columns:
        top_5 = df.nlargest(5, "mispricing_score")[["ticker", "mispricing_score"]]
        dashboard_data["top_performers"]["most_undervalued"] = [
            {"ticker": row["ticker"], "score": float(row["mispricing_score"])}
            for _, row in top_5.iterrows()
        ]

        bottom_5 = df.nsmallest(5, "mispricing_score")[["ticker", "mispricing_score"]]
        dashboard_data["top_performers"]["most_overvalued"] = [
            {"ticker": row["ticker"], "score": float(row["mispricing_score"])}
            for _, row in bottom_5.iterrows()
        ]

    # Data quality summary
    dashboard_data["data_quality"]["total_rows"] = int(len(df))
    dashboard_data["data_quality"]["total_columns"] = int(len(df.columns))
    dashboard_data["data_quality"]["missing_values"] = int(df.isnull().sum().sum())
    dashboard_data["data_quality"]["completeness_pct"] = float(
        (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    )

    return dashboard_data


def apply_dashboard_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to DataFrame for interactive dashboard.

    Supports filters:
    - sectors: List of sectors to include
    - regions: List of regions to include
    - min_market_cap: Minimum market cap
    - max_market_cap: Maximum market cap
    - min_mispricing: Minimum mispricing score
    - max_mispricing: Maximum mispricing score

    Args:
        df: DataFrame with financial data
        filters: Dictionary of filter criteria

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Sector filter
    if "sectors" in filters and filters["sectors"]:
        if "sector" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["sector"].isin(filters["sectors"])]

    # Region filter
    if "regions" in filters and filters["regions"]:
        if "region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["region"].isin(filters["regions"])]

    # Market cap range filter
    if "min_market_cap" in filters and filters["min_market_cap"] is not None:
        if "market_cap" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["market_cap"] >= filters["min_market_cap"]]

    if "max_market_cap" in filters and filters["max_market_cap"] is not None:
        if "market_cap" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["market_cap"] <= filters["max_market_cap"]]

    # Mispricing score range filter
    if "min_mispricing" in filters and filters["min_mispricing"] is not None:
        if "mispricing_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["mispricing_score"] >= filters["min_mispricing"]]

    if "max_mispricing" in filters and filters["max_mispricing"] is not None:
        if "mispricing_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["mispricing_score"] <= filters["max_mispricing"]]

    return filtered_df


def calculate_peer_comparisons(df: pd.DataFrame, ticker: str, n_peers: int = 5) -> Dict:
    """
    Calculate peer comparisons for a given stock.

    Finds similar stocks in the same sector and compares key metrics.

    Args:
        df: DataFrame with financial data
        ticker: Ticker symbol to analyze
        n_peers: Number of peer stocks to include (default: 5)

    Returns:
        Dictionary with stock data, sector average, and peer comparisons
    """
    if "ticker" not in df.columns:
        return {"error": "Column 'ticker' not found in DataFrame"}

    # Get the stock
    stock_data = df[df["ticker"] == ticker]
    if len(stock_data) == 0:
        return {"error": f"Ticker '{ticker}' not found in DataFrame"}

    stock_row = stock_data.iloc[0]

    result = {
        "stock": {},
        "sector_avg": {},
        "peers": [],
    }

    # Stock data
    metrics = ["market_cap", "p_e", "p_b", "roe", "revenue_growth", "mispricing_score"]
    for metric in metrics:
        if metric in stock_row.index:
            value = stock_row[metric]
            result["stock"][metric] = float(value) if pd.notna(value) else None

    result["stock"]["ticker"] = ticker
    if "sector" in stock_row.index:
        result["stock"]["sector"] = stock_row["sector"]

    # Sector average
    if "sector" in stock_row.index and pd.notna(stock_row["sector"]):
        sector_df = df[df["sector"] == stock_row["sector"]]

        for metric in metrics:
            if metric in sector_df.columns:
                result["sector_avg"][metric] = float(sector_df[metric].mean())

        # Find peers (stocks in same sector, sorted by similarity in market cap)
        if "market_cap" in df.columns and pd.notna(stock_row.get("market_cap")):
            peers_df = sector_df[sector_df["ticker"] != ticker].copy()

            if len(peers_df) > 0:
                # Calculate market cap similarity
                peers_df["market_cap_diff"] = (
                    peers_df["market_cap"] - stock_row["market_cap"]
                ).abs()
                peers_df = peers_df.sort_values("market_cap_diff").head(n_peers)

                # Build peer list
                for _, peer_row in peers_df.iterrows():
                    peer_data = {"ticker": peer_row["ticker"]}
                    for metric in metrics:
                        if metric in peer_row.index:
                            value = peer_row[metric]
                            peer_data[metric] = float(value) if pd.notna(value) else None
                    result["peers"].append(peer_data)

    return result


# ============================================================================
# Phase 9.3: Future Enhancements
# ============================================================================


def perform_time_series_hypothesis_tests(
    df: pd.DataFrame,
    date_column: str,
    metrics: list,
    group_by: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform comprehensive time-series hypothesis tests for temporal trends.

    Tests performed:
    - Mann-Kendall trend test: Detects monotonic trends
    - Augmented Dickey-Fuller test: Tests for stationarity
    - Ljung-Box test: Tests for autocorrelation

    Args:
        df: DataFrame with time-series data
        date_column: Name of date/datetime column
        metrics: List of metric columns to test
        group_by: Optional column to group by (e.g., 'ticker', 'sector')
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results for each metric

    Example:
        >>> result = perform_time_series_hypothesis_tests(
        ...     df, date_column='date', metrics=['price', 'volume']
        ... )
        >>> print(result['trend_tests']['price']['has_trend'])
    """
    try:
        from scipy import stats
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return {"error": "Required packages not available. Install scipy and statsmodels."}

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    result = {
        "trend_tests": {},
        "stationarity_tests": {},
        "autocorrelation_tests": {},
    }

    # Helper function for Mann-Kendall trend test
    def mann_kendall_test(data):
        """Simple Mann-Kendall trend test implementation."""
        n = len(data)
        if n < 3:
            return {"has_trend": False, "p_value": 1.0, "test_statistic": 0}

        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])

        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "has_trend": p_value < alpha,
            "p_value": float(p_value),
            "test_statistic": float(z),
            "direction": "increasing" if z > 0 else "decreasing" if z < 0 else "none",
        }

    # Process each metric
    def process_metric(data_df, metric_name):
        """Process a single metric."""
        if metric_name not in data_df.columns:
            return None

        # Sort by date
        sorted_df = data_df.sort_values(date_column)
        series = sorted_df[metric_name].dropna()

        if len(series) < 3:
            return None

        metric_results = {}

        # 1. Trend test (Mann-Kendall)
        trend_result = mann_kendall_test(series.values)
        metric_results["trend"] = trend_result

        # 2. Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(series, autolag="AIC")
            metric_results["stationarity"] = {
                "is_stationary": adf_result[1] < alpha,
                "adf_statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            }
        except Exception as e:
            metric_results["stationarity"] = {"error": str(e)}

        # 3. Autocorrelation test (Ljung-Box)
        try:
            # Test up to lag 10 or min(len(series)//4, 10)
            # Ensure we have enough data: need at least 2*n_lags observations
            n_lags = min(10, max(1, len(series) // 4))

            # Ljung-Box needs at least n_lags + 1 observations
            if len(series) > n_lags:
                lb_result = acorr_ljungbox(series, lags=n_lags, return_df=False)
                # lb_result is a tuple: (lb_stat, lb_pvalue)
                # Each is an array, use the last lag result
                lb_stat = lb_result[0]
                lb_pvalue = lb_result[1]

                # Handle both array and scalar returns
                if hasattr(lb_stat, "__len__") and len(lb_stat) > 0:
                    lb_stat_val = float(lb_stat[-1])
                    lb_pvalue_val = float(lb_pvalue[-1])
                else:
                    lb_stat_val = float(lb_stat)
                    lb_pvalue_val = float(lb_pvalue)

                metric_results["autocorrelation"] = {
                    "has_autocorrelation": lb_pvalue_val < alpha,
                    "ljung_box_statistic": lb_stat_val,
                    "p_value": lb_pvalue_val,
                }
            else:
                metric_results["autocorrelation"] = {
                    "error": "Insufficient data for autocorrelation test"
                }
        except Exception as e:
            metric_results["autocorrelation"] = {"error": str(e)}

        return metric_results

    if group_by and group_by in df.columns:
        # Group-wise analysis
        result["by_group"] = {}
        for group_name, group_df in df.groupby(group_by):
            group_results = {}
            for metric in metrics:
                metric_result = process_metric(group_df, metric)
                if metric_result:
                    group_results[metric] = metric_result
            if group_results:
                result["by_group"][group_name] = group_results
    else:
        # Overall analysis
        for metric in metrics:
            metric_result = process_metric(df, metric)
            if metric_result:
                result["trend_tests"][metric] = metric_result["trend"]
                result["stationarity_tests"][metric] = metric_result["stationarity"]
                result["autocorrelation_tests"][metric] = metric_result["autocorrelation"]

    return result


def perform_multi_factor_anova(
    df: pd.DataFrame, dependent_var, factors: list, alpha: float = 0.05, post_hoc: bool = False
) -> Dict:
    """
    Perform multi-factor ANOVA to test for interaction effects.

    Tests main effects and interaction effects between multiple factors.
    Useful for understanding how sector, region, and other categorical
    variables interact to influence financial metrics.

    Args:
        df: DataFrame with financial data
        dependent_var: Dependent variable(s) - string or list of strings
        factors: List of factor columns (categorical variables)
        alpha: Significance level (default: 0.05)
        post_hoc: Whether to perform post-hoc pairwise comparisons (default: False)

    Returns:
        Dictionary with main effects, interaction effects, and model summary

    Example:
        >>> result = perform_multi_factor_anova(
        ...     df, dependent_var='p_e', factors=['sector', 'region']
        ... )
        >>> print(result['interaction_effects']['sector:region']['significant'])
    """
    try:
        from scipy import stats
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        return {"error": "Required packages not available. Install scipy and statsmodels."}

    # Handle multiple dependent variables
    if isinstance(dependent_var, list):
        results = {}
        for var in dependent_var:
            results[var] = perform_multi_factor_anova(df, var, factors, alpha, post_hoc)
        return results

    # Single dependent variable case
    if dependent_var not in df.columns:
        return {"error": f"Dependent variable '{dependent_var}' not found in DataFrame"}

    for factor in factors:
        if factor not in df.columns:
            return {"error": f"Factor '{factor}' not found in DataFrame"}

    # Prepare data - remove missing values
    columns_needed = [dependent_var] + factors
    clean_df = df[columns_needed].dropna()

    if len(clean_df) < 10:
        return {"error": "Insufficient data after removing missing values"}

    result = {
        "main_effects": {},
        "interaction_effects": {},
        "model_summary": {},
        "effect_sizes": {},
    }

    # Build formula for ANOVA
    # Main effects: factor1 + factor2 + ...
    # Two-way interactions: factor1:factor2
    # Three-way interactions: factor1:factor2:factor3
    formula_parts = []

    # Main effects
    formula_parts.extend(factors)

    # Two-way interactions
    if len(factors) >= 2:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                formula_parts.append(f"{factors[i]}:{factors[j]}")

    # Three-way interactions (only if 3+ factors)
    if len(factors) >= 3:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                for k in range(j + 1, len(factors)):
                    formula_parts.append(f"{factors[i]}:{factors[j]}:{factors[k]}")

    formula = f"{dependent_var} ~ " + " + ".join(formula_parts)

    try:
        # Fit model
        model = ols(formula, data=clean_df).fit()
        anova_table = anova_lm(model, typ=2)

        # Total sum of squares for effect size calculation
        total_ss = anova_table["sum_sq"].sum()

        # Parse results
        for idx, row in anova_table.iterrows():
            if idx == "Residual":
                continue

            effect_data = {
                "f_statistic": float(row["F"]) if pd.notna(row["F"]) else None,
                "p_value": float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else None,
                "df": int(row["df"]) if pd.notna(row["df"]) else None,
                "sum_sq": float(row["sum_sq"]) if pd.notna(row["sum_sq"]) else None,
            }

            # Determine if significant
            if effect_data["p_value"] is not None:
                effect_data["significant"] = effect_data["p_value"] < alpha

            # Calculate effect size (eta-squared)
            if effect_data["sum_sq"] is not None and total_ss > 0:
                eta_squared = effect_data["sum_sq"] / total_ss
                result["effect_sizes"][idx] = {"eta_squared": float(eta_squared)}

            # Categorize as main or interaction effect
            if ":" in idx:
                result["interaction_effects"][idx] = effect_data
            else:
                result["main_effects"][idx] = effect_data

        # Model summary
        result["model_summary"] = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "n_observations": int(model.nobs),
        }

        # Post-hoc tests (Tukey HSD) for main effects if requested
        if post_hoc:
            result["post_hoc"] = {}
            for factor in factors:
                try:
                    tukey = pairwise_tukeyhsd(
                        endog=clean_df[dependent_var], groups=clean_df[factor], alpha=alpha
                    )
                    # Convert to dictionary format
                    post_hoc_results = []
                    for i in range(len(tukey.summary().data) - 1):  # Skip header
                        row = tukey.summary().data[i + 1]
                        post_hoc_results.append(
                            {
                                "group1": str(row[0]),
                                "group2": str(row[1]),
                                "meandiff": float(row[2]),
                                "p_adj": float(row[3]),
                                "significant": row[5] == "True",
                            }
                        )
                    result["post_hoc"][factor] = post_hoc_results
                except Exception as e:
                    result["post_hoc"][factor] = {"error": str(e)}

    except Exception as e:
        return {"error": f"ANOVA failed: {str(e)}"}

    return result


def correct_outliers_with_validation(
    df: pd.DataFrame,
    columns: list,
    method: str = "winsorize",
    n_std: float = 3.0,
    limits: tuple = (0.05, 0.05),
    impute_strategy: str = "median",
    by_group: Optional[str] = None,
    return_mapping: bool = False,
) -> Dict:
    """
    Automated outlier correction with validation metrics.

    Corrects outliers using various methods and provides before/after
    validation to assess the impact of corrections.

    Args:
        df: DataFrame with data to correct
        columns: List of columns to check for outliers
        method: Correction method - 'winsorize', 'clip', or 'impute'
        n_std: Number of standard deviations for 'clip' method (default: 3.0)
        limits: Percentile limits for 'winsorize' method (default: (0.05, 0.05))
        impute_strategy: Strategy for 'impute' method - 'median', 'mean', or 'mode'
        by_group: Optional column to group by for group-specific correction
        return_mapping: Whether to return mapping of corrected indices

    Returns:
        Dictionary with corrected_data, outlier_report, validation metrics, and optional mapping

    Example:
        >>> result = correct_outliers_with_validation(
        ...     df, columns=['price', 'volume'], method='winsorize'
        ... )
        >>> corrected_df = result['corrected_data']
        >>> print(result['validation']['improvement'])
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return {"error": "scipy required for outlier correction"}

    corrected_df = df.copy()
    outlier_report = {}
    correction_mapping = {}
    validation = {"before": {}, "after": {}, "improvement": {}}

    def detect_outliers_zscore(series, threshold=3.0):
        """Detect outliers using z-score method."""
        if len(series) < 3:
            return np.array([False] * len(series))

        mean = series.mean()
        std = series.std()
        if std == 0:
            return np.array([False] * len(series))

        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold

    def calculate_validation_metrics(series):
        """Calculate validation metrics for a series."""
        if len(series) == 0:
            return {}

        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skewness": float(sp_stats.skew(series.dropna())),
            "kurtosis": float(sp_stats.kurtosis(series.dropna())),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    def correct_outliers_in_series(series, col_name):
        """Correct outliers in a single series."""
        # Store original for validation
        original_series = series.copy()

        # Detect outliers
        outliers_mask = detect_outliers_zscore(series, threshold=n_std)
        n_outliers = outliers_mask.sum()

        outlier_indices = series.index[outliers_mask].tolist()

        # Apply correction method
        corrected_series = series.copy()

        if method == "winsorize":
            # Winsorize: cap at percentiles
            lower_limit = series.quantile(limits[0])
            upper_limit = series.quantile(1 - limits[1])
            corrected_series = series.clip(lower=lower_limit, upper=upper_limit)

        elif method == "clip":
            # Clip: cap at mean ± n_std
            mean = series.mean()
            std = series.std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            corrected_series = series.clip(lower=lower_bound, upper=upper_bound)

        elif method == "impute":
            # Impute: replace outliers with central tendency
            if impute_strategy == "median":
                fill_value = series.median()
            elif impute_strategy == "mean":
                fill_value = series.mean()
            elif impute_strategy == "mode":
                fill_value = series.mode()[0] if len(series.mode()) > 0 else series.median()
            else:
                fill_value = series.median()

            corrected_series[outliers_mask] = fill_value

        else:
            raise ValueError(f"Unknown correction method: {method}")

        # Store report
        outlier_report[col_name] = {
            "n_outliers": int(n_outliers),
            "pct_outliers": float(n_outliers / len(series) * 100),
            "method": method,
            "outlier_values": {
                "min": float(series[outliers_mask].min()) if n_outliers > 0 else None,
                "max": float(series[outliers_mask].max()) if n_outliers > 0 else None,
            },
        }

        # Store mapping if requested
        if return_mapping:
            correction_mapping[col_name] = {
                "outlier_indices": outlier_indices,
                "original_values": original_series[outliers_mask].to_dict(),
                "corrected_values": corrected_series[outliers_mask].to_dict(),
            }

        # Validation metrics - store both flat (for single column) and nested (for multi-column)
        col_metrics_before = calculate_validation_metrics(original_series)
        col_metrics_after = calculate_validation_metrics(corrected_series)

        validation["before"][col_name] = col_metrics_before
        validation["after"][col_name] = col_metrics_after

        # For single column case, also expose metrics at root level
        if len(columns) == 1:
            validation["before"].update(col_metrics_before)
            validation["after"].update(col_metrics_after)

        # Calculate improvement
        if col_metrics_before and col_metrics_after:
            improvement_metrics = {
                "skewness_reduction": float(
                    abs(col_metrics_before["skewness"]) - abs(col_metrics_after["skewness"])
                ),
                "kurtosis_reduction": float(
                    abs(col_metrics_before["kurtosis"]) - abs(col_metrics_after["kurtosis"])
                ),
                "std_reduction_pct": (
                    float(
                        (col_metrics_before["std"] - col_metrics_after["std"])
                        / col_metrics_before["std"]
                        * 100
                    )
                    if col_metrics_before["std"] > 0
                    else 0
                ),
            }
            validation["improvement"][col_name] = improvement_metrics
            if len(columns) == 1:
                validation["improvement"].update(improvement_metrics)

        return corrected_series

    # Process each column
    if by_group and by_group in df.columns:
        # Group-wise correction
        result_by_group = {}

        for group_name, group_df in df.groupby(by_group):
            group_report = {}
            for col in columns:
                if col in group_df.columns:
                    corrected_series = correct_outliers_in_series(
                        group_df[col].copy(), f"{group_name}_{col}"
                    )
                    corrected_df.loc[group_df.index, col] = corrected_series
                    group_report[col] = outlier_report[f"{group_name}_{col}"]

            result_by_group[group_name] = group_report

        result = {
            "corrected_data": corrected_df,
            "by_group": result_by_group,
            "validation": validation,
        }
    else:
        # Overall correction
        for col in columns:
            if col in df.columns:
                corrected_df[col] = correct_outliers_in_series(df[col].copy(), col)

        result = {
            "corrected_data": corrected_df,
            "outlier_report": outlier_report,
            "validation": validation,
        }

    if return_mapping:
        result["correction_mapping"] = correction_mapping

    return result


def prepare_plotly_dashboard_data(
    df: pd.DataFrame, include_timeseries: bool = False, color_scheme: str = "plotly"
) -> Dict:
    """
    Prepare structured data for interactive Plotly dashboards.

    Generates data structures optimized for various Plotly chart types:
    scatter plots, histograms, box plots, heatmaps, sunburst charts, and treemaps.

    Args:
        df: DataFrame with financial data
        include_timeseries: Whether to include time-series data (requires 'date' column)
        color_scheme: Color scheme for visualizations (default: 'plotly')

    Returns:
        Dictionary with data structured for different Plotly chart types

    Example:
        >>> data = prepare_plotly_dashboard_data(df)
        >>> # Use with Plotly:
        >>> # fig = px.scatter(**data['scatter_data'])
    """
    result = {
        "scatter_data": {},
        "histogram_data": {},
        "box_data": {},
        "heatmap_data": {},
        "sunburst_data": {},
        "treemap_data": {},
        "color_scales": {"default": color_scheme},
    }

    # 1. Scatter plot data (mispricing vs market cap)
    if all(col in df.columns for col in ["last_price", "market_cap", "mispricing_score"]):
        result["scatter_data"] = {
            "x": df["market_cap"].tolist(),
            "y": df["mispricing_score"].tolist(),
            "text": df["ticker"].tolist() if "ticker" in df.columns else None,
            "color": df["sector"].tolist() if "sector" in df.columns else None,
            "size": df["last_price"].tolist(),
        }

    # 2. Histogram data (mispricing by sector)
    if "mispricing_score" in df.columns and "sector" in df.columns:
        hist_by_sector = []
        for sector in df["sector"].dropna().unique():
            sector_data = df[df["sector"] == sector]["mispricing_score"].dropna()
            if len(sector_data) > 0:
                hist_by_sector.append(
                    {"sector": sector, "values": sector_data.tolist(), "name": sector}
                )
        result["histogram_data"]["mispricing_by_sector"] = hist_by_sector

    # 3. Box plot data (sector and region comparisons)
    box_comparisons = {}

    if "sector" in df.columns and "p_e" in df.columns:
        sector_box = []
        for sector in df["sector"].dropna().unique():
            sector_data = df[df["sector"] == sector]["p_e"].dropna()
            if len(sector_data) > 0:
                sector_box.append({"name": sector, "y": sector_data.tolist()})
        box_comparisons["sector_comparisons"] = sector_box

    if "region" in df.columns and "roe" in df.columns:
        region_box = []
        for region in df["region"].dropna().unique():
            region_data = df[df["region"] == region]["roe"].dropna()
            if len(region_data) > 0:
                region_box.append({"name": region, "y": region_data.tolist()})
        box_comparisons["region_comparisons"] = region_box

    result["box_data"] = box_comparisons

    # 4. Heatmap data (correlation matrix)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Select key financial metrics if available
        key_metrics = ["p_e", "p_b", "roe", "revenue_growth", "mispricing_score", "market_cap"]
        available_metrics = [col for col in key_metrics if col in numeric_cols]

        if len(available_metrics) >= 2:
            corr_matrix = df[available_metrics].corr()
            result["heatmap_data"]["correlation_matrix"] = {
                "z": corr_matrix.values.tolist(),
                "x": corr_matrix.columns.tolist(),
                "y": corr_matrix.index.tolist(),
            }

    # 5. Sunburst chart data (hierarchical: region > sector > ticker)
    if all(col in df.columns for col in ["region", "sector", "ticker"]):
        labels = ["All"]
        parents = [""]
        values = [len(df)]

        # Region level
        for region in df["region"].dropna().unique():
            labels.append(str(region))
            parents.append("All")
            region_count = len(df[df["region"] == region])
            values.append(region_count)

            # Sector level within region
            region_df = df[df["region"] == region]
            for sector in region_df["sector"].dropna().unique():
                sector_label = f"{region}_{sector}"
                labels.append(str(sector))
                parents.append(str(region))
                sector_count = len(region_df[region_df["sector"] == sector])
                values.append(sector_count)

        result["sunburst_data"] = {"labels": labels, "parents": parents, "values": values}

    # 6. Treemap data (sector/region breakdown with market cap)
    if all(col in df.columns for col in ["sector", "region", "market_cap"]):
        labels = []
        parents = []
        values = []

        # Add root
        labels.append("All")
        parents.append("")
        values.append(df["market_cap"].sum())

        # Add sectors
        for sector in df["sector"].dropna().unique():
            sector_df = df[df["sector"] == sector]
            labels.append(str(sector))
            parents.append("All")
            values.append(sector_df["market_cap"].sum())

            # Add regions within sectors
            for region in sector_df["region"].dropna().unique():
                region_df = sector_df[sector_df["region"] == region]
                labels.append(f"{sector}_{region}")
                parents.append(str(sector))
                values.append(region_df["market_cap"].sum())

        result["treemap_data"] = {"labels": labels, "parents": parents, "values": values}

    # 7. Time-series data (optional)
    if include_timeseries and "date" in df.columns:
        ts_data = {}

        # Try common price columns
        price_col = None
        for col_name in ["price", "last_price", "close", "adj_close"]:
            if col_name in df.columns:
                price_col = col_name
                break

        if price_col:
            # Aggregate by date
            daily_avg = df.groupby("date")[price_col].mean().reset_index()
            ts_data = {"dates": daily_avg["date"].tolist(), "values": daily_avg[price_col].tolist()}
        elif len(df.select_dtypes(include=[np.number]).columns) > 0:
            # Fallback: use first numeric column
            numeric_col = df.select_dtypes(include=[np.number]).columns[0]
            daily_avg = df.groupby("date")[numeric_col].mean().reset_index()
            ts_data = {
                "dates": daily_avg["date"].tolist(),
                "values": daily_avg[numeric_col].tolist(),
            }

        result["timeseries_data"] = ts_data

    return result


def generate_enhanced_pdf_report(
    df: pd.DataFrame,
    pdf_path: Path,
    title: str = "Enhanced Financial Analysis Report",
    include_financial_dashboard: bool = False,
    include_quality_alerts: bool = False,
    include_hypothesis_tests: bool = False,
    include_charts: bool = False,
    include_toc: bool = False,
    template: str = "default",
) -> Dict:
    """
    Generate enhanced PDF report integrating Phase 9.2 and 9.3 features.

    Extends the existing generate_pdf_report() with comprehensive analytics:
    - Financial metrics dashboard
    - Data quality alerts
    - Statistical hypothesis testing results
    - Interactive chart embeddings
    - Table of contents

    Args:
        df: DataFrame with financial data and predictions
        pdf_path: Path to save the PDF report
        title: Report title
        include_financial_dashboard: Include Phase 9.2 financial dashboard
        include_quality_alerts: Include Phase 9.2 data quality alerts
        include_hypothesis_tests: Include Phase 9.2/9.3 hypothesis tests
        include_charts: Include embedded charts
        include_toc: Include table of contents
        template: Template style - 'default', 'modern', 'classic'

    Returns:
        Dictionary with report metadata and section information

    Example:
        >>> result = generate_enhanced_pdf_report(
        ...     df, Path('report.pdf'),
        ...     include_financial_dashboard=True,
        ...     include_quality_alerts=True
        ... )
        >>> print(f"Report saved: {result['page_count']} pages")
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        return {
            "error": "reportlab required for PDF generation. Install with: pip install reportlab"
        }

    # Initialize document
    pagesize = A4 if template == "modern" else letter
    doc = SimpleDocTemplate(str(pdf_path), pagesize=pagesize)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=30,
        alignment=1,  # Center
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    # Track sections for metadata
    sections = {}
    page_count = 0

    # 1. Title Page
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3 * inch))

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Total Stocks Analyzed: {len(df)}", styles["Normal"]))
    story.append(PageBreak())
    page_count += 1

    # 2. Table of Contents (if requested)
    if include_toc:
        toc_entries = ["1. Executive Summary"]
        if include_financial_dashboard:
            toc_entries.append("2. Financial Metrics Dashboard")
        if include_quality_alerts:
            toc_entries.append("3. Data Quality Analysis")
        if include_hypothesis_tests:
            toc_entries.append("4. Statistical Hypothesis Testing")
        if include_charts:
            toc_entries.append("5. Visualizations")

        story.append(Paragraph("Table of Contents", heading_style))
        story.append(Spacer(1, 0.2 * inch))
        for entry in toc_entries:
            story.append(Paragraph(entry, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
        story.append(PageBreak())
        page_count += 1
        sections["table_of_contents"] = toc_entries
        toc_list = toc_entries  # Store for root-level exposure

    # 3. Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    summary_text = f"""
    This report provides a comprehensive analysis of {len(df)} financial instruments across 
    multiple dimensions including valuation, profitability, growth, and risk metrics.
    """
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Basic statistics
    if "mispricing_score" in df.columns:
        avg_mispricing = df["mispricing_score"].mean()
        story.append(
            Paragraph(f"<b>Average Mispricing:</b> {avg_mispricing:.2f}%", styles["Normal"])
        )

    if "sector" in df.columns:
        n_sectors = df["sector"].nunique()
        story.append(Paragraph(f"<b>Sectors Covered:</b> {n_sectors}", styles["Normal"]))

    story.append(PageBreak())
    page_count += 1

    # 4. Financial Metrics Dashboard
    if include_financial_dashboard:
        story.append(Paragraph("Financial Metrics Dashboard", heading_style))
        story.append(Spacer(1, 0.2 * inch))

        dashboard = calculate_financial_metrics_dashboard(df)
        sections["financial_dashboard"] = "included"

        # Valuation metrics
        if "valuation" in dashboard:
            story.append(Paragraph("<b>Valuation Metrics</b>", styles["Heading3"]))

            valuation_data = [["Metric", "Mean", "Median", "Std Dev"]]
            for metric, stats in dashboard["valuation"].items():
                if stats:
                    valuation_data.append(
                        [
                            metric.upper().replace("_", "/"),
                            f"{stats.get('mean', 0):.2f}",
                            f"{stats.get('median', 0):.2f}",
                            f"{stats.get('std', 0):.2f}",
                        ]
                    )

            if len(valuation_data) > 1:
                t = Table(valuation_data)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 0.3 * inch))

        story.append(PageBreak())
        page_count += 1

    # 5. Data Quality Alerts
    if include_quality_alerts:
        story.append(Paragraph("Data Quality Analysis", heading_style))
        story.append(Spacer(1, 0.2 * inch))

        alerts = generate_data_quality_alerts(df)
        sections["quality_alerts"] = "included"

        # Group by severity
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
        high_alerts = [a for a in alerts if a.get("severity") == "high"]

        story.append(Paragraph(f"<b>Total Alerts:</b> {len(alerts)}", styles["Normal"]))
        story.append(Paragraph(f"<b>Critical:</b> {len(critical_alerts)}", styles["Normal"]))
        story.append(Paragraph(f"<b>High Priority:</b> {len(high_alerts)}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Show top 5 critical alerts
        if critical_alerts:
            story.append(Paragraph("<b>Top Critical Issues:</b>", styles["Heading3"]))
            for i, alert in enumerate(critical_alerts[:5], 1):
                story.append(Paragraph(f"{i}. {alert.get('message', 'N/A')}", styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))

        story.append(PageBreak())
        page_count += 1

    # 6. Hypothesis Testing Results
    if include_hypothesis_tests:
        story.append(Paragraph("Statistical Hypothesis Testing", heading_style))
        story.append(Spacer(1, 0.2 * inch))

        sections["hypothesis_tests"] = "included"

        # Perform sector comparisons if possible
        if "sector" in df.columns and "p_e" in df.columns:
            hyp_results = perform_comprehensive_hypothesis_tests(
                df, group_column="sector", metrics=["p_e"]
            )

            if "sector_tests" in hyp_results:
                story.append(Paragraph("<b>Sector Comparison Results</b>", styles["Heading3"]))

                pe_test = hyp_results["sector_tests"].get("p_e", {})
                if "anova" in pe_test:
                    anova_result = pe_test["anova"]
                    story.append(
                        Paragraph(
                            f"ANOVA F-statistic: {anova_result.get('statistic', 0):.2f}",
                            styles["Normal"],
                        )
                    )
                    story.append(
                        Paragraph(
                            f"P-value: {anova_result.get('p_value', 1):.4f}", styles["Normal"]
                        )
                    )

                    if anova_result.get("significant"):
                        story.append(
                            Paragraph(
                                "<b>Result:</b> Sectors show significantly different P/E ratios",
                                styles["Normal"],
                            )
                        )

        story.append(PageBreak())
        page_count += 1

    # 7. Charts (if requested)
    if include_charts:
        sections["charts"] = []
        story.append(Paragraph("Visualizations", heading_style))
        story.append(Spacer(1, 0.2 * inch))

        # Note: In a full implementation, charts would be generated and embedded
        story.append(
            Paragraph(
                "Chart visualizations would be embedded here in production version.",
                styles["Normal"],
            )
        )
        sections["charts"].append("placeholder")

        story.append(PageBreak())
        page_count += 1

    # Build PDF
    try:
        doc.build(story)

        result = {
            "status": "success",
            "pdf_path": str(pdf_path),
            "page_count": page_count,
            "sections": sections,
            "template": template,
            "timestamp": timestamp,
        }

        # Expose table_of_contents at root level if included
        if include_toc and "table_of_contents" in sections:
            result["table_of_contents"] = sections["table_of_contents"]

        return result
    except Exception as e:
        return {"status": "error", "error": f"PDF generation failed: {str(e)}"}


def create_structured_output_directory(base_dir: Path, run_id: str = None) -> dict:
    """
    Create organized output directory structure for ML workflow artifacts.

    Structure:
    outputs/
    ├── {run_id}/
    │   ├── data/
    │   │   ├── processed_data.csv
    │   │   └── imputation_report.json
    │   ├── regression/
    │   │   ├── checkpoints/
    │   │   └── feature_importance.csv
    │   ├── reporting/
    │   │   ├── ml_workflow_report.html
    │   │   ├── eda_report.html
    │   │   └── data_quality_dashboard.html
    │   ├── visualizations/
    │   │   ├── eda/
    │   │   ├── predictions/
    │   │   ├── residuals/
    │   │   └── feature_importance/
    │   ├── analytics/
    │   │   ├── predictions.csv
    │   │   ├── stock_rankings.csv
    │   │   └── prediction_analyst_comparison.xlsx
    │   └── logs/
    │       └── pipeline.log

    Args:
        base_dir: Base directory for outputs (default: 'outputs')
        run_id: Unique identifier for this run (default: timestamp)

    Returns:
        Dict with paths to each subdirectory
    """
    if run_id is None:
        run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(base_dir) / run_id

    structure = {
        "run_dir": run_dir,
        "data": run_dir / "data",
        "regression": run_dir / "regression",
        "model_checkpoints": run_dir / "regression" / "checkpoints",
        "reporting": run_dir / "reporting",
        "visualizations": run_dir / "visualizations",
        "eda_viz": run_dir / "visualizations" / "eda",
        "prediction_viz": run_dir / "visualizations" / "predictions",
        "residual_viz": run_dir / "visualizations" / "residuals",
        "feature_viz": run_dir / "visualizations" / "feature_importance",
        "analytics": run_dir / "analytics",
        "logs": run_dir / "logs",
    }

    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    # Create README
    readme_path = run_dir / "README.md"
    readme_content = f"""# ML Workflow Run: {run_id}

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure

- **data/**: Processed datasets and imputation reporting
- **regression/**: Trained model artifacts and checkpoints
- **reporting/**: HTML/PDF reporting (workflow, EDA, data quality)
- **visualizations/**: All plots and charts organized by category
- **analytics/**: Prediction results, rankings, and comparison tables
- **logs/**: Pipeline execution logs

## Key Files

- `reporting/ml_workflow_report.html` - Comprehensive ML workflow report
- `analytics/predictions.csv` - Stock predictions with confidence intervals
- `analytics/stock_rankings.csv` - Undervalued/overvalued stock rankings
- `visualizations/predictions/mispricing_scatter.png` - Mispricing analysis
"""
    readme_path.write_text(readme_content)

    logging.info(f"Created structured output directory: {run_dir}")
    return structure


def generate_imputation_report(
    imputation_stats: dict, df_before: pd.DataFrame, df_after: pd.DataFrame, output_dir: Path
) -> dict:
    """
    Generate comprehensive imputation analysis report with visualizations.

    Tracks:
    - Which columns were imputed and by which method
    - Before/after NaN heatmaps
    - Imputation time per column
    - Distribution changes (before/after histograms)
    - Emergency fallback usage

    Args:
        imputation_stats: Dict with imputation metadata
        df_before: DataFrame before imputation
        df_after: DataFrame after imputation
        output_dir: Directory to save visualizations

    Returns:
        Dict with report metrics and saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "columns_imputed": [],
        "methods_used": {},
        "total_nans_filled": 0,
        "emergency_fallbacks": [],
        "visualizations": [],
    }

    # Track imputation by column
    for col in df_before.columns:
        nans_before = df_before[col].isna().sum()
        nans_after = df_after[col].isna().sum()

        if nans_before > nans_after:
            method = imputation_stats.get(col, {}).get("method", "unknown")
            report["columns_imputed"].append(
                {
                    "column": col,
                    "nans_filled": int(nans_before - nans_after),
                    "method": method,
                    "fill_rate": (
                        float((nans_before - nans_after) / nans_before) if nans_before > 0 else 0.0
                    ),
                }
            )

            report["methods_used"][method] = report["methods_used"].get(method, 0) + 1
            report["total_nans_filled"] += int(nans_before - nans_after)

            # Track emergency fallbacks
            if method == "emergency_zero":
                report["emergency_fallbacks"].append(col)

    # Generate before/after NaN heatmaps
    if plt is not None:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Before heatmap
            if sns is not None:
                sns.heatmap(df_before.isnull(), cbar=False, yticklabels=False, ax=ax1)
            else:
                ax1.imshow(df_before.isnull(), aspect="auto", cmap="YlOrRd")
            ax1.set_title("Missing Values BEFORE Imputation")

            # After heatmap
            if sns is not None:
                sns.heatmap(df_after.isnull(), cbar=False, yticklabels=False, ax=ax2)
            else:
                ax2.imshow(df_after.isnull(), aspect="auto", cmap="YlOrRd")
            ax2.set_title("Missing Values AFTER Imputation")

            heatmap_path = output_dir / "imputation_heatmap.png"
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close()

            report["visualizations"].append(str(heatmap_path))
            logging.info(f"Saved imputation heatmap: {heatmap_path}")
        except Exception as e:
            logging.warning(f"Failed to generate heatmap: {e}")

    # Generate distribution comparison for imputed columns
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    imputed_numeric = [
        c["column"] for c in report["columns_imputed"] if c["column"] in numeric_cols
    ]

    if plt is not None and imputed_numeric:
        try:
            n_cols = min(len(imputed_numeric), 6)  # Show top 6
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, col in enumerate(imputed_numeric[:n_cols]):
                # Before distribution (excluding NaNs)
                before_clean = df_before[col].dropna()
                axes[idx].hist(before_clean, bins=30, alpha=0.5, label="Before", color="red")

                # After distribution
                axes[idx].hist(
                    df_after[col].dropna(), bins=30, alpha=0.5, label="After", color="green"
                )

                # Find the column info for this column
                col_info = next((c for c in report["columns_imputed"] if c["column"] == col), None)
                nans_filled = col_info["nans_filled"] if col_info else 0

                axes[idx].set_title(f"{col}\n({nans_filled} filled)")
                axes[idx].legend()

            # Hide unused subplots
            for idx in range(n_cols, len(axes)):
                axes[idx].axis("off")

            dist_path = output_dir / "imputation_distributions.png"
            plt.tight_layout()
            plt.savefig(dist_path, dpi=150, bbox_inches="tight")
            plt.close()

            report["visualizations"].append(str(dist_path))
            logging.info(f"Saved imputation distributions: {dist_path}")
        except Exception as e:
            logging.warning(f"Failed to generate distribution plots: {e}")

    # Save JSON report
    json_path = output_dir / "imputation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logging.info(f"Generated imputation report: {json_path}")
    return report
