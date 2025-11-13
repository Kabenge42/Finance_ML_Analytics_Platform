### Enhancement Suggestions for Imputation Functions in `imputation.py`

Based on the analysis of `preprocessed_stocks_metadata.json` and the warning about **733 NaN values remaining in 11
categorical/string/date columns**, here are comprehensive enhancements to handle all data types properly before Phase
9.3's `engineer_temporal_features()` function.

---

### Problem Analysis

The current 6-step imputation strategy in `imputation.py` (lines 578-660) only processes **numeric columns**:

- Step 1: Zero imputation (exceptional events)
- Step 2: KNN imputation (financial metrics)
- Step 3: Price imputation (price targets)
- Step 4: Median imputation (remaining numerics)

**Affected columns with NaN values:**

1. **Identifiers/Text**: `ticker`, `isin`, `description`
2. **Date columns**: `last_updated`, `income_statement_report_date`, `next_earnings`
3. **Categorical**: `style_class`, `next_earnings_status`, `size_class`, `flag`, `country`

The `engineer_temporal_features()` function (lines 502-543 in `advanced.py`) requires proper datetime formatting and
will fail on NaN values in date columns.

---

### Enhancement 1: Add Categorical/String Imputation Functions

Add these new functions to `imputation.py` after line 107:

```python
def get_categorical_imputation_config() -> dict:
    """Return configuration for categorical column imputation strategies.
    
    Returns:
        Dictionary mapping column names to imputation strategies:
        - 'most_frequent': Use mode (most common value)
        - 'constant': Use a specific constant value
        - 'forward_fill': Use forward fill (for ordered data)
    
    Examples:
        >>> config = get_categorical_imputation_config()
        >>> config['style_class']  # Returns 'most_frequent'
    """
    return {
        # Classification categories - use most frequent (mode)
        'style_class': 'most_frequent',
        'size_class': 'most_frequent',
        'next_earnings_status': 'most_frequent',
        'sector': 'most_frequent',
        'industry': 'most_frequent',
        'region': 'most_frequent',
        'country': 'most_frequent',
        'trading_country': 'most_frequent',

        # Flags - use constant 'Unknown' or most frequent
        'flag': ('constant', 'Unknown'),

        # Identifiers - use constant 'MISSING'
        'ticker': ('constant', 'N/A'),
        'isin': ('constant', 'N/A'),

        # Text descriptions - use constant
        'description': ('constant', 'No description available'),
        'name': ('constant', 'Unknown'),
        }


def apply_categorical_imputation(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: str = 'most_frequent',
        fill_value: Optional[str] = None
        ) -> pd.DataFrame:
    """Apply imputation to categorical/string columns (Step 5 of enhanced strategy).
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to impute. If None, auto-detects object dtype columns
        strategy: Imputation strategy - 'most_frequent', 'constant', or 'drop'
        fill_value: Value to use for 'constant' strategy
        
    Returns:
        DataFrame with imputed categorical columns
        
    Examples:
        >>> # Impute categorical columns with mode
        >>> df = apply_categorical_imputation(df, strategy='most_frequent')
        >>> 
        >>> # Impute specific columns with constant
        >>> df = apply_categorical_imputation(
        ...     df, 
        ...     columns=['flag', 'ticker'], 
        ...     strategy='constant',
        ...     fill_value='Unknown'
        ... )
    """
    from sklearn.impute import SimpleImputer

    result = df.copy()

    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = result.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude date columns that should be handled separately
        date_keywords = ['date', 'updated', 'earnings']
        columns = [col for col in columns
                   if not any(keyword in col.lower() for keyword in date_keywords)]

    if len(columns) == 0:
        logger.info("No categorical columns found for imputation")
        return result

    # Filter to columns that exist and have missing values
    columns_to_impute = [col for col in columns
                         if col in result.columns and result[col].isna().any()]

    if len(columns_to_impute) == 0:
        logger.info("No missing values found in specified categorical columns")
        return result

    total_imputed = 0

    for col in columns_to_impute:
        n_missing = result[col].isna().sum()

        if strategy == 'most_frequent':
            # Use mode (most common value)
            mode_value = result[col].mode()
            if len(mode_value) > 0:
                result[col] = result[col].fillna(mode_value[0])
                logger.debug(f"Imputed {n_missing} values in '{col}' with mode: {mode_value[0]}")
            else:
                # Fallback to constant if no mode exists
                result[col] = result[col].fillna('Unknown')
                logger.debug(f"Imputed {n_missing} values in '{col}' with 'Unknown' (no mode found)")

        elif strategy == 'constant':
            fill = fill_value if fill_value is not None else 'Unknown'
            result[col] = result[col].fillna(fill)
            logger.debug(f"Imputed {n_missing} values in '{col}' with constant: {fill}")

        elif strategy == 'drop':
            # Drop rows with missing values (use cautiously)
            result = result.dropna(subset=[col])
            logger.debug(f"Dropped {n_missing} rows with missing values in '{col}'")

        total_imputed += n_missing

    logger.info(
        f"Applied {strategy} imputation to {total_imputed} categorical missing values across {len(columns_to_impute)} columns")
    return result
```

---

### Enhancement 2: Add Date/Datetime Imputation and Formatting

Add after the categorical imputation function:

```python
def apply_datetime_imputation_and_formatting(
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None,
        strategy: str = 'forward_fill',
        reference_date: Optional[pd.Timestamp] = None
        ) -> pd.DataFrame:
    """Apply imputation and formatting to datetime columns (Step 6 of enhanced strategy).
    
    This ensures proper datetime formatting for engineer_temporal_features() in Phase 9.3.
    
    Args:
        df: Input DataFrame
        date_columns: List of date columns. If None, auto-detects from common patterns
        strategy: Imputation strategy - 'forward_fill', 'backward_fill', 'median', 'constant'
        reference_date: Optional reference date for 'constant' strategy
        
    Returns:
        DataFrame with properly formatted datetime columns and no NaN values
        
    Examples:
        >>> # Format and impute date columns
        >>> df = apply_datetime_imputation_and_formatting(
        ...     df,
        ...     date_columns=['last_updated', 'income_statement_report_date', 'next_earnings'],
        ...     strategy='forward_fill'
        ... )
    """
    result = df.copy()

    # Auto-detect date columns if not specified
    if date_columns is None:
        date_columns = []
        # Common date column patterns
        date_patterns = ['date', 'updated', 'earnings', 'report']
        for col in result.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                date_columns.append(col)

    if len(date_columns) == 0:
        logger.info("No date columns found for imputation")
        return result

    # Filter to columns that exist
    date_columns = [col for col in date_columns if col in result.columns]

    total_imputed = 0

    for col in date_columns:
        n_missing_initial = result[col].isna().sum()

        # Step 1: Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(result[col]):
            try:
                # Try parsing with various formats
                result[col] = pd.to_datetime(result[col], errors='coerce')
                logger.debug(f"Converted '{col}' to datetime format")
            except Exception as e:
                logger.warning(f"Could not convert '{col}' to datetime: {e}")
                continue

        # Step 2: Impute missing datetime values
        if result[col].isna().any():
            n_missing = result[col].isna().sum()

            if strategy == 'forward_fill':
                result[col] = result[col].fillna(method='ffill')
                # If still missing (at start), use backward fill
                if result[col].isna().any():
                    result[col] = result[col].fillna(method='bfill')
                logger.debug(f"Forward-filled {n_missing} missing dates in '{col}'")

            elif strategy == 'backward_fill':
                result[col] = result[col].fillna(method='bfill')
                # If still missing (at end), use forward fill
                if result[col].isna().any():
                    result[col] = result[col].fillna(method='ffill')
                logger.debug(f"Backward-filled {n_missing} missing dates in '{col}'")

            elif strategy == 'median':
                # Use median timestamp
                median_ts = result[col].dropna().median()
                result[col] = result[col].fillna(median_ts)
                logger.debug(f"Median-imputed {n_missing} missing dates in '{col}' with {median_ts}")

            elif strategy == 'constant':
                # Use reference date or current date
                fill_date = reference_date if reference_date else pd.Timestamp.now()
                result[col] = result[col].fillna(fill_date)
                logger.debug(f"Constant-imputed {n_missing} missing dates in '{col}' with {fill_date}")

            # If still missing after imputation, use current date as last resort
            if result[col].isna().any():
                n_still_missing = result[col].isna().sum()
                result[col] = result[col].fillna(pd.Timestamp.now())
                logger.warning(f"Used current date to fill {n_still_missing} remaining missing dates in '{col}'")

            total_imputed += n_missing

    logger.info(
        f"Applied datetime imputation and formatting to {total_imputed} missing values across {len(date_columns)} date columns")
    return result
```

---

### Enhancement 3: Update Main 6-step Function to 6-Step Strategy

**Replace** the existing `apply_enhanced_imputation_strategy_4step()` function (lines 578-660) with this enhanced
version:

```python
def apply_enhanced_imputation_strategy_6step(
        df: pd.DataFrame,
        sector_column: str = "sector",
        n_neighbors: int = 5,
        price_column: str = "last_price",
        handle_categoricals: bool = True,
        handle_dates: bool = True,
        categorical_strategy: str = 'most_frequent',
        date_strategy: str = 'forward_fill',
        ) -> pd.DataFrame:
    """Apply complete 6-step imputation strategy from Phase 9.1 (ENHANCED).
    
    Step 1: Zero imputation for exceptional event columns (48 columns)
    Step 2: Sector-aware KNN imputation for core financial metrics (148 columns)
    Step 3: Price imputation for price target columns (5 columns)
    Step 4: Median imputation for all remaining numerical columns
    Step 5: Categorical imputation for string/object columns (NEW)
    Step 6: Datetime imputation and formatting for date columns (NEW)
    
    This ensures ZERO missing values in the output dataframe across ALL data types.
    
    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation
        handle_categoricals: Whether to apply categorical imputation (Step 5)
        handle_dates: Whether to apply datetime imputation (Step 6)
        categorical_strategy: Strategy for categorical imputation ('most_frequent' or 'constant')
        date_strategy: Strategy for date imputation ('forward_fill', 'median', 'constant')
        
    Returns:
        DataFrame with complete 6-step imputation applied (zero missing values)
        
    Examples:
        >>> # Apply complete 6-step imputation pipeline
        >>> df_complete = apply_enhanced_imputation_strategy_6step(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5,
        ...     price_column='last_price',
        ...     handle_categoricals=True,
        ...     handle_dates=True
        ... )
        >>> # Verify no missing values remain in ANY column
        >>> assert df_complete.isna().sum().sum() == 0
    """
    logger.info("Starting Phase 9.1 enhanced 6-step imputation strategy")

    # Track missing values at each step
    missing_initial = df.isna().sum().sum()
    missing_numeric_initial = df.select_dtypes(include=[np.number]).isna().sum().sum()
    missing_categorical_initial = df.select_dtypes(include=['object', 'category']).isna().sum().sum()

    logger.info(
        f"Initial missing values: {missing_initial} total ({missing_numeric_initial} numeric, {missing_categorical_initial} categorical)")

    # Step 1: Zero imputation for exceptional events
    logger.info("Step 1: Applying zero imputation for exceptional event columns (48 cols)")
    result = apply_zero_imputation(df)
    missing_after_step1 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 1: {missing_after_step1} numeric missing values remain")

    # Step 2: KNN imputation for core financial metrics
    logger.info("Step 2: Applying sector-aware KNN imputation for financial metrics (148 cols)")
    result = apply_knn_imputation_enhanced(
            result,
            sector_column=sector_column,
            n_neighbors=n_neighbors,
            )
    missing_after_step2 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 2: {missing_after_step2} numeric missing values remain")

    # Step 3: Price imputation for price targets
    logger.info("Step 3: Applying price imputation for price target columns (5 cols)")
    result = apply_price_imputation(result, price_column=price_column)
    missing_after_step3 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 3: {missing_after_step3} numeric missing values remain")

    # Step 4: Median imputation for remaining numerical columns
    logger.info("Step 4: Applying median imputation for remaining numerical columns")
    result = apply_median_imputation(result)
    missing_after_step4 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 4: {missing_after_step4} numeric missing values remain")

    # Step 5: Categorical imputation (NEW)
    if handle_categoricals:
        logger.info(f"Step 5: Applying {categorical_strategy} imputation for categorical columns")

        # Get configuration for specific columns
        cat_config = get_categorical_imputation_config()

        # Group columns by strategy
        most_frequent_cols = [col for col, strat in cat_config.items()
                              if strat == 'most_frequent' and col in result.columns]
        constant_cols = [(col, val) for col, strat in cat_config.items()
                         if isinstance(strat, tuple) and strat[0] == 'constant' and col in result.columns]

        # Apply most_frequent strategy
        if most_frequent_cols:
            result = apply_categorical_imputation(
                    result,
                    columns=most_frequent_cols,
                    strategy='most_frequent'
                    )

        # Apply constant strategy for specific columns
        for col, fill_value in constant_cols:
            result = apply_categorical_imputation(
                    result,
                    columns=[col],
                    strategy='constant',
                    fill_value=fill_value
                    )

        # Catch any remaining categorical columns
        remaining_cat_cols = result.select_dtypes(include=['object', 'category']).columns
        remaining_with_na = [col for col in remaining_cat_cols if result[col].isna().any()]
        if remaining_with_na:
            result = apply_categorical_imputation(
                    result,
                    columns=remaining_with_na,
                    strategy=categorical_strategy
                    )

        missing_after_step5 = result.select_dtypes(include=['object', 'category']).isna().sum().sum()
        logger.info(f"After Step 5: {missing_after_step5} categorical missing values remain")
    else:
        missing_after_step5 = result.select_dtypes(include=['object', 'category']).isna().sum().sum()

    # Step 6: Datetime imputation and formatting (NEW)
    if handle_dates:
        logger.info(f"Step 6: Applying {date_strategy} imputation and formatting for datetime columns")

        # Specify critical date columns for temporal features
        critical_date_cols = ['last_updated', 'income_statement_report_date', 'next_earnings']

        result = apply_datetime_imputation_and_formatting(
                result,
                date_columns=critical_date_cols,
                strategy=date_strategy
                )

        missing_after_step6 = result.isna().sum().sum()
        logger.info(f"After Step 6: {missing_after_step6} total missing values remain")
    else:
        missing_after_step6 = result.isna().sum().sum()

    # Final verification and reporting
    missing_final = result.isna().sum().sum()
    total_reduction = missing_initial - missing_final

    logger.info(
            f"6-step imputation complete: Reduced missing values from {missing_initial} "
            f"to {missing_final} (reduction: {total_reduction})"
            )

    if missing_final > 0:
        # Identify which columns still have missing values
        cols_with_missing = result.columns[result.isna().any()].tolist()
        logger.warning(
                f"WARNING: {missing_final} NaN values still present\n"
                f"  Affected columns ({len(cols_with_missing)}): {cols_with_missing[:11]}..."
                )
        logger.warning("  Applying final median imputation...")

        # Emergency fallback: fill any remaining with appropriate defaults
        for col in cols_with_missing:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
            else:
                result[col] = result[col].fillna('Unknown')

        missing_final = result.isna().sum().sum()
        logger.info(f"After emergency fallback: {missing_final} missing values remain")

    return result


# Maintain backward compatibility - alias to new function
def apply_enhanced_imputation_strategy_4step(
        df: pd.DataFrame,
        sector_column: str = "sector",
        n_neighbors: int = 5,
        price_column: str = "last_price",
        ) -> pd.DataFrame:
    """Backward compatibility wrapper for 6-step imputation.
    
    DEPRECATED: Use apply_enhanced_imputation_strategy_6step() instead.
    This wrapper calls the 6-step function with categorical and date handling enabled.
    """
    logger.warning(
            "apply_enhanced_imputation_strategy_4step is deprecated. "
            "Use apply_enhanced_imputation_strategy_6step for full imputation coverage."
            )
    return apply_enhanced_imputation_strategy_6step(
            df,
            sector_column=sector_column,
            n_neighbors=n_neighbors,
            price_column=price_column,
            handle_categoricals=True,
            handle_dates=True,
            )
```

---

### Enhancement 4: Add Validation Function

Add a new validation function at the end of the file:

```python
def validate_imputation_completeness(
        df: pd.DataFrame,
        critical_date_columns: Optional[List[str]] = None
        ) -> dict:
    """Validate that imputation is complete and datetime columns are properly formatted.
    
    Args:
        df: DataFrame to validate
        critical_date_columns: List of date columns required for temporal features
        
    Returns:
        Dictionary with validation results including:
        - 'is_complete': bool - True if no missing values
        - 'missing_count': int - Total missing values
        - 'missing_by_type': dict - Missing values by data type
        - 'datetime_formatted': dict - Status of datetime columns
        - 'ready_for_temporal_features': bool - Ready for engineer_temporal_features()
    """
    if critical_date_columns is None:
        critical_date_columns = ['last_updated', 'income_statement_report_date', 'next_earnings']

    missing_total = df.isna().sum().sum()
    missing_numeric = df.select_dtypes(include=[np.number]).isna().sum().sum()
    missing_categorical = df.select_dtypes(include=['object', 'category']).isna().sum().sum()

    # Check datetime formatting
    datetime_status = {}
    for col in critical_date_columns:
        if col in df.columns:
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
            has_missing = df[col].isna().any()
            datetime_status[col] = {
                'is_datetime': is_datetime,
                'has_missing': has_missing,
                'ready': is_datetime and not has_missing
                }
        else:
            datetime_status[col] = {'is_datetime': False, 'has_missing': True, 'ready': False}

    ready_for_temporal = all(status['ready'] for status in datetime_status.values())

    result = {
        'is_complete': missing_total == 0,
        'missing_count': missing_total,
        'missing_by_type': {
            'numeric': missing_numeric,
            'categorical': missing_categorical,
            'other': missing_total - missing_numeric - missing_categorical
            },
        'datetime_formatted': datetime_status,
        'ready_for_temporal_features': ready_for_temporal
        }

    # Log results
    if result['is_complete'] and ready_for_temporal:
        logger.info("✓ Imputation validation PASSED - Ready for Phase 9.3 feature engineering")
    else:
        logger.warning(f"✗ Imputation validation FAILED - {missing_total} missing values remain")
        if not ready_for_temporal:
            logger.warning("✗ Date columns not ready for temporal feature engineering")

    return result
```

---

### Usage Example for Notebook

Update your notebook Phase 9.1 section to use the enhanced 6-step strategy:

```python
# Phase 9.1: Enhanced Imputation with Categorical and Date Handling
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    validate_imputation_completeness
    )

# Apply comprehensive 6-step imputation
all_stocks_imputed = apply_enhanced_imputation_strategy_6step(
        all_stocks,
        sector_column='sector',
        n_neighbors=5,
        price_column='last_price',
        handle_categoricals=True,
        handle_dates=True,
        categorical_strategy='most_frequent',
        date_strategy='forward_fill'
        )

# Validate imputation completeness
validation_results = validate_imputation_completeness(
        all_stocks_imputed,
        critical_date_columns=['last_updated', 'income_statement_report_date', 'next_earnings']
        )

print(f"Imputation Complete: {validation_results['is_complete']}")
print(f"Total Missing: {validation_results['missing_count']}")
print(f"Ready for Temporal Features: {validation_results['ready_for_temporal_features']}")

# Display datetime column status
import pandas as pd

datetime_df = pd.DataFrame(validation_results['datetime_formatted']).T
print("\nDatetime Column Status:")
print(datetime_df)
```

---

### Summary of Enhancements

#### ✅ **What's New:**

1. **`get_categorical_imputation_config()`** - Centralized configuration for categorical column strategies
2. **`apply_categorical_imputation()`** - Handles string/object/category columns with most_frequent or constant
   strategies
3. **`apply_datetime_imputation_and_formatting()`** - Ensures date columns are properly formatted as datetime and
   NaN-free
4. **`apply_enhanced_imputation_strategy_6step()`** - Comprehensive pipeline handling ALL data types (numeric +
   categorical + dates)
5. **`validate_imputation_completeness()`** - Validation function to ensure readiness for Phase 9.3
6. **Backward compatibility** - Old 6-step function maintained as wrapper

#### ✅ **Benefits:**

- **Eliminates all 733 remaining NaN values** in categorical/date columns
- **Prevents errors** in `engineer_temporal_features()` by ensuring proper datetime formatting
- **Configurable strategies** per column type (most_frequent, constant, forward_fill, etc.)
- **Full validation** before proceeding to feature engineering
- **Maintains existing API** while adding enhanced functionality

#### ⚠️ **Important Notes:**

1. For **identifier columns** (ticker, isin) - consider if imputation is appropriate or if rows should be filtered
2. For **critical date columns** - forward_fill is safest; constant date may introduce bias
3. Test the enhanced functions with your actual data to ensure the imputation strategies align with your domain
   requirements
4. Update unit tests to cover the new categorical and datetime imputation functions

This comprehensive enhancement ensures your data pipeline is robust and ready for Phase 9.3 feature engineering without
any NaN-related errors.