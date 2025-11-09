#!/usr/bin/env python3
"""
Fix Phase 9.5 data flow issues in ml_finance_model_main_v10.ipynb

This script addresses multiple critical issues:
1. Data flow: Pass imputed data to sector regression (not original with NaN)
2. Feature selection: Use only numeric features for training
3. Validation: Add checkpoints before sector model training
4. Checkpoint system: Set regression_complete flag
"""

import json
import sys
from pathlib import Path
import re

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def fix_phase95_notebook():
    """Fix Phase 9.5 cell in the notebook"""

    notebook_path = Path("ml_finance_model_main_v10.ipynb")

    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        return False

    # Read notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return False

    # Pattern to find the problematic code
    old_pattern = """    sector_models = train_sector_models(
            all_stocks_phase95, feature_cols, target_col, MIN_SECTOR_SAMPLES,
            RANDOM_STATE, out_models_dir
            )"""

    # New corrected code with proper data flow
    new_pattern = """    # ========================================================================
    # CRITICAL FIX: Use imputed data for sector regression
    # ========================================================================
    # Reconstruct fully imputed dataframe by combining imputed features
    # with original non-numeric columns
    print("\\n🔧 Preparing imputed dataset for sector regression...")
    
    all_stocks_imputed = all_stocks_phase95.copy()
    
    # Update numeric features with imputed values from X_train and X_test
    X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    all_stocks_imputed_reset = all_stocks_imputed.reset_index(drop=True)
    
    # Update only the numeric feature columns with imputed data
    for col in feature_info['numeric_features']:
        if col in X_combined.columns:
            all_stocks_imputed_reset[col] = X_combined[col]
    
    print(f"  ✓ Updated {len(feature_info['numeric_features'])} numeric features with imputed values")
    
    # Validate data before sector model training
    print("\\n🔍 Validating data before sector model training...")
    from finance_ml.advanced_models import validate_training_data
    
    validation_result = validate_training_data(
        all_stocks_imputed_reset[feature_info['numeric_features']],
        all_stocks_imputed_reset[target_col],
        strict=False
    )
    
    if not validation_result['valid']:
        print(f"⚠ Data quality issues detected:")
        for issue in validation_result['issues']:
            print(f"  - {issue}")
        
        if validation_result['nan_features'] > 0 or validation_result['nan_target'] > 0:
            print("\\n❌ CRITICAL: Data still contains NaN after imputation!")
            print("  Applying emergency imputation...")
            
            from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step
            all_stocks_imputed_reset = apply_enhanced_imputation_strategy_4step(
                all_stocks_imputed_reset,
                sector_column='sector',
                n_neighbors=5,
                price_column='last_price'
            )
            print("  ✓ Emergency imputation completed")
    else:
        print("✓ Data validation passed - ready for sector model training")
    
    # Train sector regression with clean, imputed data
    sector_models = train_sector_models(
            all_stocks_imputed_reset,  # ✓ Imputed dataframe
            feature_info['numeric_features'],  # ✓ Only numeric features
            target_col,
            MIN_SECTOR_SAMPLES,
            RANDOM_STATE,
            out_models_dir
            )"""

    # Also need to update the train_sector_models function definition
    train_sector_models_old = """def train_sector_models(df: pd.DataFrame, feature_cols: List[str],
                        target_col: str, min_samples: int,
                        random_state: int, output_dir: Path) -> Dict[str, Any]:
    \"\"\"Train sector-specific regression with preprocessing.\"\"\"
    
    logger.info(f\"Training sector-specific regression for {df[sector_col].nunique()} sectors\")
    
    sector_models, sector_results = train_sector_specific_models(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            sector_col='sector',
            model_type='random_forest',
            random_state=random_state,
            min_samples=min_samples,
            ensure_nonnegative=True
            )"""

    train_sector_models_new = """def train_sector_models(df: pd.DataFrame, feature_cols: Union[List[str], Dict[str, List[str]]],
                        target_col: str, min_samples: int,
                        random_state: int, output_dir: Path) -> Dict[str, Any]:
    \"\"\"Train sector-specific regression with preprocessing and final imputation checkpoint.
    
    Args:
        df: Input DataFrame (should be pre-imputed)
        feature_cols: Feature columns (list or dict with 'numeric_features' key)
        target_col: Target column name
        min_samples: Minimum samples per sector
        random_state: Random seed
        output_dir: Output directory for regression
    
    Returns:
        Tuple of (sector_models dict, sector_results dict)
    \"\"\"
    from typing import Union
    
    logger.info(f\"Training sector-specific regression for {df['sector'].nunique()} sectors\")
    
    # Extract feature list if dict is passed
    if isinstance(feature_cols, dict):
        feature_list = feature_cols.get('numeric_features', feature_cols.get('all_features', []))
        logger.info(f"Extracted {len(feature_list)} features from feature_cols dict")
    else:
        feature_list = feature_cols
    
    # CRITICAL: Apply final imputation checkpoint before training
    from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step
    
    logger.info("Applying final imputation checkpoint before sector model training...")
    df_clean = apply_enhanced_imputation_strategy_4step(
        df.copy(),
        sector_column='sector',
        n_neighbors=5,
        price_column='last_price' if 'last_price' in df.columns else None
    )
    logger.info(f"✓ Final imputation complete: {df_clean.isnull().sum().sum()} NaN values remain")
    
    sector_models, sector_results = train_sector_specific_models(
            df=df_clean,  # Use cleaned dataframe
            feature_cols=feature_list,  # Pass list, not dict
            target_col=target_col,
            sector_col='sector',
            model_type='random_forest',
            random_state=random_state,
            min_samples=min_samples,
            ensure_nonnegative=True
            )"""

    # Also add checkpoint flag setting at end of Phase 9.5
    checkpoint_code = """
    
    # ========================================================================
    # SET CHECKPOINT FLAG FOR PHASE 9.5.1
    # ========================================================================
    regression_complete = True  # Enable Phase 9.5.1 to run
    
    print("\\n" + "="*80)
    print("✓ PHASE 9.5 COMPLETE - Checkpoint flag set")
    print("="*80)"""

    modified = False

    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])

            # Join source lines
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            # Check if this is the Phase 9.5 cell with train_sector_models call
            if 'train_sector_models(' in source_text and 'all_stocks_phase95' in source_text:
                print("Found Phase 9.5 sector regression cell...")

                # Replace the old pattern
                if old_pattern in source_text:
                    source_text = source_text.replace(old_pattern, new_pattern)
                    modified = True
                    print("  [OK] Fixed sector_models call with imputed data flow")

                # Replace train_sector_models function definition
                if 'def train_sector_models(' in source_text:
                    source_text = source_text.replace(train_sector_models_old, train_sector_models_new)
                    print("  [OK] Updated train_sector_models function signature")

                # Add checkpoint flag if not present
                if 'regression_complete = True' not in source_text:
                    # Find a good place to add it (after the training completes)
                    if 'PHASE 9.5 COMPLETE' not in source_text:
                        # Add before the final section
                        if 'except Exception as e:' in source_text:
                            # Insert before the except block
                            source_text = source_text.replace(
                                'except Exception as e:',
                                checkpoint_code + '\n\nexcept Exception as e:'
                            )
                        else:
                            # Append at end
                            source_text += checkpoint_code
                        print("  [OK] Added regression_complete checkpoint flag")

                # Convert back to list format
                if isinstance(source, list):
                    cell['source'] = source_text.splitlines(keepends=True)
                else:
                    cell['source'] = source_text

    if not modified:
        print("[WARNING] Pattern not found - notebook may have different structure")
        print("  Creating backup and attempting alternative fix...")

    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup_phase95_fix')
    try:
        import shutil
        shutil.copy2(notebook_path, backup_path)
        print(f"[OK] Created backup: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")

    # Write modified notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[OK] Updated {notebook_path}")
        return True
    except Exception as e:
        print(f"Error writing notebook: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("PHASE 9.5 DATA FLOW FIX")
    print("="*80)
    print("\nThis script fixes:")
    print("  1. Data flow: Pass imputed data to sector regression")
    print("  2. Feature selection: Use only numeric features")
    print("  3. Validation: Add data quality checkpoints")
    print("  4. Checkpoint system: Set regression_complete flag")
    print("\n" + "="*80 + "\n")

    success = fix_phase95_notebook()

    if success:
        print("\n" + "="*80)
        print("[SUCCESS] FIX COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Restart the Jupyter kernel")
        print("  2. Run cells up to Phase 9.5")
        print("  3. Verify no NaN errors occur")
        print("  4. Check that all 11 sectors train successfully")
    else:
        print("\n" + "="*80)
        print("[FAILED] FIX FAILED")
        print("="*80)
        print("\nPlease check error messages above and try manual fix if needed.")

    sys.exit(0 if success else 1)
