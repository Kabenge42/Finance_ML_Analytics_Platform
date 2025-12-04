"""Update notebook with Phase 9.5 and 9.6 enhancements."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]

# ============================================================================
# 1. Update Cell 4 - Add comprehensive regression imports
# ============================================================================
print("Updating Cell 4 with comprehensive regression imports...")

# Find the regression imports section in cell 4 and enhance it
cell4_source = "".join(cells[4]["source"])

# New comprehensive regression imports to add
new_regression_imports = """# Phase 9.5: Regression models - Comprehensive imports
from finance_ml import (
    regression_prepare_data,
    regression_compare_regressors,
    regression_save_model,
    regression_load_model,
    regression_create_classification_interactions,
    regression_train_stacking,
    regression_train_quantile,
    winsorize_target,
)

# Phase 9.5: Individual regression model training functions
from finance_ml.ml_workflow.regression.models import (
    # Linear Models
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor,
    train_bayesian_ridge_regressor,
    train_polynomial_regressor,
    # Gradient Boosting Models
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_histgb_regressor,
    # Tree Models
    train_random_forest_regressor,
    train_extra_trees_regressor,
    # Neural Network
    train_neural_network_regressor,
    # Ensemble Methods
    train_voting_regressor,
    train_stacking_regressor,
    # Model Comparison
    compare_regressors,
)

# Phase 9.5: Regression utilities
from finance_ml.ml_workflow.regression import (
    NonNegativeRegressionWrapper,
)
from finance_ml.ml_workflow.regression.dataset import integrate_classification_features
from finance_ml.ml_workflow.regression.robust import (
    adaptive_clip_predictions,
    clip_predictions,
    enforce_non_negative,
)
from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor
from finance_ml.ml_workflow.regression.sector_models import (
    train_sector_optimized_regressors,
)"""

# Replace old regression imports with new comprehensive ones
old_pattern_start = "# Phase 9.5: Regression models"
old_pattern_end = "from finance_ml.ml_workflow.regression.robust import"

if old_pattern_start in cell4_source:
    # Find start and end positions
    start_idx = cell4_source.find(old_pattern_start)
    # Find the end of the robust import block
    end_search = cell4_source.find("adaptive_clip_predictions,", start_idx)
    if end_search != -1:
        # Find the closing paren and newline after it
        end_idx = cell4_source.find(")", end_search) + 1
        # Replace the section
        cell4_source = cell4_source[:start_idx] + new_regression_imports + cell4_source[end_idx:]
        cells[4]["source"] = [cell4_source]
        print("  [OK] Updated regression imports in Cell 4")
    else:
        print("  [!] Could not find end of regression imports section")
else:
    print("  [!] Could not find regression imports section start")

# ============================================================================
# 2. Find where to insert new cells after cell 79 (compare_regressors)
# ============================================================================
print("\nAdding new model comparison cells after Section 6.3...")

# New cell: Linear Models Comparison
linear_models_markdown = """### 6.3.1 Linear Models Comparison
Compare linear regression models for baseline performance and interpretability."""

linear_models_code = """# ============================================================================
# PHASE 9.5: Linear Models Comparison
# ============================================================================
print("=" * 80)
print("PHASE 9.5: LINEAR MODELS COMPARISON")
print("=" * 80)

# Ensure we have the regression-ready data
if 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    from finance_ml.ml_workflow.regression.models import (
        train_ridge_regressor,
        train_lasso_regressor,
        train_elastic_net_regressor,
        train_bayesian_ridge_regressor,
    )
    
    linear_results = {}
    
    # 1. Ridge Regression
    print("\\n📊 Training Ridge Regressor...")
    try:
        ridge_result = train_ridge_regressor(X_train, y_train, alpha=1.0, cv=CV_FOLDS)
        linear_results['Ridge'] = ridge_result
        print(f"   R² Score: {ridge_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ Ridge failed: {e}")
    
    # 2. Lasso Regression
    print("\\n📊 Training Lasso Regressor...")
    try:
        lasso_result = train_lasso_regressor(X_train, y_train, alpha=0.1, cv=CV_FOLDS)
        linear_results['Lasso'] = lasso_result
        print(f"   R² Score: {lasso_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ Lasso failed: {e}")
    
    # 3. ElasticNet Regression
    print("\\n📊 Training ElasticNet Regressor...")
    try:
        elastic_result = train_elastic_net_regressor(X_train, y_train, alpha=0.1, l1_ratio=0.5, cv=CV_FOLDS)
        linear_results['ElasticNet'] = elastic_result
        print(f"   R² Score: {elastic_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ ElasticNet failed: {e}")
    
    # 4. Bayesian Ridge
    print("\\n📊 Training Bayesian Ridge Regressor...")
    try:
        bayesian_result = train_bayesian_ridge_regressor(X_train, y_train)
        linear_results['BayesianRidge'] = bayesian_result
        print(f"   R² Score: {bayesian_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ BayesianRidge failed: {e}")
    
    # Summary
    print("\\n" + "=" * 60)
    print("LINEAR MODELS SUMMARY")
    print("=" * 60)
    for name, result in linear_results.items():
        if result and 'r2_score' in result:
            print(f"  {name:15s}: R² = {result['r2_score']:.4f}")
    
    print("\\n✓ Linear models comparison complete")"""

# New cell: Tree Models Comparison
tree_models_markdown = """### 6.3.2 Tree-Based Models Comparison
Compare ensemble tree models (RandomForest, ExtraTrees) for robust predictions."""

tree_models_code = """# ============================================================================
# PHASE 9.5: Tree-Based Models Comparison
# ============================================================================
print("=" * 80)
print("PHASE 9.5: TREE-BASED MODELS COMPARISON")
print("=" * 80)

if 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    from finance_ml.ml_workflow.regression.models import (
        train_random_forest_regressor,
        train_extra_trees_regressor,
    )
    
    tree_results = {}
    
    # 1. Random Forest
    print("\\n🌲 Training Random Forest Regressor...")
    try:
        rf_result = train_random_forest_regressor(
            X_train, y_train, 
            n_estimators=100, 
            max_depth=10,
            random_state=RANDOM_SEED
        )
        tree_results['RandomForest'] = rf_result
        print(f"   R² Score: {rf_result.get('r2_score', 'N/A'):.4f}")
        print(f"   MAE: {rf_result.get('mae', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ RandomForest failed: {e}")
    
    # 2. Extra Trees
    print("\\n🌲 Training Extra Trees Regressor...")
    try:
        et_result = train_extra_trees_regressor(
            X_train, y_train,
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED
        )
        tree_results['ExtraTrees'] = et_result
        print(f"   R² Score: {et_result.get('r2_score', 'N/A'):.4f}")
        print(f"   MAE: {et_result.get('mae', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ ExtraTrees failed: {e}")
    
    # Feature importance from best tree model
    if tree_results:
        best_tree = max(tree_results.items(), key=lambda x: x[1].get('r2_score', 0))
        print(f"\\n📊 Best Tree Model: {best_tree[0]} (R² = {best_tree[1].get('r2_score', 0):.4f})")
        
        if 'feature_importance' in best_tree[1]:
            print("\\n📊 Top 10 Feature Importances:")
            fi = best_tree[1]['feature_importance']
            if hasattr(fi, 'items'):
                sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                for feat, imp in sorted_fi:
                    print(f"   {feat:30s}: {imp:.4f}")
    
    print("\\n✓ Tree-based models comparison complete")"""

# New cell: Gradient Boosting Models
gb_models_markdown = """### 6.3.3 Gradient Boosting Models Comparison
Compare gradient boosting models (XGBoost, LightGBM, CatBoost, HistGradientBoosting)."""

gb_models_code = """# ============================================================================
# PHASE 9.5: Gradient Boosting Models Comparison
# ============================================================================
print("=" * 80)
print("PHASE 9.5: GRADIENT BOOSTING MODELS COMPARISON")
print("=" * 80)

if 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    from finance_ml.ml_workflow.regression.models import (
        train_xgboost_regressor,
        train_lightgbm_regressor,
        train_catboost_regressor,
        train_histgb_regressor,
    )
    
    gb_results = {}
    
    # 1. XGBoost
    print("\\n🚀 Training XGBoost Regressor...")
    try:
        xgb_result = train_xgboost_regressor(
            X_train, y_train,
            params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            random_state=RANDOM_SEED
        )
        gb_results['XGBoost'] = xgb_result
        print(f"   R² Score: {xgb_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ XGBoost failed: {e}")
    
    # 2. LightGBM
    print("\\n🚀 Training LightGBM Regressor...")
    try:
        lgb_result = train_lightgbm_regressor(
            X_train, y_train,
            params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            random_state=RANDOM_SEED
        )
        gb_results['LightGBM'] = lgb_result
        print(f"   R² Score: {lgb_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ LightGBM failed: {e}")
    
    # 3. CatBoost
    print("\\n🚀 Training CatBoost Regressor...")
    try:
        cb_result = train_catboost_regressor(
            X_train, y_train,
            params={'iterations': 100, 'depth': 6, 'learning_rate': 0.1},
            random_state=RANDOM_SEED
        )
        gb_results['CatBoost'] = cb_result
        print(f"   R² Score: {cb_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ CatBoost failed: {e}")
    
    # 4. HistGradientBoosting
    print("\\n🚀 Training HistGradientBoosting Regressor...")
    try:
        hgb_result = train_histgb_regressor(
            X_train, y_train,
            max_iter=100,
            max_depth=6,
            random_state=RANDOM_SEED
        )
        gb_results['HistGradientBoosting'] = hgb_result
        print(f"   R² Score: {hgb_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ HistGradientBoosting failed: {e}")
    
    # Summary comparison
    print("\\n" + "=" * 60)
    print("GRADIENT BOOSTING MODELS SUMMARY")
    print("=" * 60)
    for name, result in gb_results.items():
        if result and 'r2_score' in result:
            r2 = result.get('r2_score', 0)
            mae = result.get('mae', 'N/A')
            print(f"  {name:20s}: R² = {r2:.4f}, MAE = {mae}")
    
    if gb_results:
        best_gb = max(gb_results.items(), key=lambda x: x[1].get('r2_score', 0))
        print(f"\\n🏆 Best Gradient Boosting Model: {best_gb[0]}")
    
    print("\\n✓ Gradient boosting models comparison complete")"""

# New cell: Neural Network Regression
nn_markdown = """### 6.3.4 Neural Network Regression (Optional)
Train feedforward neural network for price target prediction using TensorFlow/Keras."""

nn_code = """# ============================================================================
# PHASE 9.5: Neural Network Regression (Optional)
# ============================================================================
print("=" * 80)
print("PHASE 9.5: NEURAL NETWORK REGRESSION")
print("=" * 80)

# Skip if TensorFlow not available or dataset too small
ENABLE_NN = False  # Set to True to enable neural network training

if not ENABLE_NN:
    print("⚠️  Neural network training disabled. Set ENABLE_NN = True to enable.")
elif 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    try:
        from finance_ml.ml_workflow.regression.models import train_neural_network_regressor
        
        print("\\n🧠 Training Neural Network Regressor...")
        print("   Architecture: [128, 64, 32] hidden layers")
        print("   Dropout: 0.3, Learning rate: 0.001")
        
        nn_result = train_neural_network_regressor(
            X_train, y_train,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            random_state=RANDOM_SEED
        )
        
        if nn_result and 'r2_score' in nn_result:
            print(f"\\n   R² Score: {nn_result['r2_score']:.4f}")
            print(f"   MAE: {nn_result.get('mae', 'N/A')}")
            print("\\n✓ Neural network training complete")
        else:
            print("\\n⚠️  Neural network training returned no results")
            
    except ImportError as e:
        print(f"\\n⚠️  TensorFlow not available: {e}")
        print("   Install with: pip install tensorflow")
    except Exception as e:
        print(f"\\n⚠️  Neural network training failed: {e}")"""

# New cell: Ensemble Methods Summary
ensemble_markdown = """### 6.3.5 Ensemble Methods Summary
Summary of voting and stacking ensemble performance."""

ensemble_code = """# ============================================================================
# PHASE 9.5: Ensemble Methods Summary
# ============================================================================
print("=" * 80)
print("PHASE 9.5: ENSEMBLE METHODS SUMMARY")
print("=" * 80)

if 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    from finance_ml.ml_workflow.regression.models import (
        train_voting_regressor,
    )
    
    ensemble_results = {}
    
    # Voting Regressor
    print("\\n🗳️ Training Voting Regressor (XGBoost + LightGBM + Ridge)...")
    try:
        voting_result = train_voting_regressor(
            X_train, y_train,
            weights=[0.4, 0.4, 0.2],  # XGB, LGB, Ridge
            random_state=RANDOM_SEED
        )
        ensemble_results['Voting'] = voting_result
        print(f"   R² Score: {voting_result.get('r2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️ Voting ensemble failed: {e}")
    
    # Summary of all model types
    print("\\n" + "=" * 60)
    print("ALL MODELS SUMMARY (Phase 9.5)")
    print("=" * 60)
    
    all_results = {}
    
    # Collect results from previous cells if available
    if 'linear_results' in dir():
        all_results.update(linear_results)
    if 'tree_results' in dir():
        all_results.update(tree_results)
    if 'gb_results' in dir():
        all_results.update(gb_results)
    if 'ensemble_results' in dir():
        all_results.update(ensemble_results)
    
    if all_results:
        # Sort by R² score
        sorted_results = sorted(
            [(k, v) for k, v in all_results.items() if v and 'r2_score' in v],
            key=lambda x: x[1]['r2_score'],
            reverse=True
        )
        
        print(f"\\n{'Model':<25} {'R² Score':>10} {'MAE':>12}")
        print("-" * 50)
        for name, result in sorted_results:
            r2 = result.get('r2_score', 0)
            mae = result.get('mae', 'N/A')
            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
            print(f"{name:<25} {r2:>10.4f} {mae_str:>12}")
        
        if sorted_results:
            best = sorted_results[0]
            print(f"\\n🏆 Best Overall Model: {best[0]} (R² = {best[1]['r2_score']:.4f})")
    else:
        print("\\n⚠️  No model results available. Run model comparison cells first.")
    
    print("\\n✓ Phase 9.5 model comparison complete")"""

# ============================================================================
# 3. Create new cells for Phase 9.6 enhancements
# ============================================================================

# New cell: Feature Importance by Sector
fi_sector_markdown = """### 7.1 Feature Importance Analysis by Sector
Analyze feature importance patterns across different sectors."""

fi_sector_code = """# ============================================================================
# PHASE 9.6: Feature Importance Analysis by Sector
# ============================================================================
print("=" * 80)
print("PHASE 9.6: FEATURE IMPORTANCE BY SECTOR")
print("=" * 80)

if 'all_stocks_enhanced' not in dir():
    print("⚠️  Enhanced dataset not available. Run Phase 9.5 first.")
elif 'best_model' not in dir() and 'stacking_model' not in dir():
    print("⚠️  No trained model available. Run model training cells first.")
else:
    import pandas as pd
    import numpy as np
    
    # Get feature importance from best available model
    model = stacking_model if 'stacking_model' in dir() else best_model
    
    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns.tolist() if 'X_train' in dir() else []
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        fi_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\\n📊 Top 20 Most Important Features (Overall):")
        print("-" * 50)
        for idx, row in fi_df.head(20).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.4f}")
        
        # Sector-specific analysis if sector data available
        if 'sector' in all_stocks_enhanced.columns:
            print("\\n📊 Analyzing feature importance patterns by sector...")
            
            sectors = all_stocks_enhanced['sector'].dropna().unique()
            sector_top_features = {}
            
            for sector in sectors[:5]:  # Top 5 sectors
                sector_mask = all_stocks_enhanced['sector'] == sector
                sector_count = sector_mask.sum()
                if sector_count > 100:
                    # Get correlation with target for this sector
                    sector_data = all_stocks_enhanced[sector_mask]
                    if 'price_target' in sector_data.columns:
                        numeric_cols = sector_data.select_dtypes(include=[np.number]).columns
                        correlations = sector_data[numeric_cols].corrwith(
                            sector_data['price_target']
                        ).abs().sort_values(ascending=False)
                        sector_top_features[sector] = correlations.head(5).to_dict()
            
            if sector_top_features:
                print("\\n📊 Top Correlated Features by Sector:")
                for sector, features in sector_top_features.items():
                    print(f"\\n  {sector}:")
                    for feat, corr in features.items():
                        if feat != 'price_target':
                            print(f"    {feat:35s}: {corr:.3f}")
    else:
        print("\\n⚠️  Model does not have feature_importances_ attribute")
        print("   Try using SHAP values or permutation importance instead")
    
    print("\\n✓ Feature importance analysis complete")"""

# New cell: Model Performance by Region
perf_region_markdown = """### 7.2 Model Performance by Sector and Region
Evaluate model performance segmented by sector and geographic region."""

perf_region_code = """# ============================================================================
# PHASE 9.6: Model Performance by Sector and Region
# ============================================================================
print("=" * 80)
print("PHASE 9.6: MODEL PERFORMANCE BY SECTOR AND REGION")
print("=" * 80)

if 'all_stocks_enhanced' not in dir():
    print("⚠️  Enhanced dataset not available. Run Phase 9.5 first.")
elif 'y_pred' not in dir() and 'y_pred_stacking' not in dir():
    print("⚠️  Predictions not available. Run model training first.")
else:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Get predictions
    predictions = y_pred_stacking if 'y_pred_stacking' in dir() else y_pred
    actuals = y_test if 'y_test' in dir() else None
    
    if actuals is not None and len(predictions) == len(actuals):
        # Create evaluation DataFrame
        eval_df = pd.DataFrame({
            'actual': actuals,
            'predicted': predictions,
            'error': actuals - predictions,
            'abs_error': np.abs(actuals - predictions),
            'pct_error': np.abs(actuals - predictions) / np.maximum(actuals, 1) * 100
        })
        
        # Add sector and region if available
        if 'X_test' in dir() and hasattr(X_test, 'index'):
            test_idx = X_test.index
            if 'sector' in all_stocks_enhanced.columns:
                eval_df['sector'] = all_stocks_enhanced.loc[test_idx, 'sector'].values
            if 'region' in all_stocks_enhanced.columns:
                eval_df['region'] = all_stocks_enhanced.loc[test_idx, 'region'].values
        
        # Overall metrics
        print("\\n📊 Overall Model Performance:")
        print("-" * 40)
        print(f"  R² Score:     {r2_score(actuals, predictions):.4f}")
        print(f"  MAE:          {mean_absolute_error(actuals, predictions):.4f}")
        print(f"  RMSE:         {np.sqrt(mean_squared_error(actuals, predictions)):.4f}")
        print(f"  Mean % Error: {eval_df['pct_error'].mean():.2f}%")
        
        # Performance by sector
        if 'sector' in eval_df.columns:
            print("\\n📊 Performance by Sector:")
            print("-" * 60)
            print(f"{'Sector':<25} {'Count':>8} {'MAE':>10} {'R²':>10}")
            print("-" * 60)
            
            for sector in eval_df['sector'].dropna().unique():
                sector_data = eval_df[eval_df['sector'] == sector]
                if len(sector_data) >= 10:
                    mae = sector_data['abs_error'].mean()
                    r2 = r2_score(sector_data['actual'], sector_data['predicted'])
                    print(f"{sector:<25} {len(sector_data):>8} {mae:>10.2f} {r2:>10.4f}")
        
        # Performance by region
        if 'region' in eval_df.columns:
            print("\\n📊 Performance by Region:")
            print("-" * 60)
            print(f"{'Region':<30} {'Count':>8} {'MAE':>10} {'MAPE':>10}")
            print("-" * 60)
            
            for region in eval_df['region'].dropna().unique():
                region_data = eval_df[eval_df['region'] == region]
                if len(region_data) >= 10:
                    mae = region_data['abs_error'].mean()
                    mape = region_data['pct_error'].mean()
                    print(f"{region:<30} {len(region_data):>8} {mae:>10.2f} {mape:>9.2f}%")
        
        print("\\n✓ Performance analysis by segment complete")
    else:
        print("\\n⚠️  Cannot perform segment analysis - predictions/actuals mismatch")"""

# ============================================================================
# 4. Insert new cells at appropriate positions
# ============================================================================

# Find cell 79 (compare_regressors section) to insert after
insert_pos = 80  # After cell 79

new_cells = [
    # Linear Models
    {"cell_type": "markdown", "metadata": {}, "source": [linear_models_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [linear_models_code],
        "outputs": [],
        "execution_count": None,
    },
    # Tree Models
    {"cell_type": "markdown", "metadata": {}, "source": [tree_models_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [tree_models_code],
        "outputs": [],
        "execution_count": None,
    },
    # Gradient Boosting
    {"cell_type": "markdown", "metadata": {}, "source": [gb_models_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [gb_models_code],
        "outputs": [],
        "execution_count": None,
    },
    # Neural Network
    {"cell_type": "markdown", "metadata": {}, "source": [nn_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [nn_code],
        "outputs": [],
        "execution_count": None,
    },
    # Ensemble Summary
    {"cell_type": "markdown", "metadata": {}, "source": [ensemble_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [ensemble_code],
        "outputs": [],
        "execution_count": None,
    },
]

# Insert cells after position 79 (before stacking ensemble)
for i, cell in enumerate(new_cells):
    cells.insert(insert_pos + i, cell)

print(f"  [OK] Inserted {len(new_cells)} new cells for Phase 9.5 model comparisons")

# Find Phase 9.6 section and add feature importance cells
# Phase 9.6 starts around cell 96 (now shifted due to insertions)
phase96_insert_pos = 96 + len(new_cells) + 10  # After basic Phase 9.6 cells

phase96_cells = [
    # Feature Importance by Sector
    {"cell_type": "markdown", "metadata": {}, "source": [fi_sector_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [fi_sector_code],
        "outputs": [],
        "execution_count": None,
    },
    # Performance by Region
    {"cell_type": "markdown", "metadata": {}, "source": [perf_region_markdown]},
    {
        "cell_type": "code",
        "metadata": {},
        "source": [perf_region_code],
        "outputs": [],
        "execution_count": None,
    },
]

# Insert Phase 9.6 cells
for i, cell in enumerate(phase96_cells):
    cells.insert(phase96_insert_pos + i, cell)

print(f"  [OK] Inserted {len(phase96_cells)} new cells for Phase 9.6 enhancements")

# ============================================================================
# 5. Save updated notebook
# ============================================================================
print("\nSaving updated notebook...")

with open("ml_finance_model_main2_0.ipynb", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook updated successfully!")
print(f"     Total cells: {len(cells)}")
print(f"     New Phase 9.5 cells: {len(new_cells)}")
print(f"     New Phase 9.6 cells: {len(phase96_cells)}")
