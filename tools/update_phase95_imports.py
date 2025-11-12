"""Update Phase 9.5 regression imports in ml_finance_model_main.ipynb"""
import json

# Read notebook
with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New comprehensive Phase 9.5 imports
new_imports = '''# Phase 9.5: Regression models
# Direct import from regression subpackage (Phase 9.5 refactor)
from finance_ml import (
    regression_prepare_data,
    regression_train_xgboost,
    regression_train_lightgbm,
    regression_train_catboost,
    regression_compare_regressors,
    regression_train_sector_models,
    regression_save_model,
    regression_load_model,
    regression_create_classification_interactions,
    regression_train_stacking,
    regression_train_quantile,
    )
# Import additional regression functions directly from subpackage
from finance_ml.ml_workflow.regression import (
    # Dataset preparation and validation
    prepare_regression_data,
    validate_training_data,
    prepare_features_for_training,
    extract_numeric_feature_columns,
    extract_classification_features,
    integrate_classification_features_into_dataframe,
    create_classification_interactions,
    train_sector_specific_models,
    )
# Import model training functions from regression.models
from finance_ml.ml_workflow.regression.models import (
    # Linear models
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor,
    train_bayesian_ridge_regressor,
    train_polynomial_regressor,
    # Gradient boosting models
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_histgb_regressor,
    # Tree models
    train_random_forest_regressor,
    train_extra_trees_regressor,
    # Neural network
    train_neural_network_regressor,
    # Ensemble methods
    train_voting_regressor,
    train_stacking_regressor,
    # Model comparison
    compare_regressors,
    )
# Import quantile regression
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor,
    )
# Import hyperparameter tuning
from finance_ml.ml_workflow.regression.tuning import (
    optimize_hyperparameters_optuna,
    )
# Import model IO
from finance_ml.ml_workflow.regression.io import (
    save_model,
    load_model,
    )
# Import constraints
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper,
    )
'''

# Find cell 4 and update
cell_4 = notebook['cells'][4]
source = ''.join(cell_4['source'])

# Find the section to replace
start_marker = '# Phase 9.5: Regression models'
end_marker = '# Phase 9.6: Evaluation'

start_idx = source.find(start_marker)
end_idx = source.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("ERROR: Could not find Phase 9.5 section markers")
    exit(1)

# Replace the section
new_source = source[:start_idx] + new_imports + '\n' + source[end_idx:]

# Update cell source (split by lines for JSON format)
cell_4['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

# Save updated notebook
with open('ml_finance_model_main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Updated Phase 9.5 imports in ml_finance_model_main.ipynb")
print(f"  - Added 17 new regression functions")
print(f"  - Organized by category: Dataset, Models (Linear/GB/Tree/NN/Ensemble), Quantile, Tuning, IO, Constraints")
print(f"  - Total regression functions: 28 (was 11)")
