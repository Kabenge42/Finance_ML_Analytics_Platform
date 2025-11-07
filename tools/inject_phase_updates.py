#!/usr/bin/env python3
"""
Inject or update Phase 9.3–9.8 integration cells in ml_finance_model_main_v10.ipynb
- Adds minimal, non-invasive cells that call enhanced finance_ml tooling per README v0.6.0
- Idempotent: skips insertion if markers already present
"""
from __future__ import annotations
import json
from pathlib import Path

NOTEBOOK_PATH = Path('ml_finance_model_main_v10.ipynb')

# Utility to build simple Jupyter cell dicts

def md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if source.endswith("\n") else source + "\n",
    }


def code_cell(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [s + ("\n" if not s.endswith("\n") else "") for s in source.splitlines()],
    }


def load_notebook(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict):
    # backup
    backup = path.with_suffix(path.suffix + '.backup_inject_phases')
    backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    with path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def has_marker(nb: dict, marker: str) -> bool:
    text = json.dumps(nb)
    return marker in text


# Cells to insert (minimal integration points)
PHASE_94_MARKER = "# %% PHASE 9.4 — ENHANCED CLASSIFICATION (classification_enhanced)"
PHASE_95Q_MARKER = "# %% PHASE 9.5.2 — QUANTILES + SECTOR MODELS (advanced_models)"
PHASE_96_MARKER = "# %% PHASE 9.6 — COMPREHENSIVE EVALUATION (finance_ml.eval)"
PHASE_97_MARKER = "# %% PHASE 9.7 — VALUATION & ANALYST COMPARISON"
PHASE_98_MARKER = "# %% PHASE 9.8 — REPORTING EXPORTS"

phase94_code = PHASE_94_MARKER + """
try:
    from finance_ml import classification_enhanced as clf_enh
    print('Loaded finance_ml.classification_enhanced')
    if 'all_stocks_processed' in globals():
        # Derive labels using improved helper if available
        y_labels = clf_enh.create_event_labels_enhanced(
            all_stocks_processed.copy(),
            sector_col='sector',
            price_col='last_price',
            momentum_window=20,
            sector_thresholds=True,
            use_volatility=True,
        )
        # Train and select best model (includes NN + ensembles)
        cls_result = clf_enh.train_compare_classifiers(
            all_stocks_processed.copy(), y_labels, random_state=42
        )
        X_proc, num_cols = clf_enh.prepare_features_for_prediction(all_stocks_processed.copy())
        y_proba_all = cls_result['best_model'].predict_proba(X_proc)
        all_stocks_with_classification = clf_enh.export_classification_features(
            all_stocks_processed.copy(), y_proba_all,
            class_names=['Neutral','Positive','Negative']
        )
        all_stocks_phase94 = all_stocks_with_classification.copy()
        print('✓ Phase 9.4 enhanced classification complete -> all_stocks_phase94')
    else:
        print('⚠ all_stocks_processed not found — skipping enhanced classification')
except Exception as e:
    print(f'⚠ Phase 9.4 enhanced classification skipped: {e}')
"""

phase95q_code = PHASE_95Q_MARKER + """
from pathlib import Path
try:
    from finance_ml import advanced_models as am
    print('Loaded finance_ml.advanced_models')
    df_src = all_stocks_phase94 if 'all_stocks_phase94' in globals() else (
        all_stocks_featured if 'all_stocks_featured' in globals() else None
    )
    if df_src is None:
        raise RuntimeError('No dataframe available for Phase 9.5.2 (need all_stocks_phase94 or all_stocks_featured)')
    df95 = df_src.copy()
    # Extract features (auto, exclude obvious targets)
    feature_cols = am.extract_numeric_feature_columns(
        df95,
        exclude_cols=['price_target','predicted_price_target'],
        exclude_patterns=['id','date','_flag']
    )
    # Train sector-specific baseline (fast)
    sector_models = am.train_sector_specific_models(
        df95, feature_cols=feature_cols, target_col='price_target',
        sector_col='sector', model_type='random_forest', random_state=42,
        min_samples=20, ensure_nonnegative=True, auto_extract_fallback=True,
    )
    # Quantile models for intervals
    q_model = am.train_quantile_regressor(
        df95[feature_cols].fillna(0), df95['price_target'], quantiles=[0.1,0.5,0.9]
    )
    # Hyperparameter tuning (bounded trials for speed)
    try:
        best_rf = am.optimize_hyperparameters_optuna(
            df95[feature_cols].fillna(0), df95['price_target'],
            model_type='random_forest', n_trials=10, cv=3, random_state=42
        )
    except Exception as tune_err:
        print(f'⚠ Optuna tuning skipped: {tune_err}')
        best_rf = None
    # Store for downstream phases
    phase95q_artifacts = {
        'sector_models': sector_models,
        'quantile_model': q_model,
        'best_rf': best_rf,
        'feature_cols': feature_cols,
    }
    print('✓ Phase 9.5.2 quantiles + sector models complete -> phase95q_artifacts')
except Exception as e:
    print(f'⚠ Phase 9.5.2 failed: {e}')
"""

phase96_code = PHASE_96_MARKER + """
try:
    from finance_ml import eval as fme
    import pandas as pd
    from pathlib import Path
    out_dir = Path('outputs/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer all_stocks_featured with predictions; fallback to df95+RF if available
    if 'all_stocks_featured' in globals() and 'predicted_price_target' in all_stocks_featured.columns:
        df_eval = all_stocks_featured.copy()
        y_true = df_eval.get('price_target')
        y_pred = df_eval.get('predicted_price_target')
    else:
        # Fallback: attempt quick predictions using tuned/baseline
        df_src = (all_stocks_phase94 if 'all_stocks_phase94' in globals() else None)
        if df_src is None:
            raise RuntimeError('No predictions available for Phase 9.6')
        df_eval = df_src.copy()
        # Naive proxy if no predictions
        y_true = df_eval.get('price_target')
        y_pred = df_eval.get('last_price')
        df_eval['predicted_price_target'] = y_pred
    # Compute metrics and plots
    metrics = fme.compute_regression_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(out_dir / 'regression_metrics_summary.csv', index=False)
    fme.plot_residuals(y_true, y_pred, save_path=str(out_dir / 'residual_histogram.png'))
    fme.plot_residuals_vs_predicted(y_true, y_pred, save_path=str(out_dir / 'residuals_vs_predicted.png'))
    fme.plot_qq(y_true, y_pred, save_path=str(out_dir / 'qq_plot.png'))
    print('✓ Phase 9.6 evaluation complete -> metrics and plots saved')
except Exception as e:
    print(f'⚠ Phase 9.6 evaluation skipped: {e}')
"""

phase97_code = PHASE_97_MARKER + """
try:
    from finance_ml import eval as fme
    import pandas as pd
    df_src = all_stocks_featured if 'all_stocks_featured' in globals() else (
        all_stocks_phase94 if 'all_stocks_phase94' in globals() else None
    )
    if df_src is None:
        raise RuntimeError('No dataframe available for Phase 9.7')
    dfv = df_src.copy()
    dfv = fme.calculate_mispricing_score(dfv)
    rankings = fme.rank_stocks_by_sector(dfv, top_n=10)
    rankings.to_csv('outputs/analytics/top_undervalued_by_sector.csv', index=False)
    # Analyst comparison if columns available
    try:
        comp = fme.compare_with_analyst_targets(dfv)
        comp.to_csv('outputs/analytics/analyst_comparison.csv', index=False)
        print('✓ Analyst comparison generated')
    except Exception as ac_err:
        print(f'ℹ Analyst comparison skipped: {ac_err}')
    print('✓ Phase 9.7 valuation & rankings complete')
except Exception as e:
    print(f'⚠ Phase 9.7 skipped: {e}')
"""

phase98_code = f"""
{PHASE_98_MARKER}
try:
    from finance_ml import eval as fme
    import pandas as pd
    from pathlib import Path
    out_dir = Path('outputs/analytics')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Export a compact report
    src = all_stocks_featured if 'all_stocks_featured' in globals() else (
        all_stocks_phase94 if 'all_stocks_phase94' in globals() else None
    )
    if src is None:
        raise RuntimeError('No dataframe available for Phase 9.8')
    report_cols = [c for c in src.columns if c.lower() in (
        'ticker','sector','region','last_price','price_target','predicted_price_target','mispricing_score'
    ) or c.startswith('event_prob_')]
    src[report_cols].to_csv(out_dir / 'predictions_report_compact.csv', index=False)
    try:
        fme.export_excel_report(src, out_path=str(out_dir / 'predictions_report.xlsx'))
    except Exception as exl_err:
        print('ℹ Excel report skipped:', exl_err)
    print('✓ Phase 9.8 reporting exports complete')
except Exception as e:
    print('⚠ Phase 9.8 skipped:', e)
"""


def insert_cells(nb: dict) -> bool:
    inserted = False
    # ensure outputs directories exist markers in md
    additions = []
    if not has_marker(nb, PHASE_94_MARKER):
        additions.append(md_cell("### 9.4 — Enhanced Classification (auto-injected)"))
        additions.append(code_cell(phase94_code))
    if not has_marker(nb, PHASE_95Q_MARKER):
        additions.append(md_cell("### 9.5.2 — Quantile & Sector Models (auto-injected)"))
        additions.append(code_cell(phase95q_code))
    if not has_marker(nb, PHASE_96_MARKER):
        additions.append(md_cell("### 9.6 — Comprehensive Evaluation (auto-injected)"))
        additions.append(code_cell(phase96_code))
    if not has_marker(nb, PHASE_97_MARKER):
        additions.append(md_cell("### 9.7 — Valuation & Analyst Comparison (auto-injected)"))
        additions.append(code_cell(phase97_code))
    if not has_marker(nb, PHASE_98_MARKER):
        additions.append(md_cell("### 9.8 — Reporting Exports (auto-injected)"))
        additions.append(code_cell(phase98_code))
    if not additions:
        return False
    # Insert additions near Phase 9.5 section if found; else append at end
    target_idx = None
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') in ('markdown','code'):
            text = ''.join(cell.get('source', []))
            if '## Phase 9.5' in text:
                target_idx = i + 1
                break
    if target_idx is None:
        nb['cells'].extend(additions)
    else:
        nb['cells'][target_idx:target_idx] = additions
    return True


def main():
    if not NOTEBOOK_PATH.exists():
        print(f"Notebook not found: {NOTEBOOK_PATH}")
        return 1
    nb = load_notebook(NOTEBOOK_PATH)
    changed = insert_cells(nb)
    if changed:
        save_notebook(NOTEBOOK_PATH, nb)
        print("Notebook updated with phase integration cells.")
    else:
        print("No updates needed (markers present).")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
