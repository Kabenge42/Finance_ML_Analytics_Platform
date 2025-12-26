Below is a mechanical, cell-by-cell style checklist for updating `ml_finance_model_main.ipynb` so it fully aligns with
the new ml_workflow utilities and `code_guidelines.md`, followed by concrete Week 4 actions.

I’ll keep it concise and implementation-focused.

---

### 1. Notebook update checklist (Phase 9.9 alignment)

Treat these as targeted edits to the appropriate sections/cells in `ml_finance_model_main.ipynb`. The exact cell numbers
may differ, but the structure typically follows: config → data load → preprocessing → features → classification →
regression → evaluation/analytics.

#### 1.1 Imports cell (top of notebook)

**Goal:** Ensure all new helpers are imported from the canonical locations.

In the main imports cell:

1. Add classification meta-feature export helper:
   ```python
   from finance_ml.ml_workflow.classification.evaluation import (
       evaluate_classification,
       analyze_calibration,
       export_classification_probabilities,  # NEW
   )
   ```

2. Add regression meta-feature integration helper:
   ```python
   from finance_ml.ml_workflow.regression.dataset import (
       prepare_regression_data,
       extract_classification_features,
       integrate_classification_features_into_dataframe,
       integrate_classification_features,  # NEW convenience wrapper
   )
   ```

3. Confirm that you are using the new regression entry point and calibration utilities:
   ```python
   from finance_ml.ml_workflow.regression.models import train_and_evaluate_regression
   from finance_ml.ml_workflow.regression.calibration import (
       calibrate_predictions_by_sector,
       market_cap_bias_correction,
       temporal_bias_adjustment,
   )
   ```

4. Confirm split/safety rails imports are via the new modules (if the notebook calls them directly):
   ```python
   from finance_ml.ml_workflow.validation.splits import create_train_test_split, time_series_cv_or_grouped_split
   from finance_ml.ml_workflow.regression.safety_rails import (
       winsorize_target,
       clip_predictions,
       enforce_non_negative,
   )
   ```

#### 1.2 Data loading and preprocessing cells

The notebook already uses the Phase 9.1–9.2 preprocessing pipeline. Just ensure:

- You do **not** perform manual `train_test_split` calls for modeling; instead rely on:
    - `prepare_regression_data` for the low-level case, or
    - `create_train_test_split` and `time_series_cv_or_grouped_split` where time-aware/grouped splits are needed.
- Any manual splits should be replaced by
  `create_train_test_split(df, date_col="snapshot_date", group_col="ticker", stratify_col="sector", ...)` where
  applicable.

No new code is strictly required here if the notebook already calls the modern helpers.

#### 1.3 Classification step: export probabilities via the standard interface

Locate the section where the classification model is trained and probabilities are obtained, e.g. something like:

```python
cls_result = fit_classifier(X_cls, y_cls, model="lightgbm", ...)
y_pred_cls = cls_result["y_pred"]
y_proba_cls = cls_result["y_proba"]
```

Replace any ad-hoc probability export logic with:

```python
from finance_ml.ml_workflow.classification.evaluation import export_classification_probabilities

# Export standardized classification probabilities
probs_df = export_classification_probabilities(
		y_true=y_cls,
		y_pred=y_pred_cls,
		y_proba=y_proba_cls,
		index=df.index,  # ensure df index corresponds to the same universe
		)

# Save artifact for diagnostics
probs_path = outputs_dir / "classification" / "classification_probabilities.csv"
probs_path.parent.mkdir(parents=True, exist_ok=True)
probs_df.to_csv(probs_path, index=False)
```

Checklist:

- [ ] Remove any hand-built `DataFrame` for probabilities.
- [ ] Use `export_classification_probabilities` everywhere instead.

#### 1.4 Integrate classification meta-features before regression

Right after exporting probabilities, integrate them into the regression dataframe.

If you currently build a regression dataframe `df_reg` via manual concatenation, replace that logic with:

```python
from finance_ml.ml_workflow.regression.dataset import integrate_classification_features

# df should be your main stock universe with price_target, sector, etc.
df_reg = integrate_classification_features(df, y_proba_cls)
```

Then, ensure all downstream regression steps use `df_reg` (or an appropriately named variable) rather than the original
`df` where you expect classification meta-features:

- [ ] Replace references to plain `df` in regression modeling cells with `df_reg` when classification features are
  needed.

#### 1.5 Regression modeling cell: use default pipeline + stacking

In the regression modeling section, ensure you use the new unified entry point and stacking parameter.

Replace any legacy call such as:

```python
from finance_ml.ml_workflow.models import train_and_evaluate_regression

reg_result = train_and_evaluate_regression(df, out_dir=outputs_dir, n_jobs=CONFIG_N_JOBS, dry_run=False)
```

with a call that uses the meta-feature-enriched dataframe and stacking:

```python
from finance_ml.ml_workflow.models import train_and_evaluate_regression

reg_result = train_and_evaluate_regression(
		df_reg,  # with classification meta-features integrated
		out_dir=outputs_dir,
		n_jobs=CONFIG_N_JOBS,
		dry_run=False,
		use_stacking=True,  # default; matches CLI behavior
		)

preds_df = reg_result["predictions"]
full_preds_df = reg_result["full_predictions"]
artifacts = reg_result["artifacts"]
stacking_used = artifacts.get("stacking_enabled", False)
```

Checklist:

- [ ] Ensure the notebook uses `df_reg` here, not the raw `df`.
- [ ] Pass `use_stacking=True` or explicitly control it as desired.
- [ ] Use `reg_result["predictions"]` as the single source of standardized regression predictions.

#### 1.6 Sector bias calibration & metrics cells

Confirm that the notebook already uses the enhanced calibration utilities:

```python
from finance_ml.ml_workflow.regression.calibration import calibrate_predictions_by_sector

calibrated_df = calibrate_predictions_by_sector(preds_df, method="isotonic")
```

And that it:

- Uses `market_cap_bias_correction` and `temporal_bias_adjustment` where appropriate.
- Writes out `regression_metrics_by_sector.csv` via existing sector metrics pipeline.

No new code needed here if the notebook already matches the Phase 10 functions, but you should:

- [ ] Verify that any calibration logic calls these helpers rather than ad-hoc code.

#### 1.7 Predictions and analytics cells

Ensure that all evaluation, analytics, and plotting cells read from the standardized predictions DataFrame(s) created
by:

- CLI/packaged pipeline (`regression/regression_predictions_detailed.csv`), or
- `reg_result["predictions"]` in the notebook.

Key alignment points:

- [ ] Use the standardized columns from `build_predictions_frame` (`y_true`, `y_pred`, `abs_error`, `pct_error`,
  ticker/sector/region/last_price) as per `code_guidelines.md` Section 2.4.
- [ ] Avoid parallel alternative schemas (e.g., `regression_predictions.csv` with a different layout).

---

### 3. Week 4: Integration & Validation – how to run

Once the notebook is updated as above, you can move into Week 4 tasks.

#### 3.1 End-to-end pipeline runs

**CLI pipeline:**

From project root:

```bash
python -m finance_ml.cli main \
  --data-source auto \
  --output-dir outputs \
  --n-jobs 4 \
  --seed 42
```

Options:

- Add `--disable-stacking` to compare stacking vs baseline quickly.
- Add `--skip-sector-regression` or `--skip-eda` if you want faster runs during development.

Verify:

- [ ] CLI completes without errors.
- [ ] `outputs/regression/regression_predictions_detailed.csv` exists and passes `validate_predictions_schema` checks.
- [ ] `outputs/regression/regression_predictions_full.csv` exists.
- [ ] `outputs/regression/regression_metrics_by_sector.csv` is non-empty and conforms to the sector metrics schema.

**Notebook pipeline:**

In `ml_finance_model_main.ipynb`:

- [ ] Re-run all cells in order with the updated imports and helpers.
- [ ] Confirm that:
    - Standardized predictions artifacts are written under `outputs/`.
    - Classification probabilities CSV is generated.
    - Sector metrics and bias-calibration plots are generated as before.

#### 3.2 Performance benchmarking vs baseline

You can use two main levers:

1. **Stacking vs baseline**
    - CLI:
      ```bash
      # Baseline only
      python -m finance_ml.cli main --disable-stacking --output-dir outputs_baseline
 
      # Stacking enabled (default)
      python -m finance_ml.cli main --output-dir outputs_stacking
      ```
    - Compare metrics in the resulting predictions files or in your analytics notebook (e.g., mean/median errors,
      sector-level MAE/RMSE).

2. **Meta-features vs no meta-features**
    - In the notebook, you can temporarily skip the `integrate_classification_features` call and compare regression
      performance with vs without classification meta-features.

Checklist:

- [ ] Record baseline and stacked metrics (MAE, RMSE, R²) and note the relative improvement.
- [ ] Optionally record sector-wise metrics pre/post stacking or meta-features.

#### 3.3 Final documentation sweep

Files to update after confirming behavior:

1. **`README.md`**
    - Add a short section describing:
        - Stacking default behavior and `--disable-stacking` flag.
        - Classification meta-features: how probabilities flow from classifier to regression via
          `export_classification_probabilities` and `integrate_classification_features`.

2. **`CHANGELOG.md`**
    - Add an entry describing:
        - Phase 9.9 completion items: schema standardization, split policy, classification meta-features, stacking
          default.
        - Any observed performance improvements vs previous baseline.

3. **`docs/code_guidelines.md`** (optional refinement)
    - If needed, add small cross-references in the Phase 9.4 / 9.5 sections pointing explicitly to the new helpers:
        - `classification.evaluation.export_classification_probabilities`.
        - `regression.dataset.integrate_classification_features`.

4. **`Finance ML_implementation_plan.md`**
    - Once you’ve confirmed the notebook uses the standard interfaces, you can flip:
      ```markdown
      - [ ] Update notebook to use standard interface
      ```
      to:
      ```markdown
      - [x] Update notebook to use standard interface
      ```

---

If you’d like, you can now run the recommended Week 4 test bundles (fast+medium) and, based on their outcome, I can help
you interpret any remaining discrepancies between the plan and the real behavior (e.g., coverage targets, performance
deltas).
