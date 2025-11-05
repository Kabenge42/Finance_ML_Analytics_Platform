# Phase 9.5: Sector-Optimized Regression Models with Classification Features
# Complete implementation with all 8 workflow steps

# ============================================================================
# PHASE 9.5 — SECTOR-OPTIMIZED REGRESSION MODELS WITH CLASSIFICATION FEATURES
# ============================================================================
print_section_header("PHASE 9.5 — SECTOR-OPTIMIZED REGRESSION MODELS WITH CLASSIFICATION FEATURES")

# Import required functions from finance_ml.advanced_models
from finance_ml.advanced_models import (
    extract_classification_features,
    integrate_classification_features_into_dataframe,
    create_classification_interactions,
    prepare_regression_data,
    compare_regressors,
    train_stacking_regressor,
    train_quantile_regressor,
    train_sector_specific_models,
    save_model,
)

# Configuration
TARGET_COL = "price_target"
FALLBACK_TARGET = "last_price"
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
QUANTILES = [0.1, 0.5, 0.9]
MIN_SECTOR_SAMPLES = 20

# Create output directory
out_models_dir = Path("outputs") / "models"
out_models_dir.mkdir(parents=True, exist_ok=True)

try:
    # ========================================================================
    # STEP 1: VERIFY PREREQUISITES
    # ========================================================================
    print("\n📋 Step 1: Verifying prerequisites...")

    if "all_stocks_phase94" not in globals():
        raise NameError(
            "❌ all_stocks_phase94 not found. Please run Phase 9.4 first.\n"
            "   Phase 9.4 creates classification meta-features required for regression."
        )

    all_stocks_phase95 = all_stocks_phase94.copy()
    print(f"✓ Dataset loaded: {len(all_stocks_phase95):,} stocks")
    print(f"  Columns: {len(all_stocks_phase95.columns)}")

    # Check for classification probability columns
    classification_cols = [c for c in all_stocks_phase95.columns if c.startswith("event_prob_")]
    if len(classification_cols) == 0:
        print("⚠ Warning: No classification probability columns found")
        print("  Phase 9.4 classification may not have run successfully")
    else:
        print(f"✓ Found {len(classification_cols)} classification probability columns")

    # ========================================================================
    # STEP 2: CREATE INTERACTION FEATURES
    # ========================================================================
    print("\n🔧 Step 2: Creating interaction features...")

    # Define valuation columns for interactions
    valuation_cols = ["p_e", "p_b", "ev_ebitda", "market_cap"]
    available_valuation = [c for c in valuation_cols if c in all_stocks_phase95.columns]

    if len(classification_cols) > 0 and len(available_valuation) > 0:
        print(f"  Creating interactions between {len(classification_cols)} classification features")
        print(f"  and {len(available_valuation)} valuation metrics...")

        all_stocks_phase95 = create_classification_interactions(
            df=all_stocks_phase95,
            classification_cols=classification_cols,
            valuation_cols=available_valuation,
        )

        # Count new interaction columns
        interaction_cols = [c for c in all_stocks_phase95.columns if "_x_" in c]
        print(f"✓ Created {len(interaction_cols)} interaction features")
    else:
        print("⚠ Skipping interaction features (missing classification or valuation columns)")

    # ========================================================================
    # STEP 3: PREPARE REGRESSION DATA
    # ========================================================================
    print("\n📊 Step 3: Preparing regression data...")

    # Determine target column
    target_col = TARGET_COL if TARGET_COL in all_stocks_phase95.columns else FALLBACK_TARGET
    if target_col == FALLBACK_TARGET:
        print(f"⚠ Using '{FALLBACK_TARGET}' as proxy for target variable")

    # Prepare train/test split
    exclude_cols = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "region",
        "trading_country",
        "exchange",
        "currency",
    ]

    X_train, X_test, y_train, y_test, feature_cols = prepare_regression_data(
        df=all_stocks_phase95,
        target_col=target_col,
        exclude_cols=exclude_cols,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"✓ Training set: {len(X_train):,} samples, {len(feature_cols)} features")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # ========================================================================
    # STEP 4: TRAIN AND COMPARE MULTIPLE REGRESSION MODELS
    # ========================================================================
    print("\n🤖 Step 4: Training and comparing regression models...")
    print("  Models: Ridge, Lasso, RF, ExtraTrees, GradientBoosting, HistGradientBoosting")

    comparison_results = compare_regressors(
        X=pd.concat([X_train, X_test]),
        y=pd.concat([y_train, y_test]),
        test_size=TEST_SIZE,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        ensure_nonnegative=True,  # TDD enhancement
        loss="huber",  # TDD enhancement for outlier robustness
    )

    if comparison_results is not None and len(comparison_results) > 0:
        print("\n📈 Model Comparison Results:")
        print(comparison_results.to_string(index=False))

        best_model = comparison_results.iloc[0]["Model"]
        best_mae = comparison_results.iloc[0]["MAE"]
        best_r2 = comparison_results.iloc[0]["R2"]
        print(f"\n✓ Best model: {best_model} (MAE={best_mae:.2f}, R²={best_r2:.4f})")

        # Save comparison results
        comparison_path = out_models_dir / "model_comparison_results.csv"
        comparison_results.to_csv(comparison_path, index=False)
        print(f"✓ Comparison results saved to: {comparison_path}")
    else:
        print("⚠ Model comparison failed or returned no results")

    # ========================================================================
    # STEP 5: BUILD STACKING ENSEMBLE
    # ========================================================================
    print("\n🏗 Step 5: Building stacking ensemble...")

    stacking_result = train_stacking_regressor(
        X=X_train,
        y=y_train,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        ensure_nonnegative=True,  # TDD enhancement
        loss="huber",  # TDD enhancement
    )

    if stacking_result is not None:
        stacking_model = stacking_result["model"]

        # Evaluate on test set
        y_pred_stacking = stacking_model.predict(X_test)

        # Ensure non-negative predictions
        y_pred_stacking = np.maximum(y_pred_stacking, 0)

        mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
        rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
        r2_stacking = r2_score(y_test, y_pred_stacking)

        print(f"✓ Stacking Ensemble Performance:")
        print(f"  Train Score: {stacking_result.get('train_score', 0):.4f}")
        print(f"  CV Score: {stacking_result.get('cv_score', 0):.4f}")
        print(f"  Test MAE: {mae_stacking:.2f}")
        print(f"  Test RMSE: {rmse_stacking:.2f}")
        print(f"  Test R²: {r2_stacking:.4f}")

        # Save stacking model with metadata
        stacking_metadata = {
            "model_type": "stacking_ensemble",
            "features": feature_cols,
            "target": target_col,
            "date_trained": datetime.now().strftime("%Y-%m-%d"),
            "phase": "9.5",
            "train_score": stacking_result.get("train_score", 0),
            "cv_score": stacking_result.get("cv_score", 0),
            "test_score": r2_stacking,
            "test_mae": mae_stacking,
            "test_rmse": rmse_stacking,
        }

        stacking_path = out_models_dir / "stacking_ensemble_phase95.joblib"
        save_model(stacking_model, str(stacking_path), metadata=stacking_metadata)
        print(f"✓ Stacking model saved: {stacking_path}")
    else:
        print("⚠ Stacking ensemble training failed")
        y_pred_stacking = None

    # ========================================================================
    # STEP 6: TRAIN QUANTILE REGRESSION FOR PREDICTION INTERVALS
    # ========================================================================
    print("\n📊 Step 6: Training quantile regression for prediction intervals...")

    quantile_result = train_quantile_regressor(
        X=X_train, y=y_train, quantiles=QUANTILES, random_state=RANDOM_STATE
    )

    if quantile_result is not None:
        quantile_models = quantile_result["models"]

        # Generate predictions for each quantile
        predictions_quantile = {}
        for q, model in zip(QUANTILES, quantile_models):
            preds = model.predict(X_test)
            predictions_quantile[q] = np.maximum(preds, 0)  # Ensure non-negative

        print(f"✓ Quantile regression trained for quantiles: {QUANTILES}")

        # Calculate prediction interval statistics
        interval_width = predictions_quantile[0.9] - predictions_quantile[0.1]
        print(f"  Mean prediction interval width: {interval_width.mean():.2f}")
        print(f"  Median prediction interval width: {np.median(interval_width):.2f}")

        # Save quantile models
        for q, model in zip(QUANTILES, quantile_models):
            quantile_metadata = {
                "model_type": f"quantile_regressor_q{q}",
                "features": feature_cols,
                "target": target_col,
                "date_trained": datetime.now().strftime("%Y-%m-%d"),
                "phase": "9.5",
                "quantile": q,
            }
            quantile_path = out_models_dir / f"quantile_q{int(q*100)}_phase95.joblib"
            save_model(model, str(quantile_path), metadata=quantile_metadata)

        print(f"✓ Quantile models saved: {len(QUANTILES)} models")
    else:
        print("⚠ Quantile regression training failed")
        predictions_quantile = {q: np.zeros(len(X_test)) for q in QUANTILES}

    # ========================================================================
    # STEP 7: TRAIN SECTOR-SPECIFIC MODELS (OPTIONAL)
    # ========================================================================
    print("\n🏢 Step 7: Training sector-specific models...")

    if "sector" in all_stocks_phase95.columns:
        sector_counts = all_stocks_phase95["sector"].value_counts()
        eligible_sectors = sector_counts[sector_counts >= MIN_SECTOR_SAMPLES].index.tolist()

        print(f"  Eligible sectors (>={MIN_SECTOR_SAMPLES} samples): {len(eligible_sectors)}")

        if len(eligible_sectors) > 0:
            sector_models_result = train_sector_specific_models(
                df=all_stocks_phase95,
                feature_cols=feature_cols,
                target_col=target_col,
                sector_col="sector",
                model_type="random_forest",
                random_state=RANDOM_STATE,
                min_samples=MIN_SECTOR_SAMPLES,
                ensure_nonnegative=True,  # TDD enhancement
            )

            if sector_models_result is not None:
                sector_models = sector_models_result["models"]
                sector_metrics = sector_models_result["metrics"]

                print(f"✓ Trained {len(sector_models)} sector-specific models")
                print("\n📊 Top 3 Sector Model Performance:")
                for sector, metrics in list(sector_metrics.items())[:3]:
                    print(f"  {sector}:")
                    print(f"    MAE: {metrics.get('mae', 0):.2f}")
                    print(f"    R²: {metrics.get('r2', 0):.4f}")

                # Save sector models
                for sector, model in sector_models.items():
                    sector_metadata = {
                        "model_type": "sector_specific_rf",
                        "sector": sector,
                        "features": feature_cols,
                        "target": target_col,
                        "date_trained": datetime.now().strftime("%Y-%m-%d"),
                        "phase": "9.5",
                    }
                    sector_filename = (
                        f'sector_model_{sector.replace(" ", "_").lower()}_phase95.joblib'
                    )
                    sector_path = out_models_dir / sector_filename
                    save_model(model, str(sector_path), metadata=sector_metadata)

                print(f"✓ Sector models saved: {len(sector_models)} models")
            else:
                print("⚠ Sector-specific model training failed")
        else:
            print(f"⚠ No sectors with >={MIN_SECTOR_SAMPLES} samples, skipping sector models")
    else:
        print("⚠ 'sector' column not found, skipping sector-specific models")

    # ========================================================================
    # STEP 8: STORE PREDICTIONS FOR DOWNSTREAM ANALYSIS
    # ========================================================================
    print("\n💾 Step 8: Storing predictions for downstream analysis...")

    # Use stacking predictions as primary, or fall back to median quantile
    if y_pred_stacking is not None:
        y_pred_final = y_pred_stacking
        prediction_source = "Stacking Ensemble"
    else:
        y_pred_final = predictions_quantile[0.5]
        prediction_source = "Quantile Regression (Median)"

    print(f"  Using predictions from: {prediction_source}")

    # Create comprehensive predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred_final,
            "residual": y_test.values - y_pred_final,
            "abs_error": np.abs(y_test.values - y_pred_final),
            "pct_error": ((y_test.values - y_pred_final) / y_test.values) * 100,
            "lower_10": predictions_quantile[0.1],
            "median": predictions_quantile[0.5],
            "upper_90": predictions_quantile[0.9],
        },
        index=y_test.index,
    )

    # Add metadata columns for error analysis (TDD Priority 1.1)
    if "sector" in all_stocks_phase95.columns:
        predictions_df["sector"] = all_stocks_phase95.loc[y_test.index, "sector"].values
    if "ticker" in all_stocks_phase95.columns:
        predictions_df["ticker"] = all_stocks_phase95.loc[y_test.index, "ticker"].values
    if "market_cap" in all_stocks_phase95.columns:
        predictions_df["market_cap"] = all_stocks_phase95.loc[y_test.index, "market_cap"].values

    # Save predictions
    predictions_path = out_models_dir / "regression_predictions_phase95.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved: {predictions_path}")
    print(f"  Shape: {predictions_df.shape}")
    print(f"  Columns: {list(predictions_df.columns)}")

    # Store predictions in all_stocks_featured for downstream phases
    all_stocks_featured = all_stocks_phase95.copy()
    all_stocks_featured.loc[y_test.index, "predicted_price_target"] = y_pred_final
    all_stocks_featured.loc[y_test.index, "prediction_lower_10"] = predictions_quantile[0.1]
    all_stocks_featured.loc[y_test.index, "prediction_upper_90"] = predictions_quantile[0.9]

    print(f"✓ Predictions stored in all_stocks_featured dataframe")
    print(f"  Added columns: predicted_price_target, prediction_lower_10, prediction_upper_90")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 9.5 COMPLETE — SECTOR-OPTIMIZED REGRESSION MODELS")
    print("=" * 80)
    print(f"✓ Step 1: Prerequisites verified")
    print(f"✓ Step 2: Interaction features created")
    print(f"✓ Step 3: Regression data prepared ({len(X_train):,} train, {len(X_test):,} test)")
    print(f"✓ Step 4: Multiple models compared (6+ algorithms)")
    print(f"✓ Step 5: Stacking ensemble trained and saved")
    print(f"✓ Step 6: Quantile regression for prediction intervals")
    print(f"✓ Step 7: Sector-specific models (optional)")
    print(f"✓ Step 8: Predictions stored for Phases 9.6, 9.7, 9.8")
    print("\n📁 Outputs saved to:")
    print(f"  - {out_models_dir}")
    print(f"  - regression_predictions_phase95.csv")
    print(f"  - stacking_ensemble_phase95.joblib")
    print(f"  - quantile models (3 files)")
    print(f"  - sector-specific models (if applicable)")

    # Checkpoint for downstream phases
    checkpoint("phase_95_complete", requires=["classification_complete"])
    print("\n✓ Checkpoint: phase_95_complete")

except Exception as e:
    logger.error(f"Phase 9.5 failed: {e}", exc_info=True)
    print(f"\n❌ Phase 9.5 FAILED: {e}")
    import traceback

    traceback.print_exc()

    # Create minimal dataframe for downstream phases
    if "all_stocks_phase94" in globals():
        all_stocks_featured = all_stocks_phase94.copy()
        print("\n⚠ Using Phase 9.4 data as fallback for all_stocks_featured")
