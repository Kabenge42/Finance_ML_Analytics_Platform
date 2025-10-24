"""
Verification script for preprocessing pipeline improvements
Demonstrates the proper use of returned feature lists for preprocessing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_feature_return_signature():
    """Verify that build_features_and_target returns 4 values"""
    print("=" * 80)
    print("VERIFICATION: build_features_and_target return signature")
    print("=" * 80)

    try:
        import pandas as pd
        from finance_ml.features import build_features_and_target

        # Create test dataframe
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'sector': ['Technology', 'Technology', 'Technology'],
            'region': ['US', 'US', 'US'],
            'last_price': [150.0, 200.0, 100.0],
            'market_cap': [1e9, 2e9, 1.5e9],
            'p_e_ntm': [25.0, 30.0, 22.0],
            'price_target': [160.0, 220.0, 110.0]
        })

        # Call function - should return 4 values
        result = build_features_and_target(df)

        if len(result) != 4:
            print(f"❌ FAIL: Expected 4 return values, got {len(result)}")
            return False

        X, y, numeric_features, categorical_features = result

        print(f"✅ SUCCESS: Function returns 4 values")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - Numeric features: {len(numeric_features)}")
        print(f"  - Categorical features: {len(categorical_features)}")
        print(f"\nNumeric features list: {numeric_features}")
        print(f"Categorical features list: {categorical_features}")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_preprocessing_pipeline():
    """Verify that preprocessing pipeline works with feature lists"""
    print("\n" + "=" * 80)
    print("VERIFICATION: Preprocessing pipeline with feature lists")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        from finance_ml.features import build_features_and_target

        # Create larger test dataset
        np.random.seed(42)
        n_samples = 100

        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy']
        regions = ['US', 'EU', 'APAC']

        df = pd.DataFrame({
            'ticker': [f'STOCK{i:03d}' for i in range(n_samples)],
            'sector': np.random.choice(sectors, n_samples),
            'region': np.random.choice(regions, n_samples),
            'last_price': np.random.uniform(50, 500, n_samples),
            'market_cap': np.random.lognormal(20, 2, n_samples),
            'p_e_ntm': np.random.uniform(5, 50, n_samples),
            'profit_margin': np.random.uniform(-0.1, 0.3, n_samples),
            'price_target': np.random.uniform(60, 550, n_samples)
        })

        # Build features
        X, y, numeric_features, categorical_features = build_features_and_target(df)

        print(f"\n📊 Dataset prepared:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")

        # Build preprocessing pipeline
        print(f"\n🔧 Building ColumnTransformer with separate transformers...")

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', StandardScaler(with_mean=False), numeric_features),
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 categorical_features)
            ],
            remainder='drop'
        )

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n📈 Training pipeline...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n✅ SUCCESS: Pipeline trained and evaluated")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

        # Verify transformation
        X_transformed = preprocessor.transform(X_test)
        print(f"\n🔄 Feature transformation verified:")
        print(f"  Original features: {X_test.shape[1]}")
        print(f"  Transformed features: {X_transformed.shape[1]}")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  One-hot encoded categorical: {X_transformed.shape[1] - len(numeric_features)}")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model_functions():
    """Verify that model functions use the feature lists correctly"""
    print("\n" + "=" * 80)
    print("VERIFICATION: Model functions with preprocessing")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from finance_ml.models import build_regression_pipeline

        # Test build_regression_pipeline
        numeric_features = ['last_price', 'market_cap', 'p_e_ntm']
        categorical_features = ['sector', 'region']

        print(f"\n🔧 Testing build_regression_pipeline...")
        print(f"  Numeric features: {numeric_features}")
        print(f"  Categorical features: {categorical_features}")

        pipeline = build_regression_pipeline(numeric_features, categorical_features, n_jobs=1)

        print(f"\n✅ SUCCESS: Pipeline created")
        print(f"  Pipeline steps: {[step[0] for step in pipeline.steps]}")

        # Verify preprocessor transformers
        preprocessor = pipeline.named_steps['preprocessor']
        print(f"\n🔍 Preprocessor transformers:")
        for name, transformer, columns in preprocessor.transformers:
            if name == 'remainder':
                continue
            print(f"  - {name}: {type(transformer).__name__} on {len(columns)} features")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE IMPROVEMENTS - VERIFICATION SUITE")
    print("=" * 80)

    results = {
        'Feature return signature': verify_feature_return_signature(),
        'Preprocessing pipeline': verify_preprocessing_pipeline(),
        'Model functions': verify_model_functions()
    }

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30}: {status}")

    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100

    print(f"\n{'=' * 80}")
    print(f"Overall: {passed}/{total} tests passed ({success_rate:.1f}%)")

    if success_rate == 100:
        print("🎉 All verifications passed!")
        print("✅ Preprocessing pipeline improvements are working correctly")
    else:
        print("⚠️ Some verifications failed - review output above")

    print("=" * 80)

    return success_rate == 100


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
