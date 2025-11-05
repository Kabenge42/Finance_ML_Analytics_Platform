### ml_finance_model_main_notebook_implementation guide.md

#### Purpose and scope

This guide describes how to upgrade the `ml_finance_model_main.ipynb` notebook into a robust, modular workflow aligned
with the project’s README and IMPROVEMENT_PLAN. It consolidates data into a single `all_stocks` dataframe, adds
validation, advanced feature engineering, multi-model training (classification + sector-optimized regression with
quantiles and stacking), and delivers rich analytics (mispricing, rankings, interactive visuals, and exportable
reports).

It also outlines how these steps map to maintainable `finance_ml` modules so logic can be reused by the CLI scripts and
tests.

---

### High-level architecture (modules and notebook sections)

- `finance_ml.data` — Data access, validation, and preprocessing
- `finance_ml.features` — Feature engineering and encoders
- `finance_ml.models` — Model definitions, training loops, CV strategies
- `finance_ml.eval` — Metrics, diagnostics, SHAP/feature importance
- `finance_ml.analytics` — Mispricing, rankings, interactive visuals
- `finance_ml.reporting` — Exports to CSV/Excel and JSON summaries

Notebook section map (cell groups):

1) Setup and config (env vars, paths, logging, seed)
2) Data management (DB/CSV loaders → `all_stocks`) + validation
3) EDA essentials (counts, missingness, correlations)
4) Feature engineering (ratios, volatility, encodings, interactions)
5) Event classification labels (leak-safe) + classifier training
6) Sector-optimized regression (with classification meta-features)
7) Quantile models and stacking ensembles
8) Evaluation and error analysis (global + per sector)
9) Analytics: mispricing scores, rankings, visuals
10) Reporting: CSV/Excel/JSON artifacts
11) Run summary, reproducibility, and next steps

Each section presents notebook-ready code that can later be promoted into `finance_ml` package modules.

---

### 0) Setup and configuration

- Python 3.10–3.11 recommended.
- Environment variables: see `environment_variables.txt`. Common: `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`, `RANDOM_SEED`,
  `N_JOBS`, `DB_URL`, `MODEL_VERSION` (e.g., `v8_3`).
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, optional `catboost`, `sqlalchemy`,
  `psycopg2-binary`, `plotly`, `matplotlib`, `seaborn`, `openpyxl`/`xlsxwriter`, `shap`.

Example notebook cell:

```python
import os
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)

# Optional gradient boosting backends
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except Exception:
    CatBoostClassifier = CatBoostRegressor = Pool = None

# DB access
try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None

import logging

warnings.filterwarnings("ignore")

# Config and paths
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v8_3")
N_JOBS = int(os.getenv("N_JOBS", 4))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("finance_ml")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

np.random.seed(RANDOM_SEED)
```

---

### 1) Data management: load, validate, and prepare `all_stocks`

Goals:

- Load from PostgreSQL if available, else CSV fallback.
- Normalize column names to Python-friendly format while retaining original columns if helpful for reporting.
- Coerce numerics with `pd.to_numeric(errors='coerce')` and perform NA handling.
- Deduplicate on `Ticker` with Region/Exchange preference.
- Validate schema, basic quality checks, and output a `data_quality.json`.

Notebook cells (promotable to `finance_ml.data`):

```python
EXPECTED_REGIONS = {"US", "EU", "APAC", "ROTW"}
CRITICAL_COLS = ["Ticker", "Region", "Sector", "Last_Price"]

DB_URL = os.getenv("DB_URL")  # e.g., 'postgresql+psycopg2://postgres:@localhost:5432/postgres'

REGION_FILES = {
    "US": DATA_DIR / "screening_us.csv",
    "EU": DATA_DIR / "screening_eu.csv",
    "APAC": DATA_DIR / "screening_apac.csv",
    "ROTW": DATA_DIR / "screening_rotw.csv",
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve a copy of original names for reporting if needed
    df = df.copy()
    df.columns_original = df.columns
    df.columns = (
        df.columns
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return df


def load_from_db(limit: int | None = None) -> pd.DataFrame:
    assert create_engine is not None, "SQLAlchemy is required for DB access"
    assert DB_URL, "DB_URL env var must be set for DB access"
    engine = create_engine(DB_URL)
    q = 'SELECT * FROM equities WHERE "Region" IN (\'US\',\'EU\',\'APAC\',\'ROTW\')'
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql(q, engine)
    return df


def load_from_csvs() -> pd.DataFrame:
    parts = []
    for region, path in REGION_FILES.items():
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            # Normalize NULLs
            df = df.replace({"": np.nan})
            # Backfill Region if missing
            if "Region" in df.columns:
                df.loc[df["Region"].isna(), "Region"] = region
            else:
                df["Region"] = region
            parts.append(df)
        else:
            logger.warning(f"Missing CSV for region {region}: {path}")
    if not parts:
        raise FileNotFoundError("No CSVs found in data/ for regions US/EU/APAC/ROTW")
    return pd.concat(parts, ignore_index=True)


# Auto data-source selection
def load_all_stocks(limit: int | None = None, prefer_db: bool = True) -> pd.DataFrame:
    df = None
    if prefer_db and DB_URL and create_engine is not None:
        try:
            logger.info("Loading from PostgreSQL via DB_URL...")
            df = load_from_db(limit=limit)
        except Exception as e:
            logger.warning(f"DB load failed ({e}); falling back to CSVs.")
    if df is None:
        logger.info("Loading from CSVs...")
        df = load_from_csvs()
    return df


raw = load_all_stocks(limit=None)
logger.info(f"Raw rows loaded: {len(raw):,}")

# Basic validation
issues = {}
missing_critical = raw[CRITICAL_COLS].isna().any().to_dict()
issues["missing_critical_columns"] = missing_critical

region_counts = raw.get("Region").value_counts(dropna=False).to_dict() if "Region" in raw.columns else {}
issues["region_counts"] = region_counts

# Normalize columns for processing
all_stocks = normalize_columns(raw)

# Coerce numerics (heuristic: try to_numeric for any column with >60% numeric-like values)
num_like_cols = []
for c in all_stocks.columns:
    if all_stocks[c].dropna().empty:
        continue
    sample = all_stocks[c].dropna().astype(str).str.replace(",", "", regex=False)
    frac_num = sample.str.match(r"^[+-]?(\d+\.?\d*|\.\d+)$").mean()
    if frac_num > 0.6:
        num_like_cols.append(c)
all_stocks[num_like_cols] = all_stocks[num_like_cols].apply(pd.to_numeric, errors="coerce")

# Filter unusable rows
all_stocks = all_stocks.dropna(subset=["ticker", "sector", "last_price"])  # keys for modeling

# Deduplicate by ticker with region/exchange preference
PREFERRED_REGION_ORDER = ["US", "EU", "APAC", "ROTW"]
all_stocks["region"] = all_stocks.get("region", pd.Series(index=all_stocks.index))
all_stocks["_region_rank"] = all_stocks["region"].map({r: i for i, r in enumerate(PREFERRED_REGION_ORDER)}).fillna(999)

all_stocks = (all_stocks
              .sort_values(["ticker", "_region_rank"])  # add exchange priority if available
              .drop_duplicates(subset=["ticker"], keep="first")
              .drop(columns=["_region_rank"], errors="ignore")
              .reset_index(drop=True)
              )

# Save quality report
with open(OUTPUT_DIR / "data_quality.json", "w", encoding="utf-8") as f:
    json.dump(issues, f, indent=2)

logger.info(f"Prepared all_stocks: {all_stocks.shape}")
```

Optional: add SQLite quick path as in the project’s addendum when PostgreSQL isn’t available.

---

### 2) EDA essentials (quick sanity checks)

Keep EDA lightweight to support batch runs, but provide helpful snapshots.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Counts by region and sector
region_counts = all_stocks["region"].value_counts().sort_index()
sector_counts = all_stocks["sector"].value_counts().sort_index()

logger.info({"region_counts": region_counts.to_dict()})
logger.info({"sector_counts": sector_counts.to_dict()})

# Missingness overview (top 20 most-missing columns)
missing_rate = all_stocks.isna().mean().sort_values(ascending=False).head(20)
print("Top missingness:")
print(missing_rate)

# Correlations for selected core numeric fields
core_num_cols = [c for c in all_stocks.columns if all_stocks[c].dtype.kind in "if"]
if len(core_num_cols) >= 5:
    corr = all_stocks[core_num_cols].corr(method="spearman").clip(-1, 1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Spearman correlations (core numeric)")
    plt.tight_layout()
    plt.show()
```

---

### 3) Feature engineering (baseline + sector-aware)

Goals: robust, leak-safe features from fundamentals and categorical context.

Key families:

- Ratios: `ev_over_ebitda`, `net_debt_over_ebitda`, `p_e`, `p_b`, `p_tbv`, margins (gross/ebitda/net)
- Growth: revenue CAGR (e.g., 1Y/3Y), `eps_growth`
- Risk/volatility: rolling volatility if time series present; otherwise proxies from snapshot (e.g., beta if available)
- Size: market cap buckets, log transforms
- Categorical: one-hot for `sector`, target-encoding for `industry` with CV
- Interactions: `region x sector`, `size_class x sector`
- Scaling: robust or standard scaling per numeric block; winsorization

Notebook cells (promotable to `finance_ml.features`):

```python
def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_div(a, b):
        return np.where((b == 0) | np.isnan(b), np.nan, a / b)

    # Examples (only create if source columns exist)
    colmap = df.columns
    if set(["enterprise_value", "ebitda"]).issubset(colmap):
        df["ev_over_ebitda"] = safe_div(df.get("enterprise_value"), df.get("ebitda"))
    if set(["net_debt", "ebitda"]).issubset(colmap):
        df["net_debt_over_ebitda"] = safe_div(df.get("net_debt"), df.get("ebitda"))
    if set(["last_price", "eps_ttm"]).issubset(colmap):
        df["p_e"] = safe_div(df.get("last_price"), df.get("eps_ttm"))
    if set(["last_price", "book_value_per_share"]).issubset(colmap):
        df["p_b"] = safe_div(df.get("last_price"), df.get("book_value_per_share"))
    if set(["tangible_book_value_per_share", "last_price"]).issubset(colmap):
        df["p_tbv"] = safe_div(df.get("last_price"), df.get("tangible_book_value_per_share"))
    # Margins
    if set(["ebitda", "revenue"]).issubset(colmap):
        df["ebitda_margin"] = safe_div(df.get("ebitda"), df.get("revenue"))
    if set(["net_income", "revenue"]).issubset(colmap):
        df["net_margin"] = safe_div(df.get("net_income"), df.get("revenue"))
    # Log transforms for skewed metrics
    for c in ["market_cap", "enterprise_value", "revenue", "ebitda"]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    return df


# Winsorization helper
def winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    if s.dropna().empty:
        return s
    lo, hi = s.quantile([lower, upper])
    return s.clip(lower=lo, upper=hi)


# Apply feature building
features = add_financial_ratios(all_stocks)

# Sector-wise winsorization for selected numeric columns
wins_cols = [c for c in ["ev_over_ebitda", "net_debt_over_ebitda", "p_e", "p_b", "p_tbv",
                         "ebitda_margin", "net_margin", "log_market_cap", "log_enterprise_value"] if
             c in features.columns]
if wins_cols:
    features[wins_cols] = (features
                           .groupby("sector")[wins_cols]
                           .transform(lambda g: g.apply(winsorize_series)))

# Categorical encodings
cat_cols = [c for c in ["sector", "industry", "region"] if c in features.columns]
num_cols = [c for c in features.columns if features[c].dtype.kind in "if" and c not in wins_cols] + wins_cols
num_cols = sorted(set(num_cols))

# Basic ColumnTransformer (OHE for cats, robust scaling for nums)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocess = ColumnTransformer([
    ("num", RobustScaler(), num_cols),
    ("cat", ohe, cat_cols)
    ], remainder="drop")
```

Notes:

- If time series prices are accessible, add recent rolling volatility windows (e.g., 21d, 63d) prior to deduplication.
- Optionally add target encoding for `industry` using out-of-fold means of training target with KFold to prevent
  leakage.

---

### 4) Event classification labels and training

Define a 3-class event label using only information available at or before the training snapshot.

Possible label design (customize to your columns):

- 0 = Neutral
- 1 = Positive catalyst (e.g., analyst target raised vs. prior, improved consensus rating, or volatility spike down)
- 2 = Negative catalyst (e.g., target cut, rating downgrade, or volatility spike up)

Notebook cells (promotable to `finance_ml.models.events`):

```python
# Example label construction (customize to data availability)
labels = features.copy()

# Heuristics based on available columns
labels["pt_change"] = np.nan
if {"price_target", "price_target_ytd_ago"}.issubset(labels.columns):
    labels["pt_change"] = labels["price_target"] - labels["price_target_ytd_ago"]

labels["rating_change"] = 0.0
if {"analyst_rating", "analyst_rating_ytd_ago"}.issubset(labels.columns):
    labels["rating_change"] = labels["analyst_rating_ytd_ago"] - labels["analyst_rating"]  # improvement if positive

# Volatility proxy (if beta or stdev available)
vol_proxy = None
for vcol in ["beta", "volatility_63d", "volatility_21d"]:
    if vcol in labels.columns:
        vol_proxy = vcol
        break


def label_row(row):
    score = 0
    if not np.isnan(row.get("pt_change", np.nan)):
        if row["pt_change"] > 0:
            score += 1
        elif row["pt_change"] < 0:
            score -= 1
    if "rating_change" in row and not np.isnan(row["rating_change"]):
        if row["rating_change"] > 0:
            score += 1
        elif row["rating_change"] < 0:
            score -= 1
    if vol_proxy and not np.isnan(row.get(vol_proxy, np.nan)):
        # Treat higher vol as negative catalyst
        if row[vol_proxy] > np.nanmedian(labels[vol_proxy]):
            score -= 1
        else:
            score += 0  # neutral/weak positive
    # Map to 0/1/2
    return 2 if score < 0 else (1 if score > 0 else 0)


labels["event_class"] = labels.apply(label_row, axis=1)

# Train-validation split with GroupKFold by ticker to avoid leakage
groups = labels["ticker"]
X = labels
y = labels["event_class"].astype(int)

# Build numerical/categorical transformer from prior step
clf_preprocess = preprocess  # reuse from features section

# Choose a classifier backend
use_lgb = lgb is not None
use_xgb = xgb is not None and not use_lgb

if use_lgb:
    clf_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=N_JOBS,
            )
elif use_xgb:
    clf_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            scale_pos_weight=None,  # class weights handled separately if needed
            )
else:
    from sklearn.ensemble import RandomForestClassifier

    clf_model = RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, n_jobs=N_JOBS,
                                       class_weight="balanced")

clf = Pipeline(steps=[("prep", clf_preprocess), ("clf", clf_model)])

cv = GroupKFold(n_splits=5)
probas_oof = np.zeros((len(X), 3))
for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups))):
    logger.info(f"Training event classifier fold {fold + 1}/5...")
clf.fit(X.iloc[tr], y.iloc[tr])
probas = clf.predict_proba(X.iloc[va])
probas_oof[va] = probas

labels[["p_neutral", "p_pos", "p_neg"]] = probas_oof  # order as produced
```

The predicted class probabilities (`p_neutral`, `p_pos`, `p_neg`) will feed the regression models as meta-features.

---

### 5) Sector-optimized regression (targets + meta-features)

Targets: choose one primary target (e.g., `price_target` or `price_target_median`).

- Train one regressor per `sector` with core engineered features + event-class probabilities.
- Use Gradient Boosting (LightGBM/XGBoost/CatBoost) and/or ElasticNet baselines.
- Apply `GroupKFold` (by `ticker`) to avoid leakage; compute OOF predictions for stacking.

Notebook cells (promotable to `finance_ml.models.regression`):

```python
TARGET_COL = "price_target" if "price_target" in features.columns else None
assert TARGET_COL is not None, "Target column not found. Provide price_target or price_target_median."

reg_base = features.join(labels[["p_neutral", "p_pos", "p_neg"]])
reg_base = reg_base.dropna(subset=[TARGET_COL])

sectors = sorted(reg_base["sector"].dropna().unique())

sector_models = {}
sector_metrics = {}
reg_oof = pd.Series(index=reg_base.index, dtype=float)

for sector in sectors:
    df_s = reg_base[reg_base["sector"] == sector].copy()
    if len(df_s) < 100:  # skip tiny sectors or adjust threshold
        logger.info(f"Skipping small sector: {sector} ({len(df_s)})")
        continue

    y = df_s[TARGET_COL].astype(float)
    X = df_s.drop(columns=[TARGET_COL])

    # Rebuild preprocess for sector (can share same transformers)
    cat_cols_s = [c for c in ["sector", "industry", "region"] if c in X.columns]
    num_cols_s = [c for c in X.columns if X[c].dtype.kind in "if"]
    preprocess_s = ColumnTransformer([
        ("num", RobustScaler(), num_cols_s),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_s)
        ], remainder="drop")

    # Choose regressor backend
    if lgb is not None:
        reg_model = lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                )
    elif xgb is not None:
        reg_model = xgb.XGBRegressor(
                n_estimators=1300,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                )
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        reg_model = HistGradientBoostingRegressor(max_depth=None, l2_regularization=1.0, random_state=RANDOM_SEED)

    pipe = Pipeline(steps=[("prep", preprocess_s), ("reg", reg_model)])

    cv = GroupKFold(n_splits=5)
    groups = df_s["ticker"]

    preds_s = pd.Series(index=df_s.index, dtype=float)
    fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
        logger.info(f"Sector {sector}: training reg fold {fold + 1}/5...")
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[va])
        preds_s.iloc[va] = p
        mae = mean_absolute_error(y.iloc[va], p)
        rmse = mean_squared_error(y.iloc[va], p, squared=False)
        r2 = r2_score(y.iloc[va], p)
        fold_metrics.append({"mae": mae, "rmse": rmse, "r2": r2})

    sector_models[sector] = pipe
    reg_oof.loc[df_s.index] = preds_s
    # Aggregate metrics
    sector_metrics[sector] = {
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "n": int(len(df_s)),
        }

logger.info({"regression_sector_metrics": sector_metrics})
```

---

### 6) Quantile regression and stacking ensembles

Add predictive intervals and stacking to improve robustness.

Quantile LightGBM example (if available):

```python
quantile_models = {}
quantiles = [0.1, 0.5, 0.9]

if lgb is not None:
    for sector in sectors:
        if sector not in sector_models:  # skipped small sectors
            continue
        df_s = reg_base[reg_base["sector"] == sector]
        y = df_s[TARGET_COL].astype(float)
        X = df_s.drop(columns=[TARGET_COL])

        cat_cols_s = [c for c in ["sector", "industry", "region"] if c in X.columns]
        num_cols_s = [c for c in X.columns if X[c].dtype.kind in "if"]
        preprocess_s = ColumnTransformer([
            ("num", RobustScaler(), num_cols_s),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_s)
            ], remainder="drop")

        q_models = {}
        for q in quantiles:
            # LGBM with quantile objective
            q_model = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                        n_estimators=800, learning_rate=0.05, num_leaves=63,
                                        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
                                        n_jobs=N_JOBS)
            q_models[q] = Pipeline(steps=[("prep", preprocess_s), ("reg", q_model)])
        quantile_models[sector] = q_models
```

Simple stacking skeleton (OOF meta-features from base learners):

```python
# Suppose we have two base regressors (lgb and xgb) — create OOF preds and train a meta-ridge
from sklearn.linear_model import Ridge

stack_oof = pd.DataFrame(index=reg_base.index)
stack_meta = {}

# Generate base OOFs (example with lgb-only fallback already stored in reg_oof)
stack_oof["base_gbm"] = reg_oof

# Meta learner per sector
for sector in sectors:
    idx = reg_base[reg_base["sector"] == sector].index
    y = reg_base.loc[idx, TARGET_COL].astype(float)
    X_meta = stack_oof.loc[idx]
    if X_meta.isna().any().any() or len(X_meta) < 50:
        continue
    meta = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    cv = GroupKFold(n_splits=5)
    groups = reg_base.loc[idx, "ticker"]
    preds_meta = pd.Series(index=idx, dtype=float)
    for tr, va in cv.split(X_meta, y, groups=groups):
        meta.fit(X_meta.iloc[tr], y.iloc[tr])
        preds_meta.iloc[va] = meta.predict(X_meta.iloc[va])
    stack_meta[sector] = meta

# Combined prediction preference: meta if available else base
final_oof = stack_oof["base_gbm"].copy()
for sector, meta in stack_meta.items():
    idx = reg_base[reg_base["sector"] == sector].index
    final_oof.loc[idx] = preds_meta.loc[idx]
```

---

### 7) Evaluation and diagnostics

Global and per-sector metrics and plots. Include residuals, error buckets, and SHAP (if supported).

```python
# Global metrics
true_vals = reg_base.loc[reg_oof.index, TARGET_COL].astype(float)
mae = mean_absolute_error(true_vals, reg_oof)
rmse = mean_squared_error(true_vals, reg_oof, squared=False)
r2 = r2_score(true_vals, reg_oof)
logger.info({"regression_overall": {"mae": mae, "rmse": rmse, "r2": r2}})

# Residuals per sector
residuals = true_vals - reg_oof
res_by_sector = residuals.groupby(reg_base.loc[reg_oof.index, "sector"]).agg(
        ["mean", "median", "std", "count"]).to_dict()
logger.info({"residuals_by_sector": res_by_sector})

# Optional: SHAP summary for one trained sector model (requires shap and tree-based model)
try:
    import shap

    if lgb is not None and sectors:
        s0 = sectors[0]
        pipe = sector_models.get(s0)
        if pipe is not None and hasattr(pipe.named_steps["reg"], "booster_"):
            # Build a small background set for SHAP
            df_s = reg_base[reg_base["sector"] == s0]
            X_s = df_s.drop(columns=[TARGET_COL])
            explainer = shap.Explainer(pipe.named_steps["reg"].booster_)
            # IMPORTANT: explain on the numeric/cat-transformed space is preferable; here we show model-native tree explainer
            shap_values = explainer(pipe.named_steps["prep"].transform(X_s.head(500)))
            shap.plots.beeswarm(shap_values, max_display=20)
except Exception as e:
    logger.warning(f"SHAP skipped: {e}")
```

---

### 8) Analytics: mispricing, rankings, interactive visuals

Compute mispricing and produce ranked tables and charts.

```python
pred_col = "predicted_target_oof"
reg_base[pred_col] = reg_oof

# Mispricing score
reg_base["mispricing_score"] = (reg_base[pred_col] - reg_base["last_price"]) / reg_base["last_price"]


# Rankings
def top_n_by_sector(df, n=10, undervalued=True):
    order = df["mispricing_score"].sort_values(ascending=not undervalued)
    idx = order.groupby(df["sector"]).head(n).index
    return df.loc[idx].sort_values(["sector", "mispricing_score"], ascending=[True, undervalued])


undervalued = top_n_by_sector(reg_base, n=10, undervalued=True)
overvalued = top_n_by_sector(reg_base, n=10, undervalued=False)

# Interactive visuals with plotly
import plotly.express as px

fig1 = px.scatter(
        reg_base, x="last_price", y=pred_col, color="sector", hover_data=["ticker", "region"],
        title="Predicted Target vs Last Price"
        )
fig1.show()

fig2 = px.bar(
        undervalued.sort_values("mispricing_score", ascending=False).head(30),
        x="ticker", y="mispricing_score", color="sector", title="Top Undervalued (Mispricing Score)"
        )
fig2.show()
```

---

### 9) Reporting: CSV/Excel/JSON artifacts

Mirror the example `Stock_Prediction_Analysis_Report_YYYYMMDD_HHMMSS.xlsx` with multiple sheets.

```python
ts = time.strftime("%Y%m%d_%H%M%S")
report_path = OUTPUT_DIR / f"Stock_Prediction_Analysis_Report_{ts}.xlsx"

sheets = {
    "summary_metrics": pd.DataFrame([
        {"metric": "MAE", "value": mae},
        {"metric": "RMSE", "value": rmse},
        {"metric": "R2", "value": r2},
        ]),
    "sector_metrics": pd.DataFrame.from_dict(sector_metrics, orient="index").reset_index().rename(
        columns={"index": "sector"}),
    "undervalued": undervalued,
    "overvalued": overvalued,
    "all_predictions": reg_base[["ticker", "sector", "region", "last_price", pred_col, "mispricing_score"]].sort_values(
        "mispricing_score", ascending=False)
    }

with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name[:31], index=False)

# JSON summaries
with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump({"region_counts": region_counts.to_dict(), "sector_counts": sector_counts.to_dict()}, f, indent=2)

reg_base[["ticker", "sector", "region", "last_price", pred_col, "mispricing_score"]]
    .to_csv(OUTPUT_DIR / "regression_predictions.csv", index=False)

logger.info(f"Report written: {report_path}")
```

---

### 10) Reproducibility, configuration, and performance

- Use `RANDOM_SEED`, `N_JOBS`, and `MODEL_VERSION` env vars.
- Prefer `GroupKFold` by `ticker` to avoid leakage.
- Consider caching transformed matrices for large datasets (e.g., `joblib.dump` of preprocessed arrays).
- Start with fewer features, then scale up; monitor memory and use `RobustScaler` for heavy-tailed data.
- Optional GPU: enable for XGBoost/LightGBM if configured.

---

### 11) Testing alignment

- Ensure functions are factored to be importable by existing tests (e.g., `tests/test_features.py`,
  `tests/test_preprocess_and_training.py`).
- Keep DB-dependent code optional; enable via `DB_URL` when available.
- Run tests:

```bash
python -m unittest -v
```

- Coverage (option A):

```bash
pip install coverage
coverage run -m unittest -v
coverage report -m
```

---

### 12) Versioning and documentation

- Bump `MODEL_VERSION` when materially changing modeling behavior (e.g., `v8_3`).
- Update: `README.md`, `IMPROVEMENT_PLAN.md`, `environment_variables.txt` (if new vars), and any CLI references.
- Log key metric deltas (MAE/RMSE/R2) and notable drivers from SHAP/feature importance.

---

### 13) Implementation checklist (acceptance criteria)

- Data management
    - [ ] Auto-loads from DB (when `DB_URL` set) else CSV fallback
    - [ ] Consolidates into `all_stocks` with normalized columns and types
    - [ ] Saves `data_quality.json` with critical checks
- Feature engineering
    - [ ] Creates core ratios, margins, log-size, sector-wise winsorization
    - [ ] ColumnTransformer with numeric scaling + categorical OHE
    - [ ] Optional target encoding (leak-safe) for `industry`
- Models
    - [ ] Event classifier trained with `GroupKFold`; saves OOF class probabilities
    - [ ] Sector regressors trained with `GroupKFold`; produces OOF predictions
    - [ ] Optional quantile models per sector
    - [ ] Optional stacking meta-learner on OOF predictions
- Evaluation
    - [ ] Overall MAE/RMSE/R2 and per-sector metrics
    - [ ] Residual diagnostics; optional SHAP
- Analytics & Reporting
    - [ ] Mispricing score computed; top-N under/overvalued per sector
    - [ ] Interactive visuals (Plotly)
    - [ ] Exports: `regression_predictions.csv`, `eda_summary.json`, and Excel report with multiple sheets
- Reproducibility & Docs
    - [ ] `MODEL_VERSION` bumped and recorded; guide synced with README and IMPROVEMENT_PLAN

---

### 14) How to run the updated notebook (Windows PowerShell)

1) Create and activate venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install/upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

3) Install dependencies

```powershell
pip install -r requirements.txt
# Optional for DB
pip install psycopg2-binary SQLAlchemy
# Optional for analysis
pip install shap xlsxwriter plotly
```

4) Ensure data availability

- PostgreSQL path: load equities via the provided SQL scripts, then set `DB_URL` (e.g.,
  `postgresql+psycopg2://postgres:@localhost:5432/postgres`).
- Or place CSVs under `data/` as `screening_us.csv`, `screening_eu.csv`, `screening_apac.csv`, `screening_rotw.csv`.

5) Launch Jupyter and run cells in order

```powershell
jupyter notebook ml_finance_model_main.ipynb
```

- Confirm outputs under `outputs/`: `data_quality.json`, `eda_summary.json`, `regression_predictions.csv`, and
  `Stock_Prediction_Analysis_Report_*.xlsx`.

6) Run tests (optional but recommended)

```powershell
python -m unittest -v
```

---

### 15) Notes and alignment with reference materials

- The feature engineering and model selection reference best practices from:
    - Hands-on ML chapters (classification, linear models, SVM, trees/ensembles, dimensionality reduction)
    - TensorFlow/Keras chapters are optional; for tabular finance data the boosting models generally perform strongly.
- Consider benchmarking a simple ElasticNet and RandomForest baseline to validate gains from GBMs and stacking.
- For dimensionality reduction, PCA can be explored for high-cardinality one-hot expansions.

---

### 16) Future extensions (roadmap hooks)

- Time-aware splitting by snapshot date if multiple snapshots become available.
- Integration with SQLite import tools for lightweight local runs.
- AutoML-style hyperparameter sweeps with constrained budgets.
- Probability calibration for the event classifier and quantile coverage calibration for regression.

---

### Quick summary of key improvements to make in ml_finance_model_main.ipynb

- Introduce a unified data loader with DB→CSV fallback and robust validation → `all_stocks` dataframe
- Add comprehensive, sector-aware feature engineering and encodings with leak-safe practices
- Add multi-class event classifier; feed class probabilities into sector-specific regressors
- Add quantile prediction and optional stacking for improved accuracy and uncertainty estimates
- Provide thorough evaluation, diagnostics, and SHAP insights
- Deliver analytics: mispricing scores, stock rankings, and interactive plots
- Export professional Excel/CSV/JSON artifacts mirroring the provided report example
- Ensure reproducibility with env vars, seeds, and versioning; align with README and IMPROVEMENT_PLAN

This guide can be pasted (as-is) into a new repository file named
`ml_finance_model_main_notebook_implementation guide.md`, and the code cells can be integrated into
`ml_finance_model_main.ipynb` step by step.