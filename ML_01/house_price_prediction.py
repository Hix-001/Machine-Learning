"""
========================================================================================
[PIPELINE] PRODIGY INFOTECH - TASK 01: ADVANCED REGRESSION KAGGLE GRANDMASTER (V8)
[ARCHITECTURE] 7-Model SLSQP Optimal Blend + 25+ High-Impact Domain Features
[GOAL] Target: Kaggle Test Leaderboard RMSLE < 0.10000
========================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.optimize import minimize

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & CLOUD DATA SOURCES
# =====================================================================
DATASET_URLS: Dict[str, str] = {
    "train": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329704/train.csv",
    "test": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329703/test.csv",
    "sample_submission": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329703/sample_submission.csv",
    "data_description": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329700/data_description.txt"
}

BASE_DIR: Path = Path(__file__).resolve().parent
SUBMISSIONS_DIR: Path = BASE_DIR / "submissions"
ASSETS_DIR: Path = BASE_DIR / "assets"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 2. DATA INGESTION & OUTLIER TREATMENT
# =====================================================================
def load_and_prune_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print("[1/5] Ingesting cloud datasets & pruning documented Ames outliers...")
    start = time.perf_counter()
    train = pd.read_csv(DATASET_URLS['train'])
    test = pd.read_csv(DATASET_URLS['test'])
    test_ids = test['Id']
    elapsed = time.perf_counter() - start
    print(f"  [+] Data streamed in {elapsed:.3f} seconds.")

    # Remove documented partial sale outliers recommended by Dean De Cock
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    train = train.drop(train[(train['TotalBsmtSF'] > 4000) & (train['SalePrice'] < 300000)].index)

    print(f"  [+] Cleaned Train Shape: {train.shape} | Test Shape: {test.shape}\n")
    return train, test, test_ids


# =====================================================================
# 3. ADVANCED 25+ DOMAIN FEATURE ENGINEERING
# =====================================================================
def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    print("[2/5] Creating 25+ advanced domain features & Box-Cox skewness correction...")
    
    y_train = np.log1p(train['SalePrice'].values)
    ntrain = len(train)
    
    all_data = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), test.drop(['Id'], axis=1)], ignore_index=True)

    # 1. NA Imputations
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'
    ]
    for col in none_cols:
        all_data[col] = all_data[col].fillna('None')

    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        all_data[col] = all_data[col].fillna(0)

    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    for col in ['KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'Functional', 'Utilities']:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # 2. Quality Quantifications & Aggregations
    qual_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for q in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'PoolQC']:
        all_data[q + '_Num'] = all_data[q].map(qual_mapping).fillna(0)

    all_data['TotalQual'] = (
        all_data['OverallQual'] +
        all_data['ExterQual_Num'] +
        all_data['ExterCond_Num'] +
        all_data['BsmtQual_Num'] +
        all_data['GarageQual_Num'] +
        all_data['FireplaceQu_Num'] +
        all_data['KitchenQual_Num'] +
        all_data['HeatingQC_Num'] +
        all_data['PoolQC_Num']
    )

    # 3. Area & Living Space Aggregations
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
    all_data['TotalInsideSF'] = all_data['TotalSF'] + all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']

    # 4. Bathrooms
    all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath']
    all_data['TotalBath2'] = all_data['FullBath'] + all_data['BsmtFullBath']

    # 5. Garage & Basement Finishes
    all_data['GarageAge'] = (all_data['YrSold'] - all_data['GarageYrBlt']).clip(lower=0)
    all_data['GarageScore'] = all_data['GarageCars'] * all_data['GarageArea']
    all_data['BsmtFinSF'] = all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']
    all_data['BsmtUnfToTotal'] = all_data['BsmtUnfSF'] / (all_data['TotalBsmtSF'] + 1)
    all_data['BsmtFinToTotal'] = all_data['BsmtFinSF'] / (all_data['TotalBsmtSF'] + 1)

    # 6. Age Features & Renovation Flags
    all_data['HouseAge'] = (all_data['YrSold'] - all_data['YearBuilt']).clip(lower=0)
    all_data['RenovationAge'] = (all_data['YrSold'] - all_data['YearRemodAdd']).clip(lower=0)
    all_data['AgeDiff'] = (all_data['YearRemodAdd'] - all_data['YearBuilt']).clip(lower=0)
    all_data['IsNew'] = (all_data['YrSold'] == all_data['YearBuilt']).astype(int)
    all_data['IsRenovated'] = (all_data['YearRemodAdd'] > all_data['YearBuilt']).astype(int)

    # 7. Predictive Real Estate Ratios
    all_data['LivLotRatio'] = all_data['GrLivArea'] / (all_data['LotArea'] + 1)
    all_data['SFPerRoom'] = all_data['GrLivArea'] / (all_data['TotRmsAbvGrd'] + 1)
    all_data['GaragePerCar'] = all_data['GarageArea'] / (all_data['GarageCars'] + 1)
    all_data['BathPerRoom'] = all_data['TotalBath'] / (all_data['TotRmsAbvGrd'] + 1)
    all_data['AreaPerBath'] = all_data['GrLivArea'] / (all_data['TotalBath'] + 1)

    # 8. High-Impact Valuation Interactions
    all_data['Qual_GrLiv'] = all_data['OverallQual'] * all_data['GrLivArea']
    all_data['Qual_Bsmt'] = all_data['OverallQual'] * all_data['TotalBsmtSF']
    all_data['Qual_Garage'] = all_data['OverallQual'] * all_data['GarageArea']
    all_data['Year_Qual'] = all_data['YearBuilt'] * all_data['OverallQual']
    all_data['SF_Qual'] = all_data['GrLivArea'] * all_data['OverallQual']
    all_data['Bath_Qual'] = all_data['TotalBath'] * all_data['OverallQual']

    # 9. Binary Indicator Flags
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasBasement'] = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
    all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
    all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
    all_data['HasPorch'] = (all_data['TotalPorchSF'] > 0).astype(int)

    # 10. Neighborhood Target Encoding with Bayesian Smoothing
    train_df_sub = all_data.iloc[:ntrain].copy()
    train_df_sub['y'] = y_train
    neigh_means = train_df_sub.groupby('Neighborhood')['y'].mean()
    neigh_counts = train_df_sub.groupby('Neighborhood')['y'].count()
    global_mean_y = y_train.mean()
    smoothed_neigh = (neigh_means * neigh_counts + global_mean_y * 10) / (neigh_counts + 10)
    all_data['Neighborhood_TargetEnc'] = all_data['Neighborhood'].map(smoothed_neigh).fillna(global_mean_y)

    # 11. Type conversions
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # 12. Box-Cox Power Transformations (λ=0.15)
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_features = skewed_feats[abs(skewed_feats) > 0.5].index

    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data = pd.get_dummies(all_data).reset_index(drop=True)
    all_data = all_data.fillna(all_data.median()).fillna(0)

    X_train = all_data.iloc[:ntrain].copy()
    X_test = all_data.iloc[ntrain:].copy()

    # Drop uninformative sparse features (>99.5% identical values)
    overfit = [col for col in X_train.columns if (X_train[col].value_counts().iloc[0] / len(X_train)) > 0.995]
    X_train = X_train.drop(overfit, axis=1)
    X_test = X_test.drop(overfit, axis=1)

    print(f"  [+] Grandmaster Feature Space: {X_train.shape[1]} features (pruned {len(overfit)} uninformative columns)\n")
    return X_train, X_test, y_train


# =====================================================================
# 4. CROSS-VALIDATION & 7-MODEL ENSEMBLE TRAINING
# =====================================================================
def train_and_optimize_blend(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    print("[3/5] Benchmarking 7-Model Ensemble with 5-Fold Cross Validation...")
    
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00045, max_iter=10000, random_state=1))
    elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00045, l1_ratio=0.88, max_iter=10000, random_state=3))
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=12, random_state=42))
    bayesian = make_pipeline(RobustScaler(), BayesianRidge(max_iter=500))

    gbr = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=3, max_features='sqrt',
        min_samples_leaf=12, min_samples_split=10, loss='huber', random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        colsample_bytree=0.45, gamma=0.045, learning_rate=0.02, max_depth=3,
        min_child_weight=1.8, n_estimators=1200, reg_alpha=0.45, reg_lambda=0.85,
        subsample=0.52, random_state=7, n_jobs=-1
    )

    lgb_model = lgb.LGBMRegressor(
        objective='regression', num_leaves=5, learning_rate=0.02, n_estimators=1000,
        max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.25,
        random_state=42, verbose=-1
    )

    models: Dict[str, Any] = {
        'Lasso': lasso,
        'ElasticNet': elastic,
        'Ridge': ridge,
        'BayesianRidge': bayesian,
        'GBR': gbr,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(y_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}

    print("\n--- 5-Fold Cross Validation Benchmarks ---")
    for name, model in models.items():
        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            oof_preds[name][val_idx] = val_pred
            scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        
        print(f"  [>] {name:15s}: 5-Fold CV RMSLE = {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")
        
    print("\n[4/5] Fitting all models on full training data...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_preds[name] = model.predict(X_test)

    # SLSQP Mathematical Optimization of Ensemble Blending Weights
    model_names = list(models.keys())
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])

    def obj_func(w):
        w_norm = w / np.sum(w)
        blend = np.dot(oof_matrix, w_norm)
        return np.sqrt(mean_squared_error(y_train, blend))

    init_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(obj_func, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    opt_w = res.x / np.sum(res.x)

    print(f"\n[OPTIMAL BLEND] Out-Of-Fold CV RMSLE = {res.fun:.5f}")
    print("SLSQP Blending Weights:")
    for name, w in zip(model_names, opt_w):
        print(f"  - {name:15s}: {w*100:.2f}%")

    test_matrix = np.column_stack([test_preds[name] for name in model_names])
    final_log_test = np.dot(test_matrix, opt_w)
    
    # Also calculate in-sample train predictions
    train_matrix = np.column_stack([model.predict(X_train) for model in models.values()])
    final_log_train = np.dot(train_matrix, opt_w)

    return models, final_log_train, final_log_test


# =====================================================================
# 5. DIAGNOSTICS & SUBMISSION EXPORT
# =====================================================================
def export_results(
    final_log_train: np.ndarray,
    final_log_test: np.ndarray,
    y_train: np.ndarray,
    test_ids: pd.Series,
    output_dir: Path
) -> None:
    print("\n[5/5] Generating diagnostics and test predictions...")
    
    y_train_actual = np.expm1(y_train)
    y_train_pred = np.expm1(final_log_train)
    
    train_rmsle = np.sqrt(mean_squared_error(y_train, final_log_train))
    train_dollar_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_dollar_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    
    print("=" * 70)
    print("[METRICS] V8 GRANDMASTER ENSEMBLE BENCHMARKS")
    print("=" * 70)
    print(f"  [+] Log-Scale RMSLE (Kaggle Metric):   {train_rmsle:.5f}")
    print(f"  [+] Actual Dollar RMSE:                ${train_dollar_rmse:,.2f}")
    print(f"  [+] Actual Dollar MAE:                 ${train_dollar_mae:,.2f}")
    print(f"  [+] R2 Score:                          {train_r2:.6f}")
    print("=" * 70)
    
    # Save diagnostic plot
    diag_path = output_dir / "model_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.set_theme(style="whitegrid")
    
    axes[0].scatter(y_train_actual, y_train_pred, alpha=0.5, color='#2563eb', edgecolors='none', s=25)
    min_v, max_v = min(y_train_actual.min(), y_train_pred.min()), max(y_train_actual.max(), y_train_pred.max())
    axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit')
    axes[0].set_title("Actual vs Predicted SalePrice ($)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual SalePrice ($)")
    axes[0].set_ylabel("Predicted SalePrice ($)")
    axes[0].legend()
    
    residuals = y_train_actual - y_train_pred
    axes[1].scatter(y_train_pred, residuals, alpha=0.5, color='#dc2626', edgecolors='none', s=25)
    axes[1].axhline(0, color='black', linestyle='--', lw=1.8)
    axes[1].set_title("Residual Distribution ($)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted SalePrice ($)")
    axes[1].set_ylabel("Residual ($)")
    
    plt.tight_layout()
    plt.savefig(diag_path, dpi=200)
    plt.close()
    print(f"  [+] Diagnostics exported: {diag_path}")
    
    # Exponentiate predictions to dollar scale
    final_dollar_test = np.expm1(final_log_test)
    
    sub = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': np.round(final_dollar_test, 2)
    })
    sub_path = output_dir / "my_submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  [+] Kaggle test submission saved: {sub_path}")
    print("\nFirst 5 Records of Output Submission:")
    print(sub.head())


# =====================================================================
# MAIN PIPELINE ENTRYPOINT
# =====================================================================
def main() -> None:
    total_start = time.perf_counter()
    print("=" * 70)
    print("[PIPELINE] PRODIGY INFOTECH - TASK 01: ADVANCED REGRESSION KAGGLE GRANDMASTER (V8)")
    print("[ARCHITECTURE] 7-Model SLSQP Optimal Blend + 25+ High-Impact Domain Features")
    print("=" * 70)
    
    train, test, test_ids = load_and_prune_data()
    X_train, X_test, y_train = engineer_features(train, test)
    models, train_preds, test_preds = train_and_optimize_blend(X_train, X_test, y_train)
    export_results(train_preds, test_preds, y_train, test_ids, output_dir=SUBMISSIONS_DIR)
    
    elapsed = time.perf_counter() - total_start
    print(f"\n[DONE] Pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
