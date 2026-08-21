"""
========================================================================================
🚀 PRODIGY INFOTECH - MACHINE LEARNING INTERNSHIP (TASK 01)
🏡 Enterprise-Grade House Price Prediction Pipeline with Linear Regression
========================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from scipy import stats


# =====================================================================
# 1. CLOUD DATA SOURCES CONFIGURATION
# =====================================================================
# Direct Cloudinary endpoints for seamless cross-platform execution (Linux / Mac / Windows / CI/CD)
DATASET_URLS: Dict[str, str] = {
    "train": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329704/train.csv",
    "test": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329703/test.csv",
    "sample_submission": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329703/sample_submission.csv",
    "data_description": "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787329700/data_description.txt"
}

# Relative Project Directories (Fully portable & GitHub-safe)
BASE_DIR: Path = Path(__file__).resolve().parent
SUBMISSIONS_DIR: Path = BASE_DIR / "submissions"
ASSETS_DIR: Path = BASE_DIR / "assets"


# =====================================================================
# 2. HIGH-PERFORMANCE DATA INGESTION
# =====================================================================
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Streams datasets directly from Cloudinary raw endpoints using pandas.
    """
    print("[1/5] Ingesting datasets from Cloud Storage...")
    print(f"  -> Train: {DATASET_URLS['train']}")
    print(f"  -> Test:  {DATASET_URLS['test']}")

    start_time = time.perf_counter()
    train_df = pd.read_csv(DATASET_URLS['train'])
    test_df = pd.read_csv(DATASET_URLS['test'])
    elapsed = time.perf_counter() - start_time

    print(f"  [+] Data streamed in {elapsed:.3f} seconds.")
    print(f"  [+] Training Shape: {train_df.shape} | Test Shape: {test_df.shape}\n")
    return train_df, test_df


# =====================================================================
# 3. ADVANCED FEATURE ENGINEERING & OUTLIER REMOVAL
# =====================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to capture non-linear relationships and interactions.
    """
    df = df.copy()
    
    # --- Size features ---
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
    df['TotalBath'] = (df['FullBath'] + 0.5*df['HalfBath'] + 
                       df['BsmtFullBath'].fillna(0) + 0.5*df['BsmtHalfBath'].fillna(0))
    df['TotalPorchSF'] = (df['OpenPorchSF'].fillna(0) + df['EnclosedPorch'].fillna(0) + 
                          df['3SsnPorch'].fillna(0) + df['ScreenPorch'].fillna(0))
    df['TotalSF_Sq'] = df['TotalSF'] ** 2
    
    # --- Age features ---
    df['HouseAge'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
    df['RemodAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)
    df['IsNew'] = (df['HouseAge'] <= 5).astype(int)
    df['IsRemodeled'] = (df['RemodAge'] < df['HouseAge']).astype(int)
    df['AgeRatio'] = np.where(df['HouseAge'] + 1 == 0, 0, df['RemodAge'] / (df['HouseAge'] + 1))
    
    # --- Quality interactions ---
    df['Qual_Area'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_Bath'] = df['OverallQual'] * df['TotalBath']
    df['Qual_Bed'] = df['OverallQual'] * df['BedroomAbvGr']
    df['Qual_Sq'] = df['OverallQual'] ** 2
    
    # --- Binary flags ---
    df['HasBasement'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int)
    df['Has2ndFlr'] = (df['2ndFlrSF'].fillna(0) > 0).astype(int)
    
    return df


def remove_outliers(df: pd.DataFrame, target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Removes extreme outliers using IQR method and domain knowledge.
    """
    df = df.copy()
    print(f"  Original shape: {df.shape}")
    
    # Remove extreme SalePrice outliers (1st and 99th percentile)
    q1 = df[target_col].quantile(0.01)
    q99 = df[target_col].quantile(0.99)
    df = df[(df[target_col] > q1) & (df[target_col] < q99)]
    
    # Remove extreme GrLivArea outliers (domain knowledge: > 4000 sq ft is rare)
    df = df[df['GrLivArea'] < 4000]
    
    # Remove properties with unrealistic TotalSF (less than 500 or more than 5000)
    df = df[df['GrLivArea'] > 500]
    df = df[df['TotalBsmtSF'].fillna(0) < 3000]
    
    print(f"  After outlier removal: {df.shape}")
    return df


# =====================================================================
# 4. VECTORIZED PREPROCESSING & PIPELINE SETUP
# =====================================================================
def build_and_transform_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "SalePrice"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, bool]:
    """
    Constructs an optimized scikit-learn transformer pipeline with median imputation
    and z-score standard scaling. Now includes log transformation for skewed features.
    """
    print("=" * 65)
    print("[2/5] FEATURE ENGINEERING & TRANSFORMATION PIPELINE")
    print("=" * 65)
    print(f"Selected Features : {feature_cols}")
    print(f"Target Column     : {target_col}\n")

    X_train_raw = train_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    y_train = train_df[target_col].values
    X_test_raw = test_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

    # Apply log transformation to skewed features for better linearity
    skewed_features = ['TotalSF', 'GrLivArea', 'TotalPorchSF']
    for feat in skewed_features:
        if feat in X_train_raw.columns:
            # Add small constant to avoid log(0)
            X_train_raw[feat] = np.log1p(np.clip(X_train_raw[feat].fillna(0), 0, None))
            X_test_raw[feat] = np.log1p(np.clip(X_test_raw[feat].fillna(0), 0, None))

    # Log transform target variable (crucial for Linear Regression)
    y_train = np.log1p(y_train)
    use_log_target = True

    # Preprocessing Pipeline (Impute + Scale)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_scaled = pipeline.fit_transform(X_train_raw)
    X_test_scaled = pipeline.transform(X_test_raw)

    return X_train_scaled, y_train, X_test_scaled, test_df['Id'], use_log_target


def select_top_features(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray,
    feature_names: List[str],
    k: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Selects top K features using SelectKBest with f_regression.
    """
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"  [+] Selected {len(selected_features)} features from {len(feature_names)}")
    return X_train_selected, X_test_selected, selected_features


# =====================================================================
# 5. MODEL TRAINING, EVALUATION & BENCHMARKING
# =====================================================================
def train_and_evaluate_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    feature_cols: List[str],
    use_log_target: bool = True
) -> Tuple[LinearRegression, np.ndarray, Dict[str, Any]]:
    """
    Fits standard OLS Linear Regression and computes RMSE, MAE, R-squared.
    Handles log-transformed target by converting predictions back to original scale.
    """
    print("=" * 65)
    print("[3/5] MODEL TRAINING & METRICS EVALUATION")
    print("=" * 65)

    start_time = time.perf_counter()
    model = LinearRegression()
    model.fit(X_train, y_train)
    fit_time_ms = (time.perf_counter() - start_time) * 1000

    # Predictions on training data (in log scale)
    y_train_pred_log = model.predict(X_train)

    # Convert predictions back to original scale
    if use_log_target:
        y_train_pred = np.expm1(y_train_pred_log)
        y_train_actual = np.expm1(y_train)
    else:
        y_train_pred = y_train_pred_log
        y_train_actual = y_train

    # Compute regression metrics (in original dollar scale)
    mse = mean_squared_error(y_train_actual, y_train_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train_actual, y_train_pred)
    r2 = r2_score(y_train_actual, y_train_pred)

    # Cross-validation on log scale
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    metrics = {
        "RMSE": rmse, 
        "MAE": mae, 
        "R2": r2, 
        "CV_RMSE": cv_rmse,
        "FitTime_ms": fit_time_ms
    }

    print(f"  [+] Training Latency : {metrics['FitTime_ms']:.2f} ms")
    print(f"  [+] R2 Score (Train) : {r2:.4f}")
    print(f"  [+] RMSE             : ${rmse:,.2f}")
    print(f"  [+] MAE              : ${mae:,.2f}")
    print(f"  [+] CV RMSE (log)    : {cv_rmse:.4f}\n")

    print("Learned Feature Weights (Top 10):")
    weights = sorted(zip(feature_cols, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:10]
    for name, weight in weights:
        print(f"  - {name:15s}: {weight:12.4f}")
    print(f"  - {'Intercept':15s}: {model.intercept_:12.4f}\n")

    return model, y_train_pred, metrics


# =====================================================================
# 6. DIAGNOSTICS & VISUALIZATION
# =====================================================================
def generate_diagnostics_plot(
    y_actual: np.ndarray, 
    y_pred: np.ndarray, 
    output_dir: Path
) -> Path:
    """
    Generates publication-quality regression diagnostics plots.
    """
    print("=" * 65)
    print("[4/5] GENERATING DIAGNOSTICS VISUALIZATIONS")
    print("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    diag_path = output_dir / "model_diagnostics.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted
    axes[0].scatter(y_actual, y_pred, alpha=0.5, color='#2563eb', edgecolors='none', s=30)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line')
    axes[0].set_title("Actual vs Predicted SalePrice ($)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Actual SalePrice ($)", fontsize=11)
    axes[0].set_ylabel("Predicted SalePrice ($)", fontsize=11)
    axes[0].legend()

    # 2. Residual Distribution
    residuals = y_actual - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='#dc2626', edgecolors='none', s=30)
    axes[1].axhline(0, color='black', linestyle='--', lw=1.8)
    axes[1].set_title("Residual Analysis (Errors vs Predicted)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted SalePrice ($)", fontsize=11)
    axes[1].set_ylabel("Residual Error ($)", fontsize=11)

    plt.tight_layout()
    plt.savefig(diag_path, dpi=250)
    plt.close()

    print(f"  [+] Diagnostics chart saved: {diag_path}\n")
    return diag_path


# =====================================================================
# 7. INFERENCE & SUBMISSION
# =====================================================================
def execute_submission_pipeline_with_preds(
    predictions: np.ndarray, 
    test_ids: pd.Series, 
    output_dir: Path
) -> pd.DataFrame:
    """
    Exports submission.csv with predictions already transformed.
    """
    print("=" * 65)
    print("[5/5] EXPORTING SUBMISSION")
    print("=" * 65)
    
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': np.round(predictions, 2)
    })
    
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "my_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"  [+] Submission exported to : {submission_path}")
    print("\nFirst 5 Records of Output Submission:")
    print(submission_df.head())
    return submission_df


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def main() -> None:
    total_start = time.perf_counter()
    
    # Step 1: Load data
    train_df, test_df = load_datasets()
    
    # Step 2: Feature Engineering
    print("=" * 65)
    print("[1.5/5] ENGINEERING FEATURES...")
    print("=" * 65)
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Step 3: Remove outliers from training data
    train_df = remove_outliers(train_df)
    print()
    
    # Step 4: Define expanded feature list
    numeric_features = ['TotalSF', 'TotalBath', 'TotalPorchSF', 'HouseAge', 'RemodAge',
                        'IsNew', 'IsRemodeled', 'AgeRatio', 'Qual_Area', 'Qual_Bath',
                        'Qual_Bed', 'Qual_Sq', 'HasBasement', 'HasGarage', 
                        'HasFireplace', 'Has2ndFlr', 'TotalSF_Sq', 'GrLivArea']
    
    # Add original base features (some may duplicate but that's fine)
    base_features = ['OverallQual', 'YearBuilt', 'FullBath', 'BedroomAbvGr']
    features = list(dict.fromkeys(base_features + numeric_features))  # Remove duplicates
    
    # Step 5: Transform
    X_train, y_train, X_test, test_ids, use_log_target = build_and_transform_features(
        train_df=train_df, 
        test_df=test_df, 
        feature_cols=features
    )
    
    # Step 6: Feature Selection (keep top 20)
    X_train, X_test, selected_features = select_top_features(
        X_train, y_train, X_test, features, k=20
    )
    
    # Step 7: Train & Benchmark
    model, y_train_pred, metrics = train_and_evaluate_model(
        X_train, y_train, selected_features, use_log_target
    )
    
    # Step 8: Diagnostic Visualizations
    # Get actual y_train in original scale for plotting
    y_train_actual = np.expm1(y_train) if use_log_target else y_train
    generate_diagnostics_plot(y_train_actual, y_train_pred, output_dir=SUBMISSIONS_DIR)
    
    # Step 9: Inference & Output
    # Predict on test set (in log scale)
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log) if use_log_target else test_preds_log
    
    execute_submission_pipeline_with_preds(test_preds, test_ids, output_dir=SUBMISSIONS_DIR)
    
    total_time = time.perf_counter() - total_start
    print(f"\n[DONE] End-to-End Pipeline Execution Time: {total_time:.3f} seconds.")


if __name__ == "__main__":
    main()
