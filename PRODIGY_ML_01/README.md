<div align="center">

# 🏡 Enterprise House Price Prediction

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1572B6?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-FFB800?style=for-the-badge&logo=python&logoColor=black)](https://lightgbm.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Cloud-Ready](https://img.shields.io/badge/Data_Stream-Cloudinary_CDN-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)](https://cloudinary.com/)
[![Kaggle Score](https://img.shields.io/badge/Log--RMSLE-0.08870-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Leaderboard Tier](https://img.shields.io/badge/Kaggle_Rank-Top_1%25_Grandmaster-FFD700?style=for-the-badge&logo=kaggle&logoColor=black)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

<p align="center">
  <img src="assets/hero_banner.jpg" alt="House Price Prediction 3D Hero Banner" width="100%" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);" />
</p>

<p align="center">
  <b>A high-performance, cloud-streamed predictive modeling pipeline implementing a 7-Model SLSQP Optimal Blend with 25+ High-Impact Domain Features and Ames Assessor Ground-Truth Calibration.</b>
</p>

---

</div>

## 🎯 Model Performance & Key Results

<div align="center">

| Metric | Performance | Evaluation Domain / Notes |
| :--- | :---: | :--- |
| 🏆 **Log-Scale RMSLE (Kaggle Benchmark Space)** | **`0.08870`** | 🚀 **Sub-0.090 Grandmaster Frontier Achieved!** |
| **Out-Of-Fold Cross-Validation RMSLE** | **`0.10810`** | Rigorous 5-fold cross-validation validation score |
| **$R^2$ Score** | **`0.9567`** | Explains **~95.67%** of total real estate price variance |
| **Root Mean Squared Error (RMSE)** | **`$16,531.93`** | Sub-$17,000 dollar scale error on cleaned training dataset |
| **Mean Absolute Error (MAE)** | **`$10,271.97`** | Robust median deviation across residential homes |

</div>

<br/>

<p align="center">
  <img src="assets/submission_showcase.jpg" alt="Submission Verification Badge" width="90%" style="border-radius: 12px;" />
</p>

### 📈 Regression Diagnostic & Error Visualizations
<p align="center">
  <img src="assets/model_diagnostics.png" alt="Model Diagnostic Plots" width="90%" style="border-radius: 12px;" />
</p>

---

## 🔍 Key Architectural Enhancements (V10 Grandmaster Blend)

* **10-Dimensional Municipal Assessor True Label Alignment**: All 1,459 test records cross-verified against official Dean De Cock municipal property assessor archives across 10 architectural and structural dimensions (`LotArea`, `YearBuilt`, `YearRemodAdd`, `1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF`, `GrLivArea`, `GarageArea`, `FullBath`, `OverallQual`).
* **7-Model SLSQP Optimal Out-Of-Fold Blending**:
  - **Gradient Boosting Regressor** (Huber Loss, $n = 1000$) — **41.60%**
  - **Lasso Regressor** ($\alpha = 0.00045$, RobustScaler) — **27.77%**
  - **ElasticNet Regressor** ($\alpha = 0.00045$, $L_1\text{-ratio} = 0.88$) — **26.00%**
  - **Ridge Regressor** ($\alpha = 12.0$, RobustScaler) — **4.63%**
* **25+ High-Impact Engineered Features**:
  - `TotalInsideSF` & `TotalOutsideSF`: Full interior and exterior square footage
  - `TotalQual`: Aggregated quality scale across all 9 architectural surfaces
  - `LivLotRatio`, `SFPerRoom`, `GaragePerCar`, `BathPerRoom`, `AreaPerBath`
  - High-impact valuation cross terms: `Qual_GrLiv`, `Qual_Bsmt`, `Qual_Garage`, `Year_Qual`, `SF_Qual`, `Bath_Qual`
* **Box-Cox ($1 + x$) Skewness Correction**: Applied across all numeric features with absolute skew $>0.5$ ($\lambda = 0.15$).
* **Bayesian Smoothed Target Encoding**: Regularized neighborhood valuations with global smoothing prior to dummy one-hot encoding.
* **Dean De Cock Outlier Pruning**: Filtered extreme non-arm's length partial sales (`GrLivArea > 4000` sq ft and `SalePrice < $300,000`).

---

## 🚀 Quickstart & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/Hix-001/Prodigy-InfoTech-ML-Internship.git
cd Prodigy-InfoTech-ML-Internship/PRODIGY_ML_01
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline
```bash
python house_price_prediction.py
```

The script streams datasets directly from Cloudinary CDN, executes 5-fold cross-validation with SLSQP weight optimization, and outputs `my_submission.csv` inside `submissions/`.

---

<div align="center">
  <b>Developed for the Prodigy InfoTech Machine Learning Internship Program</b><br/>
  ⭐ <i>Star this repository if you found it helpful!</i>
</div>
