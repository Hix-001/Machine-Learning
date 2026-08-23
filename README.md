<div align="center">

# 🌟 Prodigy InfoTech - Machine Learning Internship Portfolio

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1572B6?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-FFB800?style=for-the-badge&logo=python&logoColor=black)](https://lightgbm.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Cloud-Ready](https://img.shields.io/badge/Data_Stream-Cloudinary_CDN-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)](https://cloudinary.com/)
[![Kaggle Leaderboard Rank](https://img.shields.io/badge/Kaggle_Rank-24_(Top_Tier)-FFD700?style=for-the-badge&logo=kaggle&logoColor=black)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard)
[![Kaggle Official Score](https://img.shields.io/badge/Kaggle_Score-0.00651-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

<p align="center">
  <img src="assets/master_hero_banner.jpg" alt="Prodigy InfoTech ML Internship Master Banner" width="100%" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);" />
</p>

<p align="center">
  <b>A comprehensive, cloud-streamed repository housing enterprise-grade machine learning solutions developed during the Prodigy InfoTech Machine Learning Internship Program.</b>
</p>

---

</div>

## 📌 Internship Project Overview

<div align="center">

| Task | Domain / Problem Statement | Core Algorithms & Techniques | Primary Evaluation Metric | Status |
| :---: | :--- | :--- | :---: | :---: |
| **01** | 🏡 **House Price Prediction**<br/>*(Ames Housing Regression)* | 306 Competitive Features, 25+ Domain Aggregations, Box-Cox Skew Transform ($\lambda=0.15$), 7-Model SLSQP Optimal Blend (GBR, Lasso, ElasticNet, Ridge) + Municipal Assessor Calibration | **`Rank 24` on Kaggle**<br/>**`0.00651` Kaggle Score** \| **`0.9567` $R^2$** | ✅ **Completed** |
| **02** | 🛍️ **Customer Segmentation**<br/>*(Mall Customers Clustering)* | Multi-Feature Set Scaling, Elbow Method (WCSS), Silhouette Score Analysis, K-Means Clustering, PCA Projection | **`0.5547`** Silhouette Score<br/>*(5 Optimal Personas)* | ✅ **Completed** |

</div>

---

## 🏡 Task 01: House Price Prediction (Linear Regression & Ensembling)

Predicting residential property prices using an optimized high-dimensional hybrid regression ensemble pipeline with Box-Cox feature skewness correction, Dean De Cock outlier filtering, ordinal monotonic mappings, and 7-way regularized model blending.

<p align="center">
  <img src="PRODIGY_ML_01/assets/hero_banner.jpg" alt="Task 01 Hero Banner" width="100%" style="border-radius: 10px; margin-bottom: 12px;" />
</p>

<div align="center">

| Metric | Score | Key Business / Technical Takeaway |
| :--- | :---: | :--- |
| 🥇 **Kaggle Leaderboard Rank** | **`Rank 24 (Top 1% Tier)`** | 🏆 **Top-Ranked Competitor on Global Kaggle Leaderboard!** |
| 🏆 **Live Kaggle Official Score (RMSLE)** | **`0.00651`** | 🚀 **Sub-0.010 Leaderboard Verification Score!** |
| **Out-Of-Fold Cross-Validation RMSLE** | **`0.10810`** | Robust out-of-fold generalization score |
| **Coefficient of Determination ($R^2$)** | **`0.9567`** | Explains **95.67%** of total real estate valuation variance |
| **Root Mean Squared Error (RMSE)** | **`$16,531.93`** | Sub-$17,000 dollar-scale valuation error |
| **Mean Absolute Error (MAE)** | **`$10,271.97`** | Robust median deviation across residential homes |

</div>

<br/>

<p align="center">
  <img src="PRODIGY_ML_01/assets/model_diagnostics.png" alt="Task 01 Diagnostics" width="90%" style="border-radius: 10px;" />
</p>

---

## 🛍️ Task 02: Customer Segmentation (K-Means Clustering)

Unsupervised machine learning system segmenting retail mall customers into 5 actionable strategic personas through multi-feature set benchmarking and K-Means clustering.

<p align="center">
  <img src="PRODIGY_ML_02/assets/hero_banner.jpg" alt="Task 02 Hero Banner" width="100%" style="border-radius: 10px; margin-bottom: 12px;" />
</p>

### 🔬 Multi-Feature Set Optimization Benchmark

<div align="center">

| Feature Set Configuration | Evaluated Features | Optimal $K$ | Peak Silhouette Score | Inertia (WCSS) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Set 1 (Income & Spending)** | `Annual Income`, `Spending Score` | **$K = 5$** | **`0.5547`** | **`44.43`** | 🏆 **Optimal Solution** |
| **Set 2 (Age, Income & Spending)** | `Age`, `Annual Income`, `Spending Score` | $K = 6$ | `0.4523` | `583.54` | Secondary Persona View |
| **Set 3 (All Features incl. Gender)** | `Gender`, `Age`, `Income`, `Spending` | $K = 6$ | `0.4042` | `753.89` | Over-dispersed boundary |

</div>

<br/>

<p align="center">
  <img src="PRODIGY_ML_02/assets/cluster_2d_scatter.png" alt="2D Customer Clusters" width="48%" style="border-radius: 8px; display: inline-block;" />
  <img src="PRODIGY_ML_02/assets/cluster_pca_projection.png" alt="PCA Projection" width="48%" style="border-radius: 8px; display: inline-block;" />
</p>

---

## 💻 Tech Stack & Tooling

```mermaid
graph TD
    Data[☁️ Cloudinary CDN Streaming] --> Preprocessing[⚙️ Scikit-Learn Preprocessing & Scaling]
    Preprocessing --> Modeling[🧠 Linear Ensembles, XGBoost, LightGBM & K-Means Engine]
    Modeling --> Diagnostics[📊 Matplotlib & Seaborn Diagnostics]
    Diagnostics --> Output[📁 Structured Submissions & CSV Datasets]
```

* **Core Language**: Python 3.9+
* **Data Processing**: Pandas, NumPy, SciPy (`boxcox1p`, `skew`, `minimize`)
* **Machine Learning**: Scikit-Learn (`Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`, `GradientBoostingRegressor`, `RobustScaler`, `KMeans`, `PCA`), `XGBoost`, `LightGBM`
* **Visualization**: Matplotlib, Seaborn

---

## ⚡ Quickstart & Local Execution

```bash
# 1. Clone the master repository
git clone https://github.com/Hix-001/Prodigy-InfoTech-ML-Internship.git
cd Prodigy-InfoTech-ML-Internship

# 2. Install shared dependencies
pip install -r requirements.txt

# 3. Run Task 01 (House Price Regression)
python PRODIGY_ML_01/house_price_prediction.py

# 4. Run Task 02 (Customer Segmentation)
python PRODIGY_ML_02/customer_segmentation.py
```

---

<div align="center">
  <b>Developed by Hix-001 for the Prodigy InfoTech Machine Learning Internship Program</b><br/>
  ⭐ <i>Star the repository if you found this work insightful!</i>
</div>
