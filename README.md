<div align="center">

# 🚀 Prodigy InfoTech — Machine Learning Internship Portfolio

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Cloud-Ready](https://img.shields.io/badge/Data_Stream-Cloudinary_CDN-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)](https://cloudinary.com/)
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
| **01** | 🏡 **House Price Prediction**<br/>*(Ames Housing Regression)* | Advanced Feature Engineering, IQR Trimming, Log1p Target Transform, SelectKBest, OLS Linear Regression | **`0.16256` RMSLE** *(Kaggle Live)*<br/>**`0.8822` $R^2$** | ✅ **Completed** |
| **02** | 🛍️ **Customer Segmentation**<br/>*(Mall Customers Clustering)* | Multi-Feature Set Scaling, Elbow Method (WCSS), Silhouette Score Analysis, K-Means Clustering, PCA Projection | **`0.5547`** Silhouette Score<br/>*(5 Optimal Personas)* | ✅ **Completed** |

</div>

---

## 🏡 Task 01: House Price Prediction (Linear Regression)

Predicting residential property prices using an optimized linear regression pipeline with high-dimensional feature engineering and logarithmic target normalization.

<p align="center">
  <img src="PRODIGY_ML_01/assets/hero_banner.jpg" alt="Task 01 Hero Banner" width="100%" style="border-radius: 10px; margin-bottom: 12px;" />
</p>

<div align="center">

| Metric | Score | Key Business / Technical Takeaway |
| :--- | :---: | :--- |
| 🏆 **Kaggle Leaderboard Score (RMSLE)** | **`0.16256`** | Top-tier competitive baseline on 1,459 test houses |
| **Coefficient of Determination ($R^2$)** | **`0.8822`** | Explains **88.22%** of total real estate valuation variance |
| **Root Mean Squared Error (RMSE)** | **`$23,489.44`** | Reduced baseline error by over 41% via interaction features |
| **5-Fold Cross-Validation (log RMSE)** | **`0.1329`** | Verified consistent generalization across all cross-validation folds |

</div>

<br/>

<p align="center">
  <img src="PRODIGY_ML_01/assets/model_diagnostics.png" alt="Task 01 Diagnostics" width="90%" style="border-radius: 10px;" />
</p>

- 🔗 **Full Project Details & Source**: [`PRODIGY_ML_01/`](./PRODIGY_ML_01)
- 📜 **Pipeline Script**: [`PRODIGY_ML_01/house_price_prediction.py`](./PRODIGY_ML_01/house_price_prediction.py)

---

## 🛍️ Task 02: Customer Segmentation (K-Means Clustering)

Unsupervised behavioral clustering on retail shoppers across multi-feature spaces with Elbow and Silhouette coefficient validation.

<p align="center">
  <img src="PRODIGY_ML_02/assets/hero_banner.jpg" alt="Task 02 Hero Banner" width="100%" style="border-radius: 10px; margin-bottom: 12px;" />
</p>

<div align="center">

| Cluster ID | Persona Archetype | Size (% Share) | Avg Age | Avg Income | Avg Spending Score | Dominant Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | 🎯 **Moderate Middle-Class** | 81 (40.5%) | 42.7 yrs | $55.3k | 49.5 / 100 | Loyalty Rewards |
| **1** | 💎 **Target VIPs / Whales** | 39 (19.5%) | 32.7 yrs | $86.5k | 82.1 / 100 | Luxury Exclusives |
| **2** | 🚀 **Impulsive Trendsetters** | 22 (11.0%) | 25.3 yrs | $25.7k | 79.4 / 100 | Flash Sales & Social |
| **3** | 🛡️ **Frugal Affluents** | 35 (17.5%) | 41.1 yrs | $88.2k | 17.1 / 100 | High-Utility Value |
| **4** | 🏷️ **Budget Economical** | 23 (11.5%) | 45.2 yrs | $26.3k | 20.9 / 100 | Clearance & Bulk |

</div>

<br/>

<p align="center">
  <img src="PRODIGY_ML_02/assets/cluster_2d_scatter.png" alt="2D Clusters" width="49%" style="border-radius: 10px;" />
  <img src="PRODIGY_ML_02/assets/cluster_pca_projection.png" alt="PCA Cluster Space" width="49%" style="border-radius: 10px;" />
</p>

<p align="center">
  <img src="PRODIGY_ML_02/assets/elbow_and_silhouette_analysis.png" alt="Elbow & Silhouette Multi-Feature Comparison" width="98%" style="border-radius: 10px;" />
</p>

- 🔗 **Full Project Details & Source**: [`PRODIGY_ML_02/`](./PRODIGY_ML_02)
- 📜 **Clustering Script**: [`PRODIGY_ML_02/customer_segmentation.py`](./PRODIGY_ML_02/customer_segmentation.py)

---

## 🚀 Quickstart & Reproduction

### 1. Clone the master repository
```bash
git clone https://github.com/Hix-001/Prodigy-InfoTech-ML-Internship.git
cd Prodigy-InfoTech-ML-Internship
```

### 2. Install shared dependencies
```bash
pip install -r requirements.txt
```

### 3. Run individual tasks
```bash
# Run Task 01: House Price Linear Regression Pipeline
python PRODIGY_ML_01/house_price_prediction.py

# Run Task 02: Mall Customers K-Means Segmentation Engine
python PRODIGY_ML_02/customer_segmentation.py
```

*All datasets stream automatically over high-speed Cloudinary CDNs with zero manual file downloads required.*

---

## 🛠️ Technology Stack

- **Core Runtime**: Python 3.9+
- **Data Engineering**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-Learn (`LinearRegression`, `KMeans`, `PCA`, `SelectKBest`, `StandardScaler`, `Pipeline`)
- **Data Visualization**: Matplotlib, Seaborn
- **Cloud Infrastructure**: Cloudinary Raw CDN Storage

---

<div align="center">
  <b>Developed by Hix-001 for the Prodigy InfoTech Machine Learning Internship</b><br/>
  ⭐ <i>Star this repository if you find it helpful!</i>
</div>
