# Machine Learning Projects

A comprehensive collection of production-grade Machine Learning implementations, advanced predictive modeling architectures, unsupervised clustering pipelines, and competitive Kaggle benchmarks.

---

## 📌 Repository Overview

This repository contains end-to-end Machine Learning pipelines focusing on statistical rigour, high-precision feature engineering, ensemble optimization, and unsupervised pattern discovery.

| Module | Domain | Core Techniques / Models | Key Metric / Result |
| :--- | :--- | :--- | :--- |
| **[ML_01](./ML_01/)** | Advanced House Price Regression | SLSQP Scipy Ensemble (Ridge + ElasticNet + Lasso + SVR + LightGBM + XGBoost) | **0.00651 Kaggle Score (Rank 24 Top 0.6%)** |
| **[ML_02](./ML_02/)** | Unsupervised Customer Segmentation | K-Means Clustering, PCA Dimensionality Reduction, Silhouette Analysis | **Optimal $K=5$, Silhouette Score 0.554** |

---

## 📂 Repository Architecture

```text
Machine Learning/
├── ML_01/                                      # Advanced House Price Regression Pipeline
│   ├── assets/                                 # Visualizations and diagnostic artifacts
│   ├── submissions/                            # Kaggle submission archives & logs
│   ├── house_price_prediction.py               # Complete end-to-end regression script
│   ├── requirements.txt                        # Dedicated dependencies
│   └── README.md                               # Detailed task documentation & benchmarks
│
├── ML_02/                                      # Customer Segmentation Unsupervised Pipeline
│   ├── assets/                                 # High-resolution cluster plots & EDA figures
│   ├── outputs/                                # Analytical artifacts
│   ├── customer_segmentation.py                # End-to-end K-Means clustering script
│   ├── requirements.txt                        # Dedicated dependencies
│   └── README.md                               # Detailed clustering documentation
│
├── requirements.txt                            # Master environment dependencies
└── README.md                                   # Repository documentation
```

---

## 🚀 Projects Breakdown

### 1. [ML_01: Advanced House Price Regression](./ML_01/)
* **Objective**: Predict housing sale prices with minimal log-loss error using advanced exploratory data analysis, domain-driven feature engineering, and non-linear multi-model ensembling.
* **Methodology**:
  * Outlier curation and robust preprocessing pipeline.
  * Box-Cox target & skewed feature normalization.
  * Scipy SLSQP Quadratic Optimization for constrained meta-model blending (Ridge, Lasso, ElasticNet, Kernel Ridge, Gradient Boosting, XGBoost, LightGBM, CatBoost).
* **Performance**: Achieved **0.00651 RMSLE Kaggle Score**, placing in the **Top 24 worldwide**.

---

### 2. [ML_02: Customer Segmentation & Behavioral Clustering](./ML_02/)
* **Objective**: Discover distinct consumer spending personas and optimize customer relationship management strategy through unsupervised learning.
* **Methodology**:
  * Exploratory Data Analysis with correlation heatmaps and distribution profiling.
  * K-Means clustering with Elbow Method (Inertia WCSS) and Silhouette Coefficient validation.
  * PCA (Principal Component Analysis) 2D projection and cluster behavioral profiling.
* **Performance**: Identified 5 distinct customer segments with a validated **Silhouette Score of 0.554**.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hix-001/Machine-Learning.git
   cd "Machine Learning"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Any Module**:
   ```bash
   # Run House Price Regression Pipeline
   python ML_01/house_price_prediction.py

   # Run Customer Segmentation Clustering Pipeline
   python ML_02/customer_segmentation.py
   ```

---

## 👤 Author
* **Harsh Jha (Hix-001)**
* GitHub: [@Hix-001](https://github.com/Hix-001)
* Portfolio: [https://harshjha.in](https://harshjha.in)
