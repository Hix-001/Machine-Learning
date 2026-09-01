# ML_02: Unsupervised Customer Segmentation Pipeline

A robust, production-ready unsupervised machine learning pipeline for retail customer segmentation using K-Means clustering, PCA dimensionality reduction, and multi-metric cluster validation.

---

## 📊 Key Highlights & Results

* **Optimal Number of Clusters ($K$)**: **`5 Clusters`**
* **Silhouette Coefficient**: **`0.554`** (high inter-cluster separation & intra-cluster cohesion)
* **Davies-Bouldin Index**: **`0.572`** (optimal compactness)
* **Techniques**: MinMax/Standard feature scaling, Elbow Method (WCSS Inertia), Silhouette Analysis, PCA 2D feature projection.

---

## 🔍 Identified Customer Personas

| Cluster | Segment Persona | Annual Income (k$) | Spending Score (1-100) | Recommended Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Balanced / Standard** | Moderate ($40k–$70k) | Moderate (40–60) | Cross-selling, loyalty rewards programs |
| **1** | **Target / High Value** | High ($70k–$140k) | High (65–100) | Premium concierge, luxury product launches |
| **2** | **Frugal / Savers** | High ($70k–$140k) | Low (1–40) | Value-oriented promotions, investment tiers |
| **3** | **Enthusiasts / Spenders** | Low ($15k–$40k) | High (60–100) | Micro-promotions, seasonal trend merchandising |
| **4** | **Conservative / Budget** | Low ($15k–$40k) | Low (1–40) | Discount alerts, essential bundles |

---

## 📁 Directory Structure

```text
ML_02/
├── assets/                                 # Cluster scatter plots, heatmaps, and EDA figures
├── outputs/                                # Analytical artifacts and cluster assignments
├── customer_segmentation.py                # End-to-end unsupervised pipeline
├── requirements.txt                        # Dedicated dependencies
└── README.md                               # Project documentation
```

---

## 🛠️ Execution Guide

```bash
# Navigate to ML_02 directory
cd ML_02

# Install requirements
pip install -r requirements.txt

# Run the complete segmentation pipeline
python customer_segmentation.py
```

---

## 👤 Author
* **Harsh Jha (Hix-001)**
* GitHub: [@Hix-001](https://github.com/Hix-001)
* Portfolio: [https://harshjha.in](https://harshjha.in)
