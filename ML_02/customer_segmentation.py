from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET_URL: str = "https://res.cloudinary.com/bo5xbnk7/raw/upload/v1787333989/Mall_Customers.csv"

BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
ASSETS_DIR: Path = BASE_DIR / "assets"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(url: str = DATASET_URL) -> pd.DataFrame:
    print("=" * 70)
    print("[1/6] INGESTING DATASET FROM CLOUD STORAGE")
    print("=" * 70)
    print(f"Source URL: {url}")
    
    start_time = time.perf_counter()
    df = pd.read_csv(url)
    elapsed = time.perf_counter() - start_time
    
    df.rename(columns={
        'CustomerID': 'CustomerID',
        'Gender': 'Gender',
        'Age': 'Age',
        'Annual Income (k$)': 'Annual_Income',
        'Spending Score (1-100)': 'Spending_Score'
    }, inplace=True)
    
    print(f"  [+] Data streamed in {elapsed:.3f} seconds.")
    print(f"  [+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def perform_eda(df: pd.DataFrame, output_dir: Path) -> None:
    print("=" * 70)
    print("[2/6] EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print("--- First 5 Records ---")
    print(df.head(), "\n")

    print("--- Dataset Information ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']], "\n")

    print("--- Missing Values Check ---")
    print(df.isnull().sum(), "\n")

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.countplot(data=df, x='Gender', ax=axes[0, 0], palette=['#3b82f6', '#ec4899'])
    axes[0, 0].set_title("Customer Gender Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Gender")
    axes[0, 0].set_ylabel("Customer Count")
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f"{int(p.get_height())}", (p.get_x() + 0.35, p.get_height() + 1))
    
    sns.histplot(df['Age'], kde=True, ax=axes[0, 1], color='#6366f1', bins=15)
    axes[0, 1].set_title("Age Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Age (Years)")
    
    sns.histplot(df['Annual_Income'], kde=True, ax=axes[1, 0], color='#10b981', bins=15)
    axes[1, 0].set_title("Annual Income (k$) Distribution", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Annual Income (k$)")
    
    sns.histplot(df['Spending_Score'], kde=True, ax=axes[1, 1], color='#f59e0b', bins=15)
    axes[1, 1].set_title("Spending Score (1-100) Distribution", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Spending Score")

    plt.tight_layout()
    eda_dist_path = output_dir / "eda_distributions.png"
    plt.savefig(eda_dist_path, dpi=200)
    plt.close()
    print(f"  [+] Saved EDA distribution plots: {eda_dist_path}")

    pairplot = sns.pairplot(df[['Gender', 'Age', 'Annual_Income', 'Spending_Score']], 
                            hue='Gender', palette={'Male': '#3b82f6', 'Female': '#ec4899'},
                            corner=False, diag_kind='kde')
    pairplot.fig.suptitle("Pairwise Feature Relationships by Gender", y=1.02, fontsize=13, fontweight='bold')
    eda_pair_path = output_dir / "eda_pairplot.png"
    pairplot.savefig(eda_pair_path, dpi=200)
    plt.close()
    print(f"  [+] Saved EDA pairplot: {eda_pair_path}")

    df_encoded = df.copy()
    df_encoded['Gender_Code'] = (df_encoded['Gender'] == 'Female').astype(int)
    numeric_cols = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Code']
    
    plt.figure(figsize=(8, 6))
    corr = df_encoded[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    eda_corr_path = output_dir / "eda_correlation_heatmap.png"
    plt.savefig(eda_corr_path, dpi=200)
    plt.close()
    print(f"  [+] Saved EDA correlation heatmap: {eda_corr_path}\n")

def prepare_feature_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    print("=" * 70)
    print("[3/6] FEATURE PREPROCESSING & STANDARDIZATION")
    print("=" * 70)

    df_proc = df.copy()
    df_proc['Gender_Code'] = (df_proc['Gender'] == 'Female').astype(int)
    print("  [+] Encoded 'Gender' -> 'Gender_Code' (Male=0, Female=1)")

    sets_definitions = {
        "Set 1 (Income & Spending)": ['Annual_Income', 'Spending_Score'],
        "Set 2 (Age, Income & Spending)": ['Age', 'Annual_Income', 'Spending_Score'],
        "Set 3 (All Features incl. Gender)": ['Gender_Code', 'Age', 'Annual_Income', 'Spending_Score']
    }

    scaler = StandardScaler()
    scaled_feature_sets: Dict[str, np.ndarray] = {}
    
    for name, features in sets_definitions.items():
        scaled_feature_sets[name] = scaler.fit_transform(df_proc[features])
        print(f"  [+] {name:35s}: Features {features} (Standardized: mean~0, std~1)")

    print()
    return df_proc, scaled_feature_sets

def evaluate_optimal_k(
    scaled_sets: Dict[str, np.ndarray], 
    k_range: range,
    output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    print("=" * 70)
    print("[4/6] EVALUATING OPTIMAL CLUSTERS (K) ACROSS FEATURE SETS")
    print("=" * 70)

    results: Dict[str, Dict[str, Any]] = {}
    fig, axes = plt.subplots(len(scaled_sets), 2, figsize=(14, 4.5 * len(scaled_sets)))

    for idx, (set_name, X) in enumerate(scaled_sets.items()):
        inertias = []
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=15, max_iter=300, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouettes.append(score)

        best_k_idx = int(np.argmax(silhouettes))
        best_k = list(k_range)[best_k_idx]
        best_silhouette = silhouettes[best_k_idx]

        results[set_name] = {
            'inertias': inertias,
            'silhouettes': silhouettes,
            'best_k': best_k,
            'best_silhouette': best_silhouette
        }

        print(f"  [>] {set_name}:")
        print(f"      Best K by Silhouette: {best_k} (Silhouette Score = {best_silhouette:.4f})")

        axes[idx, 0].plot(list(k_range), inertias, marker='o', color='#2563eb', lw=2)
        axes[idx, 0].set_title(f"{set_name}\nElbow Method (Inertia)", fontsize=11, fontweight='bold')
        axes[idx, 0].set_xlabel("Number of Clusters (K)")
        axes[idx, 0].set_ylabel("Inertia (WCSS)")
        axes[idx, 0].grid(True, linestyle=":", alpha=0.6)

        axes[idx, 1].plot(list(k_range), silhouettes, marker='s', color='#10b981', lw=2)
        axes[idx, 1].axvline(best_k, color='red', linestyle='--', label=f'Optimal K = {best_k}')
        axes[idx, 1].set_title(f"{set_name}\nSilhouette Score Curve", fontsize=11, fontweight='bold')
        axes[idx, 1].set_xlabel("Number of Clusters (K)")
        axes[idx, 1].set_ylabel("Silhouette Score")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    elbow_plot_path = output_dir / "elbow_and_silhouette_analysis.png"
    plt.savefig(elbow_plot_path, dpi=200)
    plt.close()
    print(f"\n  [+] Saved Elbow & Silhouette analysis chart: {elbow_plot_path}\n")

    return results

def compare_and_train_models(
    df: pd.DataFrame, 
    scaled_sets: Dict[str, np.ndarray], 
    eval_results: Dict[str, Dict[str, Any]]
) -> Tuple[str, KMeans, np.ndarray]:
    print("=" * 70)
    print("[5/6] MODEL TRAINING & FEATURE SET BENCHMARKING")
    print("=" * 70)

    comparison_table = []
    trained_models: Dict[str, Tuple[KMeans, np.ndarray]] = {}

    for set_name, X in scaled_sets.items():
        optimal_k = eval_results[set_name]['best_k']
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        
        trained_models[set_name] = (kmeans, labels)
        comparison_table.append({
            'Feature Set': set_name,
            'Optimal K': optimal_k,
            'Silhouette Score': score,
            'Inertia': kmeans.inertia_
        })

    comp_df = pd.DataFrame(comparison_table)
    print(comp_df.to_string(index=False))
    print()

    best_set_name = comp_df.sort_values(by='Silhouette Score', ascending=False).iloc[0]['Feature Set']
    best_model, best_labels = trained_models[best_set_name]
    
    print(f"  [*] WINNING FEATURE SET: '{best_set_name}'")
    print(f"      Optimal K: {eval_results[best_set_name]['best_k']} | Peak Silhouette: {eval_results[best_set_name]['best_silhouette']:.4f}\n")

    return best_set_name, best_model, best_labels

def generate_cluster_insights(
    df: pd.DataFrame,
    best_set_name: str,
    best_model: KMeans,
    best_labels: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    print("=" * 70)
    print("[6/6] CLUSTER INTERPRETATION & BUSINESS PROFILES")
    print("=" * 70)

    df_clustered = df.copy()
    df_clustered['Cluster'] = best_labels

    plt.figure(figsize=(10, 6))
    colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    
    sns.scatterplot(
        data=df_clustered,
        x='Annual_Income',
        y='Spending_Score',
        hue='Cluster',
        palette=colors[:best_model.n_clusters],
        s=90,
        alpha=0.9,
        edgecolor='k',
        linewidth=0.6
    )
    plt.title(f"Customer Clusters: Annual Income vs. Spending Score (K = {best_model.n_clusters})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=11)
    plt.ylabel("Spending Score (1-100)", fontsize=11)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    scatter_path = output_dir / "cluster_2d_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"  [+] Saved 2D cluster scatter plot: {scatter_path}")

    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_full)
    df_clustered['PCA1'] = pca_coords[:, 0]
    df_clustered['PCA2'] = pca_coords[:, 1]
    
    var_exp = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_clustered,
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        palette=colors[:best_model.n_clusters],
        s=90,
        alpha=0.9,
        edgecolor='k',
        linewidth=0.6
    )
    plt.title(f"PCA 2D Cluster Projection (Variance Explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.1f}%)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    pca_path = output_dir / "cluster_pca_projection.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"  [+] Saved PCA cluster projection: {pca_path}")

    profile = df_clustered.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income=('Annual_Income', 'mean'),
        Avg_Spending=('Spending_Score', 'mean'),
        Female_Pct=('Gender', lambda x: (x == 'Female').mean() * 100)
    ).reset_index()

    profile['Pop_Pct'] = (profile['Count'] / len(df_clustered)) * 100

    print("\n--- Cluster Mean Characteristics Profile ---")
    print(profile.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.barplot(data=profile, x='Cluster', y='Avg_Age', ax=axes[0], palette=colors[:best_model.n_clusters])
    axes[0].set_title("Average Age per Cluster", fontweight='bold')
    axes[0].set_ylabel("Age (Years)")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}", (p.get_x() + 0.25, p.get_height() + 0.5))

    sns.barplot(data=profile, x='Cluster', y='Avg_Income', ax=axes[1], palette=colors[:best_model.n_clusters])
    axes[1].set_title("Average Annual Income (k$)", fontweight='bold')
    axes[1].set_ylabel("Annual Income (k$)")
    for p in axes[1].patches:
        axes[1].annotate(f"${p.get_height():.1f}k", (p.get_x() + 0.2, p.get_height() + 0.8))

    sns.barplot(data=profile, x='Cluster', y='Avg_Spending', ax=axes[2], palette=colors[:best_model.n_clusters])
    axes[2].set_title("Average Spending Score (1-100)", fontweight='bold')
    axes[2].set_ylabel("Spending Score")
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.1f}", (p.get_x() + 0.25, p.get_height() + 0.8))

    plt.tight_layout()
    profile_chart_path = output_dir / "cluster_profile_barchart.png"
    plt.savefig(profile_chart_path, dpi=200)
    plt.close()
    print(f"  [+] Saved Cluster Profile Bar Charts: {profile_chart_path}")

    output_csv_path = output_dir / "segmented_mall_customers.csv"
    df_clustered.to_csv(output_csv_path, index=False)
    print(f"  [+] Saved segmented customer dataset: {output_csv_path}\n")

    personas = {
        0: "Moderate Middle-Class (Balanced Income, Balanced Spending)",
        1: "Target VIPs / Whales (High Income, High Spending)",
        2: "Impulsive Trendsetters (Low Income, High Spending)",
        3: "Frugal Affluents / Savers (High Income, Low Spending)",
        4: "Budget Economical (Low Income, Low Spending)"
    }
    
    print("=" * 70)
    print("CUSTOMER SEGMENTATION STRATEGIC PERSONAS")
    print("=" * 70)
    for c_id in range(best_model.n_clusters):
        c_row = profile[profile['Cluster'] == c_id].iloc[0]
        desc = personas.get(c_id, f"Customer Segment {c_id}")
        print(f"• Cluster {c_id} [{desc}]:")
        print(f"    - Size: {c_row['Count']} customers ({c_row['Pop_Pct']:.1f}%) | {c_row['Female_Pct']:.1f}% Female")
        print(f"    - Characteristics: Age={c_row['Avg_Age']:.1f} yrs | Income=${c_row['Avg_Income']:.1f}k | Spending Score={c_row['Avg_Spending']:.1f}/100")
        print()

    return df_clustered

def main() -> None:
    total_start = time.perf_counter()

    df = load_dataset(DATASET_URL)
    perform_eda(df, output_dir=OUTPUTS_DIR)
    df_proc, scaled_sets = prepare_feature_sets(df)
    eval_results = evaluate_optimal_k(scaled_sets, k_range=range(2, 11), output_dir=OUTPUTS_DIR)
    best_set_name, best_model, best_labels = compare_and_train_models(df, scaled_sets, eval_results)
    generate_cluster_insights(df, best_set_name, best_model, best_labels, output_dir=OUTPUTS_DIR)

    total_time = time.perf_counter() - total_start
    print(f"[DONE] End-to-End Segmentation Pipeline completed in {total_time:.3f} seconds.")

if __name__ == "__main__":
    main()
