import pandas as pd
from src.pipeline.features import extract_all_features
from src.pipeline.train_model import train_model
from src.pipeline.pseudo_labeling import generate_labels
from src.utils.helper import load_from_env
import seaborn as sns
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt

def load_data(jd_path, resume_path):
    jd_df = pd.read_csv(jd_path)
    resume_df = pd.read_csv(resume_path)

    jd_df = jd_df.rename(columns={
        jd_df.columns[0]: "JD_ID",
        jd_df.columns[1]: "Job Title",
        jd_df.columns[2]: "Job Description"
    })
    resume_df = resume_df.reset_index().rename(columns={"index": "Resume_ID"})

    # --- Cross join JD and resumes ---
    jd_df["key"] = 1
    resume_df["key"] = 1
    pair_df = pd.merge(jd_df, resume_df, on="key").drop(columns="key")

    return pair_df

def run_data_quality_checks(df: pd.DataFrame):
    print("🧪 Running Data Quality Checks...\n")
    
    # === 1. Missing Values ===
    print("🔍 Missing values per column:")
    print(df.isnull().sum())
    print("\n" + "="*50)

    # === 2. Constant / Low-Cardinality Features ===
    def safe_nunique(df):
        result = {}
        for col in df.columns:
            try:
                result[col] = df[col].nunique()
            except TypeError:
                result[col] = "❌ Unhashable (e.g. list/array)"
        return pd.Series(result)

    n_unique = safe_nunique(df)
    print("🧂 Features with low cardinality (≤ 2 unique values):")
    print(n_unique[n_unique.apply(lambda x: isinstance(x, int) and x <= 2)])
    print("\n" + "="*50)

    # === 3. Feature Distributions ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Label", errors="ignore")
    print("📊 Feature distributions:")
    df[numeric_cols].hist(figsize=(16, 10), bins=30)
    plt.suptitle("Histograms of Numerical Features")
    plt.tight_layout()
    plt.show()

    skew_vals = df[numeric_cols].skew().sort_values(ascending=False)
    print("📐 Skewness of features (high skew > 1):")
    print(skew_vals[skew_vals > 1])
    print("\n" + "="*50)

    # === 4. Target Distribution ===
    if "Label" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["Label"], kde=True)
        plt.title("Label Distribution")
        plt.tight_layout()
        plt.show()
    
    # === 5. Outliers with Boxplots and Z-Score ===
    print("🚨 Outlier Analysis using Boxplots:")

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5))

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)

    # Hide any empty subplots
    total_axes = n_rows * n_cols
    for j in range(len(numeric_cols), total_axes):
        row, col_idx = divmod(j, n_cols)
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)

    plt.suptitle("Boxplots of Numerical Features", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    z_scores = np.abs(zscore(df[numeric_cols]))
    outliers = (z_scores > 3).sum(axis=0)
    print("⚠️ Outlier count per feature (z-score > 3):")
    print(outliers[outliers > 0])
    print("\n" + "="*50)

    # === 6. Correlation Matrix ===
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

def transform_skewed_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Apply log1p to features with values ≥ 0
    log_features = [
        "soft_skill_coverage_ratio",
        "soft_skill_precision_ratio",
        "tech_skill_precision_ratio",
        "tech_skill_coverage_ratio",
        "jd_length",
        "keyword_frequency_density"
    ]

    for col in log_features:
        df[col] = np.log1p(df[col])

    # Apply sqrt to count-like features
    sqrt_features = [
        "soft_matching_skill_count",
        "tech_matching_skill_count"
    ]

    for col in sqrt_features:
        df[col] = np.sqrt(df[col])

    return df

def drop_extreme_outliers(df: pd.DataFrame, numeric_cols: list, z_thresh: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    z_scores = np.abs(zscore(df[numeric_cols]))
    mask = (z_scores < z_thresh).all(axis=1)
    dropped_count = (~mask).sum()
    print(f"🧹 Dropping {dropped_count} rows with extreme outliers (z > {z_thresh})")
    return df[mask]

def clip_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    return df

def clean_dataset(feature_df):
    # Step 1: Transform skewed features
    feature_df = transform_skewed_features(feature_df)

    # Step 2: Clip remaining outliers
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.drop(["Label", "JD_ID", "Resume_ID"], errors="ignore")
    feature_df = clip_outliers(feature_df, numeric_cols)

    # Step 3: Drop rows with extreme z-score outliers
    feature_df = drop_extreme_outliers(feature_df, numeric_cols)

    return feature_df

def main(skip_to_training=True):
    if skip_to_training:
        print("Loading pre-extracted features...")
        feature_df = pd.read_parquet("feature_df.parquet")

    else:
        RESUME_PATH = load_from_env("ResumeDatasetPath")
        JD_PATH = load_from_env("JDDatasetPath")
        df = load_data(JD_PATH, RESUME_PATH)  # Step 1
        df = generate_labels(df)              # Step 2
        feature_df = extract_all_features(df) # Step 3

        # Save to reuse
        feature_df.to_parquet("feature_df.parquet")

    feature_df = clean_dataset(feature_df)  # Step 4

    # run_data_quality_checks(feature_df) # Step 5
    train_model(feature_df) # Step 6

if __name__ == "__main__":
    main()