import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from helper import load_dataframe_from_env

JDDatasetPath = load_dataframe_from_env("JDDatasetPath")

# --- Load & Prepare Dataset ---
def load_and_prepare_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"])
    df.columns = ["Title", "Description"]
    df["DescLength"] = df["Description"].apply(lambda x: len(str(x).split()))
    return df

# --- Visualizations ---
def plot_top_job_titles(df: pd.DataFrame, top_n: int = 20):
    top_titles = df["Title"].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_titles.index, x=top_titles.values)
    plt.title(f"Top {top_n} Job Titles")
    plt.xlabel("Count")
    plt.ylabel("Job Title")
    plt.tight_layout()
    plt.show()

def plot_description_length_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.hist(df["DescLength"], bins=30, edgecolor='black')
    plt.title("Distribution of Job Description Lengths (Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Descriptions")
    plt.tight_layout()
    plt.show()

def plot_top_keywords(df: pd.DataFrame, max_features: int = 30):
    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(df["Description"])
    keywords = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    top_words = pd.Series(counts, index=keywords).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_words.values, y=top_words.index)
    plt.title(f"Top {max_features} Most Common Words in Job Descriptions")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_avg_length_by_title(df: pd.DataFrame, top_n: int = 20):
    avg_length = df.groupby("Title")["DescLength"].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_length.values, y=avg_length.index)
    plt.title("Average Description Length by Job Title")
    plt.xlabel("Average Word Count")
    plt.ylabel("Job Title")
    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    df_jd = load_and_prepare_dataset(JDDatasetPath)
    plot_top_job_titles(df_jd)
    plot_description_length_distribution(df_jd)
    plot_top_keywords(df_jd, max_features=30)
    plot_avg_length_by_title(df_jd)

if __name__ == "__main__":
    main()
