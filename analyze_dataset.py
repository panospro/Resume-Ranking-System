import pandas as pd
from collections import Counter
import re
from helper import load_dataframe_from_env

ResumeDatasetPath = load_dataframe_from_env("ResumeDatasetPath")

def load_dataset(path: str) -> pd.DataFrame:
    """Load resume dataset from CSV."""
    df = pd.read_csv(path)
    return df

def print_dataset_summary(df: pd.DataFrame) -> None:
    print("📌 ===== Dataset Summary =====")
    print(f"Total resumes: {len(df)}")

def show_resume_word_stats(df: pd.DataFrame) -> pd.Series:
    resume_lengths = df['Resume'].apply(lambda x: len(str(x).split()))
    print("\n📊 Resume Word Count Statistics:")
    print(resume_lengths.describe().round(2))
    df['WordCount'] = resume_lengths
    return resume_lengths

def show_top_categories(df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n📚 Top {n} Categories:")
    print(df['Category'].value_counts().head(n).to_string())

def show_common_words(df: pd.DataFrame, n: int = 20) -> None:
    print(f"\n🔤 Top {n} Most Common Words Across All Resumes:")
    all_words = " ".join(df['Resume']).lower()
    all_words = re.findall(r'\b\w+\b', all_words)
    common_words = Counter(all_words).most_common(n)
    for word, count in common_words:
        print(f"{word:<15} {count}")

def show_avg_length_per_category(df: pd.DataFrame) -> None:
    print("\n📏 Average Resume Word Count by Category:")
    avg_lengths = df.groupby('Category')['WordCount'].mean().sort_values(ascending=False)
    print(avg_lengths.round(2).to_string())

def main():
    df = load_dataset(ResumeDatasetPath)

    print_dataset_summary(df)
    show_resume_word_stats(df)
    show_top_categories(df)
    show_common_words(df)
    show_avg_length_per_category(df)


if __name__ == "__main__":
    main()
