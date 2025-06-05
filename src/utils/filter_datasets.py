import pandas as pd
from helper import load_dataframe_from_env

ResumeDatasetPath = load_dataframe_from_env("ResumeDatasetPath")
CLEANED_RESUME_PATH = "Cleaned_ResumeDataset.csv"

TECH_CATEGORIES = [
    "Java Developer", "Python Developer", "Web Designing", "DevOps Engineer",
    "Data Science", "Testing", "Automation Testing", "SAP Developer",
    "ETL Developer", "Hadoop", "Blockchain", "DotNet Developer", "Database",
    "Network Security Engineer"
]

def clean_resume_dataset(df: pd.DataFrame, min_words=150, max_words=800) -> pd.DataFrame:
    # Filter by tech categories
    tech_df = df[df["Category"].isin(TECH_CATEGORIES)].copy()
    
    # Calculate word count
    tech_df["WordCount"] = tech_df["Resume"].apply(lambda x: len(str(x).split()))
    
    # Filter by word count
    tech_df = tech_df[(tech_df["WordCount"] >= min_words) & (tech_df["WordCount"] <= max_words)]
    
    return tech_df.reset_index(drop=True)

def main():
    resume_df = pd.read_csv(ResumeDatasetPath)
    cleaned_resume_df = clean_resume_dataset(resume_df)
    cleaned_resume_df.to_csv(CLEANED_RESUME_PATH, index=False)
    print(f"✅ Saved {len(cleaned_resume_df)} tech resumes to '{CLEANED_RESUME_PATH}'")

if __name__ == "__main__":
    main()
