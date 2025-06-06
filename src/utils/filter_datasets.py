import pandas as pd
from helper import load_dataframe_from_env

TECH_CATEGORIES = [
    "Java Developer", "Python Developer", "Web Designing", "DevOps Engineer",
    "Data Science", "Testing", "Automation Testing", "SAP Developer",
    "ETL Developer", "Hadoop", "Blockchain", "DotNet Developer", "Database",
    "Network Security Engineer"
]

def clean_resume_dataset(df: pd.DataFrame, min_words=150, max_words=800) -> pd.DataFrame:
    df["Normalized"] = df["Resume"].str.lower().str.strip()
    df = df.drop_duplicates(subset="Normalized").drop(columns="Normalized")
    # Filter by tech categories
    tech_df = df[df["Category"].isin(TECH_CATEGORIES)].copy()
    
    # Calculate word count
    tech_df["WordCount"] = tech_df["Resume"].apply(lambda x: len(str(x).split()))
    
    # Filter by word count
    tech_df = tech_df[(tech_df["WordCount"] >= min_words) & (tech_df["WordCount"] <= max_words)]
    
    return tech_df.reset_index(drop=True)

def clean_job_dataset(df: pd.DataFrame, min_words=100, max_words=1500) -> pd.DataFrame:
    # Normalize and remove duplicates
    df["Normalized"] = df["Job Description"].str.lower().str.strip()
    df = df.drop_duplicates(subset="Normalized").drop(columns="Normalized")

    # Calculate word count and store it
    df["WordCount"] = df["Job Description"].apply(lambda x: len(str(x).split()))

    # Filter by word count
    df = df[(df["WordCount"] >= min_words) & (df["WordCount"] <= max_words)]

    return df.reset_index(drop=True)

def main():
    # Clean resumes
    CLEANED_RESUME_PATH = "Cleaned_ResumeDataset.csv"
    ResumeDatasetPath = load_dataframe_from_env("ResumeDatasetPath")
    resume_df = pd.read_csv(ResumeDatasetPath)
    cleaned_resume_df = clean_resume_dataset(resume_df)
    cleaned_resume_df.to_csv(CLEANED_RESUME_PATH, index=False)
    print(f"✅ Saved {len(cleaned_resume_df)} tech resumes to '{CLEANED_RESUME_PATH}'")

    # Clean jd
    CLEANED_JD_PATH = "Cleaned_ResumeDataset.csv"
    JD_DATASET_PATH = load_dataframe_from_env("JDDatasetPath")
    job_df = pd.read_csv(JD_DATASET_PATH)
    cleaned_job_df = clean_job_dataset(job_df)
    cleaned_job_df.to_csv(CLEANED_JD_PATH, index=False)
    print(f"✅ Saved {len(cleaned_job_df)} cleaned job posts to '{CLEANED_JD_PATH}'")


if __name__ == "__main__":
    main()
