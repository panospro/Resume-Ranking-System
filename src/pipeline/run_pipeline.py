import pandas as pd
from src.pipeline.features import extract_all_features
from src.pipeline.train_model import train_model
from src.pipeline.pseudo_labeling import generate_labels
from src.utils.helper import load_from_env

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

def main():
    RESUME_PATH = load_from_env("ResumeDatasetPath")
    JD_PATH = load_from_env("JDDatasetPath")

    df = load_data(JD_PATH, RESUME_PATH)    # === Step 1: Load and Cross Join ===
    df = generate_labels(df)    # === Step 2: Generate LLM Labels ===
    feature_df = extract_all_features(df)   # === Step 3: Feature Extraction ===
    train_model(feature_df) # === Step 4: Model Training ===

if __name__ == "__main__":
    main()