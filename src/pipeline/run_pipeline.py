import pandas as pd
from src.pipeline.features import extract_all_features
# from src.pipeline.reg_train_model import train_model
from src.pipeline.clas_train_model import train_model
from src.pipeline.pseudo_labeling import generate_labels
from src.utils.helper import load_from_env
from src.pipeline.feature_cleaning import clean_dataset, run_data_quality_checks

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
    skip_to_training = True
    skip_labels = True
    use_colab_bert = True

    if skip_to_training:
        print("Loading pre-extracted features...")
        feature_df = pd.read_json("feature_df.jsonl", lines=True)

    else:
        RESUME_PATH = load_from_env("ResumeDatasetPath")
        JD_PATH = load_from_env("JDDatasetPath")
        df = load_data(JD_PATH, RESUME_PATH)  # Step 1

        df = generate_labels(df, skip_labels)  # Step 2, true skips labels
        df.to_parquet("google_colab.parquet", index=False)

        feature_df = extract_all_features(df, use_colab_bert) # Step 3, true uses colab_bert
        feature_df.to_json("feature_df.jsonl", orient="records", lines=True)

    feature_df = clean_dataset(feature_df)  # Step 4
    # run_data_quality_checks(feature_df) # Step 5
    train_model(feature_df) # Step 6

if __name__ == "__main__":
    main()