import pandas as pd
from src.utils.education import satisfies_education_requirement
from src.utils.skills import get_skill_metrics
from src.utils.bert import compute_bert_similarity
from src.utils.keyword import compute_keyword_alignment

def extract_features(row) -> dict:
    jd_text = str(row["Job Description"])
    resume_text = str(row["Resume"])

    # === Skill features ===
    tech_metrics, soft_metrics = get_skill_metrics(resume_text, jd_text)

    # === Keywords feature ===
    keywords = compute_keyword_alignment(jd_text, resume_text)

    return {
        "satisfies_education": satisfies_education_requirement(jd_text, resume_text),
        "resume_length": len(resume_text.split()),
        "jd_length": len(jd_text.split()),
        "tech_matching_skill_count": tech_metrics["count"],
        "soft_matching_skill_count": soft_metrics["count"],
        "bert_similarity": compute_bert_similarity(jd_text, resume_text),

        # Helps answer: “Does this person cover what we asked for?”
        "tech_skill_coverage_ratio": tech_metrics["coverage"],
        "soft_skill_coverage_ratio": soft_metrics["coverage"],

        # NOTE: For app use later 
        "tech_stack_overlap": tech_metrics["overlap"],
        "soft_stack_overlap": soft_metrics["overlap"], 

        # Helps answer: “Are the resume’s skills actually relevant to this job?”
        "tech_skill_precision_ratio": tech_metrics["count"] / (len(tech_metrics["resume"]) + 1e-5),
        "soft_skill_precision_ratio": soft_metrics["count"] / (len(soft_metrics["resume"]) + 1e-5),

        # What fraction of JD keywords are present at least once in the resume.
        "keyword_coverage_ratio": keywords["coverage_ratio"],    

        # How many times JD keywords appear overall in the resume, normalized by resume length
        "keyword_frequency_density": keywords["frequency_density"]
    }

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Extracting features...")
    feature_rows = []
    for _, row in df.iterrows():
        features = extract_features(row)
        features.update({
            "JD_ID": row["JD_ID"],
            "Resume_ID": row["Resume_ID"],
            "Label": row["Label"]
        })
        feature_rows.append(features)
    return pd.DataFrame(feature_rows)