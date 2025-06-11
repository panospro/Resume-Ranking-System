import pandas as pd
from src.utils.education import satisfies_education_requirement
from src.utils.skills import get_skill_metrics

def extract_features(row) -> dict:
    jd_text = str(row["Job Description"])
    resume_text = str(row["Resume"])

    # === Skill features ===
    tech_metrics, soft_metrics = get_skill_metrics(resume_text, jd_text)

    return {
        "satisfies_education": satisfies_education_requirement(jd_text, resume_text),
        "resume_length": len(resume_text.split()),
        "tech_matching_skill_count": tech_metrics["count"],
        "tech_skill_coverage_ratio": tech_metrics["coverage"],
        "soft_matching_skill_count": soft_metrics["count"],
        "soft_skill_coverage_ratio": soft_metrics["coverage"],
        "tech_stack_overlap": len(tech_metrics["overlap"]),  # NOTE: For app use later
        "soft_stack_overlap": len(soft_metrics["overlap"])  # NOTE: For app use later 
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