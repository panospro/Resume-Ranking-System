import pandas as pd
from src.utils.education import satisfies_education_requirement
from src.utils.skills import get_skill_metrics, get_domain_term_overlap
from src.utils.bert import compute_bert_similarity, compute_title_similarity, resume_contains_role_title
from src.utils.keyword import compute_keyword_alignment
from src.utils.sections import extract_structure_features

def extract_features(row) -> dict:
    jd_text = str(row["Job Description"])
    resume_text = str(row["Resume"])
    jd_title = str(row.get("Job Title", ""))
    resume_category = str(row.get("Category", ""))

    # === Skill features ===
    tech_metrics, soft_metrics = get_skill_metrics(resume_text, jd_text)

    # === Keyword features ===
    keywords = compute_keyword_alignment(jd_text, resume_text)

    # === Structure features ===
    structure = extract_structure_features(resume_text)
    domain_metrics = get_domain_term_overlap(jd_text, resume_text)

    return {
        "satisfies_education": satisfies_education_requirement(jd_text, resume_text),
        "resume_length": len(resume_text.split()),
        "jd_length": len(jd_text.split()),
        "tech_matching_skill_count": tech_metrics["count"],
        "soft_matching_skill_count": soft_metrics["count"],
        "bert_similarity": compute_bert_similarity(jd_text, resume_text),
        "tech_skill_coverage_ratio": tech_metrics["coverage"],
        "soft_skill_coverage_ratio": soft_metrics["coverage"],

        # NOTE: For app use later 
        "tech_stack_overlap": tech_metrics["overlap"],
        "soft_stack_overlap": soft_metrics["overlap"], 

        # What fraction of JD keywords are present at least once in the resume.
        "keyword_coverage_ratio": keywords["coverage_ratio"],    

        # How many times JD keywords appear overall in the resume, normalized by resume length
        "keyword_frequency_density": keywords["frequency_density"],

        "tfidf_score": keywords["tfidf_score"],
        "has_projects_section": structure["has_projects_section"],
        "has_certifications_section": structure["has_certifications_section"],
        "num_sections": structure["num_sections"],
        "has_cover_letter": structure["has_cover_letter"],
        "title_similarity": compute_title_similarity(jd_title, resume_category),
        "resume_contains_role_title": resume_contains_role_title(jd_title, resume_text),
        "domain_term_overlap_count": domain_metrics["domain_term_overlap_count"],
        "domain_term_overlap_ratio": domain_metrics["domain_term_overlap_ratio"],
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