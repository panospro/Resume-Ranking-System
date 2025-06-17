import pandas as pd
from tqdm import tqdm
from src.utils.education import satisfies_education_requirement
from src.utils.skills import get_skill_metrics, get_domain_term_overlap, get_responsibility_verb_overlap, get_seniority_alignment_score
from src.utils.bert import batch_compute_embeddings, fast_cosine_sim, compute_title_similarity, resume_contains_role_title
from src.utils.keyword import compute_keyword_alignment
from src.utils.sections import extract_structure_features

def extract_all_features(df: pd.DataFrame, use_colab_bert: bool = False) -> pd.DataFrame:
    if use_colab_bert:
        print("⚙️ Using precomputed BERT & TF-IDF features from Colab...")
        colab_bert_df = pd.read_json("bert.jsonl", lines=True)
        df = df.merge(colab_bert_df, on=["JD_ID", "Resume_ID", "Label"], how="left")
    else:
        print("⚙️ Precomputing BERT embeddings in RAM...")
        jd_texts = df["Job Description"].astype(str).tolist()
        resume_texts = df["Resume"].astype(str).tolist()
        jd_embeddings = batch_compute_embeddings(jd_texts)
        resume_embeddings = batch_compute_embeddings(resume_texts)

    print("⚙️ Extracting features row-by-row...")
    feature_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Features"):
        jd_text = str(row["Job Description"])
        resume_text = str(row["Resume"])
        jd_title = str(row.get("Job Title", ""))
        resume_category = str(row.get("Category", ""))

        tech_metrics, soft_metrics = get_skill_metrics(resume_text, jd_text)

        # === Keyword Metrics & TF-IDF ===
        if use_colab_bert:
            keyword_coverage_ratio = row.get("keyword_coverage_ratio", 0.0)
            keyword_frequency_density = row.get("keyword_frequency_density", 0.0)
            tfidf_score = row.get("tfidf_score", 0.0)
            tfidf_coverage_ratio = row.get("tfidf_coverage_ratio", 0.0)
            tfidf_frequency_density = row.get("tfidf_frequency_density", 0.0)
        else:
            keywords = compute_keyword_alignment(jd_text, resume_text)
            keyword_coverage_ratio = keywords.get("coverage_ratio", 0.0)
            keyword_frequency_density = keywords.get("frequency_density", 0.0)
            tfidf_score = keywords.get("tfidf_score", 0.0)
            tfidf_coverage_ratio = keywords.get("tfidf_coverage_ratio", 0.0)
            tfidf_frequency_density = keywords.get("tfidf_frequency_density", 0.0)

        # === Structure + Semantic Overlaps ===
        structure = extract_structure_features(resume_text)
        domain_metrics = get_domain_term_overlap(resume_text, jd_text)
        responsibility_verb = get_responsibility_verb_overlap(resume_text, jd_text)
        seniority_alignment = get_seniority_alignment_score(resume_text, jd_text)

        # === BERT-related values ===
        if use_colab_bert:
            bert_sim = row.get("bert_similarity", 0.0)
            title_similarity_val = row.get("title_similarity", 0.0)
            role_title_hit = row.get("resume_contains_role_title", 0)
            jd_length = row.get("jd_length", len(jd_text.split()))
            resume_length = row.get("resume_length", len(resume_text.split()))
        else:
            bert_sim = fast_cosine_sim(jd_embeddings[i], resume_embeddings[i])
            title_similarity_val = compute_title_similarity(jd_title, resume_category)
            role_title_hit = resume_contains_role_title(jd_title, resume_text)
            jd_length = len(jd_text.split())
            resume_length = len(resume_text.split())

        features = {
            "JD_ID": row["JD_ID"],
            "Resume_ID": row["Resume_ID"],
            "Label": row["Label"],

            # === BERT & TF-IDF Features ===
            "bert_similarity": round(bert_sim, 4),
            "keyword_coverage_ratio": keyword_coverage_ratio,
            "keyword_frequency_density": keyword_frequency_density,
            "tfidf_score": tfidf_score,
            "tfidf_coverage_ratio": tfidf_coverage_ratio,
            "tfidf_frequency_density": tfidf_frequency_density,
            "title_similarity": title_similarity_val,
            "resume_contains_role_title": role_title_hit,
            "jd_length": jd_length,
            "resume_length": resume_length,

            # === Education & Skills ===
            "satisfies_education": satisfies_education_requirement(jd_text, resume_text),
            "tech_matching_skill_count": tech_metrics["count"],
            "soft_matching_skill_count": soft_metrics["count"],
            "tech_skill_coverage_ratio": tech_metrics["coverage"],
            "soft_skill_coverage_ratio": soft_metrics["coverage"],
            "tech_stack_overlap": tech_metrics["overlap"],
            "soft_stack_overlap": soft_metrics["overlap"],

            # === Resume Structure ===
            "has_projects_section": structure["has_projects_section"],
            "has_certifications_section": structure["has_certifications_section"],
            "num_sections": structure["num_sections"],
            "has_cover_letter": structure["has_cover_letter"],

            # === Domain & Responsibility Overlap ===
            "domain_term_overlap_count": domain_metrics["domain_term_overlap_count"],
            "domain_term_overlap_ratio": domain_metrics["domain_term_overlap_ratio"],
            "responsibility_verb_overlap_count": responsibility_verb["responsibility_verb_overlap_count"],
            "responsibility_verb_overlap_ratio": responsibility_verb["responsibility_verb_overlap_ratio"],

            # === Seniority Matching ===
            "seniority_alignment_count": seniority_alignment["seniority_alignment_count"],
            "seniority_alignment_ratio": seniority_alignment["seniority_alignment_ratio"],
        }

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
