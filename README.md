# 🧾 Resume Ranking System

A hybrid ML pipeline that scores how well a resume matches a job description.  
Combines BERT embeddings, TF-IDF, rule-based features, and LLM-generated pseudo-labels to deliver interpretable match scores.

---

## 🎯 Goal

Help job seekers understand how well their resume aligns with a job post — and why.  
Designed for integration into career platforms, ATS systems, or resume feedback tools.

---

## 🧠 How It Works

1. **Input**: A pair of resume text and job description  
2. **Feature Extraction**: Rich semantic, structural, and rule-based features  
3. **Labeling**: Pseudo-labels generated using LLaMA-based prompting  
4. **Modeling**: XGBoost ranks the match quality  
5. **Output**: A match score + feature contributions (coming soon)

---

## 🚀 Feature Breakdown

### 🔍 Similarity & Relevance

| Feature                     | Description |
|----------------------------|-------------|
| `bert_similarity`          | Cosine similarity from Sentence-BERT embeddings |
| `title_similarity`         | Similarity between job title and resume heading |
| `resume_contains_role_title` | Boolean flag if role title appears in resume |
| `jd_length` / `resume_length` | Word counts of job post and resume |

### 📊 Keyword & TF-IDF Features

| Feature                         | Description |
|----------------------------------|-------------|
| `keyword_coverage_ratio`         | % of JD keywords covered in resume |
| `keyword_frequency_density`      | Normalized frequency of matched keywords |
| `tfidf_score`                    | Overall TF-IDF similarity |
| `tfidf_coverage_ratio`          | TF-IDF token match ratio |
| `tfidf_frequency_density`       | Frequency of TF-IDF-weighted terms in resume |

### 🎓 Education & Skills

| Feature                         | Description |
|----------------------------------|-------------|
| `satisfies_education`            | Checks if resume meets education requirement |
| `tech_matching_skill_count`      | Matched technical skills |
| `soft_matching_skill_count`      | Matched soft skills |
| `tech_skill_coverage_ratio`      | Coverage % of tech skills from JD |
| `soft_skill_coverage_ratio`      | Coverage % of soft skills |
| `tech_stack_overlap`             | Overlapping technologies |
| `soft_stack_overlap`             | Overlapping soft skills/traits |

### 📑 Resume Structure

| Feature                         | Description |
|----------------------------------|-------------|
| `has_projects_section`           | Does the resume include a "Projects" section? |
| `has_certifications_section`     | Does the resume list certifications? |
| `num_sections`                   | Count of distinct sections in resume |
| `has_cover_letter`               | Was a cover letter included? |

### 🧠 Domain & Responsibilities

| Feature                               | Description |
|--------------------------------------|-------------|
| `domain_term_overlap_count`          | Count of domain-specific term matches |
| `domain_term_overlap_ratio`          | Ratio of domain terms covered |
| `responsibility_verb_overlap_count`  | Matching verbs in responsibilities (e.g., “led”, “designed”) |
| `responsibility_verb_overlap_ratio`  | Normalized match ratio for verbs |

### 🏆 Seniority Alignment

| Feature                          | Description |
|----------------------------------|-------------|
| `seniority_alignment_count`      | Matched seniority-related keywords |
| `seniority_alignment_ratio`      | % of seniority cues that aligned |

---

## 🛠️ Tech Stack

- `Python`, `XGBoost`, `scikit-learn`
- `SentenceTransformers` (`all-MiniLM`)
- `LLaMA` (via Together API for pseudo-labeling)
- `Spacy`, `NLTK`, `TF-IDF Vectorizer`

---

