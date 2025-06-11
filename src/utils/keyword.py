from nltk.stem import PorterStemmer
from src.config import TECH_SKILLS, SOFT_SKILLS
import re 

stemmer = PorterStemmer()

def tokenize(text: str) -> list:
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

def extract_keywords(jd_text: str, top_n: int = 30) -> list:
    COMMON_VERBS = ["develop", "design", "implement", "build", "lead", "manage", "optimize", "analyze"]
    COMMON_TERMS = ["project", "system", "api", "pipeline", "database", "application"]

    tokens = tokenize(jd_text)
    keywords = set()

    for token in tokens:
        if token in COMMON_VERBS or token in COMMON_TERMS or token in TECH_SKILLS or token in SOFT_SKILLS:
            keywords.add(stemmer.stem(token))

    return list(keywords)[:top_n]

def compute_keyword_alignment(jd_text: str, resume_text: str) -> dict:
    jd_keywords = extract_keywords(jd_text)
    resume_tokens = tokenize(resume_text)
    resume_stems = [stemmer.stem(w) for w in resume_tokens]

    if not jd_keywords:
        return {
            "coverage_ratio": 0.0,
            "frequency_density": 0.0
        }

    coverage_hits = sum(1 for kw in jd_keywords if kw in resume_stems)
    total_mentions = sum(resume_stems.count(kw) for kw in jd_keywords)

    coverage_ratio = coverage_hits / len(jd_keywords)
    frequency_density = total_mentions / len(resume_stems) if resume_stems else 0.0

    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "frequency_density": round(frequency_density, 6)
    }