import re
from typing import Set, List, Tuple, Dict
from src.config import TECH_SKILLS, SOFT_SKILLS, DOMAIN_KEYWORDS, RESPONSIBILITY_VERBS, SENIORITY_TERMS

# === Generic Skill Extractor ===
def extract_skills(text: str, skills: List[str]) -> Set[str]:
    text = str(text).lower()
    text = re.sub(r'[^\w\s.+#]', ' ', text) # Remove non-skill special chars except dot, +, #
    tokens = set(re.findall(r'\b[\w.+#]+\b', text)) # Keep things like c++, c#, node.js
    return {skill for skill in skills if skill.lower() in tokens}

def compute_skill_metrics(resume_text: str, jd_text: str, skills: List[str]) -> dict:
    resume_skills = extract_skills(resume_text, skills)
    jd_skills = extract_skills(jd_text, skills)
    overlap = resume_skills & jd_skills
    
    # resume and jd are for debugging purposes to see the skills extracted
    return {
        'resume': resume_skills,
        'jd': jd_skills,
        'overlap': overlap,
        'count': len(overlap),
        'coverage': len(overlap) / len(jd_skills) if jd_skills else 0.0
    }

def get_skill_metrics(resume_text: str, jd_text: str) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Returns:
        tech_metrics: dict with resume skills/jd skills/overlap/count/coverage for TECH_SKILLS
        soft_metrics: same structure for SOFT_SKILLS
    """
    tech_metrics = compute_skill_metrics(resume_text, jd_text, TECH_SKILLS)
    soft_metrics = compute_skill_metrics(resume_text, jd_text, SOFT_SKILLS)
    return tech_metrics, soft_metrics

def extract_tokens(text: str) -> Set[str]:
    """Normalize and extract lowercased tokens from text."""
    text = text.lower()
    text = re.sub(r'[^\w\s.+#]', ' ', text)
    return set(re.findall(r'\b[\w.+#]+\b', text))

def get_domain_term_overlap(resume_text: str, jd_text: str) -> Dict[str, float]:
    """
    Computes domain term overlap by checking which domain's keywords are mentioned in both JD and Resume.
    Returns:
        {
            'domain_term_overlap_count': int,
            'domain_term_overlap_ratio': float
        }
    """
    jd_tokens = extract_tokens(jd_text)
    resume_tokens = extract_tokens(resume_text)

    max_count = 0
    max_ratio = 0.0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        keywords_set = set(kw.lower() for kw in keywords)
        matched_keywords = {kw for kw in keywords_set if any(kw in token for token in jd_tokens)}
        overlap_count = sum(1 for kw in matched_keywords if any(kw in token for token in resume_tokens))

        if matched_keywords:
            ratio = overlap_count / len(matched_keywords)
        else:
            ratio = 0.0

        if overlap_count > max_count:
            max_count = overlap_count
            max_ratio = ratio

    return {
        "domain_term_overlap_count": max_count,
        "domain_term_overlap_ratio": round(max_ratio, 4)
    }

def get_responsibility_verb_overlap(resume_text: str, jd_text: str) -> Dict[str, float]:
    """
    Measures overlap of action/responsibility verbs between resume and job description.
    Returns:
        {
            'responsibility_verb_overlap_count': int,
            'responsibility_verb_overlap_ratio': float
        }
    """
    resume_tokens = extract_tokens(resume_text)
    jd_tokens = extract_tokens(jd_text)

    resume_verbs = {word for word in resume_tokens if word in RESPONSIBILITY_VERBS}
    jd_verbs = {word for word in jd_tokens if word in RESPONSIBILITY_VERBS}

    overlap = resume_verbs & jd_verbs

    return {
        'responsibility_verb_overlap_count': len(overlap),
        'responsibility_verb_overlap_ratio': len(overlap) / len(jd_verbs) if jd_verbs else 0.0
    }

def get_seniority_alignment_score(resume_text: str, jd_text: str) -> Dict[str, float]:
    """
    Measures the overlap of seniority-level terms between JD and resume.
    Returns:
        {
            'seniority_alignment_count': int,
            'seniority_alignment_ratio': float
        }
    """
    resume_tokens = extract_tokens(resume_text)
    jd_tokens = extract_tokens(jd_text)

    resume_senior_terms = {term for term in SENIORITY_TERMS if term in resume_tokens}
    jd_senior_terms = {term for term in SENIORITY_TERMS if term in jd_tokens}

    overlap = resume_senior_terms & jd_senior_terms

    return {
        'seniority_alignment_count': len(overlap),
        'seniority_alignment_ratio': len(overlap) / len(jd_senior_terms) if jd_senior_terms else 0.0
    }