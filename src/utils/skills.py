import re
from typing import Set, List, Tuple, Dict
from src.config import TECH_SKILLS, SOFT_SKILLS, DOMAIN_KEYWORDS, RESPONSIBILITY_VERBS, SENIORITY_TERMS

def extract_tokens(text: str) -> Set[str]:
    """Normalize and extract tokens, preserving tech-specific symbols like +, ., #"""
    return set(re.findall(r'\b[\w.+#]+\b', re.sub(r'[^\w\s.+#]', ' ', text.lower())))

def extract_skills(text: str, skills: List[str]) -> Set[str]:
    tokens = extract_tokens(text)
    return {s for s in skills if s.lower() in tokens}

def compute_skill_metrics(resume: str, jd: str, skills: List[str]) -> Dict[str, any]:
    resume_skills, jd_skills = extract_skills(resume, skills), extract_skills(jd, skills)
    overlap = resume_skills & jd_skills
    return {
        'resume': resume_skills,
        'jd': jd_skills,
        'overlap': overlap,
        'count': len(overlap),
        'coverage': len(overlap) / len(jd_skills) if jd_skills else 0.0
    }

def get_skill_metrics(resume: str, jd: str) -> Tuple[Dict[str, any], Dict[str, any]]:
    return (
        compute_skill_metrics(resume, jd, TECH_SKILLS),
        compute_skill_metrics(resume, jd, SOFT_SKILLS)
    )

def get_domain_term_overlap(resume: str, jd: str) -> Dict[str, float]:
    resume_tokens = extract_tokens(resume)
    jd_tokens = extract_tokens(jd)

    max_count, max_ratio = 0, 0.0

    for keywords in DOMAIN_KEYWORDS.values():
        keywords_lower = {kw.lower() for kw in keywords}
        jd_matched = {kw for kw in keywords_lower if any(kw in tok for tok in jd_tokens)}
        if not jd_matched:
            continue
        overlap = {kw for kw in jd_matched if any(kw in tok for tok in resume_tokens)}
        count, ratio = len(overlap), len(overlap) / len(jd_matched)
        if count > max_count:
            max_count, max_ratio = count, ratio

    return {
        "domain_term_overlap_count": max_count,
        "domain_term_overlap_ratio": round(max_ratio, 4)
    }

def get_overlap_metric(resume: str, jd: str, term_set: Set[str], label: str) -> Dict[str, float]:
    resume_terms = extract_tokens(resume) & term_set
    jd_terms = extract_tokens(jd) & term_set
    overlap = resume_terms & jd_terms
    return {
        f'{label}_count': len(overlap),
        f'{label}_ratio': len(overlap) / len(jd_terms) if jd_terms else 0.0
    }

def get_responsibility_verb_overlap(resume: str, jd: str) -> Dict[str, float]:
    return get_overlap_metric(resume, jd, RESPONSIBILITY_VERBS, 'responsibility_verb_overlap')

def get_seniority_alignment_score(resume: str, jd: str) -> Dict[str, float]:
    return get_overlap_metric(resume, jd, SENIORITY_TERMS, 'seniority_alignment')
