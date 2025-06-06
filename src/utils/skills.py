import re
from typing import Set, List, Tuple, Dict
from ..config import TECH_SKILLS, SOFT_SKILLS

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
