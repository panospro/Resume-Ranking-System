import re
from ..config import TECH_SKILLS
from typing import Set

# === Extract matching tech skills from a given text ===
def extract_tech_skills(text: str, skill_list: list = TECH_SKILLS) -> set:
    text = str(text).lower()
    text = re.sub(r'[^\w\s.+#]', ' ', text)  # Remove non-skill special chars except dot, +, #
    words = set(re.findall(r'\b[\w.+#]+\b', text))  # Keep things like c++, c#, node.js
    return set(skill for skill in skill_list if skill.lower() in words)

# === Count how many JD skills are found in the resume ===
def matching_skill_count(resume_text: str, jd_text: str) -> int:
    resume_skills = extract_tech_skills(resume_text)
    jd_skills = extract_tech_skills(jd_text)
    return len(resume_skills & jd_skills)

# === Percentage of JD skills covered by resume ===
def skill_coverage_ratio(resume_text: str, jd_text: str) -> float:
    resume_skills = extract_tech_skills(resume_text)
    jd_skills = extract_tech_skills(jd_text)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

# === Return actual overlapping skills as a set ===
def tech_stack_overlap(resume_text: str, jd_text: str) -> Set[str]:
    resume_skills = extract_tech_skills(resume_text)
    jd_skills = extract_tech_skills(jd_text)
    return resume_skills & jd_skills
