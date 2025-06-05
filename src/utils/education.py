import re
from config import EDUCATION_LEVELS, EDU_RANK

# === Core helper function to extract education level from any text ===
# Uses regex to find explicit mentions and falls back to vague term matching
def _extract_education_level(text: str, fallback_value: str, vague_terms: list) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    for level in ['PhD', 'Masters', 'Bachelors', 'Diploma', 'High School']:
        if re.search(EDUCATION_LEVELS[level], text, re.IGNORECASE):
            return level

    if any(term in text for term in vague_terms):
        return "Possibly Bachelors"

    return fallback_value

# === Extract education level from a resume string ===
# Returns one of: PhD, Masters, Bachelors, Diploma, High School, Possibly Bachelors, or Unknown
def extract_education_level_from_resume(text: str) -> str:
    vague_terms = ['university', 'college', 'graduate', 'degree', 'education']
    return _extract_education_level(text, fallback_value="Unknown", vague_terms=vague_terms)

# === Extract required education level from a job description ===
# Same idea as resume extraction, but with broader fallback terms
def extract_education_level_from_jd(text: str) -> str:
    vague_terms = [
        'bachelor', 'b.tech', 'b.sc', 'graduate', 'degree', 'qualification', 
        'master', 'm.sc', 'm.tech', 'mba', 'education', 'phd', 'diploma', 'doctor'
    ]
    return _extract_education_level(text, fallback_value="Not Specified", vague_terms=vague_terms)

# === Determine if a resume satisfies a JD's minimum education requirement ===
# Compares the rank of both and returns 1 (satisfies) or 0 (doesn't)
def satisfies_education_requirement(jd_text: str, resume_text: str) -> int:
    resume_level = extract_education_level_from_resume(resume_text)
    jd_level = extract_education_level_from_jd(jd_text)

    resume_rank = EDU_RANK.get(resume_level, 0)
    jd_rank = EDU_RANK.get(jd_level, 0)

    return int(resume_rank >= jd_rank)
