import re

def extract_section_headers(text):
    text = text.lower().replace('\r', '\n')
    headers = set()

    # Fuzzy match against soft section labels
    soft_patterns = {
        "projects": r"\b(projects|project experience|project details)\b",
        "certifications": r"\b(certifications|certification|certified)\b",
        "education": r"\b(education details|education|academic background)\b",
        "skills": r"\b(technical skills|skill details|skills)\b",
        "experience": r"\b(work experience|experience)\b"
    }

    for section, pattern in soft_patterns.items():
        if re.search(pattern, text):
            headers.add(section)

    return headers

def has_projects_section(text):
    return int("projects" in extract_section_headers(text))

def has_certifications_section(text):
    return int("certifications" in extract_section_headers(text))

def num_sections(text):
    return len(extract_section_headers(text))
