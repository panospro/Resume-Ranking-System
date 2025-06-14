import re

def extract_structure_features(text: str) -> dict:
    text = text.lower().replace('\r', '\n')
    headers = set()

    # --- Section headers detection ---
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

    # --- Cover letter detection ---
    cover_letter_patterns = [
        r"dear (hiring manager|recruiter|sir|madam|team)",
        r"i am writing (to apply|in response|this letter)",
        r"please find (my resume|attached)",
        r"i am excited (to apply|about this opportunity)",
        r"my name is .* and i am (interested|applying)",
        r"with great enthusiasm",
        r"this letter is in regard to",
        r"i am reaching out",
        r"i would like to express",
        r"i am submitting my application",
    ]

    has_cover_letter = any(re.search(pattern, text) for pattern in cover_letter_patterns)

    return {
        "has_projects_section": int("projects" in headers),
        "has_certifications_section": int("certifications" in headers),
        "num_sections": len(headers),
        "has_cover_letter": int(has_cover_letter)
    }
