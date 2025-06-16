from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
from sentence_transformers import SentenceTransformer
print(SentenceTransformer("all-MiniLM-L6-v2").device)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_and_truncate(text: str, max_words: int = 750) -> str:
    text = normalize(text)
    words = text.split()
    return " ".join(words[:max_words])

def batch_compute_embeddings(texts: list[str], batch_size: int = 256) -> np.ndarray:
    texts = [normalize_and_truncate(t) for t in texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

def fast_cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def compute_title_similarity(jd_title: str, resume_category: str) -> float:
    jd_title = normalize_title(jd_title)
    resume_title = normalize_title(resume_category)
    return fuzz.token_sort_ratio(jd_title, resume_title) / 100

def resume_contains_role_title(job_title: str, resume_text: str) -> int:
    job_title = str(job_title).lower().strip()
    resume_text = str(resume_text).lower()
    score = fuzz.partial_ratio(job_title, resume_text)
    return int(score >= 80)
