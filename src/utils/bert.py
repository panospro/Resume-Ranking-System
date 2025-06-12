from sentence_transformers import SentenceTransformer, util
import re
from rapidfuzz import fuzz

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def naive_sent_tokenize(text: str):
    return [s.strip() for s in text.split(".") if len(s.strip()) > 10]

def compute_bert_similarity(jd_text: str, resume_text: str) -> float:
    jd_sentences = naive_sent_tokenize(normalize(jd_text))
    resume_sentences = naive_sent_tokenize(normalize(resume_text))

    if not jd_sentences or not resume_sentences:
        return 0.0

    jd_embeddings = model.encode(jd_sentences, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_sentences, convert_to_tensor=True)

    sim_matrix = util.pytorch_cos_sim(jd_embeddings, resume_embeddings)
    max_sim_per_jd = sim_matrix.max(dim=1).values  # best match for each JD sentence

    return float(max_sim_per_jd.mean())  # average of best matches

def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def compute_title_similarity(jd_title: str, resume_category: str) -> float:
    jd_title = normalize_title(jd_title)
    resume_title = normalize_title(resume_category)
    return fuzz.token_sort_ratio(jd_title, resume_title) / 100  # scaled 0 to 1
 
def resume_contains_role_title(job_title: str, resume_text: str) -> int:
    """
    Returns 1 if the job title is fuzzily found in the resume text.
    """
    job_title = str(job_title).lower().strip()
    resume_text = str(resume_text).lower()

    # Use partial fuzzy match
    score = fuzz.partial_ratio(job_title, resume_text)
    return int(score >= 80)
