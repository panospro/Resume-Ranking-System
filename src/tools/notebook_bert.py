# Cell 1 
# 
# 
# !pip install -U sentence-transformers rapidfuzz scikit-learn nltk spacy
# !python -m spacy download en_core_web_sm

from google.colab import files
import pandas as pd
import io
uploaded = files.upload()


# If you uploaded "resume_data.csv", change filename here:
df = pd.read_csv(io.BytesIO(uploaded[list(uploaded.keys())[0]]))
df = pd.read_csv("sample_resume_job_data.csv")
df.head()


# Cell 2    
import numpy as np
import nltk
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from rapidfuzz import fuzz

nltk.download("punkt")
nltk.download("stopwords")

# === Preload models and constants ===
stop_words = set(stopwords.words("english"))
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# === Tokenization ===
def simple_tokenize(text):
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

def extract_keywords_from_tokens(tokens: list, top_n: int = 30) -> list:
    token_freq = {}
    for token in tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    sorted_keywords = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:top_n]]

def compute_keyword_alignment(jd_tokens, resume_tokens, jd_keywords) -> dict:
    coverage_hits = sum(1 for kw in jd_keywords if kw in resume_tokens)
    total_mentions = sum(resume_tokens.count(kw) for kw in jd_keywords)
    coverage_ratio = coverage_hits / len(jd_keywords) if jd_keywords else 0.0
    frequency_density = total_mentions / len(resume_tokens) if resume_tokens else 0.0
    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "frequency_density": round(frequency_density, 6),
    }

def fast_cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def compute_title_similarity(jd_title: str, resume_category: str) -> float:
    return fuzz.token_sort_ratio(jd_title.lower(), resume_category.lower()) / 100

def resume_contains_role_title(job_title: str, resume_text: str) -> int:
    return int(fuzz.partial_ratio(job_title.lower(), resume_text.lower()) >= 80)

def truncate_to_token_limit(text: str, max_words: int = 300):
    return " ".join(text.split()[:max_words])

def batch_compute_embeddings(texts: list[str], batch_size: int = 128) -> np.ndarray:
    texts = [truncate_to_token_limit(t) for t in texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

# === Main feature extraction function ===
def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    jd_texts = df["Job Description"].astype(str).tolist()
    resume_texts = df["Resume"].astype(str).tolist()

    print("⚡ Tokenizing (fast)...")
    jd_tokens_list = [simple_tokenize(t) for t in tqdm(jd_texts, desc="Tokenizing JDs")]
    resume_tokens_list = [simple_tokenize(t) for t in tqdm(resume_texts, desc="Tokenizing Resumes")]

    print("🚀 Embedding on GPU...")
    jd_embeddings = batch_compute_embeddings(jd_texts)
    resume_embeddings = batch_compute_embeddings(resume_texts)

    print("⚙️ Extracting features...")
    features = []

    for i in tqdm(range(len(df)), desc="Processing Rows"):
        jd_tokens = jd_tokens_list[i]
        resume_tokens = resume_tokens_list[i]
        jd_keywords = extract_keywords_from_tokens(jd_tokens)

        keyword_metrics = compute_keyword_alignment(jd_tokens, resume_tokens, jd_keywords)
        bert_sim = fast_cosine_sim(jd_embeddings[i], resume_embeddings[i])

        features.append({
            "JD_ID": df.iloc[i]["JD_ID"],
            "Resume_ID": df.iloc[i]["Resume_ID"],
            "Label": df.iloc[i]["Label"],
            "keyword_coverage_ratio": keyword_metrics["coverage_ratio"],
            "keyword_frequency_density": keyword_metrics["frequency_density"],
            "bert_similarity": round(bert_sim, 4),
            "title_similarity": compute_title_similarity(df.iloc[i].get("Job Title", ""), df.iloc[i].get("Category", "")),
            "resume_contains_role_title": resume_contains_role_title(df.iloc[i].get("Job Title", ""), resume_texts[i]),
            "jd_length": len(jd_texts[i].split()),
            "resume_length": len(resume_texts[i].split())
        })

    return pd.DataFrame(features)



feature_df = extract_all_features(df)
feature_df.head()

feature_df.to_json("feature_df.jsonl", orient="records", lines=True)
files.download("feature_df.jsonl")