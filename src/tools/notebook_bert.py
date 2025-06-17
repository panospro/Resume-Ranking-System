# !pip install -U sentence-transformers rapidfuzz scikit-learn nltk spacy
# !python -m spacy download en_core_web_sm

from google.colab import files
import pandas as pd
import io
uploaded = files.upload()


# If you uploaded "resume_data.csv", change filename here:
df = pd.read_parquet("google_colab1.parquet")
df.head()



import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from rapidfuzz import fuzz
import spacy
from nltk.stem import PorterStemmer

# === Downloads ===
nltk.download("punkt")
nltk.download("stopwords")

# === Constants & Models ===
stop_words = set(stopwords.words("english"))
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm", disable=["ner"])

TECH_SKILLS = {"python", "java", "sql", "tensorflow", "pytorch", "react", "aws"}
SOFT_SKILLS = {"communication", "leadership", "problem-solving"}
COMMON_VERBS = {"develop", "design", "implement", "build", "lead", "manage", "optimize", "analyze"}
COMMON_TERMS = {"project", "system", "api", "pipeline", "database", "application"}
ALLOWED_POS = {"NOUN", "VERB", "ADJ"}

# === Utility Functions ===
def simple_tokenize(text):
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

def extract_keywords_from_tokens(tokens, top_n=30):
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return [kw for kw, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def compute_keyword_alignment(jd_tokens, resume_tokens, jd_keywords):
    hits = sum(1 for kw in jd_keywords if kw in resume_tokens)
    mentions = sum(resume_tokens.count(kw) for kw in jd_keywords)
    coverage = hits / len(jd_keywords) if jd_keywords else 0.0
    density = mentions / len(resume_tokens) if resume_tokens else 0.0
    return {"coverage_ratio": round(coverage, 4), "frequency_density": round(density, 6)}

def fast_cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0

def compute_title_similarity(jd_title, resume_cat):
    return fuzz.token_sort_ratio(jd_title.lower(), resume_cat.lower()) / 100

def resume_contains_role_title(job_title, resume_text):
    return int(fuzz.partial_ratio(job_title.lower(), resume_text.lower()) >= 80)

def truncate_to_token_limit(text, max_words=128):
    return " ".join(text.split()[:max_words])

def batch_compute_embeddings(texts, batch_size=128):
    texts = [truncate_to_token_limit(t) for t in texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

def tokenize_spacy_pipe(texts, desc):
    docs = nlp.pipe(texts, batch_size=32, n_process=1)  # use 1 process only
    output = []
    for doc in tqdm(docs, total=len(texts), desc=desc):
        tokens = [
            stemmer.stem(token.lemma_)
            for token in doc
            if not token.is_stop and token.is_alpha and token.pos_ in ALLOWED_POS
        ]
        output.append(tokens)
    return output

def extract_keywords_from_tokens_tfidf(tokens, top_n=30):
    freq = {}
    for token in tokens:
        if token in TECH_SKILLS or token in COMMON_VERBS or token in COMMON_TERMS or token in SOFT_SKILLS:
            freq[token] = freq.get(token, 0) + 1
    return [kw for kw, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def compute_tfidf_scores(jd_tokens, resume_tokens):
    texts = [' '.join(jd_tokens), ' '.join(resume_tokens)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return dict(zip(feature_names, tfidf[0].toarray()[0]))

def compute_keyword_alignment_tfidf(jd_tokens, resume_tokens):
    if not jd_tokens or not resume_tokens:
        return {"tfidf_coverage_ratio": 0.0, "tfidf_frequency_density": 0.0, "tfidf_score": 0.0}

    jd_keywords = extract_keywords_from_tokens_tfidf(jd_tokens)
    if not jd_keywords:  # ✅ avoid division by zero
        return {"tfidf_coverage_ratio": 0.0, "tfidf_frequency_density": 0.0, "tfidf_score": 0.0}

    hits = sum(1 for kw in jd_keywords if kw in resume_tokens)
    mentions = sum(resume_tokens.count(kw) for kw in jd_keywords)
    coverage = hits / len(jd_keywords)
    density = mentions / len(resume_tokens)
    tfidf_scores = compute_tfidf_scores(jd_tokens, resume_tokens)
    tfidf_score = sum(tfidf_scores.get(kw, 0.0) for kw in jd_keywords)

    return {
        "tfidf_coverage_ratio": round(coverage, 4),
        "tfidf_frequency_density": round(density, 6),
        "tfidf_score": round(tfidf_score, 6)
    }

# === Main Feature Extraction ===
def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    jd_texts = df["Job Description"].astype(str).tolist()
    resume_texts = df["Resume"].astype(str).tolist()

    print("⚡ Tokenizing (fast)...")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        jd_tokens_list = list(tqdm(executor.map(simple_tokenize, jd_texts), total=len(jd_texts), desc="Tokenizing JDs"))
        resume_tokens_list = list(tqdm(executor.map(simple_tokenize, resume_texts), total=len(resume_texts), desc="Tokenizing Resumes"))

    print("⚡ spaCy Tokenizing in Batches...")
    # spacy_jd_tokens_list = tokenize_spacy_pipe(jd_texts, desc="Spacy JDs")
    # spacy_resume_tokens_list = tokenize_spacy_pipe(resume_texts, desc="Spacy Resumes")
    spacy_jd_tokens_list = jd_tokens_list
    spacy_resume_tokens_list = resume_tokens_list


    print("🚀 Embedding on GPU...")
    jd_embeddings = batch_compute_embeddings(jd_texts)
    resume_embeddings = batch_compute_embeddings(resume_texts)

    print("⚙️ Extracting features...")
    features = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        jd_tokens = jd_tokens_list[i]
        resume_tokens = resume_tokens_list[i]
        jd_keywords = extract_keywords_from_tokens(jd_tokens)

        keyword_metrics = compute_keyword_alignment(jd_tokens, resume_tokens, jd_keywords)
        tfidf_metrics = compute_keyword_alignment_tfidf(spacy_jd_tokens_list[i], spacy_resume_tokens_list[i])
        bert_sim = fast_cosine_sim(jd_embeddings[i], resume_embeddings[i])

        features.append({
            "JD_ID": row["JD_ID"],
            "Resume_ID": row["Resume_ID"],
            "Label": row["Label"],
            "keyword_coverage_ratio": keyword_metrics["coverage_ratio"],
            "keyword_frequency_density": keyword_metrics["frequency_density"],
            "bert_similarity": round(bert_sim, 4),
            "title_similarity": compute_title_similarity(row.get("Job Title", ""), row.get("Category", "")),
            "resume_contains_role_title": resume_contains_role_title(row.get("Job Title", ""), resume_texts[i]),
            "jd_length": len(jd_texts[i].split()),
            "resume_length": len(resume_texts[i].split()),
            "tfidf_coverage_ratio": tfidf_metrics["tfidf_coverage_ratio"],
            "tfidf_frequency_density": tfidf_metrics["tfidf_frequency_density"],
            "tfidf_score": tfidf_metrics["tfidf_score"],
        })

    return pd.DataFrame(features)

feature_df = extract_all_features(df)
feature_df.head()
feature_df.to_json("bert.jsonl", orient="records", lines=True)
files.download("bert.jsonl")
