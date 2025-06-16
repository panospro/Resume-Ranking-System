import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from src.config import TECH_SKILLS, SOFT_SKILLS

stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm", disable=["ner"])

COMMON_VERBS = {"develop", "design", "implement", "build", "lead", "manage", "optimize", "analyze"}
COMMON_TERMS = {"project", "system", "api", "pipeline", "database", "application"}
ALLOWED_POS = {"NOUN", "VERB", "ADJ"}

# === TOKENIZATION ===
def tokenize_with_spacy(text: str) -> list:
    doc = nlp(text.lower())
    return [
        stemmer.stem(token.lemma_) 
        for token in doc 
        if not token.is_stop and token.is_alpha and token.pos_ in ALLOWED_POS
    ]

# === KEYWORD EXTRACTION ===
def extract_keywords_from_tokens(tokens: list, top_n: int = 30) -> list:
    token_freq = {}
    for token in tokens:
        if (
            token in COMMON_VERBS or
            token in COMMON_TERMS or
            token in TECH_SKILLS or
            token in SOFT_SKILLS
        ):
            token_freq[token] = token_freq.get(token, 0) + 1

    sorted_keywords = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:top_n]]

# === TF-IDF BINARY WRAPPER ===
def compute_tfidf_scores(jd_tokens: list, resume_tokens: list) -> dict:
    texts = [' '.join(jd_tokens), ' '.join(resume_tokens)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[0]
    return dict(zip(feature_names, jd_vector.toarray()[0]))

# === MAIN ENTRYPOINT ===
def compute_keyword_alignment(jd_text: str, resume_text: str,
                              jd_tokens=None, resume_tokens=None) -> dict:
    jd_tokens = jd_tokens or tokenize_with_spacy(jd_text)
    resume_tokens = resume_tokens or tokenize_with_spacy(resume_text)

    if not jd_tokens or not resume_tokens:
        return {
            "coverage_ratio": 0.0,
            "frequency_density": 0.0,
            "tfidf_score": 0.0
        }

    jd_keywords = extract_keywords_from_tokens(jd_tokens)

    # === Coverage + Frequency ===
    coverage_hits = sum(1 for kw in jd_keywords if kw in resume_tokens)
    total_mentions = sum(resume_tokens.count(kw) for kw in jd_keywords)

    coverage_ratio = coverage_hits / len(jd_keywords)
    frequency_density = total_mentions / len(resume_tokens)

    # === TF-IDF Score ===
    tfidf_scores = compute_tfidf_scores(jd_tokens, resume_tokens)
    tfidf_score = sum(tfidf_scores.get(kw, 0.0) for kw in jd_keywords)

    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "frequency_density": round(frequency_density, 6),
        "tfidf_score": round(tfidf_score, 6)
    }
