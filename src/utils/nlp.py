from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def naive_sent_tokenize(text: str):
    return [s.strip() for s in text.split(".") if len(s.strip()) > 10]

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # remove extra spaces/newlines
    return text.strip()

# Best 
def compute_bert_similarity(jd_text: str, resume_text: str) -> float:
    jd_text = normalize(jd_text)
    resume_text = normalize(resume_text)

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(jd_embedding, resume_embedding)[0][0])
