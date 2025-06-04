import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper import load_dataframe_from_env

ResumeDatasetPath = load_dataframe_from_env("ResumeDatasetPath")

EDU_KEYWORDS = [
    "bachelor", "bachelors", "b.sc", "b.s", "b.tech", "be", "bca", "beng", "ba", "b.a",
    "llb", "law", "master", "masters", "m.sc", "m.s", "m.tech", "msc", "mca",
    "mba", "phd", "doctorate", "pgdm", "diploma", "graduate", "graduation"
]
EDU_GROUP = {
    "phd": "PhD", "doctorate": "PhD", "mba": "MBA", "pgdm": "MBA",
    "m.sc": "Master's", "msc": "Master's", "m.tech": "Master's", "m.s": "Master's",
    "masters": "Master's", "master": "Master's", "mca": "Master's",
    "b.sc": "Bachelor's", "b.tech": "Bachelor's", "b.s": "Bachelor's", "bachelor": "Bachelor's",
    "bachelors": "Bachelor's", "be": "Bachelor's", "bca": "Bachelor's", "ba": "Bachelor's",
    "b.a": "Bachelor's", "beng": "Bachelor's", "graduate": "Bachelor's", "graduation": "Bachelor's",
    "llb": "Bachelor's", "law": "Bachelor's", "diploma": "Diploma"
}
EDU_RANK = {"PhD": 4, "MBA": 3, "Master's": 3, "Bachelor's": 2, "Diploma": 1}
CERT_KEYWORDS = [
    "certified", "certification", "certificate", "aws certified", "pmp",
    "scrum master", "microsoft certified", "google certified"
]

def plot_bar(x, y, title, xlabel, ylabel, figsize=(10, 5), rotate=False):
    plt.figure(figsize=figsize)
    sns.barplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate: plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_hist(data, title, xlabel, ylabel, bins=30, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def extract_keywords(text, keywords):
    text = str(text).lower()
    return list({kw for kw in keywords if kw in text})

def normalize_education(text):
    text = str(text).lower()
    found = [EDU_GROUP[kw] for kw in EDU_KEYWORDS if kw in text and kw in EDU_GROUP]
    if not found: return "Unknown"
    return max(found, key=lambda x: EDU_RANK.get(x, 0))

def main():
    df = pd.read_csv(ResumeDatasetPath)

    # 1. Bar Chart: Resume Count by Category
    counts = df['Category'].value_counts()
    plot_bar(counts.values, counts.index, "Resume Count by Category", "Number of Resumes", "Category")

    # 2. Histogram: Resume Length Distribution
    lengths = df['Resume'].apply(lambda x: len(str(x).split()))
    plot_hist(lengths, "Distribution of Resume Lengths (Word Count)", "Number of Words", "Number of Resumes")
    # Feature engineering
    df['PrimaryEducation'] = df['Resume'].apply(normalize_education)
    df['Certifications'] = df['Resume'].apply(lambda x: extract_keywords(x, CERT_KEYWORDS))
    df['CertCount'] = df['Certifications'].apply(len)
    df['HasCertification'] = df['CertCount'] > 0

    # 3. Education Level Distribution
    edu_counts = df['PrimaryEducation'].value_counts()
    plot_bar(edu_counts.index, edu_counts.values, "Distribution of Primary Education Level", "Education Level", "Number of Resumes", rotate=True)
    
    # 4. Certification presence
    cert_counts = df['HasCertification'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=cert_counts.index.map({False: "No Certification", True: "Has Certification"}),
        y=cert_counts.values
    )
    plt.title("Resumes With/Without Certifications")
    plt.ylabel("Number of Resumes")
    plt.tight_layout()
    plt.show()

    # 5. Number of Certifications per Resume
    plt.figure(figsize=(8, 5))
    sns.countplot(x='CertCount', data=df)
    plt.title("Number of Certifications per Resume")
    plt.xlabel("Certifications Count")
    plt.ylabel("Number of Resumes")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
