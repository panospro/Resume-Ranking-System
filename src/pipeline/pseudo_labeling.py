from openai import OpenAI
import pandas as pd
import json
from src.utils.helper import load_from_env
from pathlib import Path

PROMPT_JSONL = "Dataset/llm_prompts.jsonl"
RESPONSE_JSONL = "Dataset/llm_raw_outputs.jsonl"

client = OpenAI(
    api_key=load_from_env("groqKey"),
    base_url="https://api.groq.com/openai/v1"
)

def call_groq(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Rate how well a resume matches a job description on a scale of 0 to 5."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"error: {str(e)}"

def extract_score(resp: str):
    for token in str(resp).split():
        if token.isdigit() and 0 <= int(token) <= 5:
            return int(token)
    return None

def create_prompts(df: pd.DataFrame) -> pd.DataFrame:
    def format_prompt(jd, res):
        return f"Job Description:\n{jd}\n\nResume:\n{res}\n\nHow well does this resume match the job? Reply with a number from 0 to 5."
    df["Prompt"] = df.apply(lambda row: format_prompt(row["Job Description"], row["Resume"]), axis=1)
    return df

def save_prompts_jsonl(df: pd.DataFrame):
    with open(PROMPT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "JD_ID": row["JD_ID"],
                "Resume_ID": row["Resume_ID"],
                "Prompt": row["Prompt"]
            }) + "\n")

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    # === Only use this if you want to regenerate prompts ===
    # df = create_prompts(df)
    # save_prompts_jsonl(df)

    # === Load prompts directly from existing JSONL ===
    print("📂 Loading prompts from existing file...")
    prompts = []
    with open("Dataset/llm_prompts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                prompts.append(item)
            except:
                continue
    prompt_df = pd.DataFrame(prompts)
    df = pd.merge(df, prompt_df, on=["JD_ID", "Resume_ID"], how="inner")
    # ========================

    # === Load completed responses if they exist ===
    seen_pairs = set()
    if Path(RESPONSE_JSONL).exists():
        with open(RESPONSE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    seen_pairs.add((record["JD_ID"], record["Resume_ID"]))
                except:
                    continue

    print(f"🔁 Resuming from last saved progress ({len(seen_pairs)} already processed)")

    with open(RESPONSE_JSONL, "a", encoding="utf-8") as out:
        for i, row in df.iterrows():
            key = (row["JD_ID"], row["Resume_ID"])
            if key in seen_pairs:
                continue

            print(f"🔍 [{i+1}/{len(df)}] JD: {row['JD_ID']} | Resume: {row['Resume_ID']}")
            response = call_groq(row["Prompt"])
            print(f"   📥 Response: {response}")

            # If an error happens just break
            # NOTE: We are on "JD_ID": 9, "Resume_ID": 2
            if "rate limit" in response.lower() or "error code: 429" in response.lower():
                print("❌ Rate limit hit or error occurred. Stopping safely...")
                break

            record = {
                "JD_ID": row["JD_ID"],
                "Resume_ID": row["Resume_ID"],
                "Response": response,
                "Label": extract_score(response)
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    # Reload updated responses
    with open(RESPONSE_JSONL, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if "Label" in line]

    response_df = pd.DataFrame(results)
    df = pd.merge(df, response_df[["JD_ID", "Resume_ID", "Label"]], on=["JD_ID", "Resume_ID"])
    print("🏷️ Labels extracted and merged.")
    return df
