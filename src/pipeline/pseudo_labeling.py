import asyncio
import aiofiles
import time 
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from pathlib import Path
from src.utils.helper import load_from_env

PROMPT_JSONL = "Dataset/llm_prompts.jsonl"
RESPONSE_JSONL = "Dataset/llm_raw_outputs.jsonl"
MAX_CONCURRENT = 5  # Control concurrency

# Round-robin across API keys
api_credentials = [
    # {"api_key": load_from_env("groqKey1"), "base_url": "https://api.groq.com/openai/v1", "model": "llama3-8b-8192"},
    # {"api_key": load_from_env("groqKey2"), "base_url": "https://api.groq.com/openai/v1", "model": "llama3-8b-8192"},
    # {"api_key": load_from_env("groqKey3"), "base_url": "https://api.groq.com/openai/v1", "model": "llama3-8b-8192"},
    {"api_key": load_from_env("togetherKey1"), "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"},
    {"api_key": load_from_env("togetherKey2"), "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"},
    {"api_key": load_from_env("togetherKey3"), "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"},
]

api_index = 0

# === Helpers === #
def extract_score(resp: str):
    for token in str(resp).split():
        if token.isdigit() and 0 <= int(token) <= 5:
            return int(token)
    return None

def call_llm(prompt: str, max_retries: int = 6) -> str:
    global api_index
    for attempt in range(max_retries):
        creds = api_credentials[api_index]
        api_index = (api_index + 1) % len(api_credentials)

        client = OpenAI(api_key=creds["api_key"], base_url=creds["base_url"])

        try:
            response = client.chat.completions.create(
                model=creds["model"],
                messages=[
                    {"role": "system", "content": "Rate how well a resume matches a job description on a scale of 0 to 5."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                wait = 5 * (2 ** attempt)
                print(f"⚠️ Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"error: {error_str}"

    return f"error: Gave up after {max_retries} retries"

async def call_and_write(row, sem, executor, outfile_path):
    async with sem:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(executor, call_llm, row["Prompt"])
        label = extract_score(response)

        record = {
            "JD_ID": row["JD_ID"],
            "Resume_ID": row["Resume_ID"],
            "Response": response,
            "Label": label
        }

        if label is not None:
            async with aiofiles.open(outfile_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(record) + "\n")
            print(f"✅ JD: {row['JD_ID']} | Resume: {row['Resume_ID']} | Label: {label}")
        else:
            print(f"⛔ JD: {row['JD_ID']} | Resume: {row['Resume_ID']} | Skipped due to null label")

        return record

def generate_labels(df: pd.DataFrame, skip_labels: bool = False) -> pd.DataFrame:
    async def _run():
        # === Only use this if you want to regenerate prompts ===
        # df = create_prompts(df)
        # save_prompts_jsonl(df)

        print("📂 Loading prompts from existing file...")
        prompts = []
        with open(PROMPT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    prompts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        prompt_df = pd.DataFrame(prompts)
        df_prompted = pd.merge(df, prompt_df, on=["JD_ID", "Resume_ID"], how="inner")

        seen_pairs = set()
        if Path(RESPONSE_JSONL).exists():
            with open(RESPONSE_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        seen_pairs.add((record["JD_ID"], record["Resume_ID"]))
                    except json.JSONDecodeError:
                        continue

        print(f"🔁 Skipping {len(seen_pairs)} seen pairs")

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

        tasks = []

        if not skip_labels:
            for _, row in df_prompted.iterrows():
                key = (row["JD_ID"], row["Resume_ID"])
                if key in seen_pairs:
                    continue
                tasks.append(call_and_write(row, sem, executor, RESPONSE_JSONL))

        await asyncio.gather(*tasks)

        # Reload and merge labeled results
        with open(RESPONSE_JSONL, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if "Label" in line]

        response_df = pd.DataFrame(results)
        merged = pd.merge(df, response_df[["JD_ID", "Resume_ID", "Label"]], on=["JD_ID", "Resume_ID"], how="left")
        print("🏷️ Labels extracted and merged.")

        # ✅ Filter only labeled rows
        labeled_df = merged[merged["Label"].notnull()]
        return labeled_df

    return asyncio.run(_run())
