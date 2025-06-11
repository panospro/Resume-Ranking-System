from dotenv import load_dotenv
import os
import pandas as pd

def load_from_env(env_var: str) -> pd.DataFrame:
    load_dotenv()
    return os.getenv(env_var)
