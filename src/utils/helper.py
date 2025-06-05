from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd

# Loads a DataFrame from a file path specified in a .env variable.
def load_dataframe_from_env(env_var: str) -> pd.DataFrame:
    load_dotenv()
    file_path = os.getenv(env_var)

    if not file_path:
        raise ValueError(f"Environment variable '{env_var}' is not set in the .env file.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at path: {path}")

    return path
