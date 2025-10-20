# === Standard Library ===
import os
import re
import json
import base64
import mimetypes
from pathlib import Path

# === Third-Party ===
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # (kept if you need it elsewhere)
from dotenv import load_dotenv
from html import escape
from google import genai
load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_respose(model:str, prompt:str) -> str:
    """
    Get response from Gemini API.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# === Data Loading ===
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and derive date parts commonly used in charts."""
    df = pd.read_csv(csv_path)
    # Be tolerant if 'date' exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df
