"""הגדרות מרכזיות של המערכת - כל ההחלטות הטכנולוגיות במקום אחד."""
import os
from pathlib import Path

import truststore

# הסינון (NetFree) מחליף את תעודת ה-SSL בתעודה שלו. Python 3.13 דוחה אותה בבדיקה המחמירה שלו,
# ולכן מאמתים מול מאגר התעודות של Windows (שבו תעודת הסינון מותקנת)
truststore.inject_into_ssl()

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# --- נתיבים ---
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "college_regulations"

# --- מודלים ---
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

# --- חלוקה ל-Chunks ---
# בעברית כל תו הוא בערך טוקן, ולכן 512 טוקנים הם רק כ-100 מילים - פחות מסעיף אחד.
# 1024 מספיק לסעיף שלם יחד עם ה-metadata
CHUNK_SIZE = 1024     # טוקנים לכל chunk
CHUNK_OVERLAP = 100   # כ-10% חפיפה (רלוונטי רק לסעיף ארוך שמפוצל)

# --- אחזור ---
TOP_K = 4                    # כמה chunks מחזירים מה-Vector DB
# בבדיקות, שאלות שאין עליהן תשובה קיבלו ציון 0.48-0.57, כמו חלק מהשאלות הרלוונטיות - אז סף
# לא מספיק כדי להחליט אם יש תשובה (את זה ה-LLM מחליט לפי הפרומפט). הסף רק זורק קטעים שממש לא קשורים
SIMILARITY_CUTOFF = 0.40


def init_models() -> None:
    """מגדיר את המודלים של LlamaIndex. נקרא פעם אחת בתחילת כל סקריפט."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "חסר OPENAI_API_KEY. העתיקי את .env.example ל-.env והכניסי את המפתח שלך."
        )
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
