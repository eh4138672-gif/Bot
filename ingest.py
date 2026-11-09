"""שלב ההכנה (Offline): טעינת המסמכים, חלוקה ל-chunks, יצירת embeddings ושמירה ב-ChromaDB.

הרצה:  python ingest.py
"""
import re

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def load_documents():
    docs = SimpleDirectoryReader(
        str(config.DATA_DIR), required_exts=[".md", ".txt"], encoding="utf-8"
    ).load_data()
    for doc in docs:
        # השורה הראשונה בכל מסמך היא הכותרת - שומרים אותה כ-metadata לצורך הצגת מקורות
        doc.metadata["title"] = doc.text.strip().splitlines()[0].lstrip("# ").strip()
        # לא שולחים את נתיב הקובץ ואת התאריכים ל-embedding - זה רק רעש
        doc.excluded_embed_metadata_keys = [
            "file_path", "file_type", "file_size", "creation_date", "last_modified_date",
        ]
        doc.excluded_llm_metadata_keys = doc.excluded_embed_metadata_keys
    return docs


SECTION_RE = re.compile(r"סעיף\s+(\d+\.\d+)")


def split_documents(docs):
    """חלוקה לפי מבנה: קודם לפי כותרות (כל סעיף בתקנון = chunk), ורק סעיף ארוך מדי
    מפוצל שוב לפי טוקנים. כך כותרת הסעיף תמיד נשארת יחד עם התוכן שלו."""
    sections = MarkdownNodeParser().get_nodes_from_documents(docs)
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(sections)

    # כותרת בלי תוכן (למשל "## פרק 1 – מועדי בחינות") היא רק רעש בחיפוש
    nodes = [n for n in nodes if len(n.text.strip().splitlines()) > 1]
    for node in nodes:
        # מספר הסעיף נשמר כ-metadata כדי לאפשר סינון בשאלות כמו "מה אומר סעיף 1.3?"
        # תת-כותרת (כמו "שלבי הפרויקט") מקבלת את מספר הסעיף שמעליה מתוך header_path
        match = SECTION_RE.search(node.text) or SECTION_RE.search(node.metadata.get("header_path", ""))
        if match:
            node.metadata["section"] = match.group(1)

        # ל-embedding נכנס רק נושא המסמך בקצרה + טקסט הסעיף. metadata שחוזר בכל chunk
        # (שם המכללה, נתיב כותרות, שם קובץ) גורם לכל הווקטורים להיראות דומים
        node.metadata["topic"] = node.metadata["title"].split(" – ")[0]
        node.excluded_embed_metadata_keys = [k for k in node.metadata if k != "topic"]
        node.excluded_llm_metadata_keys = [k for k in node.metadata if k not in ("topic", "section")]
    return nodes


def build_index():
    config.init_models()
    docs = load_documents()
    nodes = split_documents(docs)
    print(f"נטענו {len(docs)} מסמכים, שחולקו ל-{len(nodes)} chunks")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # מוחקים את האוסף הקודם, אחרת כל הרצה מוסיפה עותק נוסף של אותם chunks
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # האוסף עוד לא קיים - הרצה ראשונה
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    print(f"נשמרו {collection.count()} וקטורים ב-ChromaDB ({config.CHROMA_DIR})")


if __name__ == "__main__":
    build_index()
