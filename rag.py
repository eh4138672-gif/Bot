"""שלב השאלה (Online): הרחבת שאלה (אופציונלי), אחזור מ-ChromaDB, סינון ויצירת תשובה."""
import re
from dataclasses import dataclass, field

import chromadb
import tiktoken
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

SECTION_RE = re.compile(r"סעיף\s*(\d+\.\d+)")

NO_ANSWER ="לא מצאתי מידע על כך בתקנון המכללה. מומלץ לפנות למזכירות הסטודנטים."

QA_PROMPT = PromptTemplate(
    "את עוזרת שעונה לסטודנטיות על שאלות לגבי תקנון מכללת אופק.\n"
    "עני רק על סמך הקטעים מהתקנון שמופיעים למטה. אל תשתמשי בידע כללי.\n"
    "אם יש בקטעים מידע שקשור לשאלה, גם אם הוא מנוסח אחרת או עונה רק על חלק ממנה - עני על סמכו.\n"
    "אם השאלה לא ברורה, עני לפי הפירוש הסביר ביותר שיש לו תשובה בקטעים.\n"
    "אם שואלים באופן כללי מה יש במאגר או במה את יכולה לעזור, תארי את הנושאים מהרשימה למטה.\n"
    f"רק אם אין בקטעים שום מידע שקשור לשאלה, עני בדיוק: \"{NO_ANSWER}\"\n"
    "עני בעברית, בקצרה, וציני את מספר הסעיף שעליו התשובה מבוססת.\n"
    "הנושאים שיש במאגר: {topics}\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "שאלה: {query_str}\n"
    "תשובה: "
)

EXPAND_PROMPT = PromptTemplate(
    "שכתבי את שאלת הסטודנטית לשאילתת חיפוש מלאה וברורה בתקנון של מכללה.\n"
    "השתמשי במונחים הרשמיים שמקובלים בתקנונים, אבל הישארי בדיוק באותו נושא:\n"
    "אל תוסיפי נושאים שהסטודנטית לא שאלה עליהם, ואל תוסיפי עובדות. שאלה כללית נשארת כללית.\n"
    "שמרי תאריכים ומספרים בדיוק כפי שהם, ואל תוסיפי מספרי סעיפים שלא הופיעו בשאלה.\n"
    "החזירי רק את השאילתה, בשורה אחת.\n"
    "שאלה: {question}\n"
    "שאילתה: "
)


@dataclass
class Source:
    title: str
    file_name: str
    score: float
    text: str


@dataclass
class Answer:
    text: str
    search_query: str
    sources: list[Source] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0


class CollegeRAG:
    def __init__(self):
        config.init_models()
        # ספירה בקירוב עם ה-tokenizer של tiktoken
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.get_encoding("cl100k_base").encode
        )
        Settings.callback_manager = CallbackManager([self.token_counter])

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_or_create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        if collection.count() == 0:
            raise RuntimeError("ה-Vector DB ריק. הריצי קודם: python ingest.py")
        self.index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))

        # שאלה כמו "מה יש לך על המכללה?" היא על המאגר כולו ולא על סעיף מסוים, ו-Top K קטעים
        # לא יכולים לענות עליה. לכן המודל מקבל גם את רשימת נושאי המסמכים
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        self.topics = sorted({m["topic"] for m in metadatas if m.get("topic")})
        self.synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=QA_PROMPT.partial_format(topics=", ".join(self.topics)),
        )

    def expand_query(self, question: str) -> str:
        return str(Settings.llm.predict(EXPAND_PROMPT, question=question)).strip()

    def retrieve(self, query: str, top_k: int, section: str | None = None):
        # חיפוש סמנטי לא מבחין בין "סעיף 1.3" ל"סעיף 6.3" - אם השאלה מזכירה מספר סעיף,
        # מסננים לפי ה-metadata של הסעיף במקום לסמוך על דמיון וקטורי
        if section:
            filters = MetadataFilters(filters=[MetadataFilter(key="section", value=section)])
            nodes = self.index.as_retriever(similarity_top_k=top_k, filters=filters).retrieve(query)
            if nodes:
                return nodes
        nodes = self.index.as_retriever(similarity_top_k=top_k).retrieve(query)
        return [n for n in nodes if (n.score or 0) >= config.SIMILARITY_CUTOFF]

    def ask(self, question: str, top_k: int = config.TOP_K, expand: bool = False) -> Answer:
        self.token_counter.reset_counts()
        search_query = self.expand_query(question) if expand else question
        # מספר הסעיף נלקח רק מהשאלה המקורית - מודל ההרחבה נוטה להמציא מספרי סעיפים
        section = SECTION_RE.search(question)
        nodes = self.retrieve(search_query, top_k, section.group(1) if section else None)

        if not nodes:
            # אין קטע רלוונטי - לא שולחים כלום ל-LLM (חוסך טוקנים ומונע המצאות)
            text = NO_ANSWER
        else:
            text = str(self.synthesizer.synthesize(question, nodes)).strip()

        sources = [
            Source(
                title=n.node.metadata.get("title", ""),
                file_name=n.node.metadata.get("file_name", ""),
                score=n.score or 0.0,
                text=n.node.get_content(),
            )
            for n in nodes
        ]
        return Answer(
            text=text,
            search_query=search_query,
            sources=sources,  # מוצגים גם כשאין תשובה - כדי לראות מה נשלף ולמה זה לא הספיק
            prompt_tokens=self.token_counter.prompt_llm_token_count,
            completion_tokens=self.token_counter.completion_llm_token_count,
            embedding_tokens=self.token_counter.total_embedding_token_count,
        )
