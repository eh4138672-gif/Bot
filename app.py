"""ממשק צ'אט ב-Streamlit.

הרצה:  streamlit run app.py
"""
import streamlit as st

import config
from rag import CollegeRAG

st.set_page_config(page_title="בוט תקנון המכללה", page_icon="🎓")

# יישור לימין עבור עברית
st.markdown(
    """
    <style>
    .stApp, .stChatMessage, .stMarkdown, .stExpander, section[data-testid="stSidebar"] {
        direction: rtl; text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bot() -> CollegeRAG:
    return CollegeRAG()


def show_sources(answer):
    if not answer.sources:
        return
    with st.expander(f"📚 מקורות ({len(answer.sources)})"):
        for i, src in enumerate(answer.sources, 1):
            st.markdown(f"**{i}. {src.title}**  \n`{src.file_name}` · ציון דמיון: {src.score:.3f}")
            st.text(src.text)
    st.caption(
        f"שאילתת חיפוש: {answer.search_query} · "
        f"טוקנים: {answer.prompt_tokens} prompt / {answer.completion_tokens} completion"
    )


st.title("🎓 בוט תקנון מכללת אופק")
st.write("שאלי שאלה על תקנון המכללה. הבוט עונה רק מתוך מסמכי התקנון.")

with st.sidebar:
    st.header("הגדרות")
    top_k = st.slider("Top K – כמה קטעים לשלוף", 1, 10, config.TOP_K)
    expand = st.toggle("הרחבת השאלה לפני החיפוש", value=True)
    if st.button("ניקוי השיחה"):
        st.session_state.messages = []

try:
    bot = get_bot()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("על מה אפשר לשאול")
    st.markdown("\n".join(f"- {topic}" for topic in bot.topics))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("answer"):
            show_sources(msg["answer"])

if question := st.chat_input("למשל: עד מתי אפשר לערער על ציון?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפשת בתקנון..."):
            answer = bot.ask(question, top_k=top_k, expand=expand)
        st.markdown(answer.text)
        show_sources(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})
