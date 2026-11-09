"""מריץ סט שאלות קבוע ושומר את התשובות והמקורות לקובץ examples.md.

הרצה:  python run_tests.py
"""
import config
from rag import CollegeRAG

QUESTIONS = [
    # (שאלה, הערה - מה בודקים)
    ("עד מתי אפשר להגיש ערעור על ציון בבחינה?", "שאלה ישירה"),
    ("איך יוצאים מקורס באמצע הסמסטר וכמה זה עולה?", "ניסוח שונה מהמסמך (במסמך: ביטול קורס)"),
    ("כמה פעמים מותר לי לא להגיע לתרגול?", "ניסוח שונה מהמסמך (במסמך: היעדרות)"),
    ("מה אומר סעיף 1.3?", "חיפוש לפי מספר סעיף"),
    ("איזה מלגות יש ומה הסכום המקסימלי שאפשר לקבל?", "תשובה שמשלבת כמה חלקים"),
    ("מילואים", "שאלה מתומצתת מאוד - מקרה לבדיקת הרחבת שאלה"),
    ("מותר להשתמש בצ'אט GPT בעבודה?", "מילה שלא מופיעה במסמך (במסמך: בינה מלאכותית)"),
    ("שכחתי סיסמה מה עושים?", "שאלה בשפה יומיומית (במסמך: איפוס סיסמה)"),
    ("אני במילואים, מה מגיע לי?", "תשובה שמפוזרת בכמה מסמכים"),
    ("מה שעות הפתיחה של הקפיטריה?", "שאלה שאין עליה תשובה במאגר"),
]


def main():
    bot = CollegeRAG()
    lines = ["# דוגמאות לשאלות ותשובות", "",
             f"מודל: {config.LLM_MODEL} · Embeddings: {config.EMBED_MODEL} · "
             f"Top K: {config.TOP_K} · סף דמיון: {config.SIMILARITY_CUTOFF}", ""]

    for i, (question, note) in enumerate(QUESTIONS, 1):
        for expand in (False, True):
            ans = bot.ask(question, expand=expand)
            mode = "עם הרחבת שאלה" if expand else "בלי הרחבת שאלה"
            print(f"\n[{i}] {question} ({mode})\n{ans.text}")
            for s in ans.sources:
                print(f"   - {s.file_name} ({s.score:.3f})")

            lines += [f"## {i}. {question} – {mode}", f"*{note}*", ""]
            if expand:
                lines += [f"**שאילתת חיפוש מורחבת:** {ans.search_query}", ""]
            lines += [f"**תשובה:** {ans.text}", "", "**מקורות שאוחזרו:**"]
            lines += [f"- {s.title} (`{s.file_name}`, ציון {s.score:.3f})" for s in ans.sources] or ["- אין"]
            lines += ["", f"טוקנים: {ans.prompt_tokens} prompt, {ans.completion_tokens} completion", ""]

    (config.BASE_DIR / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nהתוצאות נשמרו ב-examples.md")


if __name__ == "__main__":
    main()
