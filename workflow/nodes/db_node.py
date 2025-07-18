# nodes/db_node.py
import json
from utils.db import get_connection

def db_node(state: dict) -> dict:
    """
    DB 저장 노드
    - state['articles']에 있는 뉴스들을 news 테이블에 저장
    - 'summaries' 리스트를 JSON으로 summary 컬럼에 저장
    - 중복 URL은 무시 (INSERT OR IGNORE)
    """
    articles = state.get("articles", [])
    if not articles:
        print("⚠️ 저장할 뉴스가 없습니다.")
        return state

    conn = get_connection()
    cur = conn.cursor()

    saved_count = 0
    for article in articles:
        summaries = article.get("summaries", [])
        summary_json = json.dumps(summaries, ensure_ascii=False) if summaries else None
        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO news
                (title, url, content, summary, source, published, category, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.get("title"),
                    article.get("url"),
                    article.get("content", ""),   
                    summary_json,                 
                    article.get("source"),
                    article.get("published"),
                    article.get("category", ""),
                    article.get("score", None)
                )
            )
            if cur.rowcount > 0:
                saved_count += 1
        except Exception as e:
            print(f"❌ 저장 실패: {article.get('title')} - {e}")

    conn.commit()
    conn.close()
    print(f"✅ DB 저장 완료: {saved_count}/{len(articles)}개 저장됨")

    return state
