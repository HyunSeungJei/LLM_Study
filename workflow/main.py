from workflow import create_news_workflow
from utils.db import init_db
from config import RSS_URLS

def main():
    # DB 초기화 (최초 1회만)
    init_db()

    # Workflow 생성
    workflow = create_news_workflow()

    # RSS URL 하나씩 실행
    for index, rss_url in enumerate(RSS_URLS):
        print("=" * 80)
        print(f"{index} : News parsing from {rss_url}")
        print("=" * 80)

        rss_state = {
            "rss_url": rss_url, 
            "limit": 100
        }
        result = workflow.invoke(rss_state)
        print("DB Update Done")
        print(f"Updated News count: {len(result.get('articles', []))}")

if __name__ == "__main__":
    main()