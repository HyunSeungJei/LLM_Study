import feedparser
from urllib.parse import urlparse

def rss_node(state: dict) -> dict:
    """
    Node1: RSS 수집 노드
    - state['rss_url']에서 RSS를 파싱해 최신 뉴스 리스트를 반환
    - state['limit']이 있으면 해당 개수만큼 잘라서 가져옴
    - LangGraph state 딕셔너리에 articles 리스트 추가
    
    Args:
        state (dict): {
            "rss_url": str,
            "limit": int (optional)  # 가져올 기사 개수 제한
        }
        
    Returns:
        dict: {
            "articles": [
                { "title": str, "url": str, "published": str, "source": str }
            ]
        }
    """
    rss_url = state.get("rss_url")
    if not rss_url:
        raise ValueError("rss_url is required in state")
    
    # 가져올 기사 개수, 지정 안 하면 전체
    limit = state.get("limit")  
    feed = feedparser.parse(rss_url)
    
    entries = feed.entries
    if limit:
        entries = entries[:limit]  # 처음 limit개 항목만
    
    articles = []
    for entry in entries:
        title = entry.title
        url = entry.link
        published = getattr(entry, "published", "")
        
        parsed_url = urlparse(url)
        source = parsed_url.netloc.replace("www.", "")
        
        articles.append({
            "title": title,
            "url": url,
            "published": published,
            "source": source
        })
    
    state["articles"] = articles
    return state