# nodes/related_news_node.py
import json
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "../data/news.db"
CHROMA_DIR = "../data/embeddings"
COLLECTION_NAME = "news-embeddings"

# ✅ 임베딩 로드
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embed_model
)

def load_all_embeddings_from_chroma():
    """ChromaDB에서 모든 뉴스 벡터 + 메타데이터 로드"""
    data = vectordb._collection.get(include=["embeddings", "metadatas"])
    ids = data["ids"]  # 여기서는 url을 id로 사용
    titles = [m["title"] for m in data["metadatas"]]
    embeddings = np.vstack(data["embeddings"])
    return ids, titles, embeddings

def compute_related_news(ids, embeddings, threshold=0.8):
    """
    cosine_similarity 기반으로 연관 기사 찾기
    threshold 이상인 뉴스들의 id를 리스트로 반환
    """
    n = len(ids)
    sim_matrix = cosine_similarity(embeddings)
    related_dict = {}

    for i in range(n):
        sims = sim_matrix[i]
        sims[i] = -1  # 자기 자신 제외
        # threshold 이상인 기사만 필터링
        related_idx = [j for j, s in enumerate(sims) if s >= threshold]
        # 유사도 높은 순으로 정렬해서 id 리스트 생성
        related_ids = [ids[j] for j in sorted(related_idx, key=lambda x: sims[x], reverse=True)]
        related_dict[ids[i]] = related_ids

    return related_dict

def update_related_news_in_db(related_dict):
    """
    DB news 테이블에 related_news 업데이트
    related_dict: { "url_1": ["url_2", "url_3"], ... }
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for news_id, related_ids in related_dict.items():
        json_data = json.dumps(related_ids, ensure_ascii=False)
        cur.execute("UPDATE news SET related_news = ? WHERE url = ?", (json_data, news_id))

    conn.commit()
    conn.close()
    print(f"✅ DB에 {len(related_dict)}개의 related_news 업데이트 완료")

def related_news_node(state: dict) -> dict:
    """
    LangGraph용 Node
    - embedding_node 실행 이후 호출
    - DB에서 연관 기사 id 리스트 업데이트
    """
    print("🔄 연관 뉴스 분석 중...")
    ids, titles, embeddings = load_all_embeddings_from_chroma()
    related_dict = compute_related_news(ids, embeddings, threshold=0.8)
    update_related_news_in_db(related_dict)
    return state