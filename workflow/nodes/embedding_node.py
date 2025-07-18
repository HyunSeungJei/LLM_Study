import os
from typing import Dict, List
import chromadb
from chromadb.config import Settings


# For embedding 
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
# Environment variable for Chroma DB directory

# 2) Chroma 클라이언트 설정
CHROMA_DIR = "../data/embeddings"
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Get or create the collection for news embeddings
collection = chroma_client.get_or_create_collection(
    name="news-embeddings",
    metadata={"hnsw:space": "cosine"}
)

# 1. Embedding
# Load embedding model
EMBED_MODEL = SentenceTransformer("BAAI/bge-m3")

# 2. Ollama embedding
# "dengcao/Qwen3-Embedding-0.6B"
# "bge-m3", "BAAI/bge-m3"
# EMBED_MODEL = OllamaEmbeddings(model="bge-m3")

def embedding_node(state: Dict) -> Dict:
    """
    Embedding node for LangGraph:
    - state['articles']의 각 article에서 URL을 고유 ID로 사용
    - title + summaries를 결합해 임베딩 생성
    - ChromaDB에 upsert하여 벡터와 메타데이터 저장
    """
    articles: List[Dict] = state.get("articles", [])
    for art in articles:
        uid = art.get("url") or None
        if not uid:
            continue

        # Prepare text for embedding
        summaries = art.get("summaries", [])
        text = art.get("title", "")
        if summaries:
            text += " " + " ".join(summaries)
        text = text[:1000]

        # Generate embedding vector
        embedding = EMBED_MODEL.encode(text, normalize_embeddings=True).tolist()

        # Prepare metadata
        metadata = {
            "title": art.get("title", ""),
            "url": uid,
            "published": art.get("published", "")
        }

        # Upsert into Chroma collection
        collection.upsert(
            ids=[uid],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )

    print(f"✅ Upserted {len(articles)} embeddings into ChromaDB at {CHROMA_DIR}")
    return state