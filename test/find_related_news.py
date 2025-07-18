import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hdbscan
except ImportError:
    hdbscan = None

# ✅ 최신 권장 패키지
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# ✅ 1) ChromaDB 로드
CHROMA_DIR = "../data/embeddings"
COLLECTION_NAME = "news-embeddings"

# embed_model = SentenceTransformer("BAAI/bge-m3")

# vectordb = Chroma(
#     persist_directory=CHROMA_DIR,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embed_model
# )


from langchain_huggingface import HuggingFaceEmbeddings

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


# ✅ 2) ChromaDB에서 모든 벡터 가져오기
def load_all_embeddings_from_chroma():
    all_ids, all_titles, all_embeddings = [], [], []
    data = vectordb._collection.get(include=["embeddings", "metadatas", "documents"])

    for idx, emb in enumerate(data["embeddings"]):
        all_ids.append(data["ids"][idx])
        all_titles.append(data["metadatas"][idx]["title"])
        all_embeddings.append(np.array(emb))
    
    if len(all_embeddings) == 0:
        print("❌ ChromaDB에 저장된 벡터가 없습니다!")
        return [], [], np.array([])
    return all_ids, all_titles, np.vstack(all_embeddings)

# ✅ 3) 알고리즘별 클러스터링
def cluster_dbscan(embeddings, eps=0.35, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return clustering.fit_predict(embeddings)

def cluster_kmeans(embeddings, n_clusters=5):
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    return clustering.fit_predict(embeddings)

def cluster_hdbscan(embeddings, min_cluster_size=2):
    if hdbscan is None:
        raise ImportError("`pip install hdbscan` 필요")
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return clustering.fit_predict(embeddings)

# ✅ 4) 클러스터 내 유사도 계산 & 출력
def print_clusters_with_similarity(algorithm_name, titles, embeddings, labels):
    cluster_dict = {}
    for idx, label in enumerate(labels):
        cluster_dict.setdefault(label, []).append(idx)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0

    print(f"\n\n### {algorithm_name} 결과 ###")
    print(f"클러스터 개수: {n_clusters}, 노이즈 비율: {noise_ratio:.2%}")

    for cluster_id, indices in cluster_dict.items():
        if cluster_id == -1:
            print("\n❌ 독립 기사 (-1):")
        else:
            print(f"\n✅ 클러스터 {cluster_id} (연관 기사 {len(indices)}개)")

        # 클러스터 내 기사 제목과 유사도 계산
        cluster_titles = [titles[i] for i in indices]
        cluster_embeds = embeddings[indices]

        # 클러스터 중심 벡터 구하기
        cluster_centroid = np.mean(cluster_embeds, axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_centroid, cluster_embeds)[0]

        # 유사도 높은 순으로 정렬
        sorted_idx = np.argsort(-sims)

        for rank, i in enumerate(sorted_idx):
            print(f"[유사도 {sims[i]:.4f}] {cluster_titles[i]}")


def find_related_for_all_cosine_similarity(titles, embeddings, threshold=0.8):
    print(f"\n\n### Cosine similarity 결과 (유사도 {threshold} 이상만 출력) ###")
    n = len(titles)
    sim_matrix = cosine_similarity(embeddings)  # N x N 유사도 행렬

    for i in range(n):
        print(f"\n\n=== 기준 뉴스 #{i+1}: {titles[i]} ===")

        sims = sim_matrix[i]
        # 자기 자신 제외
        sims[i] = -1  

        # ✅ threshold 이상인 인덱스만 필터링
        valid_idx = [j for j, sim in enumerate(sims) if sim >= threshold]

        if not valid_idx:
            print("  ❌ 유사도 0.8 이상 연관 기사 없음")
            continue

        # ✅ 유사도 높은 순 정렬
        sorted_idx = sorted(valid_idx, key=lambda j: sims[j], reverse=True)

        for rank, j in enumerate(sorted_idx, start=1):
            print(f"  {rank}. [유사도 {sims[j]:.4f}] {titles[j]}")


def find_related_for_all_similarity_by_vector(titles, embeddings, topn=10, threshold=0.8):
    print(f"\n\n### Similarity Search by Vector (유사도 {threshold} 이상만 출력) ###")
    n = len(titles)

    for i, title in enumerate(titles):
        print(f"\n\n=== 기준 뉴스 #{i+1}: {title} ===")

        # ✅ 쿼리 벡터는 이미 로드된 embeddings[i]
        query_vector = embeddings[i]

        # ✅ ANN 검색 (자기 자신 포함)
        results = vectordb.similarity_search_by_vector(query_vector, k=topn + 1)

        found = False
        for doc in results:
            # ✅ 후보 벡터 다시 가져오기
            doc_data = vectordb._collection.get(ids=[doc.metadata["url"]], include=["embeddings"])
            if not doc_data["embeddings"]:
                continue
            candidate_vec = np.array(doc_data["embeddings"][0]).reshape(1, -1)

            # ✅ 유사도 직접 계산
            sim = cosine_similarity(query_vector.reshape(1, -1), candidate_vec)[0][0]

            # 자기 자신 제외
            if doc.metadata["title"] == title:
                continue

            # threshold 이상인 것만 출력
            if sim >= threshold:
                found = True
                print(f"  [유사도 {sim:.4f}] {doc.metadata['title']}")

        if not found:
            print("  ❌ 유사도 0.8 이상 연관 기사 없음")



# ✅ 6) 실행
if __name__ == "__main__":
    ids, titles, embeddings = load_all_embeddings_from_chroma()
    if len(embeddings) == 0:
        exit(0)

    # DBSCAN
    labels_dbscan = cluster_dbscan(embeddings, eps=0.35, min_samples=2)
    print_clusters_with_similarity("DBSCAN", titles, embeddings, labels_dbscan)

    # HDBSCAN (선택)
    if hdbscan is not None:
        labels_hdbscan = cluster_hdbscan(embeddings, min_cluster_size=2)
        print_clusters_with_similarity("HDBSCAN", titles, embeddings, labels_hdbscan)

    # KMeans
    labels_kmeans = cluster_kmeans(embeddings, n_clusters=5)
    print_clusters_with_similarity("KMeans", titles, embeddings, labels_kmeans)

    # ✅ 추가: 유사도 검색 기반 알고리즘
    find_related_for_all_cosine_similarity(titles, embeddings, threshold=0.8)    
    
    find_related_for_all_similarity_by_vector(titles, embeddings, topn=10, threshold=0.8)
