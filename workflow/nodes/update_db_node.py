# from huggingface_hub import HfApi
# from dotenv import load_dotenv
# import os

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# DATASET_REPO = "HSJay/news_db"
# DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/news.db")

# def update_db_node(state: dict) -> dict:
#     """
#     Hugging Face Dataset에 news.db 최신 버전 업로드
#     """
#     if not os.path.exists(DB_PATH):
#         print(f"❌ DB 파일을 찾을 수 없습니다: {DB_PATH}")
#         return state

#     if not HF_TOKEN:
#         print("⚠️ HF_TOKEN이 설정되지 않아 업로드를 건너뜁니다.")
#         return state

#     print("📤 Hugging Face Dataset에 news.db 업로드 중...")
#     api = HfApi()
#     api.upload_file(
#         path_or_fileobj=DB_PATH,
#         path_in_repo="news.db",        # HF Dataset 내 경로
#         repo_id=DATASET_REPO,
#         repo_type="dataset",
#         token=HF_TOKEN
#     )
#     print("🚀 HF Dataset 업데이트 완료!")

#     return state


# nodes/update_db_node.py
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "HSJay/news_db"

BASE_DIR = os.path.join(os.path.dirname(__file__), "../../data")
DB_PATH = os.path.join(BASE_DIR, "news.db")
CHROMA_DIR = os.path.join(BASE_DIR, "embeddings")  # Chroma 저장 폴더

def upload_file_to_hf(api: HfApi, local_path: str, remote_path: str):
    """HF Dataset에 단일 파일 업로드 (덮어쓰기)"""
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"✅ {remote_path} 업로드 완료")

def update_db_node(state: dict) -> dict:
    """
    HF Dataset(HSJay/news_db)에 최신 news.db + ChromaDB 전체 업로드
    - 기존 파일은 덮어쓰기됨
    - HF Dataset은 Git LFS 버전 관리됨
    """

    if not HF_TOKEN:
        print("⚠️ HF_TOKEN이 설정되지 않아 Hugging Face 업로드 건너뜁니다.")
        return state

    api = HfApi()

    # ✅ 1) news.db 업로드
    if os.path.exists(DB_PATH):
        print("📤 news.db 업로드 중...")
        upload_file_to_hf(api, DB_PATH, "news.db")
    else:
        print(f"❌ news.db를 찾을 수 없습니다: {DB_PATH}")

    # ✅ 2) ChromaDB 전체 업로드
    if os.path.exists(CHROMA_DIR):
        print("📤 Chroma 벡터 DB 업로드 중...")
        for root, _, files in os.walk(CHROMA_DIR):
            for f in files:
                local_path = os.path.join(root, f)
                # HF repo 내 상대 경로 유지
                relative_path = os.path.relpath(local_path, start=BASE_DIR)
                upload_file_to_hf(api, local_path, relative_path)
    else:
        print(f"⚠️ Chroma 벡터 DB 폴더를 찾을 수 없습니다: {CHROMA_DIR}")

    print(f"🚀 Hugging Face Dataset({DATASET_REPO}) 업데이트 완료!")
    return state

