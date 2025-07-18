from db import init_db
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# 0. Envirionment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Local DB init
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(PROJECT_ROOT, "data")
LOCAL_DB_PATH = os.path.join(DB_DIR, "news.db")
if os.path.exists(LOCAL_DB_PATH):
    os.remove(LOCAL_DB_PATH)
    print("🗑️ Deleted old local news.db")
init_db()
print("✅ Initialized empty local DB")

# 2. Empty HF Dataset upload
api = HfApi()
api.upload_file(
    path_or_fileobj=LOCAL_DB_PATH,
    path_in_repo="news.db",
    repo_id="HSJay/news_db",
    repo_type="dataset",
    token=HF_TOKEN
)
print("🚀 Uploaded empty news.db to Hugging Face Dataset")