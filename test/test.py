from huggingface_hub import hf_hub_download

repo_id = "HSJay/news_db"   # Dataset repo
filename = "news.db"

try:
    db_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename
    )
    print("✅ DB downloaded successfully!")
    print("DB local path:", db_path)
except Exception as e:
    print("❌ Download failed:", e)