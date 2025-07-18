# NewsCurator

LLM 기반 **뉴스 일일동향 AI 에이전트**  
최신 뉴스를 수집하고, 중복 필터링 & 카테고리 분류 & 요약 & 중요도 스코어링을 통해  
**가장 중요한 뉴스를 자동으로 선별해고 요약해주는 서비스**  

---

## Demo
[Hugging Face Spaces](https://huggingface.co/spaces/HSJay/NewsCurator)  

---

## Features
- 뉴스 자동 수집 (RSS, API, 크롤러)
- 중복 뉴스 필터링 및 DB 저장
- LLM 기반 뉴스 카테고리 분류
- 3줄 요약
- 중요도 분석
- 주요 뉴스만 추출하여 게시판 제공

---

## Architecture

NewsCurator/
├─ data/
│   ├─ embeddings                    # Chroma DB
│   └─ news.db                       # SQLite 뉴스 데이터베이스
├─ workflow/
│   ├─ main.py                       # LangGraph 워크플로우 엔트리 포인트
│   └─ nodes/
│       ├─ rss_node.py               # RSS 피드 수집
│       ├─ summarizer_node.py        # 멀티모델 요약
│       ├─ related_news_node.py      # 임베딩 기반 연관 기사 찾기
│       ├─ embedding_node.py         # 기사 본문 임베딩하여 chroma DB 저장
│       ├─ db_node.py                # summaries 리스트를 JSON으로 저장
│       └─ update_db_node.py         # HF Dataset 업로드
├─ scripts/
│   ├─ script_to_hf.sh               # 전체 실행
├─ utils/
│   ├─ reset_db.py                   # 로컬/원격 DB 초기화 스크립트
│   └─ db.py                         # get_connection, init_db 등 DB 유틸
└─ web/ (HF Space)
    ├─ src/
    │   ├─ streamlit_app.py          # Streamlit 앱: DB 다운로드 → HTML 임베드
    │   └─ template.html             # 커스텀 UI 템플릿 (카테고리, 페이징, 요약, 토글)
    ├─ requirements.txt              
    └─ huggingface.yaml              


## LangGraph Workflow
![LangGraph Workflow](./NewsCurator_Mermaidchart.png)


---

## Tech Stack
- **Language**: Python 3.12
- **Framework**: FastAPI
- **Database**: SQLite / ChromaDB
- **LLM**: Ollama(로컬) or OpenAI API
- **Deployment**: Hugging Face Spaces / Docker

---

## Github
git clone https://github.com/HSJay/news-curator.git
