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
