# # nodes/summarizer_node.py
# import subprocess
# from typing import Dict, List
# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate

# # Load environment variables
# load_dotenv()

# # 모델 리스트
# # MODEL_NAMES = ["llama3", "qwen3:0.6b", "qwen2.5:3b"]
# # MODEL_NAMES = ["deepseek-r1:1.5b"]
# MODEL_NAMES = ["qwen3:0.6b", "PetrosStav/gemma3-tools:4b"]

# 1. 
# SUMMARIZE_TEMPLATE = PromptTemplate(
#     input_variables=["model_name", "content"],
#     template="""
# 당신은 뉴스 요약가입니다. 오직 요약문만 출력하세요. 어떤 해설이나 Reasoning을 포함하지 마세요.
# --- 요약 형식: "{model_name}: 요약문"
# --- 세 줄을 넘지 않도록 요약해 주세요.

# {content}
# """
# )

# def summarize_with_model(model_name: str, text: str) -> str:
#     """
#     Use Ollama CLI to summarize text with the specified model.
#     Filters out any reasoning or chain-of-thought lines, keeping only the summary lines.
#     """
#     # Build prompt
#     prompt = SUMMARIZE_TEMPLATE.format(model_name=model_name, content=text[:2000])
#     # Call Ollama
#     result = subprocess.run(
#         ["ollama", "run", model_name],
#         input=prompt,
#         text=True,
#         capture_output=True,
#         check=True,
#     )
#     raw = result.stdout.strip()
#     # Post-process: keep only lines starting with "model_name:"
#     lines = raw.splitlines()
#     prefix = f"{model_name}:"
#     idx = next((i for i, line in enumerate(lines) if line.startswith(prefix)), 0)
#     summary_lines = lines[idx:]
#     return "\n".join(summary_lines).strip()


# def summarizer_node(state: Dict) -> Dict:
#     """
#     Summarizer node for LangGraph:
#     - Generates and prints summaries with multiple models
#     - Stores list in article['summaries']
#     """
#     articles: List[Dict] = state.get("articles", [])
#     if not articles:
#         print("⚠️ No articles to summarize.")
#         return state

#     for article in articles:
#         title = article.get("title", "")
#         content = article.get("content") or article.get("title", "")
#         if not content:
#             article["summaries"] = []
#             continue

#         summaries = []
#         print(f"🔍 Summarizing article: {title}")
#         for model in MODEL_NAMES:
#             summary = summarize_with_model(model, content)
#             if summary:
#                 summaries.append(summary)
#                 # Print each model's summary to console for testing
#                 print(f"{model} summary: {summary}\n")
#         article["summaries"] = summaries
#         print(f"✅ Completed summaries for: {title}")

#     state["articles"] = articles
#     return state

import subprocess
from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# 모델 리스트
MODEL_NAMES = ["qwen3:0.6b", "PetrosStav/gemma3-tools:4b"]

# Chain-of-Thought 프롬프트 템플릿
SUMMARIZE_TEMPLATE = PromptTemplate(
    input_variables=["content"],
    template="""
SYSTEM: 당신은 한국어 뉴스 요약 전문가입니다.
User: 다음 뉴스 본문에서 핵심 문장 3개를 추출하고, 각 문장을 결합해 3문장 이내로 최종 요약을 작성해주세요. 아래 형식을 따르세요.

뉴스 본문:
{content}

출력 형식:
Step 1: 핵심 문장
- 문장1
- 문장2
- 문장3

Step 2: 최종 요약
1) 요약문1
2) 요약문2
3) 요약문3
"""
)


def extract_final_summary(cot_output: str) -> str:
    """
    Parse the chain-of-thought output and extract only the final summary lines under 'Step 2: 최종 요약'.
    Returns a single string with summary sentences.
    """
    lines = cot_output.splitlines()
    summary_lines: List[str] = []
    in_summary = False
    for line in lines:
        if line.startswith("Step 2:"):
            in_summary = True
            continue
        if in_summary:
            # Stop if a new section starts
            if line.startswith("Step"):
                break
            # Remove leading numbering (e.g., '1) ')
            cleaned = line.strip()
            if cleaned:
                # Remove leading digit and punctuation
                summary_lines.append(cleaned.lstrip('0123456789) .'))
    return " ".join(summary_lines)


def summarize_with_model(model_name: str, text: str) -> str:
    """
    Use Ollama CLI to generate a chain-of-thought summary with the specified model,
    then extract and return only the final summary.
    """
    prompt = SUMMARIZE_TEMPLATE.format(content=text[:2000])
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True,
        check=True
    )
    cot_output = result.stdout.strip()
    return extract_final_summary(cot_output)


def summarizer_node(state: Dict) -> Dict:
    """
    Summarizer node for LangGraph:
    - Generates summaries with multiple models using Chain-of-Thought internally
    - Stores only the final summary strings in article['summaries']
    """
    articles: List[Dict] = state.get("articles", [])
    if not articles:
        print("⚠️ No articles to summarize.")
        return state

    for article in articles:
        title = article.get("title", "")
        content = article.get("content") or article.get("title", "")
        if not content:
            article["summaries"] = []
            continue

        summaries: List[str] = []
        print(f"🔍 Summarizing article: {title}")
        for model in MODEL_NAMES:
            summary = summarize_with_model(model, content)
            if summary:
                entry = f"{model}: {summary}"
                summaries.append(entry)
                print(f"{model} summary: {summary}\n")
        article["summaries"] = summaries
        print(f"✅ Completed summaries for: {title}")

    state["articles"] = articles
    return state
