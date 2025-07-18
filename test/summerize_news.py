NEWS_EXAMPLE = (
    "아폴로 이코노미스트 시총 상위 10개 기업, "
    "과거 버블보다 더 고평가 토르스텐 슬록 자산운용사 아폴로 글로벌 매니지먼트 수석 이코노미스트는 "
    "1990년대 IT 버블과 현 AI 버블의 차이점을 들자면 현재 뉴욕증시 시총 상위 10개 기업이 1990년대 상위 10개 기업보다 더 고평가됐다는 점이라고"
    "16일(현지시간) 밝혔다. 최근 뉴욕증시에서 인공지능 관련 주식의 버블이 1990년대 말 IT 버블 때보다 심각하다는 월가 전문가의 경고가 나온다."
    "슬록 이코노미스트의 이 같은 지적은 미국의 주가지수가 관세 불확실성에도 불구하고 다시 전고점을 돌파하고, "
    "기술주 중심의 나스닥 종합지수가 연일 사상 최고치를 경신하는 가운데 나왔다."
    "슬록 이코노미스트가 공개한 뉴욕증시 상위 10개 기업의 12개월 선행 주가수익비율은 30배에 육박,"
    "25배 언저리였던 2000년 IT 버블 정점 시기를 능가했다."
    "최근 2년여간 뉴욕증시 강세장은 AI 열풍에 힘입어 엔비디아를 필두로 마이크로소프트, 메타 등 빅테크가 이끌어왔다."
    "AI 반도체의 절대 강자인 엔비디아는 전 세계 기업 중 사상 최초로 최근 시총 4조 달러를 돌파하기도 했다."
    "다만, 월가에서는 현재 대형 빅테크의 주가가 역사적 기준으로 볼 때 매우 비싸다는 데 공감하면서도 2000년 IT 거품 붕괴와 같은 현상으로 이어질 가능성은 작다는 반론도 적지 않다."
    "존 히긴스 캐피털 이코노믹스의 수석 이코노미스트는 오늘날 AI 기업 주가의 상승은 평가가치 상승보다는 기업이익 증가에 기인하고 있다며 "
    "미 증시가 내년 말까지 빅테크 부문을 필두로 강세장을 지속할 것이라고 예상하는 이유라고 말했다."
)

import requests


def ask_ollama(prompt: str, model: str) -> str:
    """
    Send a prompt to OLLAMA API (non-streaming) and return the generated text.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 300,
            "stream": False  # disable streaming to get a single JSON response
        }
    )
    response.raise_for_status()
    data = response.json()
    # Extract the generated content
    return data.get("response", "")

# Models to test
MODELS = [
    "qwen3:0.6b",
    "PetrosStav/gemma3-tools:4b"
]

# Define prompt templates for different strategies
templates = {
    "role_instruction": {
        "description": "Role + clear instruction",
        "template": (
            "SYSTEM: 당신은 한국어 뉴스 요약 전문가입니다.\n"
            "User: 다음 뉴스를 **3문장 이내**로 요약하고, **핵심 키워드 5개**를 [ ] 형태로 제공해주세요.\n\n"
            "뉴스 본문:\n{news_text}\n\n"
            "<출력 형식 예시>\n1) 요약: …\n2) 키워드: [키워드1, 키워드2, …]"
        )
    },
    "few_shot": {
        "description": "Few-shot examples",
        "template": (
            "SYSTEM: 당신은 한국어 뉴스 요약 전문가입니다.\n"
            "User: 다음 예시를 참고하여 뉴스를 요약해 주세요.\n\n"
            "예시 1:\n"
            "뉴스: 'A, B를 공개하며...'\n"
            "요약: '...'\n\n"
            "예시 2:\n"
            "뉴스: 'C, D 발표...'\n"
            "요약: '...'\n\n"
            "이제 다음 뉴스를 **3문장 이내**로 요약하고, **핵심 키워드 5개**를 제공해주세요.\n"
            "뉴스 본문:\n{news_text}"
        )
    },
    "chain_of_thought": {
        "description": "Chain-of-Thought reasoning",
        "template": (
            "SYSTEM: 당신은 한국어 뉴스 요약 전문가입니다.\n"
            "User: 먼저 본문에서 핵심 문장 3개를 추출하고, 각 문장을 결합해 요약문을 작성해주세요.\n"
            "뉴스 본문:\n{news_text}\n\n"
            "출력 형식:\n"
            "Step 1: 핵심 문장\n"
            "- ...\n"
            "Step 2: 최종 요약 (3문장 이내)\n"
            "1) ...\n"
            "2) ...\n"
            "3) ..."
        )
    },
    "self_critique": {
        "description": "Self-Critique routine",
        "template": (
            "SYSTEM: 당신은 한국어 뉴스 요약 전문가입니다.\n"
            "User: 뉴스를 요약한 뒤, 빠진 핵심 정보를 보완해주세요.\n"
            "뉴스 본문:\n{news_text}\n\n"
            "출력 형식:\n"
            "1) 초안 요약\n"
            "2) 보완할 정보\n"
            "3) 최종 요약"
        )
    }
}

def main():
    for model in MODELS:
        print(f"### Model: {model}\n")
        for key, info in templates.items():
            template = info["template"]
            prompt = template.format(news_text=NEWS_EXAMPLE)
            print(f"[Strategy: {key} - {info['description']}]\n")
            print("=== Prompt ===")
            print(prompt)
            print("\n=== Response ===")
            try:
                summary = ask_ollama(prompt, model=model)
                print(summary)
            except Exception as e:
                print(f"Error during {key} with model {model}: {e}")
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
