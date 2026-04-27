"""
DynamicPlanGenerator — GPT-4o-mini를 호출하여 5단계 CBT 상담 플랜을 생성.
동기 함수로 구현 (pipeline에서 run_in_executor로 호출).
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# GPT-4o-mini에게 보낼 플랜 생성 프롬프트
PLAN_GENERATION_PROMPT = """\
당신은 CBT(인지행동치료) 기반 심리상담 플랜을 설계하는 전문가입니다.
아래 내담자 정보를 바탕으로, **정확히 5단계**로 구성된 상담 플랜을 JSON으로 생성하세요.

## 내담자 정보
- 상담 주제: {topic}
- 현재 기분: {mood}
- 호소 내용: {content}

## 출력 형식 (JSON)
```json
{{
  "steps": [
    {{
      "step": 1,
      "title": "단계 제목",
      "goal": "이 단계의 목표",
      "system_prompt": "이 단계에서 상담사 AI가 사용할 기본 지침 (한국어, 1~2문장). 반드시 '사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요.' 문장으로 끝낼 것.",
      "questions": [
        "이 단계에서 물어볼 질문 1 (내담자 정보에 맞게 구체적으로)",
        "질문 2",
        "질문 3"
      ]
    }},
    ...
  ]
}}
```

## 규칙
1. step 1은 반드시 '감정 탐색 및 공감' 단계로 시작
2. step 5는 반드시 '정리 및 마무리' 단계로 끝냄
3. 각 단계의 questions는 2~4개, 내담자 정보(주제/기분/내용)에 맞게 구체적으로 작성
4. questions는 순서대로 진행되므로, 자연스럽게 깊어지는 흐름으로 작성
5. 반드시 위 JSON 형식만 출력 (설명 텍스트 없이)
"""

# API 실패 시 사용할 기본 폴백 플랜
FALLBACK_PLAN: List[Dict[str, Any]] = [
    {
        "step": 1,
        "title": "감정 탐색 및 공감",
        "goal": "내담자의 현재 감정 상태를 파악하고 공감한다",
        "system_prompt": (
            "당신은 따뜻하고 공감적인 AI 심리상담사 '루나'입니다. "
            "사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요."
        ),
        "questions": [
            "지금 가장 크게 느껴지는 감정이 무엇인가요?",
            "그 감정이 언제부터, 어떤 상황에서 시작됐나요?",
            "그 순간 몸에서 어떤 느낌이 들었는지 기억하세요?",
        ],
    },
    {
        "step": 2,
        "title": "상황 분석",
        "goal": "문제 상황의 구체적 맥락을 파악한다",
        "system_prompt": (
            "당신은 CBT 기반 AI 상담사 '루나'입니다. "
            "사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요."
        ),
        "questions": [
            "그 상황에서 구체적으로 어떤 일이 있었나요?",
            "그때 주변에 다른 사람이 있었나요? 그 상황을 어떻게 바라봤을 것 같으세요?",
            "비슷한 상황이 전에도 있었나요?",
        ],
    },
    {
        "step": 3,
        "title": "인지 탐색",
        "goal": "자동적 사고와 인지 왜곡을 탐색한다",
        "system_prompt": (
            "당신은 CBT 전문 AI 상담사 '루나'입니다. "
            "사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요."
        ),
        "questions": [
            "그 순간 머릿속에 어떤 생각이 스쳐 지나갔나요?",
            "그 생각이 얼마나 사실이라고 느껴졌나요? 0~100점으로 표현한다면요?",
            "혹시 그 생각이 항상 맞다고 느껴지나요, 아니면 상황에 따라 다른가요?",
        ],
    },
    {
        "step": 4,
        "title": "대안적 사고 연습",
        "goal": "균형 잡힌 대안적 사고를 연습한다",
        "system_prompt": (
            "당신은 CBT 기반 AI 상담사 '루나'입니다. "
            "사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요."
        ),
        "questions": [
            "만약 친한 친구가 같은 상황이었다면, 뭐라고 말해줬을 것 같으세요?",
            "그 생각 외에 상황을 다르게 볼 수 있는 방법이 있을까요?",
            "새로운 시각으로 봤을 때 기분이 조금 달라지는 게 느껴지나요?",
        ],
    },
    {
        "step": 5,
        "title": "정리 및 마무리",
        "goal": "상담 내용을 정리하고 실천 과제를 제안한다",
        "system_prompt": (
            "당신은 CBT 기반 AI 상담사 '루나'입니다. "
            "사용자 말에 먼저 공감한 뒤, 아래 질문을 하나씩 자연스럽게 이어가세요."
        ),
        "questions": [
            "오늘 이야기하면서 새롭게 알게 된 것이 있다면 무엇인가요?",
            "이번 주에 일상에서 작게 실천해볼 수 있는 것이 있을까요?",
        ],
    },
]


class DynamicPlanGenerator:
    """GPT-4o-mini API를 호출하여 내담자 맞춤 5-step CBT 플랜을 생성."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key

    def generate(self, topic: str, mood: str, content: str) -> List[Dict[str, Any]]:
        """
        5-step 플랜 생성. API 실패 시 폴백 플랜 반환.
        동기 함수 — pipeline에서 run_in_executor로 호출할 것.
        """
        if not self.api_key:
            logger.warning("[PlanGen] OpenAI API 키 없음 → 폴백 플랜 사용")
            return FALLBACK_PLAN

        prompt = PLAN_GENERATION_PROMPT.format(topic=topic, mood=mood, content=content)

        t0 = time.time()
        logger.info(f"[PlanGen] GPT-4o-mini로 플랜 생성 중... (topic={topic}, mood={mood})")

        raw = self._call_api(prompt)
        if raw is None:
            logger.warning("[PlanGen] API 호출 실패 → 폴백 플랜 사용")
            return FALLBACK_PLAN

        steps = self._parse_plan(raw)
        if steps is None:
            logger.warning("[PlanGen] JSON 파싱 실패 → 폴백 플랜 사용")
            return FALLBACK_PLAN

        elapsed = time.time() - t0
        logger.info(f"[PlanGen] 플랜 생성 완료 ({elapsed:.2f}초, {len(steps)}단계)")
        for s in steps:
            logger.info(f"  Step {s['step']}: {s['title']} ({len(s.get('questions', []))}개 질문)")

        return steps

    def _call_api(self, prompt: str) -> Optional[str]:
        """OpenAI Chat Completions API 호출. 실패 시 None 반환."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a CBT counseling plan designer. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[PlanGen] OpenAI API 오류: {e}")
            return None

    def _parse_plan(self, raw: str) -> Optional[List[Dict[str, Any]]]:
        """GPT 응답 JSON 파싱. 실패 시 None 반환."""
        try:
            # ```json ... ``` 블록 제거
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            steps = data.get("steps", data) if isinstance(data, dict) else data

            if not isinstance(steps, list) or len(steps) != 5:
                logger.warning(f"[PlanGen] 플랜 단계 수 불일치: {len(steps) if isinstance(steps, list) else 'not list'}")
                return None

            # 필수 필드 검증
            for s in steps:
                for key in ("step", "title", "goal", "system_prompt", "questions"):
                    if key not in s:
                        logger.warning(f"[PlanGen] Step {s.get('step', '?')}에 '{key}' 필드 누락")
                        return None
                if not isinstance(s["questions"], list) or len(s["questions"]) < 1:
                    logger.warning(f"[PlanGen] Step {s.get('step', '?')} questions 형식 오류")
                    return None

            return steps
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"[PlanGen] JSON 파싱 오류: {e}")
            return None
