"""
DynamicPlanGenerator — GPT-4o-mini로 인지 왜곡 분석 + 5단계 CBT 상담 플랜 생성.
반환 스키마: {"analysis": {core_problem, cognitive_pattern}, "steps": [...]}
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


PLAN_GENERATION_PROMPT = """\
당신은 CBT(인지행동치료) 전문 상담사입니다.
아래 내담자 정보를 바탕으로 맞춤형 5단계 상담 계획을 JSON으로 생성하세요.

[내담자 정보]
상담 주제: {topic}
현재 감정: {mood}
상세 고민: {content}

[지시사항]
1. 내담자의 핵심 문제와 예상되는 인지 왜곡 패턴을 분석하세요 (과잉일반화, 흑백사고, 재난화, 개인화, 독심술, 감정적 추론 등)
2. 각 단계별로 이 내담자의 구체적 상황에 맞는 목표와 질문을 설계하세요
3. 질문은 한국어 상담사가 실제로 쓰는 자연스러운 구어체로 작성하세요
4. 반드시 한국어로만 작성하세요
5. 각 단계의 질문 개수는 아래 범위 내에서 내담자의 상황 복잡도에 따라 자유롭게 조절하세요:
   - Step 1 (공감 형성): 2~5개
   - Step 2 (문제 탐색): 3~7개
   - Step 3 (사고 전환): 3~8개
   - Step 4 (행동 계획): 2~5개
   - Step 5 (마무리): 1~3개

[출력 형식 - 반드시 아래 JSON만 출력. 다른 텍스트 금지]
{{
    "analysis": {{
        "core_problem": "핵심 문제 한 줄 요약",
        "cognitive_pattern": "예상되는 인지 왜곡 패턴명과 설명"
    }},
    "steps": [
        {{
            "step": 1,
            "name": "공감 형성",
            "goal": "이 내담자의 상황에 맞는 구체적 목표",
            "focus": "이 스텝에서 상담사가 집중할 포인트 (CBT 기법)",
            "key_questions": ["내담자 상황에 맞는 질문들 (2~5개)"]
        }},
        {{
            "step": 2,
            "name": "문제 탐색",
            "goal": "구체적 목표",
            "focus": "5W1H로 사건 구체화 + 자동적 사고 식별",
            "key_questions": ["내담자 상황에 맞는 질문들 (3~7개)"]
        }},
        {{
            "step": 3,
            "name": "사고 전환",
            "goal": "구체적 목표",
            "focus": "소크라테스식 질문으로 인지 왜곡 식별 및 도전",
            "key_questions": ["내담자 상황에 맞는 소크라테스식 질문들 (3~8개)"]
        }},
        {{
            "step": 4,
            "name": "행동 계획",
            "goal": "구체적 목표",
            "focus": "균형잡힌 사고에 기반한 행동 실험 설계",
            "key_questions": ["내담자 상황에 맞는 질문들 (2~5개)"]
        }},
        {{
            "step": 5,
            "name": "마무리",
            "goal": "구체적 목표",
            "focus": "통찰 정리 + 실천 다짐",
            "key_questions": ["내담자 상황에 맞는 질문들 (1~3개)"]
        }}
    ]
}}
"""


FALLBACK_PLAN: Dict[str, Any] = {
    "analysis": {
        "core_problem": "내담자의 호소를 충분히 파악하기 어려운 상황 (폴백 플랜)",
        "cognitive_pattern": "감정적 추론(emotional reasoning) 가능성 — 감정을 사실로 받아들이는 패턴",
    },
    "steps": [
        {
            "step": 1,
            "name": "감정 탐색 및 공감",
            "goal": "현재 감정과 그 강도, 신체 반응을 파악하고 안전한 분위기 형성",
            "focus": "판단 없이 감정에 라벨을 붙이고 공감, 감정의 정당성 확인",
            "key_questions": [
                "지금 가장 크게 느껴지는 감정이 무엇인가요?",
                "그 감정이 언제부터, 어떤 상황에서 시작됐나요?",
                "그 순간 몸에서 어떤 느낌이 들었는지 기억나시나요?",
            ],
        },
        {
            "step": 2,
            "name": "문제 탐색",
            "goal": "구체적 사건 맥락 파악 + 자동적 사고 식별",
            "focus": "5W1H로 사건 구체화, '그때 어떤 생각이 스쳤나요?'로 자동적 사고 노출",
            "key_questions": [
                "그 상황에서 구체적으로 어떤 일이 있었나요?",
                "그때 머릿속에 어떤 생각이 가장 먼저 떠올랐나요?",
                "그 생각이 사실이라는 증거는 무엇이었나요?",
            ],
        },
        {
            "step": 3,
            "name": "사고 전환",
            "goal": "소크라테스식 질문으로 인지 왜곡 식별 및 도전",
            "focus": "근거 검토, 대안적 관점, 친구라면 뭐라고 할지 등 소크라테스식 질문",
            "key_questions": [
                "그 생각이 100% 사실이라고 확신하나요? 0~100점으로 표현한다면요?",
                "그 생각과 반대되는 증거는 없을까요?",
                "친한 친구가 같은 상황이라면, 뭐라고 말해줄 것 같으세요?",
            ],
        },
        {
            "step": 4,
            "name": "행동 계획",
            "goal": "균형잡힌 사고에 기반한 작은 행동 실험 설계",
            "focus": "구체적이고 실현 가능한 행동, 행동 실험으로 사고 검증",
            "key_questions": [
                "새로운 시각으로 봤을 때, 이번 주에 시도해볼 수 있는 작은 행동이 있을까요?",
                "그 행동을 가로막는 것이 있다면 무엇일까요?",
                "결과가 어떻게 나올지 예상해보면, 어떤 모습일까요?",
            ],
        },
        {
            "step": 5,
            "name": "마무리",
            "goal": "통찰 정리 + 실천 다짐",
            "focus": "오늘 발견한 인지 왜곡과 대안적 사고 요약, 다음까지의 과제",
            "key_questions": [
                "오늘 이야기하면서 가장 새롭게 느낀 점은 무엇인가요?",
                "이번 주 동안 어떤 점을 기억하면 도움이 될까요?",
            ],
        },
    ],
}


class DynamicPlanGenerator:
    """GPT-4o-mini로 내담자 맞춤 5-step CBT 플랜 + 인지 왜곡 분석 생성."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key

    def generate(self, topic: str, mood: str, content: str) -> Dict[str, Any]:
        """
        반환: {"analysis": {core_problem, cognitive_pattern}, "steps": [...]}
        실패 시 FALLBACK_PLAN 반환.
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

        plan = self._parse_plan(raw)
        if plan is None:
            logger.warning("[PlanGen] JSON 파싱 실패 → 폴백 플랜 사용")
            return FALLBACK_PLAN

        elapsed = time.time() - t0
        analysis = plan.get("analysis", {})
        logger.info(f"[PlanGen] 플랜 생성 완료 ({elapsed:.2f}초)")
        logger.info(f"  핵심 문제: {analysis.get('core_problem', '?')}")
        logger.info(f"  인지 왜곡: {analysis.get('cognitive_pattern', '?')}")
        for s in plan.get("steps", []):
            logger.info(f"  Step {s['step']} {s['name']}: {len(s.get('key_questions', []))}개 질문")

        return plan

    def _call_api(self, prompt: str) -> Optional[str]:
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
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[PlanGen] OpenAI API 오류: {e}")
            return None

    def _parse_plan(self, raw: str) -> Optional[Dict[str, Any]]:
        """새 스키마(analysis + steps) 검증."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning("[PlanGen] 최상위가 dict가 아님")
                return None

            analysis = data.get("analysis")
            steps = data.get("steps")

            if not isinstance(analysis, dict) or "core_problem" not in analysis or "cognitive_pattern" not in analysis:
                logger.warning("[PlanGen] analysis 필드 누락 또는 형식 오류")
                return None

            if not isinstance(steps, list) or len(steps) != 5:
                logger.warning(f"[PlanGen] steps 5개 아님: {len(steps) if isinstance(steps, list) else 'not list'}")
                return None

            for s in steps:
                for key in ("step", "name", "goal", "focus", "key_questions"):
                    if key not in s:
                        logger.warning(f"[PlanGen] Step {s.get('step', '?')}에 '{key}' 필드 누락")
                        return None
                if not isinstance(s["key_questions"], list) or len(s["key_questions"]) < 1:
                    logger.warning(f"[PlanGen] Step {s.get('step', '?')} key_questions 형식 오류")
                    return None

            return data
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"[PlanGen] JSON 파싱 오류: {e}")
            return None
