"""
실제 AI 모델 구현체
- TextEmotionModel      : klue/bert 기반 텍스트 감정 분류 (로컬 models/text-emotion-final)
- Wav2VecEmotionModel   : wav2vec2 기반 음성 감정 분류 (로컬 models/voice-emotion-final)
- EmotionFusionModel    : 텍스트(0.40) + 음성(0.35) + 얼굴(0.25) 가중치 융합
- CBTLLMModel           : Qwen2.5-3B-Instruct + CBT LoRA + 감정별 LoRA (8bit 양자화)
"""

import os
import logging
from typing import List, Optional

import numpy as np

from ai_modules.interfaces import (
    BaseTextEmotionModel,
    BaseEmotionModel,
    BaseEmotionFusionModel,
    BaseLLMModel,
)
from ai_modules.schemas import EmotionResult, STTInput, LLMContext, LLMResponse

logger = logging.getLogger(__name__)

# 7개 감정 레이블 (알파벳 순 인덱스)
EMOTION_LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


# ──────────────────────────────────────────────────────────────
# 1. 텍스트 감정 분석 (BertForSequenceClassification, klue/bert)
#    입력: STT 결과 텍스트 (str)
#    출력: EmotionResult
# ──────────────────────────────────────────────────────────────
class TextEmotionModel(BaseTextEmotionModel):
    def __init__(self, model_path: str = "models/text-emotion-final", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        logger.info(f"[TextEmo] 로딩 완료: {self.model_path} on {self.device}")

    def analyze(self, text: str) -> EmotionResult:
        import torch

        if not text or not text.strip():
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pred_idx = int(probs.argmax())
            primary = EMOTION_LABEL_MAP.get(pred_idx, f"label_{pred_idx}")
            prob_dict = {
                EMOTION_LABEL_MAP.get(i, f"label_{i}"): round(float(p), 3)
                for i, p in enumerate(probs)
            }
            return EmotionResult(primary_emotion=primary, probabilities=prob_dict)
        except Exception as e:
            logger.error(f"[TextEmo] 분석 오류: {e}")
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})


# ──────────────────────────────────────────────────────────────
# 2. 음성 감정 분석 (Wav2Vec2ForSequenceClassification)
#    입력: STTInput.audio_data = float32 PCM bytes (16kHz, mono)
#    출력: EmotionResult
# ──────────────────────────────────────────────────────────────
class Wav2VecEmotionModel(BaseEmotionModel):
    def __init__(self, model_path: str = "models/voice-emotion-final", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()
        logger.info(f"[VoiceEmo] Wav2Vec2 로딩 완료: {self.model_path} on {self.device}")

    def analyze(self, input_data: STTInput) -> EmotionResult:
        import torch

        try:
            audio_array = np.frombuffer(input_data.audio_data, dtype=np.float32)
            if len(audio_array) < 1600:  # 최소 0.1초
                return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})

            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pred_idx = int(probs.argmax())
            primary = EMOTION_LABEL_MAP.get(pred_idx, f"label_{pred_idx}")
            prob_dict = {
                EMOTION_LABEL_MAP.get(i, f"label_{i}"): round(float(p), 3)
                for i, p in enumerate(probs)
            }
            return EmotionResult(primary_emotion=primary, probabilities=prob_dict)
        except Exception as e:
            logger.error(f"[VoiceEmo] 분석 오류: {e}")
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})


# ──────────────────────────────────────────────────────────────
# 3. 감정 융합 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25)
# ──────────────────────────────────────────────────────────────
class EmotionFusionModel(BaseEmotionFusionModel):
    TEXT_W = 0.40
    VOICE_W = 0.35
    FACE_W = 0.25

    def fuse(
        self,
        text_result: EmotionResult,
        voice_result: EmotionResult,
        face_result: EmotionResult,
    ) -> EmotionResult:
        from collections import defaultdict

        combined: dict = defaultdict(float)
        for emotion, prob in text_result.probabilities.items():
            combined[emotion] += prob * self.TEXT_W
        for emotion, prob in voice_result.probabilities.items():
            combined[emotion] += prob * self.VOICE_W
        for emotion, prob in face_result.probabilities.items():
            combined[emotion] += prob * self.FACE_W

        total = sum(combined.values()) or 1.0
        prob_dict = {k: round(v / total, 3) for k, v in combined.items()}
        primary = max(prob_dict, key=prob_dict.get)
        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)


# ──────────────────────────────────────────────────────────────
# 4. CBT LLM (Qwen2.5-3B-Instruct + CBT LoRA + 감정별 LoRA)
#    VRAM 절약: bitsandbytes 8bit 양자화 (GTX 1660 Super 6GB 기준)
#    LoRA 전략:
#      - 기본 어댑터: "cbt" (CBT 상담 스타일)
#      - 강한 감정 감지 시 해당 감정 어댑터로 전환
#      - 모든 어댑터를 메모리에 미리 로드 → 전환 속도 빠름
# ──────────────────────────────────────────────────────────────
class CBTLLMModel(BaseLLMModel):
    SYSTEM_PROMPT = (
        "당신은 따뜻하고 공감적인 AI 심리상담사 '루나'입니다. "
        "사용자의 감정을 깊이 이해하고 CBT(인지행동치료) 기반으로 상담을 진행합니다. "
        "답변은 2~3문장으로 간결하고 자연스럽게, 반드시 한국어로 대답하세요. "
        "질문은 하나만 하고, 사용자가 더 이야기할 수 있도록 열린 질문을 사용하세요."
    )

    # 감정 → LoRA 어댑터 이름 매핑
    EMOTION_TO_ADAPTER = {
        "angry": "angry", "disgust": "disgust", "fear": "fear",
        "happy": "happy", "sad": "sad", "surprise": "surprise",
        "neutral": "cbt",  # neutral은 기본 CBT 어댑터 사용
    }

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cbt_adapter_path: str = "models/cbt-counselor-final",
        lora_dir: str = "models/lora",
        device: str = "cuda",
    ):
        self.base_model_name = base_model_name
        self.cbt_adapter_path = cbt_adapter_path
        self.lora_dir = lora_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self._active_adapter = "cbt"

    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[CBT LLM] CUDA 사용 불가 → CPU 모드 (매우 느림)")
            self.device = "cpu"

        logger.info(f"[CBT LLM] {self.base_model_name} 로딩 중 (8bit 양자화, device={self.device})...")
        logger.info("[CBT LLM] 최초 실행 시 HuggingFace에서 베이스 모델 다운로드 (~6GB, 수분 소요)")

        # 8bit 양자화 설정 (6GB VRAM에서 3B 모델 구동: ~3.5GB 사용)
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            load_kwargs = dict(device_map="auto", quantization_config=bnb_config)
        else:
            load_kwargs = dict(torch_dtype=torch.float32)

        # 토크나이저는 CBT 어댑터 경로에서 로드 (fine-tuned tokenizer 사용)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cbt_adapter_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, **load_kwargs
        )
        if self.device == "cpu":
            base_model = base_model.to("cpu")

        # CBT LoRA 어댑터 로드 (기본 어댑터)
        self.model = PeftModel.from_pretrained(
            base_model, self.cbt_adapter_path, adapter_name="cbt"
        )

        # 감정별 LoRA 어댑터 메모리에 미리 로드
        loaded = []
        for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            lora_path = os.path.join(self.lora_dir, emotion)
            if os.path.exists(lora_path):
                self.model.load_adapter(lora_path, adapter_name=emotion)
                loaded.append(emotion)

        self.model.set_adapter("cbt")
        self.model.eval()
        logger.info(f"[CBT LLM] 로딩 완료. 감정 LoRA 로드: {loaded}")

    def _switch_adapter(self, fused_emotion: Optional[str]) -> None:
        """감정에 맞는 LoRA 어댑터로 전환 (같은 어댑터면 스킵)."""
        if not fused_emotion:
            return
        target = self.EMOTION_TO_ADAPTER.get(fused_emotion, "cbt")
        if target != self._active_adapter:
            try:
                self.model.set_adapter(target)
                self._active_adapter = target
                logger.info(f"[CBT LLM] LoRA 전환: {target}")
            except Exception:
                # 해당 어댑터가 없으면 cbt로 폴백
                self.model.set_adapter("cbt")
                self._active_adapter = "cbt"

    def generate_response(self, context: LLMContext) -> LLMResponse:
        import torch

        # 융합 감정으로 LoRA 어댑터 전환
        self._switch_adapter(context.fused_emotion)

        # 메시지 구성 (커스텀 시스템 프롬프트 지원)
        system_prompt = context.system_prompt or self.SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        for h in context.history:
            messages.append(h)

        # 감정 힌트 추가
        emotion_hint = ""
        if context.fused_emotion and context.fused_emotion != "neutral":
            emotion_hint = f"\n[참고 - 현재 감정: {context.fused_emotion}]"
        messages.append({"role": "user", "content": context.user_text + emotion_hint})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(reply_text=reply)
