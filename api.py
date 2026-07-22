"""
心率情绪识别 API
基于微调的 Qwen2.5 模型，通过心率和心率变异性数据识别用户情绪状态
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ── 全局模型对象 ──────────────────────────────────────────────
class ModelHolder:
    model = None
    tokenizer = None
    device = None
    base_model_name = None
    adapter_path = None
    loaded = False


holder = ModelHolder()

# ── 常量 ──────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = Path("./models/qwen_lora")
BASE_MODEL_DIR = Path("./models/qwen_base")
BASE_MODEL_CANDIDATES = (
    BASE_MODEL_DIR,
    Path("./models/Qwen2.5-1.5B-Instruct"),
)


def _resolve_base_model_path() -> str:
    for candidate in BASE_MODEL_CANDIDATES:
        if (candidate / "config.json").exists() or (candidate / "tokenizer.json").exists():
            return str(candidate)
    return str(BASE_MODEL_DIR)


def _ensure_base_model() -> str:
    base_model_path = Path(_resolve_base_model_path())
    if (base_model_path / "config.json").exists() and (base_model_path / "model.safetensors").exists():
        return str(base_model_path)

    print(f"[API] 本地基础模型未找到，开始下载: {BASE_MODEL}")
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BASE_MODEL,
        local_dir=str(BASE_MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    return str(BASE_MODEL_DIR)

SYSTEM_PROMPT = (
    "You are an emotion classification assistant. "
    "Given physiological data, classify the user's emotional state "
    "as exactly one word: happy, stressed, sad, or neutral."
)
INSTRUCTION = (
    "Based on the following physiological data, "
    "classify the user's emotional state as one of: happy, stressed, sad, neutral."
)

VALID_LABELS = {"happy", "stressed", "sad", "neutral"}

EMOTION_INFO = {
    "happy": {
        "label_cn": "开心",
        "description": "用户处于积极愉悦的情绪状态",
        "color": "#4CAF50",
        "hr_range": "通常 65-85 bpm",
        "hrv_range": "RMSSD 通常 > 40 ms",
        "suggestion": "保持当前状态，适当运动和社交",
    },
    "stressed": {
        "label_cn": "压力/紧张",
        "description": "用户可能处于焦虑、紧张或高压力状态",
        "color": "#F44336",
        "hr_range": "通常 > 90 bpm",
        "hrv_range": "RMSSD 通常 < 25 ms",
        "suggestion": "建议深呼吸、冥想或短暂休息",
    },
    "sad": {
        "label_cn": "悲伤/低落",
        "description": "用户可能处于悲伤、低落或疲惫状态",
        "color": "#2196F3",
        "hr_range": "通常 < 65 bpm",
        "hrv_range": "RMSSD 波动较大",
        "suggestion": "建议与朋友交流、进行轻度运动",
    },
    "neutral": {
        "label_cn": "平静/中性",
        "description": "用户处于平稳正常的心理状态",
        "color": "#9E9E9E",
        "hr_range": "通常 65-80 bpm",
        "hrv_range": "RMSSD 30-50 ms",
        "suggestion": "状态良好，保持规律作息",
    },
}


# ── 生命周期 ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    holder.device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    base_model = _ensure_base_model()
    print(f"[API] 设备: {holder.device}")
    print(f"[API] 加载基础模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    holder.tokenizer = tokenizer
    holder.base_model_name = BASE_MODEL

    if holder.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="cpu", local_files_only=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto", local_files_only=True,
        )

    print(f"[API] 加载 LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH), local_files_only=True)
    model.eval()
    holder.model = model
    holder.adapter_path = ADAPTER_PATH
    holder.loaded = True
    print("[API] 模型加载完成，API 就绪")

    yield

    holder.model = None
    holder.tokenizer = None
    holder.loaded = False
    print("[API] 已关闭")


# ── FastAPI 实例 ──────────────────────────────────────────────
app = FastAPI(
    title="心率情绪识别 API",
    description="通过心率和心率变异性 (HRV) 数据，使用微调的 Qwen2.5 模型识别用户情绪状态",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 请求/响应 模型 ────────────────────────────────────────────
class PredictRequest(BaseModel):
    hr: float = Field(..., description="心率 (bpm)", json_schema_extra={"examples": [75]})
    rmssd: Optional[float] = Field(None, description="RMSSD 心率变异性 (ms)", json_schema_extra={"examples": [42]})
    sdnn: Optional[float] = Field(None, description="SDNN 心率变异性 (ms)", json_schema_extra={"examples": [50]})
    pnn50: Optional[float] = Field(None, description="PNN50 (%%)", json_schema_extra={"examples": [20]})
    extra_info: Optional[dict] = Field(None, description="附加信息 (会原样返回)", json_schema_extra={"examples": [{"user_id": "u123", "source": "apple_watch"}]})


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(
        ..., description="待预测样本列表", min_length=1, max_length=50,
    )


class EmotionDetail(BaseModel):
    label_cn: str
    description: str
    color: str
    hr_range: str
    hrv_range: str
    suggestion: str


class PredictResponse(BaseModel):
    request_id: str
    latency_ms: float
    input_summary: dict
    emotion: str
    emotion_detail: EmotionDetail
    raw_model_output: str
    confidence_note: str
    extra_info: Optional[dict] = None


class BatchPredictResponse(BaseModel):
    request_id: str
    total: int
    latency_ms: float
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    base_model: str
    adapter_path: str
    device: str


# ── 工具函数 ──────────────────────────────────────────────────
def _extract_label(text: str) -> str:
    t = text.lower().strip()
    if t in VALID_LABELS:
        return t
    for label in VALID_LABELS:
        if label in t:
            return label
    return "unknown"


def _build_input_text(hr: float, rmssd=None, sdnn=None, pnn50=None) -> str:
    parts = [f"Heart Rate: {hr:.1f} bpm."]
    if rmssd is not None:
        parts.append(f"RMSSD: {rmssd:.1f} ms.")
    if sdnn is not None:
        parts.append(f"SDNN: {sdnn:.1f} ms.")
    if pnn50 is not None:
        parts.append(f"PNN50: {pnn50:.1f}%.")
    return " ".join(parts)


def _run_inference(hr, rmssd=None, sdnn=None, pnn50=None) -> tuple[str, str]:
    """执行推理，返回 (raw_output, label)"""
    if sdnn is not None and rmssd is not None and sdnn < rmssd:
        sdnn = rmssd + 10

    input_text = _build_input_text(hr, rmssd, sdnn, pnn50)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\n{input_text}"},
    ]
    prompt = holder.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = holder.tokenizer(prompt, return_tensors="pt").to(holder.model.device)

    with torch.no_grad():
        outputs = holder.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=holder.tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    raw = holder.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    label = _extract_label(raw)
    return raw, label


# ── 路由 ──────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    return HealthResponse(
        status="ok" if holder.loaded else "loading",
        model_loaded=holder.loaded,
        base_model=holder.base_model_name or "",
        adapter_path=holder.adapter_path or "",
        device=holder.device or "",
    )


@app.post("/predict", response_model=PredictResponse, tags=["推理"])
async def predict(req: PredictRequest):
    if not holder.loaded:
        raise HTTPException(503, "模型尚未加载完毕，请稍后重试")

    t0 = time.perf_counter()
    raw, label = _run_inference(req.hr, req.rmssd, req.sdnn, req.pnn50)
    latency = (time.perf_counter() - t0) * 1000

    detail = EMOTION_INFO.get(label, EMOTION_INFO["neutral"])

    return PredictResponse(
        request_id=str(uuid.uuid4()),
        latency_ms=round(latency, 2),
        input_summary={
            "hr": req.hr,
            "rmssd": req.rmssd,
            "sdnn": req.sdnn,
            "pnn50": req.pnn50,
        },
        emotion=label,
        emotion_detail=EmotionDetail(**detail),
        raw_model_output=raw,
        confidence_note=(
            "模型输出与已知标签完全匹配" if label in VALID_LABELS
            else f"模型输出 '{raw}' 未能匹配标准标签，结果可能不够准确"
        ),
        extra_info=req.extra_info,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["推理"])
async def predict_batch(req: BatchPredictRequest):
    if not holder.loaded:
        raise HTTPException(503, "模型尚未加载完毕，请稍后重试")

    t0 = time.perf_counter()
    results = []
    for sample in req.samples:
        raw, label = _run_inference(sample.hr, sample.rmssd, sample.sdnn, sample.pnn50)
        detail = EMOTION_INFO.get(label, EMOTION_INFO["neutral"])
        results.append(PredictResponse(
            request_id=str(uuid.uuid4()),
            latency_ms=0,
            input_summary={
                "hr": sample.hr,
                "rmssd": sample.rmssd,
                "sdnn": sample.sdnn,
                "pnn50": sample.pnn50,
            },
            emotion=label,
            emotion_detail=EmotionDetail(**detail),
            raw_model_output=raw,
            confidence_note=(
                "模型输出与已知标签完全匹配" if label in VALID_LABELS
                else f"模型输出 '{raw}' 未能匹配标准标签，结果可能不够准确"
            ),
            extra_info=sample.extra_info,
        ))
    latency = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        request_id=str(uuid.uuid4()),
        total=len(results),
        latency_ms=round(latency, 2),
        results=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
