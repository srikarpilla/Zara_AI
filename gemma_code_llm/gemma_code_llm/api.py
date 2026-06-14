from __future__ import annotations

import os
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import generate_code, load_generator


class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    input: str = ""
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.01, le=1.0)
    safety: bool = True


class GenerateResponse(BaseModel):
    completion: str


app = FastAPI(title="Gemma Code LLM API", version="0.1.0")

_model_state = None
_model_lock = Lock()


def _load_state():
    global _model_state
    if _model_state is not None:
        return _model_state

    with _model_lock:
        if _model_state is not None:
            return _model_state

        base_model = os.getenv("BASE_MODEL", "google/gemma-3-1b-it")
        adapter = os.getenv("ADAPTER_PATH") or None
        quantization = os.getenv("QUANTIZATION", "none")
        dtype = os.getenv("DTYPE", "auto")
        trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "0") == "1"

        _model_state = load_generator(
            base_model,
            adapter=adapter,
            quantization=quantization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return _model_state


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        model, tokenizer, torch = _load_state()
        completion = generate_code(
            model,
            tokenizer,
            torch,
            instruction=request.instruction,
            input_text=request.input,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            safety=request.safety,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(completion=completion)

