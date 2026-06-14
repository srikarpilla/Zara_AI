from __future__ import annotations

from typing import Any


def _dependency_error(package: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(
        f"Missing or broken dependency '{package}'. Install dependencies with: "
        "pip install -r requirements.txt"
    )
    error.__cause__ = exc
    return error


def import_torch():
    try:
        import torch
    except Exception as exc:
        raise _dependency_error("torch", exc)
    return torch


def import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise _dependency_error("transformers", exc)
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_dtype(torch: Any, dtype: str):
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError("dtype must be one of: auto, float32, float16, bfloat16")
    return mapping[dtype]


def build_quantization_config(torch: Any, BitsAndBytesConfig: Any, quantization: str, torch_dtype: Any):
    if quantization == "none":
        return None
    if quantization not in {"4bit", "8bit"}:
        raise ValueError("quantization must be one of: none, 4bit, 8bit")
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit and 8-bit quantization require a CUDA GPU")

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    return BitsAndBytesConfig(load_in_8bit=True)


def load_tokenizer(base_model: str, *, trust_remote_code: bool = False):
    _, AutoTokenizer, _ = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_model_and_tokenizer(
    base_model: str,
    *,
    adapter: str | None = None,
    quantization: str = "none",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    for_training: bool = False,
):
    torch = import_torch()
    AutoModelForCausalLM, _, BitsAndBytesConfig = import_transformers()
    tokenizer = load_tokenizer(base_model, trust_remote_code=trust_remote_code)
    torch_dtype = resolve_dtype(torch, dtype)
    quantization_config = build_quantization_config(torch, BitsAndBytesConfig, quantization, torch_dtype)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise _dependency_error("peft", exc)
        model = PeftModel.from_pretrained(model, adapter)

    if quantization_config is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    if not for_training:
        model.eval()

    return model, tokenizer, torch
