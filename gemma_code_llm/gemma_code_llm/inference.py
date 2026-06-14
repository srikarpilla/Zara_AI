from __future__ import annotations

import re

from .modeling import load_model_and_tokenizer

from .safety import assert_safe


_FIRST_FENCED_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*.*?```", re.DOTALL)


def clean_completion(completion: str) -> str:
    completion = completion.strip()

    fenced_match = _FIRST_FENCED_BLOCK_RE.search(completion)
    if fenced_match:
        return fenced_match.group(0).strip()

    stop_markers = ["\n### Instruction:", "\n### Input:", "\n### Response:"]
    end = len(completion)
    for marker in stop_markers:
        marker_index = completion.find(marker)
        if marker_index != -1:
            end = min(end, marker_index)
    return completion[:end].strip()


def load_generator(
    base_model: str,
    *,
    adapter: str | None = None,
    quantization: str = "none",
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    return load_model_and_tokenizer(
        base_model,
        adapter=adapter,
        quantization=quantization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        for_training=False,
    )


def generate_code(
    model,
    tokenizer,
    torch,
    *,
    instruction: str,
    input_text: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    safety: bool = True,
) -> str:
    request_text = f"{instruction}\n{input_text or ''}"
    assert_safe(request_text, enabled=safety)

    user_content = instruction
    if input_text:
        user_content += f"\n{input_text}"

    messages = [
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0][prompt_length:]
    completion = clean_completion(tokenizer.decode(completion_ids, skip_special_tokens=True))
    assert_safe(completion, enabled=safety)
    return completion
