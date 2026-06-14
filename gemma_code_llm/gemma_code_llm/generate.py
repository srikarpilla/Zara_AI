from __future__ import annotations

import argparse

from .inference import generate_code, load_generator


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate code with a Gemma LoRA adapter.")
    parser.add_argument("--base-model", default="google/gemma-3-1b-it", help="Base Hugging Face model id.")
    parser.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter.")
    parser.add_argument("--instruction", required=True, help="Coding instruction.")
    parser.add_argument("--input", default="", help="Optional extra input/context.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-safety", action="store_true")
    args = parser.parse_args()

    model, tokenizer, torch = load_generator(
        args.base_model,
        adapter=args.adapter,
        quantization=args.quantization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    completion = generate_code(
        model,
        tokenizer,
        torch,
        instruction=args.instruction,
        input_text=args.input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        safety=not args.disable_safety,
    )
    print(completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

