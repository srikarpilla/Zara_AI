from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .data import TokenizedCodeDataset, load_jsonl_records, split_records
from .modeling import load_model_and_tokenizer


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _import_training_stack():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install them with: pip install -r requirements.txt"
        ) from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training, DataCollatorForSeq2Seq, Trainer, TrainingArguments


def _build_training_args(TrainingArguments, args, torch):
    signature = inspect.signature(TrainingArguments.__init__).parameters
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not use_bf16

    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "optim": "adamw_torch",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    if "bf16" in signature:
        kwargs["bf16"] = use_bf16
    if "fp16" in signature:
        kwargs["fp16"] = use_fp16

    if args.validation_size > 0:
        strategy_name = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
        kwargs[strategy_name] = "steps"
        kwargs["eval_steps"] = args.eval_steps

    return TrainingArguments(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for code generation with LoRA.")
    parser.add_argument("--data", required=True, help="Path to JSONL training data.")
    parser.add_argument("--base-model", default="google/gemma-3-1b-it", help="Base Hugging Face model id.")
    parser.add_argument("--output-dir", default="outputs/gemma-code-lora", help="Where to save the LoRA adapter.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--validation-size", type=float, default=0.0, help="Fraction of data for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    LoraConfig, get_peft_model, prepare_model_for_kbit_training, DataCollatorForSeq2Seq, Trainer, TrainingArguments = (
        _import_training_stack()
    )

    records = load_jsonl_records(args.data, max_records=args.max_records)
    train_records, eval_records = split_records(records, args.validation_size, args.seed)

    model, tokenizer, torch = load_model_and_tokenizer(
        args.base_model,
        quantization=args.quantization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        for_training=True,
    )

    model.config.use_cache = False
    if args.quantization != "none":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_csv(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = TokenizedCodeDataset(train_records, tokenizer, args.max_length)
    eval_dataset = TokenizedCodeDataset(eval_records, tokenizer, args.max_length) if eval_records else None
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = _build_training_args(TrainingArguments, args, torch)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

