from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable




@dataclass(frozen=True)
class CodeRecord:
    instruction: str
    input: str
    output: str
    line_number: int


def _as_string(value: Any, field_name: str, line_number: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Line {line_number}: {field_name} must be a string")
    return value.strip()


def load_jsonl_records(path: str | Path, *, max_records: int | None = None) -> list[CodeRecord]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records: list[CodeRecord] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc

            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number}: record must be a JSON object")

            instruction = _as_string(item.get("instruction"), "instruction", line_number)
            input_text = _as_string(item.get("input", ""), "input", line_number)
            output = _as_string(item.get("output"), "output", line_number)

            if not instruction:
                raise ValueError(f"Line {line_number}: instruction is required")
            if not output:
                raise ValueError(f"Line {line_number}: output is required")

            records.append(
                CodeRecord(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    line_number=line_number,
                )
            )

            if max_records is not None and len(records) >= max_records:
                break

    if not records:
        raise ValueError(f"No records found in {data_path}")
    return records


def split_records(records: list[CodeRecord], validation_size: float, seed: int) -> tuple[list[CodeRecord], list[CodeRecord]]:
    if validation_size <= 0:
        return records, []
    if validation_size >= 1:
        raise ValueError("validation_size must be less than 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_size))
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]

    if not training:
        raise ValueError("validation_size leaves no training records")
    return training, validation


def encode_training_record(record: CodeRecord, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    user_content = record.instruction
    if record.input:
        user_content += f"\n{record.input}"

    messages = [
        {"role": "user", "content": user_content},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]

    messages_full = messages + [{"role": "model", "content": record.output}]
    full_ids = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
    )
    if not isinstance(full_ids, list):
        full_ids = full_ids["input_ids"]

    input_ids = list(full_ids[:max_length])
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    if not input_ids:
        raise ValueError("tokenized record is empty")
    if all(label == -100 for label in labels):
        raise ValueError(
            f"max_length={max_length} is too short; the response was fully truncated"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class TokenizedCodeDataset:
    def __init__(self, records: Iterable[CodeRecord], tokenizer: Any, max_length: int):
        self.examples = []
        for record in records:
            try:
                example = encode_training_record(record, tokenizer, max_length)
                self.examples.append(example)
            except ValueError as exc:
                print(f"Warning: Skipping record at line {record.line_number}: {exc}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]

