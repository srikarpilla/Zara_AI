from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from .data import load_jsonl_records


_FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_python_blocks(text: str) -> list[str]:
    blocks = [match.group(1).strip() for match in _FENCED_CODE_RE.finditer(text)]
    if blocks:
        return blocks

    simple_python_markers = ("def ", "class ", "import ", "from ")
    stripped = text.strip()
    if stripped.startswith(simple_python_markers):
        return [stripped]
    return []


def validate_python_outputs(records) -> list[str]:
    errors: list[str] = []
    for record in records:
        for block_index, block in enumerate(_extract_python_blocks(record.output), start=1):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                errors.append(
                    f"Line {record.line_number}, code block {block_index}: "
                    f"Python syntax error at line {exc.lineno}: {exc.msg}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Gemma code-instruction JSONL dataset.")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--check-python", action="store_true", help="Parse Python fenced code blocks with ast.")
    args = parser.parse_args()

    data_path = Path(args.data)
    records = load_jsonl_records(data_path)
    print(f"Loaded {len(records)} records from {data_path}")

    if args.check_python:
        errors = validate_python_outputs(records)
        if errors:
            print("Python syntax errors:")
            for error in errors:
                print(f"- {error}")
            return 1
        print("Python syntax check passed")

    print("Dataset validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

