# Implementation Plan - Gemma Fine-Tuning Improvements

This plan addresses several bugs and design gaps in the current Gemma fine-tuning codebase.

## User Review Required

> [!IMPORTANT]
> **Prompt Template Transition**: Changing from the custom `### Instruction:` format to the official Gemma chat template (`apply_chat_template`) will affect inference. If you have external applications querying this model with raw `### Instruction:` format, they must switch to formatting input via the tokenizer's chat template or OpenAI-style role structures.
> 
> **Relaxed Safety Filter**: The safety filter will no longer throw hard exceptions on words like `password` or calling `subprocess.run` to allow the code generation model to execute normal developer workflows.

---

## Proposed Changes

### 1. Requirements

#### [MODIFY] [requirements.txt](file:///d:/ZaraAI/gemma_code_llm/requirements.txt)
- Remove the environment marker `; platform_system != "Windows"` from `bitsandbytes` to allow Windows users with CUDA GPUs to install and use bitsandbytes 0.43.3+.

---

### 2. Dataset Processing

#### [MODIFY] [data.py](file:///d:/ZaraAI/gemma_code_llm/gemma_code_llm/data.py)
- Refactor `encode_training_record` to format data using `tokenizer.apply_chat_template` rather than custom markdown string formatting.
- Unify tokenization to use `tokenizer.apply_chat_template(..., tokenize=True)` which consistently sets the beginning-of-sequence (`<bos>`) token.
- Replace the fatal `ValueError` crash on long sequences with a warning log, skipping/filtering out sequences that exceed the maximum length.

---

### 3. Model Inference

#### [MODIFY] [inference.py](file:///d:/ZaraAI/gemma_code_llm/gemma_code_llm/inference.py)
- Refactor `generate_code` to format and tokenize input prompts using `tokenizer.apply_chat_template(..., tokenize=True, return_tensors="pt")`.
- This ensures the model receives inputs matching the exact template layout used during fine-tuning (complete with `<bos>`).

---

### 4. Safety Constraints

#### [MODIFY] [safety.py](file:///d:/ZaraAI/gemma_code_llm/gemma_code_llm/safety.py)
- Update `_DANGEROUS_PATTERNS` to remove:
  - Broad keyword matches on `"password"` and `"passwd"` from the `secret_access` checks.
  - Matches on standard coding modules like `subprocess.run` and `os.system` from the `shell_execution` checks.
- Retain truly malicious patterns (e.g. destructive deletions like `rm -rf`, reverse shells, disk formatting commands).

---

## Verification Plan

### Automated Verification
Run the validation and smoke training commands to ensure code compiles and runs:
```powershell
# Validate dataset format with updated loader
python -m gemma_code_llm.validate_dataset --data data\sample_code_instructions.jsonl

# Run a quick training smoke test on the sample dataset
python -m gemma_code_llm.train_lora --data data\sample_code_instructions.jsonl --epochs 1 --batch-size 1 --gradient-accumulation-steps 4 --max-length 1024 --quantization none
```

### Manual Verification
Test generating code through the generation script to check output:
```powershell
python -m gemma_code_llm.generate --instruction "Write a Python function that validates an email address."
```
Verify that instructions with words like "password" are no longer blocked by the safety filter.
