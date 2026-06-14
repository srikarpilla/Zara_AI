# Gemma Code LLM Starter

This folder contains a complete starter project for fine-tuning a Gemma model for code generation with LoRA.

Default model:

```text
google/gemma-3-1b-it
```

Use this first because it is smaller and easier to debug. After the full pipeline works, try a larger Gemma variant such as `google/gemma-3-4b-it`.

## 1. Create environment

Use Python 3.10 or 3.11 for the smoothest ML package support.

```powershell
cd D:\ZaraAI\gemma_code_llm
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux with an NVIDIA GPU, `bitsandbytes` enables 4-bit training. On Windows, start with `--quantization none`.

## 2. Log in to Hugging Face

Gemma models require accepting Google's license on Hugging Face.

```powershell
huggingface-cli login
```

Then open the model page and accept access for the model you want to use.

## 3. Validate the dataset

The sample dataset is tiny and only proves the pipeline works. Replace it with your real code instruction data later.

```powershell
python -m gemma_code_llm.validate_dataset --data data\sample_code_instructions.jsonl --check-python
```

## 4. Train LoRA adapter

Small CPU/GPU smoke test:

```powershell
python -m gemma_code_llm.train_lora ^
  --data data\sample_code_instructions.jsonl ^
  --base-model google/gemma-3-1b-it ^
  --output-dir outputs\gemma-code-lora ^
  --epochs 1 ^
  --batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --max-length 1024 ^
  --quantization none
```

Linux CUDA 4-bit example:

```bash
python -m gemma_code_llm.train_lora \
  --data data/sample_code_instructions.jsonl \
  --base-model google/gemma-3-1b-it \
  --output-dir outputs/gemma-code-lora \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 2048 \
  --quantization 4bit
```

## 5. Generate code

```powershell
python -m gemma_code_llm.generate ^
  --base-model google/gemma-3-1b-it ^
  --adapter outputs\gemma-code-lora ^
  --instruction "Write a Python function that returns the nth Fibonacci number."
```

## 6. Run API server

```powershell
$env:BASE_MODEL="google/gemma-3-1b-it"
$env:ADAPTER_PATH="outputs\gemma-code-lora"
$env:QUANTIZATION="none"
uvicorn gemma_code_llm.api:app --host 127.0.0.1 --port 8000
```

Test:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/generate `
  -ContentType "application/json" `
  -Body '{"instruction":"Write a Python function that checks if a number is prime.","max_new_tokens":300}'
```

## Dataset format

Use JSONL, one example per line:

```json
{"instruction":"Write a Python function that adds two numbers.","input":"","output":"```python\ndef add(a, b):\n    return a + b\n```"}
```

Required fields:

```text
instruction
output
```

Optional field:

```text
input
```

## Recommended development path

1. Run inference with the base model.
2. Validate 100 clean examples.
3. Train one tiny LoRA run.
4. Generate code and inspect failures.
5. Grow the dataset to 1,000 high-quality examples.
6. Add unit-test-style examples.
7. Train again.
8. Evaluate with HumanEval or MBPP.
9. Serve with FastAPI.
10. Add a sandbox before executing generated code.

