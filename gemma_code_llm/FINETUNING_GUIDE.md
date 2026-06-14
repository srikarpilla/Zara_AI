# Gemma Code LLM Fine-Tuning Guide

This guide describes the technical design, dataset parsing, and execution workflow for fine-tuning `google/gemma-3-1b-it` on code-generation tasks.

---

## 1. The Approach: Low-Rank Adaptation (LoRA)

Fine-tuning a large model with 1+ billion parameters requires substantial GPU memory. Instead of updating all model parameters (which would trigger Out-of-Memory errors on most hardware), this pipeline uses **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)**.

### How LoRA Works:
1. **Freezing base weights**: The pre-trained parameters of the base model (`Gemma-3-1b-it`) are frozen and do not change.
2. **Injecting low-rank adapters**: Two small trainable low-rank matrices are added to specific attention and feed-forward layers:
   * **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
3. **Trainable parameters**: This reduces the number of trained parameters from **1,012,931,712** down to **13,045,760** (only **1.28%** of the model).
4. **Saving Disk Space**: Rather than saving a ~2.5 GB model, we only save the adapter weights (`adapter_model.safetensors`), which are around **52 MB**.

---

## 2. How the Data is Taken & Tokenized

The pipeline uses a unified tokenization process to prevent train-inference mismatches.

### A. Data Format (JSONL)
The training data must be stored in a `.jsonl` file (JSON Lines) where each line represents a single training sample:
```json
{"instruction": "Write a Python function to add two numbers.", "input": "", "output": "```python\ndef add(a, b):\n    return a + b\n```"}
```
* **`instruction`** (Required): The developer's request.
* **`input`** (Optional): Extra context, code snippet with bugs, or input specifications.
* **`output`** (Required): The target response (the generated code output).

### B. Unified Chat Template Formatting
Rather than hardcoding manual instruction dividers, the data processor uses Gemma's native chat template formatting via `tokenizer.apply_chat_template`. This ensures the model receives correct beginning-of-sequence (`<bos>`) and turn-marker tokens:

```python
# User Turn
messages = [
    {"role": "user", "content": f"{instruction}\n{input_text}".strip() if input_text else instruction}
]

# Combined Turn (User + Model Response)
messages_full = messages + [{"role": "model", "content": output_code}]
```

### C. Token-Level Label Masking
During training, we want the model to learn to generate only the **response** (the code output), not the prompt. We compute loss only on the model response by masking out the prompt tokens in the training labels:

```python
# 1. Tokenize prompt only (with add_generation_prompt=True)
prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

# 2. Tokenize prompt + response together
full_ids = tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False)

# 3. Create target labels from full token IDs
labels = list(full_ids)

# 4. Mask the prompt prefix using HuggingFace's cross-entropy pad value (-100)
prompt_length = min(len(prompt_ids), len(labels))
labels[:prompt_length] = [-100] * prompt_length
```

---

## 3. Core Fine-Tuning Code

### A. Data Encoding & Loader ([data.py](file:///d:/ZaraAI/gemma_code_llm/gemma_code_llm/data.py))
The dataset encoder converts the JSONL records into token sequences. It handles sequence truncation robustly: if a sequence exceeds `max_length`, it skips the record and prints a warning rather than crashing the training process.

```python
def encode_training_record(record: CodeRecord, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    user_content = record.instruction
    if record.input:
        user_content += f"\n{record.input}"

    messages = [{"role": "user", "content": user_content}]

    # Format turns and extract raw token IDs
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]

    messages_full = messages + [{"role": "model", "content": record.output}]
    full_ids = tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False)
    if not isinstance(full_ids, list):
        full_ids = full_ids["input_ids"]

    input_ids = list(full_ids[:max_length])
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    # Mask user query
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    if not input_ids:
        raise ValueError("tokenized record is empty")
    if all(label == -100 for label in labels):
        raise ValueError(f"max_length={max_length} is too short; response was fully truncated")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

### B. Training Loop Script ([train_lora.py](file:///d:/ZaraAI/gemma_code_llm/gemma_code_llm/train_lora.py))
The training script loads the model in the selected precision, applies the PEFT LoRA adapter configurations, initializes the dataset, and runs the Hugging Face `Trainer`.

```python
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=_csv(args.target_modules),
)
model = get_peft_model(model, lora_config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    ),
)
trainer.train()
trainer.save_model(str(output_dir))
```

---

## 4. Steps to Fine-Tune the Model

### Step 1: Initialize Environment
Use Python 3.10 or 3.11:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Accept License & Login to Hugging Face
Gemma requires gate access approval on the Hugging Face Model Hub. Once accepted, login via CLI:
```powershell
huggingface-cli login
```

### Step 3: Validate the Dataset
Ensure there are no Python syntax errors or formatting issues in the dataset output:
```powershell
python -m gemma_code_llm.validate_dataset --data data\sample_code_instructions.jsonl --check-python
```

### Step 4: Run Training
Kick off the LoRA training script. Update `--data` with your real JSONL dataset path:
```powershell
python -m gemma_code_llm.train_lora `
  --data data\sample_code_instructions.jsonl `
  --base-model google/gemma-3-1b-it `
  --output-dir outputs\gemma-code-lora `
  --epochs 3 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --max-length 2048 `
  --quantization none
```
*(On Linux CUDA GPUs, you can pass `--quantization 4bit` to save VRAM).*

### Step 5: Generate Code (Test Model)
Verify code generation works on a sample prompt:
```powershell
python -m gemma_code_llm.generate `
  --base-model google/gemma-3-1b-it `
  --adapter outputs\gemma-code-lora `
  --instruction "Write a Python function that checks if a number is prime." `
  --max-new-tokens 300 `
  --temperature 0.2
```

### Step 6: Serve the API Server
Start the local FastAPI server:
```powershell
$env:BASE_MODEL="google/gemma-3-1b-it"
$env:ADAPTER_PATH="outputs\gemma-code-lora"
$env:QUANTIZATION="none"
python -m uvicorn gemma_code_llm.api:app --host 127.0.0.1 --port 8000
```
Query the server using PowerShell:
```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/generate `
  -ContentType "application/json" `
  -Body '{"instruction":"Write a Python function that reverses a string.","max_new_tokens":300}'
```
