import os
import json
import re
import argparse
from datasets import load_dataset

def convert_huggingface_dataset(dataset_name="AI-Mock-Interviewer/T5_retrain"):
    # 1. Load HF_TOKEN from main .env file if available
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.strip().split("=", 1)[1].strip()
                        os.environ["HF_TOKEN"] = token
                        print("HF_TOKEN successfully loaded from project .env")
                        break

    print(f"Loading Hugging Face dataset '{dataset_name}'...")
    try:
        ds = load_dataset(dataset_name)
        train_data = ds["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    output_path = os.path.join(os.path.dirname(__file__), "interview_questions_train.jsonl")
    print(f"Parsing and saving to {output_path}...")

    converted_count = 0
    skipped_count = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        if "T5_retrain" in dataset_name:
            for idx, row in enumerate(train_data):
                question = row.get("Question", "")
                if isinstance(question, str):
                    question = question.strip()
                target_domain = row.get("Target Domain", "")
                if isinstance(target_domain, str):
                    target_domain = target_domain.strip()

                if question and target_domain:
                    record = {
                        "instruction": f"Generate a {target_domain} interview question",
                        "input": "",
                        "output": question
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1
        else:
            # Fallback pattern matching for standard formats (like Train_data)
            pattern = re.compile(r"###\s*Instruction:\s*(.*?)\s*###\s*Response:\s*(.*)", re.DOTALL | re.IGNORECASE)
            for idx, row in enumerate(train_data):
                text = row.get("text", "")
                match = pattern.search(text)
                if match:
                    instruction = match.group(1).strip()
                    output = match.group(2).strip()
                    
                    # Check for nested responses/instructions
                    if "### Response:" in output:
                        # Strip any subsequent prompt formatting if present
                        output = output.split("### Response:")[0].strip()

                    record = {
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1

    print("\n================ CONVERSION SUMMARY ================")
    print(f"Dataset: {dataset_name}")
    print(f"Total input rows: {len(train_data)}")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped / Failed to parse: {skipped_count}")
    print(f"Saved JSONL file to: {output_path}")
    print("=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face datasets to JSONL for fine-tuning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AI-Mock-Interviewer/T5_retrain",
        help="Name of the Hugging Face dataset to load (default: AI-Mock-Interviewer/T5_retrain)"
    )
    args = parser.parse_args()
    convert_huggingface_dataset(args.dataset)
