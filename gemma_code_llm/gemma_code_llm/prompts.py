def _clean(value: str | None) -> str:
    return (value or "").strip()


def build_prompt(instruction: str, input_text: str | None = None) -> str:
    instruction = _clean(instruction)
    input_text = _clean(input_text)

    if not instruction:
        raise ValueError("instruction must not be empty")

    parts = ["### Instruction:", instruction]
    if input_text:
        parts.extend(["", "### Input:", input_text])
    parts.extend(["", "### Response:", ""])
    return "\n".join(parts)


def build_training_text(instruction: str, input_text: str | None, output: str, eos_token: str = "") -> str:
    output = _clean(output)
    if not output:
        raise ValueError("output must not be empty")
    return build_prompt(instruction, input_text) + output + eos_token

