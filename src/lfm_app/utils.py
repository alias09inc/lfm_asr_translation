import torch


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def clean_text(text: str) -> str:
    return text.replace("<|text_end|>", "").replace("<|im_end|>", "")


def normalize_prefix(text: str) -> str:
    if text and not text.endswith("\n"):
        return f"{text}\n"
    return text


def append_line(buffer: str, line: str) -> str:
    line = line.strip()
    if not line:
        return buffer
    if buffer and not buffer.endswith("\n"):
        buffer += "\n"
    return f"{buffer}{line}\n"
