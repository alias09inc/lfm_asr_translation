from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MT_PROMPT_TEMPLATE


def translation_input_ids(tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    prompt = MT_PROMPT_TEMPLATE.format(text=text.strip())
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if isinstance(input_ids, torch.Tensor):
        return input_ids
    if isinstance(input_ids, dict):
        return input_ids["input_ids"]
    return input_ids.input_ids


@dataclass
class Translator:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str
    max_new_tokens: int

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        input_ids = translation_input_ids(self.tokenizer, text).to(self.device)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
