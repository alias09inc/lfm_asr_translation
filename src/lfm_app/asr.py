from __future__ import annotations

import numpy as np
import torch
from fastrtc import AdditionalOutputs
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

from .config import ASR_PROMPT
from .state import SessionState
from .translation import Translator
from .utils import append_line, clean_text, normalize_prefix


def decode_tokens(processor: LFM2AudioProcessor, tokens: list[torch.Tensor]) -> str:
    if not tokens:
        return ""
    text = processor.text.decode(torch.cat(tokens))
    return clean_text(text)


def build_asr_chat(processor: LFM2AudioProcessor, chat_dtype: torch.dtype) -> ChatState:
    chat = ChatState(processor, dtype=chat_dtype)
    chat.new_turn("system")
    chat.add_text(ASR_PROMPT)
    chat.end_turn()
    return chat


def transcribe_response(
    audio: tuple[int, np.ndarray],
    state: SessionState,
    *,
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    chat_dtype: torch.dtype,
    max_new_tokens: int,
    translator: Translator,
):
    rate, wav = audio
    if wav.size == 0:
        return

    wav = np.asarray(wav)
    if wav.ndim == 1:
        wav = wav[None, :]
    elif wav.shape[0] != 1:
        wav = wav[:1, :]

    wave_tensor = torch.tensor(wav / 32768.0, dtype=torch.float32)

    chat = build_asr_chat(processor, chat_dtype)
    chat.new_turn("user")
    chat.add_audio(wave_tensor, rate)
    chat.end_turn()
    chat.new_turn("assistant")

    base_text = state.transcript
    base_translation = state.translation
    prefix = normalize_prefix(base_text)

    tokens: list[torch.Tensor] = []
    for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
        if t.numel() != 1:
            continue
        tokens.append(t)
        partial = decode_tokens(processor, tokens)
        state.transcript = f"{prefix}{partial}"
        yield AdditionalOutputs(state.transcript, base_translation)

    final = decode_tokens(processor, tokens).strip()
    if final:
        state.transcript = f"{prefix}{final}"
        if not state.transcript.endswith("\n"):
            state.transcript += "\n"
        translated = translator.translate(final)
        state.translation = append_line(base_translation, translated) if translated else base_translation
        yield AdditionalOutputs(state.transcript, state.translation)
    else:
        state.transcript = base_text
        state.translation = base_translation
