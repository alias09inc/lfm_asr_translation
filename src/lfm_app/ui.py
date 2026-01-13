from __future__ import annotations

import gradio as gr
from fastrtc import AlgoOptions, ReplyOnPause, WebRTC
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
import torch

from .asr import transcribe_response
from .state import SessionState
from .translation import Translator


def clear_session():
    return SessionState(), "", ""


def build_demo(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    chat_dtype: torch.dtype,
    max_new_tokens: int,
    max_segment_s: float,
    translator: Translator,
):
    fallback_state = SessionState()

    def _transcribe(*args):
        if len(args) == 2:
            audio_or_data, state = args
            audio = getattr(audio_or_data, "audio", audio_or_data)
        elif len(args) >= 3:
            audio, _webrtc_id, state = args[:3]
        elif len(args) == 1:
            audio = getattr(args[0], "audio", args[0])
            state = fallback_state
        else:
            raise ValueError(f"Unexpected args for transcribe: {len(args)}")

        return transcribe_response(
            audio,
            state,
            model=model,
            processor=processor,
            chat_dtype=chat_dtype,
            max_new_tokens=max_new_tokens,
            translator=translator,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# LFM2.5-Audio Real-Time Transcription + EN to JP Translation")

        session_state = gr.State(SessionState())
        webrtc = WebRTC(modality="audio", mode="send", full_screen=False)
        transcript_box = gr.Textbox(lines=10, label="Transcript", interactive=False)
        translation_box = gr.Textbox(lines=10, label="Japanese Translation", interactive=False)
        clear_btn = gr.Button("Clear")

        webrtc.stream(
            ReplyOnPause(
                _transcribe,
                algo_options=AlgoOptions(max_continuous_speech_s=max_segment_s),
                input_sample_rate=24_000,
                output_sample_rate=24_000,
                can_interrupt=False,
                needs_args=True,
            ),
            inputs=[webrtc, session_state],
            outputs=[webrtc],
        )
        webrtc.on_additional_outputs(
            lambda transcript, translation: (transcript, translation),
            outputs=[transcript_box, translation_box],
        )
        clear_btn.click(clear_session, outputs=[session_state, transcript_box, translation_box])

    return demo
