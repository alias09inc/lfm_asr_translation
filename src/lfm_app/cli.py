from __future__ import annotations

import argparse

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HF_REPO_DEFAULT, MT_REPO_DEFAULT
from .translation import Translator
from .ui import build_demo
from .utils import resolve_device, resolve_dtype


def main():
    parser = argparse.ArgumentParser(description="Real-time ASR + EN to JP translation.")
    parser.add_argument("--repo", default=None, help="Deprecated alias for --asr-repo.")
    parser.add_argument("--asr-repo", default=HF_REPO_DEFAULT, help="Hugging Face repo for LFM2.5-Audio.")
    parser.add_argument("--mt-repo", default=MT_REPO_DEFAULT, help="Hugging Face repo for EN-JP translation.")
    parser.add_argument("--device", default=None, help="Device override (e.g., cuda or cpu).")
    parser.add_argument("--mt-device", default=None, help="Device override for translation model.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public share link.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per segment.")
    parser.add_argument("--mt-max-new-tokens", type=int, default=256, help="Max tokens per translation segment.")
    parser.add_argument("--max-segment-s", type=float, default=10.0, help="Max seconds before forcing a segment.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    mt_device = resolve_device(args.mt_device) if args.mt_device else device
    dtype = resolve_dtype(device)
    mt_dtype = resolve_dtype(mt_device)

    asr_repo = args.asr_repo
    if args.repo:
        asr_repo = args.repo

    processor = LFM2AudioProcessor.from_pretrained(asr_repo, device=device).eval()
    model = LFM2AudioModel.from_pretrained(asr_repo, device=device, dtype=dtype).eval()
    if not device.startswith("cuda"):
        model.lfm.set_attn_implementation("sdpa")

    mt_tokenizer = AutoTokenizer.from_pretrained(args.mt_repo)
    mt_model = AutoModelForCausalLM.from_pretrained(args.mt_repo, torch_dtype=mt_dtype).to(mt_device).eval()
    translator = Translator(
        tokenizer=mt_tokenizer,
        model=mt_model,
        device=mt_device,
        max_new_tokens=args.mt_max_new_tokens,
    )

    demo = build_demo(
        model=model,
        processor=processor,
        chat_dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        max_segment_s=args.max_segment_s,
        translator=translator,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
