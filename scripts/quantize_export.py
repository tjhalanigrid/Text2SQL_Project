from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from src.quantization_utils import (
    load_bnb_quantized_model,
    load_fp32_model,
    quantize_dynamic_int8,
    quantize_dynamic_int8_decoder_only,
    save_quant_artifact,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Export quantized Seq2Seq model artifacts for CPU inference.")
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "Salesforce/codet5-base"))
    p.add_argument("--adapter", default="", help="Optional LoRA adapter directory.")
    p.add_argument("--out_dir", required=True, help="Output directory for artifact.")
    p.add_argument(
        "--mode",
        required=True,
        choices=["fp32", "int8_dynamic", "int8_decoder_dynamic", "int8_bnb", "int4_bnb"],
    )
    p.add_argument("--device", default="cpu", help="cpu|cuda (bnb requires cuda)")
    p.add_argument("--local_only", action="store_true", help="Do not hit network; use HF cache only.")
    args = p.parse_args()

    adapter = args.adapter.strip() or None
    out_dir = Path(args.out_dir)

    if args.mode == "fp32":
        tok, model = load_fp32_model(args.base_model, adapter_path=adapter, device=args.device, local_only=args.local_only)
        save_quant_artifact(out_dir, mode="fp32", base_model=args.base_model, adapter_path=adapter, tokenizer=tok, model=model)
        return

    if args.mode == "int8_dynamic":
        tok, model = load_fp32_model(args.base_model, adapter_path=adapter, device="cpu", local_only=args.local_only)
        model = quantize_dynamic_int8(model)
        save_quant_artifact(out_dir, mode="int8_dynamic", base_model=args.base_model, adapter_path=adapter, tokenizer=tok, model=model)
        return

    if args.mode == "int8_decoder_dynamic":
        tok, model = load_fp32_model(args.base_model, adapter_path=adapter, device="cpu", local_only=args.local_only)
        model = quantize_dynamic_int8_decoder_only(model)
        save_quant_artifact(
            out_dir,
            mode="int8_decoder_dynamic",
            base_model=args.base_model,
            adapter_path=adapter,
            tokenizer=tok,
            model=model,
        )
        return

    if args.mode == "int8_bnb":
        tok, model = load_bnb_quantized_model(
            args.base_model,
            adapter_path=adapter,
            device=args.device,
            local_only=args.local_only,
            load_in_8bit=True,
        )
        # Note: saving bnb quantized weights in a portable way is non-trivial; we still save state_dict for reference.
        save_quant_artifact(out_dir, mode="int8_bnb", base_model=args.base_model, adapter_path=adapter, tokenizer=tok, model=model)
        return

    if args.mode == "int4_bnb":
        tok, model = load_bnb_quantized_model(
            args.base_model,
            adapter_path=adapter,
            device=args.device,
            local_only=args.local_only,
            load_in_4bit=True,
        )
        save_quant_artifact(out_dir, mode="int4_bnb", base_model=args.base_model, adapter_path=adapter, tokenizer=tok, model=model)
        return


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

