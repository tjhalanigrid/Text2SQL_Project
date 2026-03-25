from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    PeftModel = None  # type: ignore


@dataclass(frozen=True)
class QuantArtifact:
    out_dir: Path
    mode: str  # fp32 | int8_dynamic | int8_decoder_dynamic | int8_bnb | int4_bnb
    base_model: str
    adapter_path: Optional[str]
    created_at_s: float


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in {"1", "true", "True", "yes", "Y"}


def estimate_model_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def _load_tokenizer(base_model: str, *, local_only: bool) -> Any:
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=local_only)
    if tok.pad_token_id is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp32_model(
    base_model: str,
    *,
    adapter_path: Optional[str] = None,
    device: str = "cpu",
    local_only: bool = True,
    torch_dtype: torch.dtype = torch.float32,
    merge_lora: bool = True,
) -> Tuple[Any, torch.nn.Module]:
    tok = _load_tokenizer(base_model, local_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        local_files_only=local_only,
        torch_dtype=torch_dtype,
    ).to(device)

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load adapters.")
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            model = model.to(device)

    model.eval()
    return tok, model


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    # CPU-only; quantized kernels run on CPU.
    # Ensure a quantization engine is selected (PyTorch may default to "none" on macOS).
    try:
        supported = list(getattr(torch.backends.quantized, "supported_engines", []))
        current = getattr(torch.backends.quantized, "engine", "none")
        if current in {"none", None, ""}:
            if "fbgemm" in supported:
                torch.backends.quantized.engine = "fbgemm"
            elif "qnnpack" in supported:
                torch.backends.quantized.engine = "qnnpack"
    except Exception:  # pragma: no cover
        pass
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_dynamic_int8_decoder_only(model: Any) -> Any:
    """
    Mixed-precision (Task 5): encoder fp32, decoder int8 dynamic quantized.
    """
    if not hasattr(model, "decoder"):
        raise ValueError("Model has no decoder attribute.")
    try:
        supported = list(getattr(torch.backends.quantized, "supported_engines", []))
        current = getattr(torch.backends.quantized, "engine", "none")
        if current in {"none", None, ""}:
            if "fbgemm" in supported:
                torch.backends.quantized.engine = "fbgemm"
            elif "qnnpack" in supported:
                torch.backends.quantized.engine = "qnnpack"
    except Exception:  # pragma: no cover
        pass
    model.decoder = torch.quantization.quantize_dynamic(model.decoder, {torch.nn.Linear}, dtype=torch.qint8)
    return model


def load_bnb_quantized_model(
    base_model: str,
    *,
    adapter_path: Optional[str],
    device: str,
    local_only: bool,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[Any, torch.nn.Module]:
    """
    bitsandbytes int8/int4 (requires bitsandbytes + CUDA). Not supported on CPU/MPS.
    """
    if BitsAndBytesConfig is None:
        raise RuntimeError("transformers BitsAndBytesConfig not available; upgrade transformers or install extras.")
    if device != "cuda":
        raise RuntimeError("bitsandbytes quantization requires CUDA (device=cuda).")
    if not (load_in_8bit or load_in_4bit):
        raise ValueError("Specify load_in_8bit or load_in_4bit.")

    tok = _load_tokenizer(base_model, local_only=local_only)
    qconf = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        local_files_only=local_only,
        quantization_config=qconf,
        device_map="auto",
    )
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load adapters.")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def save_quant_artifact(
    out_dir: str | Path,
    *,
    mode: str,
    base_model: str,
    adapter_path: Optional[str],
    tokenizer: Any,
    model: torch.nn.Module,
) -> QuantArtifact:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer").mkdir(exist_ok=True)

    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.state_dict(), out / "model.pt")

    meta: Dict[str, Any] = {
        "mode": mode,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "created_at_s": time.time(),
        "estimated_model_bytes": estimate_model_bytes(model),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    return QuantArtifact(
        out_dir=out,
        mode=mode,
        base_model=base_model,
        adapter_path=adapter_path,
        created_at_s=float(meta["created_at_s"]),
    )


def load_quant_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
    local_only: bool = True,
) -> Tuple[Any, torch.nn.Module, Dict[str, Any]]:
    """
    Loads a previously exported quant artifact.
    For dynamic quant modes, we reconstruct the architecture, apply the same quantization,
    then load the saved state_dict.
    """
    adir = Path(artifact_dir)
    meta = json.loads((adir / "meta.json").read_text())
    mode = meta["mode"]
    base_model = meta["base_model"]

    tok = AutoTokenizer.from_pretrained(adir / "tokenizer", local_files_only=True)
    if tok.pad_token_id is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, local_files_only=local_only).to(device)
    model.eval()

    if mode == "int8_dynamic":
        model = quantize_dynamic_int8(model)
    elif mode == "int8_decoder_dynamic":
        model = quantize_dynamic_int8_decoder_only(model)
    elif mode in {"fp32"}:
        pass
    else:
        raise RuntimeError(f"Unsupported artifact mode for local loading: {mode}")

    state = torch.load(adir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return tok, model, meta
