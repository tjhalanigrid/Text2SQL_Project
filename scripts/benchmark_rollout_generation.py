from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.prompting import encode_prompt
from src.quantization_utils import load_fp32_model, load_quant_artifact


def _load_items(root: Path, n: int, seed: int = 42) -> List[dict]:
    data = json.loads((root / "data" / "dev.json").read_text())
    if n >= len(data):
        return data
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(data), size=n, replace=False)
    return [data[int(i)] for i in idxs]


def _bench_generate(tok, model, items: List[dict], device: str) -> float:
    t0 = time.perf_counter()
    for it in items:
        input_ids = encode_prompt(tok, it["question"], it["db_id"], device=device, max_input_tokens=512).unsqueeze(0)
        _ = model.generate(input_ids=input_ids, max_new_tokens=64, num_beams=4)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark rollout generation latency for RL loops.")
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "Salesforce/codet5-base"))
    p.add_argument("--adapter", default="")
    p.add_argument("--artifact", default="", help="Quantized artifact dir (optional).")
    p.add_argument("--num_rollouts", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_only", action="store_true")
    args = p.parse_args()

    device = "cpu"
    root = Path(".")
    items = _load_items(root, args.num_rollouts, args.seed)

    tok, fp32 = load_fp32_model(
        args.base_model,
        adapter_path=args.adapter.strip() or None,
        device=device,
        local_only=args.local_only,
    )
    t_fp32 = _bench_generate(tok, fp32, items, device)
    print(f"fp32: {t_fp32:.2f}s for {len(items)} rollouts ({len(items)/max(t_fp32,1e-9):.2f} rollouts/s)")

    if args.artifact:
        tokq, mq, meta = load_quant_artifact(args.artifact, device=device, local_only=True)
        t_q = _bench_generate(tokq, mq, items, device)
        mode = meta.get("mode", "quant")
        print(f"{mode}: {t_q:.2f}s for {len(items)} rollouts ({len(items)/max(t_q,1e-9):.2f} rollouts/s)")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
