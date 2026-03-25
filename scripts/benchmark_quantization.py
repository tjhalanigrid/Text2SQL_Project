from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.execution_reward import execution_reward
from src.prompting import encode_prompt
from src.quantization_utils import load_fp32_model, load_quant_artifact


def _load_dev_items(root: Path, n: int, seed: int = 42) -> List[dict]:
    data = json.loads((root / "data" / "dev.json").read_text())
    if n >= len(data):
        return data
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(data), size=n, replace=False)
    return [data[int(i)] for i in idxs]


def _bench_variant(name: str, tok, model, items: List[dict], device: str) -> Dict[str, float]:
    latencies: List[float] = []
    ex = 0

    # Warmup (1 item)
    if items:
        it = items[0]
        _ = encode_prompt(tok, it["question"], it["db_id"], device=device, max_input_tokens=512).unsqueeze(0)

    for it in items:
        db_id = it["db_id"]
        q = it["question"]
        gold = it["query"]
        db_path = str(Path("data") / "database" / db_id / f"{db_id}.sqlite")

        input_ids = encode_prompt(tok, q, db_id, device=device, max_input_tokens=512).unsqueeze(0)
        t0 = time.perf_counter()
        out = model.generate(input_ids=input_ids, max_new_tokens=120, num_beams=8, repetition_penalty=1.2)
        dt = time.perf_counter() - t0
        latencies.append(dt)

        pred = tok.decode(out[0], skip_special_tokens=True).strip()
        r = execution_reward(pred, db_path, gold)
        if float(r) >= 1.0:
            ex += 1

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    mean = float(np.mean(latencies)) if latencies else 0.0
    return {
        "n": float(len(items)),
        "ex": float(ex / max(len(items), 1)),
        "lat_mean_s": mean,
        "lat_p50_s": p50,
        "lat_p90_s": p90,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark fp32 vs quantized artifacts (CPU-focused).")
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "Salesforce/codet5-base"))
    p.add_argument("--adapter", default="", help="Optional adapter for fp32 baseline.")
    p.add_argument("--artifact_int8", default="", help="Artifact dir exported by scripts/quantize_export.py")
    p.add_argument("--artifact_int8_decoder", default="", help="Artifact dir for decoder-only int8")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/task5_quant_bench.json")
    p.add_argument("--local_only", action="store_true")
    args = p.parse_args()

    device = "cpu"
    root = Path(".")
    items = _load_dev_items(root, args.num_samples, args.seed)

    report: Dict[str, Dict[str, float]] = {}

    tok, fp32 = load_fp32_model(
        args.base_model,
        adapter_path=args.adapter.strip() or None,
        device=device,
        local_only=args.local_only,
    )
    report["fp32"] = _bench_variant("fp32", tok, fp32, items, device)

    if args.artifact_int8:
        tok8, m8, _meta = load_quant_artifact(args.artifact_int8, device=device, local_only=True)
        report["int8_dynamic"] = _bench_variant("int8_dynamic", tok8, m8, items, device)

    if args.artifact_int8_decoder:
        tokd, md, _meta = load_quant_artifact(args.artifact_int8_decoder, device=device, local_only=True)
        report["int8_decoder_dynamic"] = _bench_variant("int8_decoder_dynamic", tokd, md, items, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

