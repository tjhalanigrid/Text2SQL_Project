from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.quantized_text2sql_engine import QuantizedText2SQLEngine


def main() -> None:
    p = argparse.ArgumentParser(description="Production-style inference harness for quantized artifacts.")
    p.add_argument("--artifact", required=True, help="Quant artifact dir from scripts/quantize_export.py")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--out", default="results/task5_quant_infer.json")
    args = p.parse_args()

    root = Path(".")
    dev = json.loads((root / "data" / "dev.json").read_text())
    dev = dev[: args.num_samples]

    engine = QuantizedText2SQLEngine(args.artifact, device="cpu")
    pairs = [(x["question"], x["db_id"]) for x in dev]

    t0 = time.perf_counter()
    results = engine.ask_batch_execute(pairs)
    dt = time.perf_counter() - t0

    out = {
        "n": len(results),
        "seconds": dt,
        "qps": len(results) / max(dt, 1e-9),
        "artifact": args.artifact,
        "meta": engine.meta,
        "results": results[:10],  # sample
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

