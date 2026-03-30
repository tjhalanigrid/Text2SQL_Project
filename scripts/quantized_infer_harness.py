# from __future__ import annotations

# import argparse
# import json
# import time
# from pathlib import Path

# from src.quantized_text2sql_engine import QuantizedText2SQLEngine


# def main() -> None:
#     p = argparse.ArgumentParser(description="Production-style inference harness for quantized artifacts.")
#     p.add_argument("--artifact", required=True, help="Quant artifact dir from scripts/quantize_export.py")
#     p.add_argument("--num_samples", type=int, default=128)
#     p.add_argument("--out", default="results/task5_quant_infer.json")
#     args = p.parse_args()

#     root = Path(".")
#     dev = json.loads((root / "data" / "dev.json").read_text())
#     dev = dev[: args.num_samples]

#     engine = QuantizedText2SQLEngine(args.artifact, device="cpu")
#     pairs = [(x["question"], x["db_id"]) for x in dev]

#     t0 = time.perf_counter()
#     results = engine.ask_batch_execute(pairs)
#     dt = time.perf_counter() - t0

#     out = {
#         "n": len(results),
#         "seconds": dt,
#         "qps": len(results) / max(dt, 1e-9),
#         "artifact": args.artifact,
#         "meta": engine.meta,
#         "results": results[:10],  # sample
#     }

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(json.dumps(out, indent=2))
#     print(json.dumps(out, indent=2))


# if __name__ == "__main__":
#     main()



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
    dev_path = root / "data" / "dev.json"
    db_root = root / "data" / "database"

    if not dev_path.exists():
        print(f"❌ ERROR: Could not find {dev_path}")
        return

    dev = json.loads(dev_path.read_text())

    # 🔥 THE FIX: Only select questions for databases you actually have downloaded
    valid_pairs = []
    for x in dev:
        db_id = x["db_id"]
        # Check if the database file exists in either standard location
        if (db_root / db_id / f"{db_id}.sqlite").exists() or (db_root / f"{db_id}.sqlite").exists():
            valid_pairs.append((x["question"], db_id))
            
        # Stop once we hit the requested number of samples
        if len(valid_pairs) >= args.num_samples:
            break

    if not valid_pairs:
        print("❌ ERROR: No valid databases found in your 'data/database' folder!")
        print("Make sure you have at least one database (like chinook_1) downloaded.")
        return

    print(f"🚀 Skipping missing databases... Found {len(valid_pairs)} valid samples to test.")

    engine = QuantizedText2SQLEngine(args.artifact, device="cpu")

    t0 = time.perf_counter()
    results = engine.ask_batch_execute(valid_pairs)
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
    print(f"✅ Benchmark complete! QPS: {out['qps']:.2f}")
    print(f"📁 Results saved to {out_path}")


if __name__ == "__main__":
    main()