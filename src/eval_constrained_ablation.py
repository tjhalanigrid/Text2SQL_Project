import argparse
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from src.schema_encoder import SchemaEncoder
from src.sql_validator import validate_sql_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"


def _normalize_sql(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    s = sql.replace('"', "'")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.rstrip(";")


def _clean_sql(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    m = re.search(r"(select|with)[\\s\\S]*", t, flags=re.IGNORECASE)
    return (m.group(0).strip() if m else t).strip()


def _connect_timeout(db_path: str, timeout_s: float = 2.0) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    start_t = time.monotonic()

    def handler():
        return 1 if (time.monotonic() - start_t) > float(timeout_s) else 0

    conn.set_progress_handler(handler, 10_000)
    return conn


def _execution_match(pred_sql: str, gold_sql: str, db_path: str, timeout_s: float = 2.0) -> bool:
    try:
        conn = _connect_timeout(db_path, timeout_s=timeout_s)
        cur = conn.cursor()
        cur.execute(pred_sql)
        pred_rows = cur.fetchall()
        cur.execute(gold_sql)
        gold_rows = cur.fetchall()
        conn.close()
        return sorted(pred_rows) == sorted(gold_rows)
    except Exception:
        return False


def _load_dev_examples(dev_json: Path) -> list[dict]:
    with dev_json.open() as f:
        return json.load(f)


def _filter_examples(examples: list[dict], db_allow: set[str] | None) -> list[dict]:
    if not db_allow:
        return examples
    return [x for x in examples if str(x.get("db_id", "")) in db_allow]


def _sample_examples(examples: list[dict], n: int, seed: int, with_replacement: bool) -> list[dict]:
    rng = random.Random(int(seed))
    if n <= 0:
        return []
    if with_replacement:
        return [examples[rng.randrange(len(examples))] for _ in range(n)]
    rng.shuffle(examples)
    if len(examples) < n:
        raise RuntimeError(f"Not enough examples: need {n}, have {len(examples)}. Use --sample_with_replacement.")
    return examples[:n]


def _prompt(schema_text: str, db_id: str, question: str) -> str:
    return (
        "You are a SQLite expert.\n\n"
        f"Database: {db_id}\n\n"
        "Schema:\n"
        f"{schema_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "SQL:"
    )


def _generate_one(
    tokenizer,
    model,
    *,
    prompt_text: str,
    db_path: str,
    constrained: bool,
    num_beams: int,
    max_new_tokens: int,
    repetition_penalty: float,
):
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    logits_processor = None
    if constrained:
        from src.constrained_decoding import SchemaConstrainedLogitsProcessor

        logits_processor = LogitsProcessorList([SchemaConstrainedLogitsProcessor(tokenizer, db_path)])
    out = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        do_sample=False,
        logits_processor=logits_processor,
    )
    return _clean_sql(tokenizer.decode(out[0], skip_special_tokens=True))


def evaluate(
    examples: list[dict],
    *,
    tokenizer,
    model,
    schema_encoder: SchemaEncoder,
    constrained: bool,
    num_beams: int,
    max_new_tokens: int,
    repetition_penalty: float,
    timeout_s: float,
):
    em = 0
    ex = 0
    constraint_ok = 0
    total = len(examples)
    t0 = time.perf_counter()

    for idx, item in enumerate(examples, 1):
        db_id = str(item["db_id"])
        question = str(item["question"])
        gold_sql = str(item["query"])
        db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")

        schema_text = schema_encoder.structured_schema(db_id)
        pred_sql = _generate_one(
            tokenizer,
            model,
            prompt_text=_prompt(schema_text, db_id, question),
            db_path=db_path,
            constrained=bool(constrained),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        try:
            ok, _ = validate_sql_schema(pred_sql, db_path)
        except Exception:
            ok = False
        if ok:
            constraint_ok += 1

        if _normalize_sql(pred_sql) == _normalize_sql(gold_sql):
            em += 1

        if ok and _execution_match(pred_sql, gold_sql, db_path, timeout_s=timeout_s):
            ex += 1

        if idx % 20 == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            per = elapsed / max(idx, 1)
            eta = per * (total - idx)
            print(
                f"  [{idx}/{total}] em={em/idx:.3f} ex={ex/idx:.3f} constraint={constraint_ok/idx:.3f} "
                f"({per:.2f}s/eg, eta {eta/60:.1f}m)",
                flush=True,
            )

    return {
        "n": total,
        "em": em / max(total, 1),
        "ex": ex / max(total, 1),
        "constraint_rate": constraint_ok / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Adapter checkpoint dir (LoRA).")
    ap.add_argument("--base_model", default="Salesforce/codet5-base")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--sample_with_replacement", action="store_true")
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--timeout_s", type=float, default=2.0)
    ap.add_argument("--mode", choices=["both", "constrained", "unconstrained"], default="both")
    ap.add_argument("--dbs", default="", help="Comma-separated db_id allowlist.")
    ap.add_argument("--out", default="results/task3_constrained_ablation.json")
    ap.add_argument("--local_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

    dev_json = PROJECT_ROOT / "data" / "dev.json"
    examples = _load_dev_examples(dev_json)

    db_allow = None
    if str(args.dbs).strip():
        db_allow = {x.strip() for x in str(args.dbs).split(",") if x.strip()}
    examples = _filter_examples(examples, db_allow)

    picked = _sample_examples(
        list(examples),
        n=int(args.num_samples),
        seed=int(args.sample_seed),
        with_replacement=bool(args.sample_with_replacement),
    )

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=bool(args.local_only))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, local_files_only=bool(args.local_only)).to(device)
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=bool(args.local_only)).to(device)
    model = model.merge_and_unload()
    model.eval()

    schema_encoder = SchemaEncoder(DB_ROOT)

    report = {
        "adapter": str(args.adapter),
        "base_model": str(args.base_model),
        "n": int(args.num_samples),
        "seed": int(args.sample_seed),
        "with_replacement": bool(args.sample_with_replacement),
        "db_allow": sorted(db_allow) if db_allow else None,
        "gen": {
            "num_beams": int(args.num_beams),
            "max_new_tokens": int(args.max_new_tokens),
            "repetition_penalty": float(args.repetition_penalty),
        },
        "timeout_s": float(args.timeout_s),
        "results": {},
    }

    if args.mode in {"both", "unconstrained"}:
        print("Evaluating: unconstrained")
        report["results"]["unconstrained"] = evaluate(
            picked,
            tokenizer=tokenizer,
            model=model,
            schema_encoder=schema_encoder,
            constrained=False,
            num_beams=int(args.num_beams),
            max_new_tokens=int(args.max_new_tokens),
            repetition_penalty=float(args.repetition_penalty),
            timeout_s=float(args.timeout_s),
        )

    if args.mode in {"both", "constrained"}:
        print("Evaluating: constrained")
        report["results"]["constrained"] = evaluate(
            picked,
            tokenizer=tokenizer,
            model=model,
            schema_encoder=schema_encoder,
            constrained=True,
            num_beams=int(args.num_beams),
            max_new_tokens=int(args.max_new_tokens),
            repetition_penalty=float(args.repetition_penalty),
            timeout_s=float(args.timeout_s),
        )

    out_path = PROJECT_ROOT / str(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
