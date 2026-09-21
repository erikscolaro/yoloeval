#!/usr/bin/env python3
"""results/*.json -> un DataFrame unico, salvato in Parquet.

Include le celle `ok`, `skipped` e `failed`, con lo `status` come colonna:
servono a distinguere una combinazione non supportata da una che e' crashata.
Aggregare solo i successi significa non poter dire, in tesi, quante celle della
matrice sono state davvero misurate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: blocchi appiattiti in colonne, con il loro prefisso
FLATTEN = {
    "axes": "",
    "latency": "lat_",
    "accuracy": "acc_",
    "energy": "energy_",
    "runtime_state": "rt_",
    "validation": "val_",
    "timing": "time_",
    "env": "env_",
}


def flatten_record(rec: dict) -> dict:
    row = {
        "cell_id": rec.get("cell_id"),
        "status": rec.get("status"),
        "schema_version": rec.get("schema_version"),
        "timestamp": rec.get("timestamp"),
        "order_index": rec.get("order_index"),
        "reason": rec.get("reason"),
        "error": rec.get("error"),
    }
    for block, prefix in FLATTEN.items():
        payload = rec.get(block) or {}
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row[f"{prefix}{key}"] = value
    cfg = rec.get("config") or {}
    row["cfg_stage"] = (cfg.get("stage") or {}).get("name")
    row["cfg_imgsz"] = (cfg.get("model") or {}).get("imgsz")
    row["cfg_iters"] = ((cfg.get("backend") or {}).get("benchmark") or {}).get("iters")
    row["cfg_bench_conf"] = (cfg.get("eval") or {}).get("bench_conf")
    return row


def load(results_dir: Path) -> list[dict]:
    rows, versions = [], Counter()
    for path in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"! {path.name} non e' JSON valido, saltato", file=sys.stderr)
            continue
        versions[rec.get("schema_version")] += 1
        rows.append(flatten_record(rec))
    if len(versions) > 1:
        print(
            f"! schema_version diversi nello stesso set: {dict(versions)} — "
            f"alcune colonne potrebbero mancare nelle run piu' vecchie",
            file=sys.stderr,
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="default: <results-dir>/../data.parquet")
    ap.add_argument("--csv", action="store_true", help="scrive anche un CSV")
    args = ap.parse_args(argv)

    rows = load(args.results_dir)
    if not rows:
        print(f"nessun risultato in {args.results_dir}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("pandas non installato: pip install pandas pyarrow", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows).sort_values(
        ["board", "backend", "quantization", "model"], na_position="last"
    )
    out = args.out or Path("data.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if args.csv:
        df.to_csv(out.with_suffix(".csv"), index=False)

    counts = df["status"].value_counts().to_dict()
    print(f"{len(df)} celle -> {out}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
