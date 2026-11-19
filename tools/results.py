#!/usr/bin/env python3
"""Stato delle misure in results/: quante per esito, quali sono fallite o saltate e perche', il dettaglio di una.

    python -m tools.results riepilogo [--results-dir results]
    python -m tools.results falliti
    python -m tools.results saltati
    python -m tools.results cella <cell_id>     # il json senza la config, poi la coda del .raw.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tools.aggregate import load

ASSI = ("model", "quantization", "backend", "board")


def riepilogo(rows: list[dict]) -> str:
    righe = [f"{len(rows)} misure: " + ", ".join(f"{k} {v}" for k, v in Counter(r["status"] for r in rows).most_common())]
    per = Counter((r.get("board"), r.get("backend"), r.get("quantization"), r["status"]) for r in rows)
    for (b, be, q, s), n in sorted(per.items(), key=lambda x: tuple(str(v) for v in x[0])):
        righe.append(f"  {b or '-':8s} {be or '-':12s} {q or '-':5s} {s:8s} {n}")
    return "\n".join(righe)


def falliti(rows: list[dict]) -> str:
    out = [f"{r['cell_id']}  " + " ".join(str(r.get(a) or "-") for a in ASSI) + "\n    "
           + str(r.get("error") or "").split("\n")[0][:150]
           for r in sorted(rows, key=lambda r: str(r.get("timestamp"))) if r["status"] == "failed"]
    return "\n".join(out) or "nessuna misura fallita"


def saltati(rows: list[dict]) -> str:
    c = Counter(str(r.get("reason")) for r in rows if r["status"] == "skipped")
    return "\n".join(f"{n:4d}  {m}" for m, n in c.most_common()) or "nessuna misura saltata"


def cella(results_dir: Path, cell_id: str) -> str:
    p = results_dir / f"{cell_id}.json"
    if not p.exists():
        return f"{p} non esiste"
    rec = json.loads(p.read_text())
    rec.pop("config", None)          # lunghissima: il resto si legge meglio senza
    raw = p.with_suffix(".raw.txt")
    coda = "\n--- raw ---\n" + raw.read_text(errors="replace")[-3000:] if raw.exists() else ""
    return json.dumps(rec, indent=2, ensure_ascii=False) + coda


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cosa", choices=("riepilogo", "falliti", "saltati", "cella"))
    ap.add_argument("cell_id", nargs="?")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    if a.cosa == "cella":
        if not a.cell_id:
            ap.error("cella vuole il cell_id")
        print(cella(a.results_dir, a.cell_id))
        return 0
    rows = load(a.results_dir)
    if not rows:
        print(f"nessun risultato in {a.results_dir}", file=sys.stderr)
        return 1
    print({"riepilogo": riepilogo, "falliti": falliti, "saltati": saltati}[a.cosa](rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
