#!/usr/bin/env python3
"""Ispezione della cache di artefatti.

    python -m tools.artifacts ls [--kind weights|exports] [--json]
    python -m tools.artifacts show <slug>
    python -m tools.artifacts prune [--apply]

`prune` rimuove gli artefatti non referenziati da alcun `results.json` ed e'
in dry-run per default: cancellare un export significa ricompilarlo sulla
board, che costa molto piu' del disco che libera.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dirs(artifacts: Path, kind: str) -> list[Path]:
    root = artifacts / kind
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def cmd_ls(args) -> int:
    kinds = [args.kind] if args.kind else ["weights", "exports"]
    payload = []
    for kind in kinds:
        for d in _dirs(args.artifacts, kind):
            meta = _read(d / "meta.json") or {}
            cfg = meta.get("config") or {}
            payload.append({
                "kind": kind,
                "slug": d.name,
                "key": meta.get("training_key") or meta.get("export_key"),
                "model": (cfg.get("model") or {}).get("name"),
                "quantization": (cfg.get("quantization") or {}).get("name"),
                "backend": (cfg.get("backend") or {}).get("name"),
                "epochs": (cfg.get("train") or {}).get("epochs"),
                "seed": (cfg.get("train") or {}).get("seed"),
                "map50": (meta.get("metrics") or {}).get("map50"),
                "validation": (meta.get("validation") or {}).get("status"),
                "e2e": (meta.get("inspection") or {}).get("actual_e2e"),
                "wall_s": (meta.get("timing") or {}).get("wall_s"),
                "created": (meta.get("timing") or {}).get("started_at"),
            })
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not payload:
        print("nessun artefatto in cache")
        return 0
    width = max(len(p["slug"]) for p in payload)
    for p in payload:
        extra = (
            f"mAP50 {p['map50']:.4f}" if p.get("map50") is not None
            else f"val {p['validation']} e2e {p['e2e']}"
        )
        print(f"{p['kind']:<8} {p['slug']:<{width}}  {extra}  {p['created'] or ''}")
    return 0


def cmd_show(args) -> int:
    for kind in ("weights", "exports"):
        d = args.artifacts / kind / args.slug
        if d.is_dir():
            print(json.dumps(_read(d / "meta.json"), indent=2, ensure_ascii=False))
            return 0
    print(f"artefatto {args.slug} non trovato", file=sys.stderr)
    return 1


def referenced_keys(results_dir: Path) -> set[str]:
    """Chiavi citate dai risultati: training_key e export_key.

    L'export_key viene ricostruita dagli `axes` del risultato invece che
    ricalcolata da `cache.export_key`: quella riparte dall'hash del dataset,
    che su una macchina diversa da quella di training non e' disponibile.
    Il formato e' lo stesso e sta scritto in `src/cache.py`.
    """
    keys: set[str] = set()
    for path in results_dir.glob("*.json"):
        rec = _read(path) or {}
        axes = rec.get("axes") or {}
        tkey = axes.get("training_key")
        if not tkey:
            continue
        keys.add(tkey)
        if axes.get("backend"):
            ekey = (
                f"{axes.get('model')}_{tkey}_{axes.get('quantization')}"
                f"_{axes.get('backend')}_{axes.get('arch')}"
            )
            if axes.get("backend") == "axelera":
                sdk = (
                    ((rec.get("config") or {}).get("backend") or {})
                    .get("build", {})
                    .get("sdk_version")
                )
                ekey += f"_sdk{sdk}"
            keys.add(ekey)
    return keys


def cmd_prune(args) -> int:
    keys = referenced_keys(args.results)
    if not keys:
        print(
            "nessun risultato in " + str(args.results) + ": mi fermo invece di "
            "considerare tutto non referenziato", file=sys.stderr,
        )
        return 1

    victims = []
    for kind in ("weights", "exports"):
        for d in _dirs(args.artifacts, kind):
            meta = _read(d / "meta.json") or {}
            key = meta.get("training_key") if kind == "weights" \
                else meta.get("export_key")
            if key is None:
                print(f"? {d} senza meta.json: lasciato dov'e'")
                continue
            if key not in keys:
                victims.append((d, key))

    if not victims:
        print("niente da rimuovere")
        return 0
    for d, key in victims:
        print(("RIMUOVO   " if args.apply else "[dry-run] ") + f"{d}  ({key})")
        if args.apply:
            shutil.rmtree(d)
    if not args.apply:
        print("\nnessuna modifica: rilancia con --apply per rimuovere davvero")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=Path("artifacts"), type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls")
    p_ls.add_argument("--kind", choices=["weights", "exports"])
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=cmd_ls)

    p_show = sub.add_parser("show")
    p_show.add_argument("slug")
    p_show.set_defaults(func=cmd_show)

    p_prune = sub.add_parser("prune")
    p_prune.add_argument("--results", default=Path("results"), type=Path)
    p_prune.add_argument("--apply", action="store_true",
                         help="senza questo flag non cancella nulla")
    p_prune.set_defaults(func=cmd_prune)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
