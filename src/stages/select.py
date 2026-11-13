"""Stadio `select` — il modello di partenza per il profilo di deploy (NEW_MASTERPLAN 2c-2d).

    python run.py stage=select target=rpi5_640

Applica la regola di src/selection.py ai dati disponibili (latenze a 640, mAP dei training e300, pavimenti) e
scrive results/select/<target.name>.json. Non misura nulla e non allena nulla.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..jsonio import atomic_write_json
from ..selection import dati, scegli, sigla
from ..timing import now_iso

log = logging.getLogger(__name__)


def run_select(cfg) -> dict:
    t = cfg.get("target")
    if t is None or t.get("model_budget_ms") is None:
        raise ValueError("stage=select vuole un profilo con il budget del modello: target=<profilo> (target.model_budget_ms)")
    if t.start_model and not t.start_reason:
        raise ValueError("target.start_model senza target.start_reason: la scelta manuale va motivata")
    root = Path(cfg.project_root)
    lat, mAP, floor = dati(root / t.lut_dir, t.latency_csv and root / t.latency_csv)
    m5095 = {a: v[1] for a, v in mAP.items()}
    out = scegli(lat, m5095, floor, t.model_budget_ms, t.floor_ratio, t.tie, t.seed_noise, t.start_model)
    out |= {"target": t.name, "start_reason": t.start_reason, "quando": now_iso(),
            "modelli": {a: {"sigla": sigla(a), "intera_ms": lat[a], "pavimento_ms": floor.get(a, (None,))[0],
                            "pavimento_misurato": floor.get(a, (None, None))[1], "map50": mAP[a][0],
                            "map50_95": mAP[a][1]} for a in mAP}}
    dest = Path(cfg.results_dir) / "select" / f"{t.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, out)
    nome = lambda a: f"{sigla(a)} ({a})" if a else "-"  # noqa: E731
    log.info("%s, B %.1f ms: caso %s; ottimo %s, tetto %s, partenza %s%s -> %s", t.name, t.model_budget_ms, out["caso"],
             nome(out["ottimo"]), nome(out["tetto"]), nome(out["partenza"]),
             f" [avviso: {out['avviso']}]" if out["avviso"] else "", dest)
    return out
