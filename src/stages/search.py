"""Stadio `search` — ricerca PIT con yolopit, solo per le strategie `method: pit`.

Parte dai pesi di `stage=train` (stesso training_key) e produce il modello potato in
`artifacts/search/`. Come il training, gira sulla workstation e non conosce l'hardware di
destinazione: la stessa ricerca vale per ogni board.

La logica della ricerca sta tutta in yolopit; qui ci sono solo cache, argomenti e meta.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from ..cache import (find_weights, is_baseline, read_dataset_hash, search_dir, search_key,
                     strategy_name, training_key, update_index, yolopit_version)
from ..env import collect_versions
from ..errors import MissingWeights
from ..jsonio import atomic_write_json
from ..timing import PhaseTimer

log = logging.getLogger(__name__)


def search_args(cfg) -> dict:
    """Config di PITYOLO dalla strategia: dataset e risoluzioni dal resto della config.

    - `data` arriva sempre da dataset.yaml (se la strategia lo mette, errore);
    - `imgsz` del training della ricerca: quello della strategia, altrimenti train.imgsz;
    - `pit.trace_imgsz` (risoluzione dei costi e dei target): model.imgsz, la risoluzione di
      deploy, se la strategia non dice altro.
    """
    conf = OmegaConf.to_container(cfg.strategy.search, resolve=True) or {}
    for vietata in ("data", "project", "name"):
        if vietata in conf:
            raise ValueError(
                f"strategy.search.{vietata} non va indicato: lo decide yoloeval "
                f"(data da dataset.yaml, project/name dalla cache)")
    conf["data"] = str(cfg.dataset.yaml)
    conf.setdefault("imgsz", int(cfg.train.imgsz))
    pit = dict(conf.get("pit") or {})
    pit.setdefault("trace_imgsz", int(cfg.model.imgsz))
    conf["pit"] = pit
    return conf


def run_search(cfg) -> Path | None:
    if is_baseline(cfg):
        log.info("strategy=%s: nessuna ricerca da fare", strategy_name(cfg))
        return None
    tkey = training_key(cfg)
    weights = find_weights(cfg.artifacts_dir, tkey)
    if weights is None:
        raise MissingWeights(
            f"nessun artefatto con training_key={tkey}: eseguire prima `stage=train` per "
            f"{cfg.model.name}")
    skey = search_key(cfg)
    out_dir = search_dir(cfg)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not cfg.force_search:
        log.info("ricerca gia' in cache: %s", out_dir.name)
        return out_dir
    if cfg.dry_run:
        log.info("[dry-run] ricerca %s -> %s", strategy_name(cfg), out_dir.name)
        return out_dir
    if yolopit_version() is None:
        raise ImportError("yolopit non installato: vedi scripts/requirements/x86_64.txt")

    out_dir.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer(device=cfg.hardware.board)
    from yolopit import PITYOLO

    with timer.phase("setup"):
        conf = search_args(cfg)
        search = PITYOLO(str(weights / "best.pt"), cfg=conf)

    with timer.phase("compute"):
        metrics = search.train(project=str(out_dir), name="run", exist_ok=True)
        search.export_pruned(path=out_dir / "pruned.pt")

    with timer.phase("teardown"):
        run_dir = out_dir / "run"
        summary = json.loads((run_dir / "pit_summary.json").read_text(encoding="utf-8"))
        meta = {
            "search_key": skey,
            "training_key": tkey,
            "strategy": strategy_name(cfg),
            "yolopit": yolopit_version(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "search_args": conf,
            "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
            # mAP dell'architettura finale, sulla validazione di fine ricerca: e' anche la mAP
            # del modello potato appena esportato, prima del fine-tuning
            "metrics": {
                "map50": float(metrics.get("metrics/mAP50(B)", float("nan"))),
                "map50_95": float(metrics.get("metrics/mAP50-95(B)", float("nan"))),
            },
            "pit_summary": summary,
            "env": collect_versions(project_root=cfg.project_root),
        }
        meta["timing"] = timer.block()
        atomic_write_json(meta_path, meta)
        update_index(Path(cfg.artifacts_dir) / "search" / "index.json", skey, meta)

    costs = summary.get("costs", {})
    log.info("ricerca completata: %s (mAP50 %.4f, ops %.1f%%, params %.1f%%, %.1f min)",
             out_dir.name, meta["metrics"]["map50"],
             100 * costs.get("ops", {}).get("final_fraction", float("nan")),
             100 * costs.get("params", {}).get("final_fraction", float("nan")),
             meta["timing"]["wall_s"] / 60)
    return out_dir
