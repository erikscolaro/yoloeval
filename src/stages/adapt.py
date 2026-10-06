"""Stadio `adapt` — fine-tuning sul dataset di destinazione (Svanstrom), per ogni modello.

Parte dai pesi di `stage=train` (allenati su `pretrain_dataset`, AOD4) e mette `best.pt` in
`artifacts/adapt/`: sono i pesi della baseline per l'export e il punto di partenza della
ricerca PIT. Obbligatorio anche per i modelli pretrained: da qui in avanti tutta la pipeline
(search, finetune, calibrazione, mAP) usa solo `dataset`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from ..cache import (adapt_dir, adapt_key, find_weights, read_dataset_hash, training_key,
                     update_index)
from ..env import collect_versions
from ..errors import MissingWeights
from ..jsonio import atomic_write_json
from ..timing import PhaseTimer

log = logging.getLogger(__name__)


def adapt_args(cfg) -> dict:
    conf = OmegaConf.to_container(cfg.adapt, resolve=True) or {}
    for vietata in ("data", "project", "name"):
        if vietata in conf:
            raise ValueError(f"adapt.{vietata} non va indicato: lo decide yoloeval")
    conf.setdefault("imgsz", int(cfg.train.imgsz))
    return conf


def run_adapt(cfg) -> Path:
    tkey = training_key(cfg)
    weights = find_weights(cfg.artifacts_dir, tkey)
    if weights is None:
        raise MissingWeights(
            f"nessun artefatto con training_key={tkey}: eseguire prima `stage=train` per "
            f"{cfg.model.name}")
    akey = adapt_key(cfg)
    out_dir = adapt_dir(cfg)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not cfg.force_adapt:
        log.info("adattamento gia' in cache: %s", out_dir.name)
        return out_dir
    if cfg.dry_run:
        log.info("[dry-run] adattamento %s su %s -> %s", cfg.model.name, cfg.dataset.name,
                 out_dir.name)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer(device=cfg.hardware.board)
    from ultralytics import YOLO

    with timer.phase("setup"):
        args = adapt_args(cfg)
        model = YOLO(str(weights / "best.pt"))

    with timer.phase("compute"):
        results = model.train(data=str(cfg.dataset.yaml), project=str(out_dir), name="run",
                              exist_ok=True, **args)

    with timer.phase("teardown"):
        best_pt = out_dir / "run" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"pesi non prodotti: {best_pt}")
        shutil.copy2(best_pt, out_dir / "best.pt")
        meta = {
            "adapt_key": akey,
            "training_key": tkey,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
            "metrics": {
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
            },
            "epochs_completed": getattr(results, "epoch", None) or int(args.get("epochs", 0)),
            "env": collect_versions(project_root=cfg.project_root),
        }
        meta["timing"] = timer.block()
        atomic_write_json(meta_path, meta)
        update_index(Path(cfg.artifacts_dir) / "adapt" / "index.json", akey, meta)

    log.info("adattamento completato: %s (mAP50 %.4f, %.1f min)", out_dir.name,
             meta["metrics"]["map50"], meta["timing"]["wall_s"] / 60)
    return out_dir
