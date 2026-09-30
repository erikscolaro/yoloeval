"""Stadio `finetune` — fine-tuning del modello potato, solo per le strategie `method: pit`.

Parte da `artifacts/search/.../pruned.pt` (stesso search_key) e mette `best.pt` in
`artifacts/finetune/`: sono i pesi che la strategia manda all'export.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from ..cache import (find_by_key, finetune_dir, finetune_key, is_baseline, read_dataset_hash,
                     search_key, strategy_name, training_key, update_index, yolopit_version)
from ..env import collect_versions
from ..errors import MissingWeights
from ..jsonio import atomic_write_json
from ..timing import PhaseTimer

log = logging.getLogger(__name__)


def finetune_args(cfg) -> dict:
    conf = OmegaConf.to_container(cfg.finetune, resolve=True) or {}
    for vietata in ("data", "project", "name", "trainer"):
        if vietata in conf:
            raise ValueError(f"finetune.{vietata} non va indicato: lo decide yoloeval")
    conf.setdefault("imgsz", int(cfg.train.imgsz))
    return conf


def run_finetune(cfg) -> Path | None:
    if is_baseline(cfg):
        log.info("strategy=%s: nessun fine-tuning da fare", strategy_name(cfg))
        return None
    skey = search_key(cfg)
    search = find_by_key(cfg.artifacts_dir, "search", "search_key", skey)
    if search is None or not (search / "pruned.pt").exists():
        raise MissingWeights(
            f"nessuna ricerca con search_key={skey}: eseguire prima `stage=search` con "
            f"strategy={strategy_name(cfg)}")
    fkey = finetune_key(cfg)
    out_dir = finetune_dir(cfg)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not cfg.force_finetune:
        log.info("fine-tuning gia' in cache: %s", out_dir.name)
        return out_dir
    if cfg.dry_run:
        log.info("[dry-run] fine-tuning %s -> %s", strategy_name(cfg), out_dir.name)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer(device=cfg.hardware.board)
    from ultralytics import YOLO
    from yolopit import PrunedTrainer

    with timer.phase("setup"):
        args = finetune_args(cfg)
        model = YOLO(str(search / "pruned.pt"))

    with timer.phase("compute"):
        results = model.train(data=str(cfg.dataset.yaml), project=str(out_dir), name="run",
                              exist_ok=True, trainer=PrunedTrainer, **args)

    with timer.phase("teardown"):
        best_pt = out_dir / "run" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"pesi non prodotti: {best_pt}")
        shutil.copy2(best_pt, out_dir / "best.pt")
        meta = {
            "finetune_key": fkey,
            "search_key": skey,
            "training_key": training_key(cfg),
            "strategy": strategy_name(cfg),
            "yolopit": yolopit_version(),
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
        update_index(Path(cfg.artifacts_dir) / "finetune" / "index.json", fkey, meta)

    log.info("fine-tuning completato: %s (mAP50 %.4f, %.1f min)", out_dir.name,
             meta["metrics"]["map50"], meta["timing"]["wall_s"] / 60)
    return out_dir
