"""Stadio `train` — wrapper sottile sulle API standard di Ultralytics.

Gira una volta per modello, sulla workstation. Non conosce l'hardware di
destinazione: se lo conoscesse, il `training_key` dipenderebbe dalla board e lo
stesso modello verrebbe riallenato per ogni cella della matrice.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from ..cache import read_dataset_hash, training_key, update_index, weights_dir
from ..env import collect_versions
from ..jsonio import atomic_write_json
from ..timing import PhaseTimer

log = logging.getLogger(__name__)


def run_training(cfg) -> Path:
    tkey = training_key(cfg)
    out_dir = weights_dir(cfg)
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not cfg.force_retrain:
        log.info("training gia' in cache: %s", out_dir.name)
        return out_dir

    if cfg.dry_run:
        log.info("[dry-run] training %s -> %s", cfg.model.name, out_dir.name)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer(device=cfg.hardware.board)

    from ultralytics import YOLO

    with timer.phase("setup"):
        if cfg.model.source == "pretrained":
            model = YOLO(str(cfg.model.weights))
        else:
            model = YOLO(str(cfg.model.cfg))
        train_args = OmegaConf.to_container(cfg.train, resolve=True)

    with timer.phase("compute"):
        results = model.train(
            data=str(cfg.dataset.yaml),
            project=str(out_dir),
            name="run",
            exist_ok=True,
            **train_args,
        )

    with timer.phase("teardown"):
        best_pt = out_dir / "run" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"pesi non prodotti: {best_pt}")
        shutil.copy2(best_pt, out_dir / "best.pt")

        meta = {
            "training_key": tkey,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
            "metrics": {
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
            },
            # Con `patience` attiva Ultralytics puo' fermarsi prima: il tempo
            # per epoca e' piu' informativo del totale.
            "epochs_completed": getattr(results, "epoch", None)
            or int(cfg.train.epochs),
            "env": collect_versions(project_root=cfg.project_root),
        }
        meta["timing"] = timer.block()
        atomic_write_json(meta_path, meta)
        update_index(Path(cfg.artifacts_dir) / "weights" / "index.json", tkey, meta)

    log.info(
        "training completato: %s (mAP50 %.4f, %.1f min)",
        out_dir.name, meta["metrics"]["map50"], meta["timing"]["wall_s"] / 60,
    )
    return out_dir
