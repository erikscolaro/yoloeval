"""Porta in artifacts/weights le baseline AOD4 della tesi, come se fossero uscite da `stage=train`.

    python scripts/import_thesis_baselines.py [n s m]

Per ogni modello calcola training_key e cartella con la stessa logica di `src.cache` (train=thesis_baseline,
pretrain_dataset corrente), copia best.pt e i file del run e scrive meta.json + index.json. Il meta porta in piu'
`imported_from` (path e data): i pesi non sono stati allenati da yoloeval.
"""
import csv
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cache import read_dataset_hash, training_key, update_index, weights_dir  # noqa: E402
from src.jsonio import atomic_write_json  # noqa: E402

THESIS = Path.home() / "Desktop/thesis/runs"

for size in sys.argv[1:] or ["n", "s", "m"]:
    src = THESIS / f"baseline-yolo26{size}/train"
    with initialize_config_dir(str(ROOT / "conf"), version_base=None):
        cfg = compose("config", overrides=[f"model=yolo26{size}", "train=thesis_baseline", "stage=train",
                                           f"project_root={ROOT}"])
    out = weights_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "weights/best.pt", out / "best.pt")
    run = out / "run"
    run.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, run / f.name)
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(src / "results.csv"))]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    wall = float(rows[-1]["time"])
    ended = datetime.fromtimestamp((src / "results.csv").stat().st_mtime).astimezone()
    meta = {
        "training_key": training_key(cfg),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_hash": read_dataset_hash(cfg.pretrain_dataset.local_path),
        "metrics": {"map50": float(best["metrics/mAP50(B)"]), "map50_95": float(best["metrics/mAP50-95(B)"])},
        "epochs_completed": len(rows),
        "env": {},
        "timing": {"wall_s": wall, "compute_s": wall, "device": "thesis",
                   "started_at": (ended - timedelta(seconds=wall)).isoformat(timespec="seconds")},
        "imported_from": {"path": str(src), "imported_at": datetime.now().astimezone().isoformat(timespec="seconds")},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/weights/index.json", meta["training_key"], meta)
    print(out.name, meta["metrics"], len(rows), "epoche")
