"""Test una tantum: finetune con finetune=<conf> partendo dall'ultima epoca salvata della search
pit_duccio_pico interrotta (e95c6991, epoca 58). Fuori dalla cache di yoloeval: tutto in
outputs/sweep_yolo26p/<conf>_test/.
    .venv/bin/python outputs/sweep_yolo26p/finetune_test.py adam|musgd|sgd
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402

from src.cache import find_weights, training_key  # noqa: E402
from src.stages.finetune import finetune_args  # noqa: E402
from src.stages.search import search_args  # noqa: E402

SEARCH = ROOT / "artifacts/search/yolo26p_aod4_pit_duccio_pico_e95c6991/run/weights/last.pt"
FT = sys.argv[1] if len(sys.argv) > 1 else "adam"
OUT = ROOT / f"outputs/sweep_yolo26p/{FT}_test"

with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
    cfg = compose("config", overrides=[
        f"project_root={ROOT}", "model=yolo26p", "train=scratch", f"finetune={FT}",
        "strategy=pit_duccio_pico", "hardware=rpi5", "quantization=int8"])

from ultralytics import YOLO  # noqa: E402
from yolopit import PITYOLO, PrunedTrainer  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
weights = find_weights(cfg.artifacts_dir, training_key(cfg)) / "best.pt"
search = PITYOLO(str(weights), cfg=search_args(cfg))
ckpt = search.load_search_checkpoint(SEARCH)
print(f"checkpoint della search: epoca {ckpt['epoch'] + 1}, costi {ckpt['costs']}")
search.export_pruned(path=OUT / "pruned.pt")

model = YOLO(str(OUT / "pruned.pt"))
results = model.train(data=str(cfg.dataset.yaml), project=str(OUT), name="run", exist_ok=True,
                      trainer=PrunedTrainer, **finetune_args(cfg))
print(f"finetune {FT}: mAP50 {results.box.map50:.4f}, mAP50-95 {results.box.map:.4f}")
