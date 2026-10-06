"""Ablation sull'optimizer del finetune del modello potato: finetune=<conf> (adam, musgd, sgd)
partendo da `pruned.pt` della search `pit_duccio_pico` di yolo26p, che a sua volta parte
dall'adattamento a Svanstrom. Le tre conf sono confrontate sullo stesso `cfg.dataset`.
Fuori dalla cache di yoloeval: tutto in runs/<conf>_test/. Grafici con plot_finetune.py.

    .venv/bin/python experiments/finetune/lr_scheduler/run.py adam|musgd|sgd

Prima: `run.py stage=train`, `stage=adapt` e `stage=search` con gli override di COMMON (se la
search manca lo script lo dice ed esce).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import chain_hint, make_cfg  # noqa: E402

HERE = Path(__file__).resolve().parent

# la search e' condivisa con gli altri esperimenti sul potato (N dal probe rpi5 int8 a 4 core)
COMMON = ["model=yolo26p", "train=scratch_musgd", "strategy=pit_duccio_pico",
          "strategy.n_source=cpu_4", "strategy.n_quantization=int8", "hardware=rpi5", "quantization=int8"]


def main() -> None:
    ft = sys.argv[1] if len(sys.argv) > 1 else "adam"
    out = HERE / "runs" / f"{ft}_test"        # ignorata da git

    from src.cache import find_by_key, search_key
    from src.stages.finetune import finetune_args

    cfg = make_cfg(*COMMON, f"finetune={ft}")
    skey = search_key(cfg)
    search = find_by_key(cfg.artifacts_dir, "search", "search_key", skey)
    if search is None or not (search / "pruned.pt").exists():
        sys.exit(f"search {skey} non in cache; lancia prima, dalla root del repo:\n"
                 + chain_hint(cfg, COMMON, upto="search"))

    from ultralytics import YOLO
    from yolopit import PrunedTrainer

    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(search / "pruned.pt"))
    results = model.train(data=str(cfg.dataset.yaml), project=str(out), name="run", exist_ok=True,
                          trainer=PrunedTrainer, **finetune_args(cfg))
    print(f"finetune {ft}: mAP50 {results.box.map50:.4f}, mAP50-95 {results.box.map:.4f}")


if __name__ == "__main__":
    main()
