"""yolo26p da zero su AOD4 per 1000 epoche (`train=run1k`), poi due rami dagli stessi pesi:
ramo1 = adapt su Svanstrom (conf/adapt/default.yaml, workers 6, patience 15, come c3b39818);
ramo2 = search PIT/DUCCIO (pit_duccio_pico, N=8) e finetune 500 epoche (finetune=musgd_500) su AOD4, poi adapt su
Svanstrom del potato con yolopit.PrunedTrainer (come ../../adapt/pruned_direct). Log in runs/.

Search e finetune su AOD4 passano dalla pipeline con `dataset=aod4`: lo stage search parte da un adapt, quindi prima
c'e' un adapt "identita'" su AOD4 (conf/adapt/identity.yaml, lr 0, tutto congelato) che copia i pesi senza cambiarli;
qui si controlla che siano identici a quelli del training prima di andare avanti.

    .venv/bin/python experiments/train/run1k/run.py --smoke        # 2 epoche di train, 1 per ogni fase dopo
    .venv/bin/python experiments/train/run1k/run.py                # tutto: train, ramo1, ramo2
    .venv/bin/python experiments/train/run1k/run.py train          # solo la fase 0
    .venv/bin/python experiments/train/run1k/run.py ramo1          # solo un ramo (gli stadi gia' in cache si saltano)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
COMMON = ["model=yolo26p", "train=run1k"]
# ramo 2: dataset AOD4 per identita', search e finetune. N fisso a 8 come la search di riferimento (5b5558e3): con
# pit.n auto oggi il probe int8 piu' recente del Pi da' N=4. Lo stesso elenco a tutti e tre: le chiavi si concatenano.
AOD4 = ["dataset=aod4", "adapt=identity", "strategy=pit_duccio_pico", "strategy.search.pit.n=8", "finetune=musgd_500"]
# workers e patience come il pico adattato di riferimento (c3b39818), il resto da conf/adapt/default.yaml
ADAPT = ["adapt.workers=6", "adapt.patience=15"]


def run(log: str, *args: str) -> None:
    print(f">>> {' '.join(args)}  (log: runs/{log}.log)", flush=True)
    with open(HERE / "runs" / f"{log}.log", "w") as f:
        subprocess.run([str(ROOT / ".venv/bin/python"), "run.py", *COMMON, *args], cwd=ROOT, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def cfg(*overrides: str):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(str(ROOT / "conf"), version_base=None):
        return compose("config", overrides=[*COMMON, f"project_root={ROOT}", *overrides])


def check_identity(c) -> None:
    """I pesi dell'adapt identita' devono essere quelli del training, bit per bit."""
    import torch
    from src.cache import adapt_dir, find_weights, training_key

    def sd(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return (ck.get("ema") or ck["model"]).state_dict()
    a, b = sd(find_weights(c.artifacts_dir, training_key(c)) / "best.pt"), sd(adapt_dir(c) / "best.pt")
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    if a.keys() != b.keys() or diff:
        sys.exit(f"adapt identita' NON identico al training: {len(diff)} tensori diversi, es. {diff[:3]}")
    print(f"adapt identita' == training ({len(a)} tensori)", flush=True)


def adapt_pruned(c, name: str, epochs: int | None) -> None:
    """Adapt su Svanstrom del potato finetunato, fuori dalla cache (come ../../adapt/pruned_direct)."""
    from ultralytics import YOLO
    from yolopit import PrunedTrainer
    from src.cache import final_weights
    from src.stages.adapt import adapt_args

    weights, fkey = final_weights(c)
    if weights is None:
        sys.exit(f"nessun finetune con finetune_key={fkey}")
    a = cfg("adapt.workers=6")   # conf/adapt/default.yaml, patience 10 come pruned_direct
    args = adapt_args(a) | ({"epochs": epochs} if epochs else {})
    print(f">>> adapt del potato {weights} (finetune {fkey}) -> runs/{name}/, args {args}", flush=True)
    YOLO(str(weights)).train(data=str(a.dataset.yaml), project=str(HERE / "runs"), name=name, exist_ok=True,
                             trainer=PrunedTrainer, **args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasi", nargs="*", choices=["train", "ramo1", "ramo2"], default=["train", "ramo1", "ramo2"])
    ap.add_argument("--smoke", action="store_true", help="2 epoche di train e 1 per ogni fase dopo, per vedere che parte")
    args = ap.parse_args()
    (HERE / "runs").mkdir(exist_ok=True)
    s = "smoke_" if args.smoke else ""
    if args.smoke:   # entrano nelle chiavi: stessi override a tutti gli stadi della catena
        COMMON.append("train.epochs=2")
        AOD4 += ["strategy.search.epochs=1", "finetune.epochs=1"]
    if "train" in args.fasi:
        run(f"{s}train", "stage=train")
    if "ramo1" in args.fasi:
        run(f"{s}adapt", "stage=adapt", *ADAPT, *(["adapt.epochs=1"] if args.smoke else []))
    if "ramo2" in args.fasi:
        run(f"{s}identity", "stage=adapt", *AOD4)
        c = cfg(*AOD4)
        check_identity(c)
        run(f"{s}search", "stage=search", *AOD4)
        run(f"{s}finetune", "stage=finetune", *AOD4)
        adapt_pruned(c, f"{s}adapt_potato", 1 if args.smoke else None)
