"""yolo26p da zero su AOD4 per 1000 epoche (`train=run1k`), poi due rami dagli stessi pesi:
ramo1 = adapt su Svanstrom (conf/adapt/default.yaml, workers 6, patience 15, come c3b39818);
ramo2 = search PIT/DUCCIO (pit_duccio_pico, N=8) e finetune 500 epoche (finetune=musgd_500) su AOD4, poi adapt su
Svanstrom del potato (conf/adapt/default.yaml, workers 6, patience 10). Tutto via pipeline; log in runs/.

    .venv/bin/python experiments/train/run1k/run.py --smoke        # 2 epoche di train, 1 per ogni fase dopo
    .venv/bin/python experiments/train/run1k/run.py                # tutto: train, ramo1, ramo2
    .venv/bin/python experiments/train/run1k/run.py train          # solo la fase 0
    .venv/bin/python experiments/train/run1k/run.py ramo1          # solo un ramo (gli stadi gia' in cache si saltano)
"""
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = ["model=yolo26p", "train=run1k"]
# workers e patience come il pico adattato di riferimento (c3b39818), il resto da conf/adapt/default.yaml
RAMO1 = ["adapt.workers=6", "adapt.patience=15"]
# N fisso a 8 come la search di riferimento (5b5558e3): con pit.n auto oggi il probe int8 piu' recente del Pi da' N=4.
# Lo stesso elenco a search, finetune e adapt: le chiavi si concatenano (adapt.* non entra in search e finetune).
RAMO2 = ["strategy=pit_duccio_pico", "strategy.search.pit.n=8", "finetune=musgd_500", "adapt.workers=6"]


def run(log: str, *args: str) -> None:
    print(f">>> {' '.join(args)}  (log: runs/{log}.log)", flush=True)
    with open(HERE / "runs" / f"{log}.log", "w") as f:
        subprocess.run([str(ROOT / ".venv/bin/python"), "run.py", *COMMON, *args], cwd=ROOT, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasi", nargs="*", choices=["train", "ramo1", "ramo2"], help="default: tutte")
    ap.add_argument("--smoke", action="store_true", help="2 epoche di train e 1 per ogni fase dopo, per vedere che parte")
    args = ap.parse_args()
    args.fasi = args.fasi or ["train", "ramo1", "ramo2"]
    (HERE / "runs").mkdir(exist_ok=True)
    s = "smoke_" if args.smoke else ""
    if args.smoke:   # entrano nelle chiavi: stessi override a tutti gli stadi della catena
        COMMON.append("train.epochs=2")
        RAMO1.append("adapt.epochs=1")
        RAMO2 += ["strategy.search.epochs=1", "finetune.epochs=1", "adapt.epochs=1"]
    if "train" in args.fasi:
        run(f"{s}train", "stage=train")
    if "ramo1" in args.fasi:
        run(f"{s}adapt", "stage=adapt", *RAMO1)
    if "ramo2" in args.fasi:
        for stage in ("search", "finetune", "adapt"):
            run(f"{s}{stage}_potato" if stage == "adapt" else f"{s}{stage}", f"stage={stage}", *RAMO2)
