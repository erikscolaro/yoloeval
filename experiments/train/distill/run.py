"""Distillazione di yolo26p (da zero, `train=scratch_musgd`) dal nano della tesi, poi adapt su Svanstrom con la ricetta
di `conf/adapt/default.yaml`; l'unica differenza dal pico di riferimento e' `+train.distill_model`. Log in runs/.

    .venv/bin/python experiments/train/distill/run.py --smoke   # 2 epoche di train + 1 di adapt, controllo
    .venv/bin/python experiments/train/distill/run.py           # 250 epoche di train, poi adapt
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# Hydra fa chdir nella cartella del run: al train il teacher va dato con path assoluto (entra nel training_key)
TEACHER = "artifacts/weights/yolo26n_aod4_e250_00a6209e/best.pt"
COMMON = ["model=yolo26p", "train=scratch_musgd", f"+train.distill_model={ROOT / TEACHER}"]
# workers e patience come il pico adattato di riferimento (c3b39818), il resto da conf/adapt/default.yaml
ADAPT = ["stage=adapt", "adapt.workers=6", "adapt.patience=15"]


def run(log: str, *args: str) -> None:
    print(f">>> {' '.join(args)}  (log: runs/{log}.log)", flush=True)
    with open(HERE / "runs" / f"{log}.log", "w") as f:
        subprocess.run([str(ROOT / ".venv/bin/python"), "run.py", *COMMON, *args], cwd=ROOT, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="2 epoche di train e 1 di adapt, per vedere che parte")
    args = ap.parse_args()
    if not (ROOT / TEACHER).exists():
        sys.exit(f"teacher mancante: {TEACHER} (importarlo con scripts/import_thesis_baselines.py)")
    (HERE / "runs").mkdir(exist_ok=True)
    smoke = ["train.epochs=2"] if args.smoke else []
    run("smoke_train" if args.smoke else "train", "stage=train", *smoke)
    run("smoke_adapt" if args.smoke else "adapt", *ADAPT, *smoke, *(["adapt.epochs=1"] if args.smoke else []))
