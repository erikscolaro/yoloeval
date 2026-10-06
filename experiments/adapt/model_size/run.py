"""Adapt su Svanstrom dei baseline AOD4 di yolo26 n, s, m (importati dalla tesi, `train=thesis_baseline`) con la ricetta
di `conf/adapt/default.yaml`; i tre adapt girano in parallelo. Log in runs/<modello>.log.

    .venv/bin/python experiments/adapt/model_size/run.py          # lancia gli adapt
    .venv/bin/python experiments/adapt/model_size/eval.py ...     # confronto, vedi eval.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
procs = []
for size in sys.argv[1:] or ["n", "s", "m"]:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", "stage=adapt", f"model=yolo26{size}", "train=thesis_baseline",
           "adapt.workers=6"]
    procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=open(HERE / "runs" / f"{size}.log", "w"), stderr=subprocess.STDOUT))
sys.exit(max(p.wait() for p in procs))
