"""Ablation optimizer dell'adattamento su Svanström (yolo26p): rete intera, poi backbone congelato.

`run.py` lancia in sequenza i 6 adapt (3 optimizer x freeze no/si) e aspetta che nessun altro
`run.py` giri; i run gia' in cache vengono saltati dalla pipeline. `run.py --report` stampa la tabella.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = str(ROOT / ".venv/bin/python")
# lr0 per optimizer: MuSGD e SGD 2e-4 (il migliore dei run su lr), Adam 10x piu' basso come nelle conf di finetune
LR = {"MuSGD": 2e-4, "SGD": 2e-4, "Adam": 2e-5}
# freeze=11: i layer 0-10 di yolo26p sono il backbone (fino a C2PSA)
FREEZE = (None, 11)
COMMON = ["adapt.epochs=40", "adapt.warmup_epochs=2", "adapt.patience=15"]


def run_all():
    while subprocess.run(["pgrep", "-f", "^[^ ]*python[^ ]* run[.]py"], capture_output=True).stdout:
        time.sleep(30)
    for freeze in FREEZE:
        for opt, lr in LR.items():
            cmd = [PY, "run.py", "stage=adapt", "model=yolo26p", "train=scratch_musgd", *COMMON,
                   f"adapt.optimizer={opt}", f"adapt.lr0={lr:.6f}"]
            if freeze:
                cmd.append(f"+adapt.freeze={freeze}")
            subprocess.run(cmd, cwd=ROOT, check=False)


def report():
    print("optimizer lr0 freeze mosaic imgsz | best(ep) mAP50 / mAP50-95 | fine run")
    for meta in sorted((ROOT / "artifacts/adapt").glob("*/meta.json")):
        a = json.loads(meta.read_text())["config"]["adapt"]
        if a.get("epochs") != 40 or a.get("warmup_epochs") != 2:
            continue
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(meta.parent / "run/results.csv"))]
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        last = rows[-1]
        f = lambda r: f"{float(r['metrics/mAP50(B)']):.3f} / {float(r['metrics/mAP50-95(B)']):.3f}"
        print(f"{a['optimizer']} {a['lr0']} {a.get('freeze') or '-'} {a.get('mosaic')} {a.get('imgsz', 640)} | {best['epoch'].strip()}: {f(best)} | {last['epoch'].strip()}: {f(last)}")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run_all()
