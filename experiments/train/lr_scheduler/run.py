"""Ablation sull'optimizer del training da zero di yolo26p: scratch_adam, scratch_musgd e
scratch_sgd (conf/train/), identiche tranne optimizer e lr0.

    .venv/bin/python experiments/train/lr_scheduler/run.py           # lancia e attende
    .venv/bin/python experiments/train/lr_scheduler/run.py --report  # solo grafici

I training passano da `run.py stage=train` su `pretrain_dataset` (AOD4), quindi finiscono nella
cache vera (artifacts/weights/, ritrovati con `find_weights`): la conf migliore diventa la
baseline, da adattare poi a Svanstrom con `stage=adapt`. Prima di lanciare
controlla RAM, GPU e disco; i tre training girano in parallelo, con i log in
runs/<nome>.log. Alla fine: grafici di mAP50 e mAP50-95 per epoca, copia
delle conf in config/ e tabella dei risultati da incollare nel README.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import ROOT, make_cfg  # noqa: E402

HERE = Path(__file__).resolve().parent

RUNS = {"Adam": "adam", "MuSGD": "musgd", "SGD": "sgd"}
COLORS = {"Adam": "#2a78d6", "MuSGD": "#eb6834", "SGD": "#1baf7a"}   # ordine categorico fisso
RUNS_DIR = HERE / "runs"          # log dei training (ignorata da git)
FIGURES = HERE / "figures"
CONFIG = HERE / "config"          # copia delle conf usate, scritta da --report

# risorse richieste per tre training in parallelo (batch 64, cache su disco)
MIN_RAM_GB = 40
MIN_GPU_FREE_GB = 36
MIN_DISK_GB = 20


def train_dir(name: str) -> Path:
    """Cartella del training in artifacts/weights/ (dipende dal training_key della conf)."""
    from src.cache import find_weights, training_key

    cfg = make_cfg("model=yolo26p", f"train=scratch_{name}")
    d = find_weights(cfg.artifacts_dir, training_key(cfg))
    if d is None:
        sys.exit(f"training scratch_{name} non in cache: .venv/bin/python run.py stage=train "
                 f"model=yolo26p train=scratch_{name}  (o senza --report)")
    return d


def check_resources() -> None:
    ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1])
    ram_gb = ram / 2**20
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpu_gb = int(out.split()[0]) / 1024
    disk_gb = shutil.disk_usage(ROOT).free / 2**30
    busy = subprocess.run(["pgrep", "-f", "run.py stage=(train|adapt|search|finetune)"],
                          capture_output=True, text=True).stdout.split()
    print(f"RAM disponibile {ram_gb:.0f} GB, GPU libera {gpu_gb:.0f} GB, disco libero "
          f"{disk_gb:.0f} GB, training gia' attivi: {len(busy)}")
    problems = [msg for ok, msg in (
        (ram_gb >= MIN_RAM_GB, f"RAM disponibile < {MIN_RAM_GB} GB"),
        (gpu_gb >= MIN_GPU_FREE_GB, f"GPU libera < {MIN_GPU_FREE_GB} GB"),
        (disk_gb >= MIN_DISK_GB, f"disco libero < {MIN_DISK_GB} GB"),
        (not busy, "altri training/search/finetune in corso"),
    ) if not ok]
    if problems:
        sys.exit("risorse insufficienti: " + "; ".join(problems))


def launch() -> None:
    check_resources()
    RUNS_DIR.mkdir(exist_ok=True)
    procs = {}
    for label, name in RUNS.items():
        log = open(RUNS_DIR / f"{name}.log", "w")
        procs[label] = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "run.py", "stage=train", "model=yolo26p",
             f"train=scratch_{name}"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"{label}: pid {procs[label].pid}, log {RUNS_DIR / f'{name}.log'}")
        time.sleep(30)   # partenze sfalsate: la scansione delle label non si sovrappone
    for label, p in procs.items():
        rc = p.wait()
        print(f"{label}: finito (exit {rc})")


def load(name: str) -> list[dict]:
    run = train_dir(name) / "run"
    path = run / "results.csv"
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(path))]


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {label: load(name) for label, name in RUNS.items()}
    for col, title, fname in (("metrics/mAP50(B)", "mAP50", "train_map50.png"),
                              ("metrics/mAP50-95(B)", "mAP50-95", "train_map50_95.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ends = []
        for label, rows in data.items():
            x = [int(r["epoch"]) for r in rows]
            y = [float(r[col]) for r in rows]
            ax.plot(x, y, color=COLORS[label], linewidth=2, label=label)
            ends.append([y[-1], y[-1], label, x[-1]])
        lo, hi = ax.get_ylim()
        gap = 0.04 * (hi - lo)       # etichette finali distanziate di almeno il 4% dell'asse
        ends.sort(reverse=True)
        for i in range(1, len(ends)):
            ends[i][1] = min(ends[i][1], ends[i - 1][1] - gap)
        for value, ypos, label, xend in ends:
            ax.annotate(f"{label} {value:.3f}", (xend, value), xytext=(xend + 3, ypos),
                        textcoords="data", va="center", fontsize=9, color="#333333")
        last = max(int(rows[-1]["epoch"]) for rows in data.values())
        ax.axvline(last - 10, color="#999999", linewidth=1, linestyle=":")   # close_mosaic: 10
        ax.text(last - 9.5, ax.get_ylim()[0], " senza mosaic", fontsize=8, color="#666666",
                va="bottom")
        ax.set_title(f"{title} per epoca, training da zero di yolo26p", fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel(title)
        ax.set_xlim(1, last * 1.12)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)
        print(FIGURES / fname)

    CONFIG.mkdir(exist_ok=True)
    print("\n| optimizer | mAP50 | mAP50-95 | migliore mAP50-95 (epoca) |\n|---|---|---|---|")
    for label, name in RUNS.items():
        shutil.copy(ROOT / f"conf/train/scratch_{name}.yaml", CONFIG / f"scratch_{name}.yaml")
        shutil.copy(train_dir(name) / "run" / "args.yaml", CONFIG / f"scratch_{name}_args.yaml")
        rows = data[label]
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        print(f"| {label} | {float(rows[-1]['metrics/mAP50(B)']):.3f} | "
              f"{float(rows[-1]['metrics/mAP50-95(B)']):.3f} | "
              f"{float(best['metrics/mAP50-95(B)']):.3f} ({best['epoch']}) |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici, conf e tabella")
    args = ap.parse_args()
    if not args.report:
        launch()
    report()
