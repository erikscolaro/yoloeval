"""Ablation sulla risoluzione: yolo26p da zero (MuSGD, 250 epoche, close_mosaic 25) allenato a
1024 (conf/train/scratch_musgd_1024.yaml, le immagini di AOD4 sono 1024x1024) contro i training
a 640 che ci sono gia': scratch_musgd (250 epoche, close_mosaic 10) e la sua estensione con 15
epoche senza mosaic (265, 25 senza mosaic in tutto, experiments/scratch_musgd_close_mosaic).
Tutti validati a 640 e a 1024: matrice train x val.

    .venv/bin/python experiments/ablation/train/imgsz/run_experiment.py           # lancia e attende
    .venv/bin/python experiments/ablation/train/imgsz/run_experiment.py --report  # solo grafici

Il training a 1024 passa da `run.py stage=train` (cache vera, artifacts/weights/), log in
outputs/ablation_imgsz/. La validazione incrociata usa model.val() del best.pt sul val, su GPU,
con le impostazioni di default di Ultralytics (batch 32, rect) a entrambe le risoluzioni.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

SIZES = (640, 1024)
# training: etichetta -> override di train (cache), risoluzione di training
TRAINS = {"640": (["train=scratch_musgd"], 640),
          "640+15": (["train=scratch_musgd", "+train.extra=close_mosaic15"], 640),
          "1024": (["train=scratch_musgd_1024"], 1024)}
COLORS = {"640": "#2a78d6", "640+15": "#1baf7a", "1024": "#eb6834"}
LOGS = ROOT / "outputs/ablation_imgsz"
RESULTS = HERE / "results.json"
MIN_RAM_GB, MIN_GPU_FREE_GB, MIN_DISK_GB = 40, 30, 20
MAP50, MAP95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def train_dir(label: str) -> Path:
    from hydra import compose, initialize_config_dir

    from src.cache import weights_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose("config", overrides=[f"project_root={ROOT}", "model=yolo26p",
                                           *TRAINS[label][0]])
    return weights_dir(cfg)


def check_resources() -> None:
    ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpu = int(out.split()[0]) / 1024
    disk = shutil.disk_usage(ROOT).free / 2**30
    busy = subprocess.run(["pgrep", "-f", "run.py stage=(train|search|finetune)"],
                          capture_output=True, text=True).stdout.split()
    print(f"RAM disponibile {ram:.0f} GB, GPU libera {gpu:.0f} GB, disco libero {disk:.0f} GB, "
          f"training attivi: {len(busy)}", flush=True)
    if ram < MIN_RAM_GB or gpu < MIN_GPU_FREE_GB or disk < MIN_DISK_GB or busy:
        sys.exit("risorse insufficienti o altri training in corso")


def launch() -> None:
    check_resources()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "train_1024.log", "w") as log:
        print(f"train 1024: log {LOGS / 'train_1024.log'}", flush=True)
        rc = subprocess.run([str(ROOT / ".venv/bin/python"), "run.py", "stage=train",
                             "model=yolo26p", "train=scratch_musgd_1024"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(f"train 1024: finito (exit {rc})", flush=True)


def validate() -> None:
    from ultralytics import YOLO

    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for t in TRAINS:
        for v in SIZES:
            key = f"train{t}_val{v}"
            if key in res:
                continue
            m = YOLO(str(train_dir(t) / "best.pt")).val(
                data=str(ROOT / "data/aod4/dataset.yaml"), imgsz=v, plots=False, verbose=False)
            res[key] = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                        "per_class_map50_95": {m.names[i]: float(x) for i, x in enumerate(m.box.maps)}}
            RESULTS.write_text(json.dumps(res, indent=1))
            print(f"{key}: mAP50 {m.box.map50:.4f}, mAP50-95 {m.box.map:.4f}", flush=True)


def load(label: str) -> list[dict]:
    run = train_dir(label) / "run"
    if label == "640":   # results.csv del run 640 e' stato esteso con le 15 epoche (251-265)
        path = run / "results_e250.csv" if (run / "results_e250.csv").exists() else run / "results.csv"
    elif label == "640+15":
        path = ROOT / "artifacts/weights" / train_dir("640").name / "run" / "results.csv"
    else:
        path = run / "results.csv"
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(path))]


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col, title, fname in ((MAP50, "mAP50", "train_map50.png"), (MAP95, "mAP50-95", "train_map50_95.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        last = 0
        for s in ("640+15", "1024"):    # 640+15: le 250 epoche a 640 piu' le 15 senza mosaic
            rows = load(s)
            x = [int(r["epoch"]) for r in rows]
            y = [float(r[col]) for r in rows]
            last = max(last, x[-1])
            ax.plot(x, y, color=COLORS[s], linewidth=2,
                    label=f"train e val a {TRAINS[s][1]}" + (" (250 + 15 senza mosaic)" if s == "640+15" else ""))
            ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#333333")
        ax.axvline(225, color="#999999", linewidth=1, linestyle=":")
        ax.text(226, ax.get_ylim()[0], " 1024 senza mosaic", fontsize=8, color="#666666", va="bottom")
        ax.set_title(f"{title} per epoca (val del training, alla risoluzione di training)", fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel(title)
        ax.set_xlim(1, last * 1.1)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(HERE / fname, dpi=150)
        plt.close(fig)
        print(HERE / fname)

    if not RESULTS.exists():
        return
    res = json.loads(RESULTS.read_text())
    labels = list(TRAINS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, title in zip(axes, ("map50", "map50_95"), ("mAP50", "mAP50-95")):
        grid = [[res[f"train{t}_val{v}"][key] for v in SIZES] for t in labels]
        ax.imshow(grid, cmap="Blues", vmin=min(map(min, grid)) - 0.02, vmax=max(map(max, grid)))
        for i in range(len(labels)):
            for j in range(2):
                ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center", fontsize=12,
                        color="#111111")
        ax.set_xticks([0, 1], [f"val {v}" for v in SIZES])
        ax.set_yticks(range(len(labels)), [f"train {t}" for t in labels])
        ax.set_title(title, fontsize=11)
    fig.suptitle("yolo26p da zero, 250 epoche: risoluzione di training contro di validazione",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "matrix.png", dpi=150)
    plt.close(fig)
    print(HERE / "matrix.png")
    print("\n| train | val | mAP50 | mAP50-95 |\n|---|---|---|---|")
    for t in TRAINS:
        for v in SIZES:
            r = res[f"train{t}_val{v}"]
            print(f"| {t} | {v} | {r['map50']:.3f} | {r['map50_95']:.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    args = ap.parse_args()
    if not args.report:
        launch()
        validate()
    report()
