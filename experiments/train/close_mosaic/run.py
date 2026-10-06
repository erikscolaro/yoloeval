"""15 epoche in piu' senza mosaic sul yolo26p allenato da zero con MuSGD (train=scratch_musgd,
experiments/ablation/train/lr_scheduler): il run ha close_mosaic 10 su 250 epoche, la tesi 100 su
1000. Con queste 15 si arriva a 25 epoche senza mosaic, il 10% finale come nella tesi.

    .venv/bin/python experiments/scratch_musgd_close_mosaic/run_experiment.py --after PID  # attende PID
    .venv/bin/python experiments/scratch_musgd_close_mosaic/run_experiment.py --report     # solo grafici

Riparte da last.pt (epoca 250) con un training nuovo di 15 epoche: last.pt a fine training non
ha piu' lo stato dell'optimizer, quindi il resume non si puo' fare. lr costante al valore finale
del run (lr0 * lrf = 1e-4, lrf 1, niente warmup), mosaic spento, il resto come scratch_musgd.
Fuori dalla cache: pesi e risultati in run/ qui accanto, best.pt del training resta quello a 250.
Alla fine le 15 righe vengono aggiunte a results.csv del run MuSGD come epoche 251-265 (copia
dell'originale in results_e250.csv, letta dai grafici dell'ablation).
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "artifacts/weights/yolo26p_aod4_e250_13fa7d10/run"   # training scratch_musgd
OUT = HERE / "run"
EPOCHS = 15
ARGS = dict(epochs=EPOCHS, close_mosaic=EPOCHS, mosaic=0.0, batch=64, workers=12, cache="disk",
            imgsz=640, optimizer="MuSGD", lr0=0.01 * 0.01, lrf=1.0, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=0, patience=EPOCHS, seed=42, mixup=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
MIN_RAM_GB = 16
MIN_GPU_FREE_GB = 16
MAP50, MAP95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def read(path: Path) -> list[dict]:
    return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(path))]


def check() -> None:
    rows = read(BASE / "results.csv")
    if int(rows[-1]["epoch"]) != 250:
        sys.exit(f"il run MuSGD e' all'epoca {rows[-1]['epoch']}, non a 250")
    ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpu = int(out.split()[0]) / 1024
    busy = subprocess.run(["pgrep", "-f", "run.py stage=(train|search|finetune)"],
                          capture_output=True, text=True).stdout.split()
    print(f"RAM disponibile {ram:.0f} GB, GPU libera {gpu:.0f} GB, training attivi: {len(busy)}")
    if ram < MIN_RAM_GB or gpu < MIN_GPU_FREE_GB or busy:
        sys.exit("risorse insufficienti o altri training in corso")


def train() -> None:
    from hydra import compose, initialize_config_dir
    from ultralytics import YOLO

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose("config", overrides=[f"project_root={ROOT}", "model=yolo26p"])
    model = YOLO(str(BASE / "weights/last.pt"))
    r = model.train(data=str(cfg.dataset.yaml), project=str(HERE), name="run", exist_ok=True, **ARGS)
    print(f"dopo {EPOCHS} epoche: mAP50 {r.box.map50:.4f}, mAP50-95 {r.box.map:.4f}")


def append() -> None:
    """Aggiunge le epoche nuove a results.csv del run MuSGD (una volta sola)."""
    backup = BASE / "results_e250.csv"
    if backup.exists():
        print("results.csv del run MuSGD gia' esteso")
        return
    shutil.copy2(BASE / "results.csv", backup)
    base, new = read(backup), read(OUT / "results.csv")
    cols = list(base[0])
    t0 = float(base[-1]["time"])
    with open(BASE / "results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        for r in new:
            w.writerow({**r, "epoch": int(r["epoch"]) + 250, "time": f"{float(r['time']) + t0:g}"})
    print(f"aggiunte {len(new)} epoche a {BASE / 'results.csv'} (originale in {backup.name})")


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, new = read(BASE / "results_e250.csv"), read(OUT / "results.csv")
    print(f"lr/pg0 ultima epoca del run {base[-1]['lr/pg0']}, prima epoca nuova {new[0]['lr/pg0']}")
    tail = base[-40:]
    for col, title, fname in ((MAP50, "mAP50", "map50.png"), (MAP95, "mAP50-95", "map50_95.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot([int(r["epoch"]) for r in tail], [float(r[col]) for r in tail], color="#eb6834",
                linewidth=2, label="run MuSGD (250 epoche)")
        ax.plot([250] + [int(r["epoch"]) + 250 for r in new],
                [float(tail[-1][col])] + [float(r[col]) for r in new], color="#2a78d6",
                linewidth=2, label=f"+{EPOCHS} epoche senza mosaic")
        for x, text in ((240, " mosaic spento"), (250, " ripartenza")):
            ax.axvline(x, color="#999999", linewidth=1, linestyle=":")
            ax.text(x + 0.3, ax.get_ylim()[0], text, fontsize=8, color="#666666", va="bottom")
        ax.set_title(f"{title} per epoca, yolo26p MuSGD: ultime 40 epoche + {EPOCHS}", fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(HERE / fname, dpi=150)
        plt.close(fig)
        print(HERE / fname)

    best = lambda rows: max(rows, key=lambda r: float(r[MAP95]))
    print("\n| | epoca | mAP50 | mAP50-95 |\n|---|---|---|---|")
    for label, r, off in (("fine run", base[-1], 0), ("migliore run", best(base), 0),
                          (f"fine +{EPOCHS}", new[-1], 250), (f"migliore +{EPOCHS}", best(new), 250)):
        print(f"| {label} | {int(r['epoch']) + off} | {float(r[MAP50]):.3f} | {float(r[MAP95]):.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    ap.add_argument("--after", type=int, help="aspetta la fine del processo con questo PID")
    args = ap.parse_args()
    if args.after:
        while Path(f"/proc/{args.after}").exists():
            time.sleep(30)
        print("ablation finita, parto", flush=True)
    if not args.report:
        check()
        train()
        append()
    report()
