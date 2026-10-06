"""Ablation: sull'architettura potata trovata dalla search, finetune dai pesi ereditati contro
training da zero (pesi reinizializzati), stessi parametri e stesso budget.

    .venv/bin/python experiments/ablation/pruned/scratch_vs_finetune/run_experiment.py           # tutto
    .venv/bin/python experiments/ablation/pruned/scratch_vs_finetune/run_experiment.py --report  # solo grafici

1. il yolo26p MuSGD con 15 epoche in piu' senza mosaic (experiments/scratch_musgd_close_mosaic)
   va in cache come training con chiave sua (`+train.extra=close_mosaic15`);
2. search `pit_duccio_pico` dalla pipeline (`run.py stage=search`, target 50% di MAC e
   params, N dal probe di rpi5 int8): finisce in artifacts/search/;
3. dal pruned.pt della search: `finetune.pt` (pesi della search) e `scratch.pt` (stessa
   architettura, pesi reinizializzati come un modello Ultralytics nuovo);
4. i due training in parallelo con PrunedTrainer, parametri di conf/train/scratch_musgd.yaml con
   500 epoche e close_mosaic 50; log in outputs/ablation_pruned/, pesi e risultati in
   finetune/ e scratch/ qui accanto (fuori dalla cache).
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

SOURCE = ROOT / "experiments/scratch_musgd_close_mosaic/run"   # 250 + 15 epoche senza mosaic
COMMON = ["model=yolo26p", "train=scratch_musgd", "+train.extra=close_mosaic15",
          "strategy=pit_duccio_pico", "hardware=rpi5", "quantization=int8"]
RUNS = {"finetune": "finetune dai pesi della search", "scratch": "da zero, stessa architettura"}
COLORS = {"finetune": "#2a78d6", "scratch": "#eb6834"}
EPOCHS, CLOSE_MOSAIC = 500, 50
LOGS = ROOT / "outputs/ablation_pruned"
MIN_RAM_GB, MIN_GPU_FREE_GB, MIN_DISK_GB = 40, 24, 20
MAP50, MAP95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def compose():
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *COMMON])


def train_args() -> dict:
    import yaml

    args = yaml.safe_load(open(ROOT / "conf/train/scratch_musgd.yaml"))
    args.update(epochs=EPOCHS, close_mosaic=CLOSE_MOSAIC, patience=EPOCHS)
    return args


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


def register() -> None:
    """best.pt delle 15 epoche in piu' come training della repo, con la sua chiave."""
    from src.cache import read_dataset_hash, training_key, update_index, weights_dir
    from src.jsonio import atomic_write_json

    cfg = compose()
    out = weights_dir(cfg)
    if (out / "meta.json").exists():
        print(f"pesi gia' registrati: {out.name}", flush=True)
        return
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(SOURCE / "results.csv"))]
    best = max(rows, key=lambda r: float(r[MAP95]))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "weights/best.pt", out / "best.pt")
    meta = {
        "training_key": training_key(cfg), "model": "yolo26p", "dataset": "aod4",
        "epochs": 265, "epochs_completed": 250 + int(best["epoch"]),
        "source": str(SOURCE), "note": "train=scratch_musgd + 15 epoche senza mosaic "
                                       "(experiments/scratch_musgd_close_mosaic)",
        "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
        "metrics": {"map50": float(best[MAP50]), "map50_95": float(best[MAP95])},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/weights/index.json", meta["training_key"], meta)
    print(f"pesi registrati: {out.name} (epoca {meta['epochs_completed']})", flush=True)


def search() -> Path:
    from src.cache import search_dir

    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", "stage=search", *COMMON]
    print(">>>", " ".join(cmd[1:]), f"(log {LOGS / 'search.log'})", flush=True)
    with open(LOGS / "search.log", "a") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL)
    return search_dir(compose())


def reinit(model) -> None:
    """Pesi come un modello Ultralytics appena costruito: init di default di PyTorch per conv e
    BN, eps/momentum delle BN di initialize_weights, bias della testa da Detect.bias_init."""
    import torch.nn as nn
    from ultralytics.nn.modules.head import Detect

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eps, m.momentum = 1e-3, 0.03
    next(m for m in model.modules() if isinstance(m, Detect)).bias_init()


def prepare(sdir: Path) -> None:
    import torch

    models = HERE / "models"
    models.mkdir(exist_ok=True)
    shutil.copy2(sdir / "pruned.pt", models / "finetune.pt")
    ckpt = torch.load(sdir / "pruned.pt", map_location="cpu", weights_only=False)
    before = [p.detach().clone() for p in ckpt["ema"].parameters()]
    reinit(ckpt["ema"])
    changed = sum(not torch.equal(a, b) for a, b in zip(before, ckpt["ema"].parameters()))
    torch.save(ckpt, models / "scratch.pt")
    meta = json.loads((sdir / "meta.json").read_text())
    (HERE / "search.json").write_text(json.dumps({"search_dir": sdir.name, **{
        k: meta.get(k) for k in ("search_key", "pit_summary", "metrics") if k in meta}},
        indent=1, default=str))
    print(f"modelli pronti in {models} (scratch: {changed}/{len(before)} tensori reinizializzati)",
          flush=True)


def train(variant: str) -> None:
    from ultralytics import YOLO
    from yolopit import PrunedTrainer

    cfg = compose()
    model = YOLO(str(HERE / "models" / f"{variant}.pt"))
    r = model.train(data=str(cfg.dataset.yaml), project=str(HERE), name=variant, exist_ok=True,
                    trainer=PrunedTrainer, **train_args())
    print(f"{variant}: mAP50 {r.box.map50:.4f}, mAP50-95 {r.box.map:.4f}", flush=True)


def launch() -> None:
    procs = {}
    for variant in RUNS:
        log = open(LOGS / f"{variant}.log", "w")
        procs[variant] = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-u", __file__, "--train", variant], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"{variant}: pid {procs[variant].pid}, log {LOGS / f'{variant}.log'}", flush=True)
        time.sleep(30)   # partenze sfalsate
    for variant, p in procs.items():
        print(f"{variant}: finito (exit {p.wait()})", flush=True)


def load(variant: str) -> list[dict]:
    path = HERE / variant / "results.csv"
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(path))]


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {v: load(v) for v in RUNS if (HERE / v / "results.csv").exists()}
    for col, title, fname in ((MAP50, "mAP50", "map50.png"), (MAP95, "mAP50-95", "map50_95.png")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ends = []
        for v, rows in data.items():
            x = [int(r["epoch"]) for r in rows]
            y = [float(r[col]) for r in rows]
            ax.plot(x, y, color=COLORS[v], linewidth=2, label=RUNS[v])
            ends.append([y[-1], y[-1], v, x[-1]])
        lo, hi = ax.get_ylim()
        gap = 0.04 * (hi - lo)       # etichette finali distanziate di almeno il 4% dell'asse
        ends.sort(reverse=True)
        for i in range(1, len(ends)):
            ends[i][1] = min(ends[i][1], ends[i - 1][1] - gap)
        for value, ypos, v, xend in ends:
            ax.annotate(f"{v} {value:.3f}", (xend, value), xytext=(xend + 6, ypos),
                        textcoords="data", va="center", fontsize=9, color="#333333")
        ax.axvline(EPOCHS - CLOSE_MOSAIC, color="#999999", linewidth=1, linestyle=":")
        ax.text(EPOCHS - CLOSE_MOSAIC + 3, ax.get_ylim()[0], " senza mosaic", fontsize=8,
                color="#666666", va="bottom")
        ax.set_title(f"{title} per epoca, yolo26p potato: finetune contro da zero", fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel(title)
        ax.set_xlim(1, EPOCHS * 1.14)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(HERE / fname, dpi=150)
        plt.close(fig)
        print(HERE / fname)

    print("\n| run | epoche | mAP50 | mAP50-95 | migliore mAP50-95 (epoca) |\n|---|---|---|---|---|")
    for v, rows in data.items():
        best = max(rows, key=lambda r: float(r[MAP95]))
        print(f"| {v} | {rows[-1]['epoch']} | {float(rows[-1][MAP50]):.3f} | "
              f"{float(rows[-1][MAP95]):.3f} | {float(best[MAP95]):.3f} ({best['epoch']}) |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    ap.add_argument("--train", choices=list(RUNS), help=argparse.SUPPRESS)   # uso interno
    args = ap.parse_args()
    if args.train:
        train(args.train)
        sys.exit()
    if not args.report:
        check_resources()
        register()
        sdir = search()
        prepare(sdir)
        check_resources()
        launch()
    report()
