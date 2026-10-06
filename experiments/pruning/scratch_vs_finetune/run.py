"""Ablation: sull'architettura potata trovata dalla search, finetune dai pesi ereditati contro
training da zero (pesi reinizializzati), con le stesse ricette della pipeline.

    .venv/bin/python experiments/pruning/scratch_vs_finetune/run.py           # tutto
    .venv/bin/python experiments/pruning/scratch_vs_finetune/run.py --report  # solo grafici

1. catena della pipeline per yolo26p con `pit_duccio_pico` (50% di MAC e params, N dal probe):
   `run.py stage=train` (AOD4), `stage=adapt` (Svanstrom) e `stage=search`; gli stadi gia' in
   cache sono saltati;
2. variante "finetune": `run.py stage=finetune` (pipeline normale, recipe `finetune`, su
   Svanstrom): finisce in artifacts/finetune/;
3. variante "scratch": dal `pruned.pt` della search, stessa architettura con pesi
   reinizializzati come un modello Ultralytics nuovo; training con la recipe `train`
   (scratch_musgd) su `pretrain_dataset` (AOD4), poi adattamento su Svanstrom con la recipe
   `adapt`, entrambi con PrunedTrainer. Log, pesi e run di Ultralytics in runs/ (fuori dalla
   cache);
4. le due varianti girano in parallelo; il confronto e' sulle curve di mAP su Svanstrom (le epoche
   di finetune contro quelle di adapt); search.json e grafici in results/ e figures/.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import ROOT, make_cfg, run_stage  # noqa: E402

HERE = Path(__file__).resolve().parent

# stessa search degli altri esperimenti sul potato (N dal probe rpi5 int8 a 4 core)
COMMON = ["model=yolo26p", "train=scratch_musgd", "strategy=pit_duccio_pico",
          "strategy.n_source=cpu_4", "strategy.n_quantization=int8", "hardware=rpi5", "quantization=int8"]
RUNS = {"finetune": "finetune dai pesi della search", "scratch": "da zero (AOD4) + adapt"}
COLORS = {"finetune": "#2a78d6", "scratch": "#eb6834"}
RUNS_DIR = HERE / "runs"          # log, modelli e run di Ultralytics (ignorata da git)
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
MIN_RAM_GB, MIN_GPU_FREE_GB, MIN_DISK_GB = 40, 24, 20
MAP50, MAP95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def compose():
    return make_cfg(*COMMON)


def pretrain_args(cfg) -> dict:
    """Recipe `train` (scratch_musgd) del baseline, per l'allenamento da zero su AOD4."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg.train, resolve=True)


def check_resources() -> None:
    ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpu = int(out.split()[0]) / 1024
    disk = shutil.disk_usage(ROOT).free / 2**30
    busy = subprocess.run(["pgrep", "-f", "run.py stage=(train|adapt|search|finetune)"],
                          capture_output=True, text=True).stdout.split()
    print(f"RAM disponibile {ram:.0f} GB, GPU libera {gpu:.0f} GB, disco libero {disk:.0f} GB, "
          f"training attivi: {len(busy)}", flush=True)
    if ram < MIN_RAM_GB or gpu < MIN_GPU_FREE_GB or disk < MIN_DISK_GB or busy:
        sys.exit("risorse insufficienti o altri training in corso")


def pipeline() -> Path:
    """train, adapt e search dalla pipeline (log in runs/pipeline.log); ritorna la cartella della search."""
    from src.cache import find_by_key, search_key

    RUNS_DIR.mkdir(exist_ok=True)
    with open(RUNS_DIR / "pipeline.log", "a") as log:
        for stage in ("train", "adapt", "search"):
            print(f"stage={stage} (log {RUNS_DIR / 'pipeline.log'})", flush=True)
            run_stage(stage, *COMMON, log=log)
    cfg = compose()
    return find_by_key(cfg.artifacts_dir, "search", "search_key", search_key(cfg))


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

    models = RUNS_DIR / "models"
    models.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(sdir / "pruned.pt", map_location="cpu", weights_only=False)
    before = [p.detach().clone() for p in ckpt["ema"].parameters()]
    reinit(ckpt["ema"])
    changed = sum(not torch.equal(a, b) for a, b in zip(before, ckpt["ema"].parameters()))
    torch.save(ckpt, models / "scratch.pt")
    meta = json.loads((sdir / "meta.json").read_text())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "search.json").write_text(json.dumps({"search_dir": sdir.name, **{
        k: meta.get(k) for k in ("search_key", "pit_summary", "metrics") if k in meta}},
        indent=1, default=str))
    print(f"modello pronto in {models} (scratch: {changed}/{len(before)} tensori reinizializzati)",
          flush=True)


def train_scratch() -> None:
    """Da zero: recipe `train` su AOD4, poi recipe `adapt` su Svanstrom (stesso trainer)."""
    from src.stages.adapt import adapt_args
    from ultralytics import YOLO
    from yolopit import PrunedTrainer

    cfg = compose()
    model = YOLO(str(RUNS_DIR / "models" / "scratch.pt"))
    r = model.train(data=str(cfg.pretrain_dataset.yaml), project=str(RUNS_DIR),
                    name="scratch_pretrain", exist_ok=True, trainer=PrunedTrainer,
                    **pretrain_args(cfg))
    print(f"scratch, train su {cfg.pretrain_dataset.name}: mAP50 {r.box.map50:.4f}", flush=True)
    model = YOLO(str(RUNS_DIR / "scratch_pretrain" / "weights" / "best.pt"))
    r = model.train(data=str(cfg.dataset.yaml), project=str(RUNS_DIR), name="scratch",
                    exist_ok=True, trainer=PrunedTrainer, **adapt_args(cfg))
    print(f"scratch, adapt su {cfg.dataset.name}: mAP50 {r.box.map50:.4f}, "
          f"mAP50-95 {r.box.map:.4f}", flush=True)


def launch() -> None:
    log = open(RUNS_DIR / "finetune.log", "w")
    procs = {"finetune": subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "run.py", "stage=finetune", *COMMON], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)}
    print(f"finetune: pid {procs['finetune'].pid}, log {RUNS_DIR / 'finetune.log'}", flush=True)
    time.sleep(30)   # partenze sfalsate
    log = open(RUNS_DIR / "scratch.log", "w")
    procs["scratch"] = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-u", __file__, "--train"], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    print(f"scratch: pid {procs['scratch'].pid}, log {RUNS_DIR / 'scratch.log'}", flush=True)
    for variant, p in procs.items():
        print(f"{variant}: finito (exit {p.wait()})", flush=True)


def load(variant: str) -> list[dict]:
    if variant == "finetune":   # run di Ultralytics dello stadio finetune, in cache
        from src.cache import find_by_key, finetune_key

        cfg = compose()
        path = find_by_key(cfg.artifacts_dir, "finetune", "finetune_key",
                           finetune_key(cfg)) / "run" / "results.csv"
    else:
        path = RUNS_DIR / variant / "results.csv"
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(path))]


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for v in RUNS:
        try:
            data[v] = load(v)
        except (OSError, AttributeError):    # variante non ancora allenata
            print(f"{v}: nessuna curva (lancia lo script senza --report)")
    cfg = compose()
    ep, cm = max(len(rows) for rows in data.values()), int(cfg.adapt.close_mosaic)
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
        ax.axvline(ep - cm, color="#999999", linewidth=1, linestyle=":")
        ax.text(ep - cm + 3, ax.get_ylim()[0], " senza mosaic", fontsize=8,
                color="#666666", va="bottom")
        ax.set_title(f"{title} per epoca, yolo26p potato, su Svanstrom: finetune contro da zero", fontsize=11)
        ax.set_xlabel("epoca")
        ax.set_ylabel(title)
        ax.set_xlim(1, ep * 1.14)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)
        print(FIGURES / fname)

    print("\n| run | epoche | mAP50 | mAP50-95 | migliore mAP50-95 (epoca) |\n|---|---|---|---|---|")
    for v, rows in data.items():
        best = max(rows, key=lambda r: float(r[MAP95]))
        print(f"| {v} | {rows[-1]['epoch']} | {float(rows[-1][MAP50]):.3f} | "
              f"{float(rows[-1][MAP95]):.3f} | {float(best[MAP95]):.3f} ({best['epoch']}) |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    ap.add_argument("--train", action="store_true", help=argparse.SUPPRESS)   # uso interno
    args = ap.parse_args()
    if args.train:
        train_scratch()
        sys.exit()
    if not args.report:
        check_resources()
        prepare(pipeline())
        check_resources()
        launch()
    report()
