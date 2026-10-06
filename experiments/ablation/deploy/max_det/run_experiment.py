"""Ablation su max_det (numero di box in uscita, fissato nel TopK del grafo all'export) per il
yolo26p potato e finetunato (experiments/rpi5_finetuned_musgd), OpenVINO fp32 640x640 sul
Raspberry Pi 5: latenza e mAP per max_det 1, 2, 4, ..., 256 e 300 (default di Ultralytics).

    .venv/bin/python experiments/ablation/deploy/max_det/run_experiment.py           # tutto
    .venv/bin/python experiments/ablation/deploy/max_det/run_experiment.py --report  # solo grafici

1. un IR per valore, esportato come fa src/backends/openvino.py (test fuori dalla pipeline: la
   chiave di export non distingue max_det);
2. mAP sul val intero con model.val() dell'IR sulla workstation (CPU, batch 1, max_det uguale a
   quello del grafo): l'accuratezza non dipende dalla board;
3. sul Pi (bench.sh): benchmark_app su ogni IR e pi_bench.py con le 300 immagini di
   experiments/rpi5_pre_post_processing ridotte a 640x640 (frame di una camera a 640).
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WEIGHTS = ROOT / "artifacts/finetune/yolo26p_aod4_pit_duccio_pico_ep58_1a905310/best.pt"
WORK = ROOT / "outputs/ablation_max_det"         # IR, fuori da git
IMAGES = ROOT / "outputs/rpi5_pre_post/images"   # 300 immagini del val (seed 0)
PI_DIR = "/home/bench/bench/maxdet"
MAX_DETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 300)
RESULTS = HERE / "results"


def sh(cmd: str) -> None:
    print(">>>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ir(md: int) -> Path:
    return WORK / "models" / f"max_det_{md}_openvino_model"


def export() -> None:
    from ultralytics import YOLO

    for md in MAX_DETS:
        if ir(md).exists():
            continue
        produced = Path(YOLO(str(WEIGHTS), task="detect").export(
            format="openvino", imgsz=640, quantize="fp32", batch=1, nms=False, max_det=md,
            device="cpu"))
        ir(md).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), ir(md))


def accuracy() -> None:
    from ultralytics import YOLO

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "map.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for md in MAX_DETS:
        if str(md) in res:
            continue
        m = YOLO(str(ir(md)), task="detect").val(
            data=str(ROOT / "data/aod4/dataset.yaml"), imgsz=640, batch=1, conf=0.001, iou=0.7,
            max_det=md, device="cpu", plots=False, verbose=False, workers=4)
        res[str(md)] = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                        "precision": float(m.box.mp), "recall": float(m.box.mr)}
        path.write_text(json.dumps(res, indent=1))
        print(f"max_det {md}: mAP50 {m.box.map50:.4f}, mAP50-95 {m.box.map:.4f}", flush=True)


def bench() -> None:
    sh(f"ssh rpi5 'mkdir -p {PI_DIR}'")
    sh(f"rsync -a --delete {WORK}/models {IMAGES} {HERE}/pi_bench.py {HERE}/bench.sh rpi5:{PI_DIR}/")
    sh(f"ssh rpi5 'bash {PI_DIR}/bench.sh'")
    RESULTS.mkdir(exist_ok=True)
    sh(f"scp rpi5:{PI_DIR}/timings.csv rpi5:{PI_DIR}/timings.json rpi5:{PI_DIR}/benchmark_app.csv "
       f"{RESULTS}/")


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    acc = json.loads((RESULTS / "map.json").read_text())
    ba: dict[int, list[float]] = {}
    for r in csv.DictReader(open(RESULTS / "benchmark_app.csv")):
        ba.setdefault(int(r["max_det"]), []).append(float(r["median_ms"]))
    py: dict[int, dict[str, list[float]]] = {}
    for r in csv.DictReader(open(RESULTS / "timings.csv")):
        d = py.setdefault(int(r["max_det"]), {k: [] for k in ("infer_ms", "total_ms", "ndet")})
        for k in d:
            d[k].append(float(r[k]))
    mds = sorted(int(k) for k in acc)

    def style(ax, title, ylabel):
        ax.set_xscale("log", base=2)
        ax.set_xticks(mds, [str(m) for m in mds])
        ax.minorticks_off()
        labels = ax.get_xticklabels()          # 256 e 300 vicini in scala log: si allontanano
        labels[-2].set_ha("right")
        labels[-1].set_ha("left")
        ax.set_xlabel("max_det")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def end_label(ax, x, y, text):
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#333333")

    # 1. latenza
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series = (("benchmark_app (solo inferenza)", "#2a78d6", [np.mean(ba[m]) for m in mds]),
              ("API Python, end-to-end (pre + inferenza + post)", "#eb6834",
               [np.median(py[m]["total_ms"]) for m in mds]))
    for name, color, y in series:
        ax.plot(mds, y, color=color, linewidth=2, marker="o", markersize=8, label=name,
                markeredgecolor="white", markeredgewidth=2)
        end_label(ax, mds[-1], y[-1], f"{y[-1]:.2f}")
    style(ax, "Latenza mediana per max_det, yolo26p potato su Pi 5 (OpenVINO fp32, 640x640)", "ms")
    ax.legend(frameon=False, loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "latency.png", dpi=150)
    plt.close(fig)

    # 2. accuratezza
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, name, color in (("map50", "mAP50", "#2a78d6"), ("map50_95", "mAP50-95", "#eb6834")):
        y = [acc[str(m)][key] for m in mds]
        ax.plot(mds, y, color=color, linewidth=2, marker="o", markersize=8, label=name,
                markeredgecolor="white", markeredgewidth=2)
        end_label(ax, mds[-1], y[-1], f"{y[-1]:.3f}")
    style(ax, "mAP sul val per max_det (val: max 36 oggetti per immagine, 1.4 in media)", "mAP")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "accuracy.png", dpi=150)
    plt.close(fig)
    print(HERE / "latency.png", HERE / "accuracy.png", sep="\n")

    env = json.loads((RESULTS / "timings.json").read_text())
    temps = [env[k] for k in env if k.startswith("temp_")]
    print(f"\ntemperatura {min(temps):.0f}-{max(temps):.0f} C\n")
    print("| max_det | benchmark_app (ms) | inferenza API Python (ms) | end-to-end (ms) | mAP50 | "
          "mAP50-95 | recall | box per frame (conf 0.25) |\n|---|---|---|---|---|---|---|---|")
    for m in mds:
        a, p = acc[str(m)], py[m]
        print(f"| {m} | {np.mean(ba[m]):.2f} | {np.median(p['infer_ms']):.2f} | "
              f"{np.median(p['total_ms']):.2f} | {a['map50']:.3f} | {a['map50_95']:.3f} | "
              f"{a['recall']:.3f} | {np.mean(p['ndet']):.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    args = ap.parse_args()
    if not args.report:
        export()
        accuracy()
        bench()
    report()
