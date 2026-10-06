"""Pre-processing, inferenza e post-processing sul Raspberry Pi 5 del yolo26p potato e finetunato
(experiments/rpi5_finetuned_musgd), OpenVINO fp32 a 640x640, piu' una piccola ablation su
max_det (300, default di Ultralytics, contro 50: il val ha al massimo 36 oggetti per immagine).

    .venv/bin/python experiments/rpi5_pre_post_processing/run_experiment.py           # tutto
    .venv/bin/python experiments/rpi5_pre_post_processing/run_experiment.py --report  # solo grafici

Test una tantum fuori dalla pipeline (la chiave di export non distingue max_det): due IR fp32
esportati come fa src/backends/openvino.py, 300 immagini del val (seed 0) copiate sul Pi,
misure con pi_bench.py (bench.sh: governor performance, 4 thread, FIFO 80, attesa termica a
50 C). Configurazioni: frame gia' 640x640 (camera a 640) e frame 1024x1024 (resize nel
pre-processing), OpenVINO diretto e model.predict() di Ultralytics.
"""
import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHTS = ROOT / "artifacts/finetune/yolo26p_aod4_pit_duccio_pico_ep58_1a905310/best.pt"
WORK = ROOT / "outputs/rpi5_pre_post"           # IR e immagini, fuori da git
PI_DIR = "/home/bench/bench/prepost"
N_IMAGES, MAX_DETS = 300, (300, 50)
STAGES = (("pre_ms", "pre-processing", "#2a78d6"), ("infer_ms", "inferenza", "#eb6834"),
          ("post_ms", "post-processing", "#1baf7a"))


def sh(cmd: str) -> None:
    print(">>>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def export() -> None:
    from ultralytics import YOLO

    for md in MAX_DETS:
        target = WORK / "models" / f"max_det_{md}_openvino_model"
        if target.exists():
            continue
        produced = Path(YOLO(str(WEIGHTS), task="detect").export(
            format="openvino", imgsz=640, quantize="fp32", batch=1, nms=False, max_det=md,
            device="cpu"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), target)
        print(f"IR max_det={md}: {target}", flush=True)


def sample() -> None:
    out = WORK / "images"
    if out.exists():
        return
    val = sorted((ROOT / "data/aod4/images/val").glob("*.jpg"))
    out.mkdir(parents=True)
    for p in random.Random(0).sample(val, N_IMAGES):
        shutil.copy2(p, out / p.name)


def bench() -> None:
    sh(f"ssh rpi5 'mkdir -p {PI_DIR}'")
    sh(f"rsync -a --delete {WORK}/models {WORK}/images {HERE}/pi_bench.py {HERE}/bench.sh rpi5:{PI_DIR}/")
    sh(f"ssh rpi5 'bash {PI_DIR}/bench.sh'")
    (HERE / "results").mkdir(exist_ok=True)
    sh(f"scp rpi5:{PI_DIR}/timings.csv rpi5:{PI_DIR}/timings.json {HERE}/results/")


def label(r) -> str:
    who = "OpenVINO" if r["pipeline"] == "openvino" else "Ultralytics predict"
    return f"{who}, frame {r['frame']}, max_det {r['max_det']}"


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(csv.DictReader(open(HERE / "results/timings.csv")))
    env = json.loads((HERE / "results/timings.json").read_text())
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(label(r), []).append(r)
    stat = {g: {k: np.array([float(r[k]) for r in rs]) for k in
                ("pre_ms", "infer_ms", "post_ms", "total_ms", "ndet")} for g, rs in groups.items()}
    names = list(groups)[::-1]                  # dall'alto: ordine delle configurazioni

    # 1. scomposizione mediana per fase
    fig, ax = plt.subplots(figsize=(9, 4.6))
    left = np.zeros(len(names))
    for key, title, color in STAGES:
        vals = np.array([np.median(stat[n][key]) for n in names])
        ax.barh(names, vals, left=left, color=color, edgecolor="white", linewidth=2, label=title,
                height=0.6)
        left += vals
    for y, n in enumerate(names):
        ax.text(left[y] + 0.3, y, f"{np.median(stat[n]['total_ms']):.1f} ms", va="center",
                fontsize=9, color="#333333")
    ax.set_xlabel("ms per frame (mediana)")
    ax.set_title("Tempi per fase, yolo26p potato su Pi 5 (OpenVINO fp32, 640x640)", fontsize=11)
    ax.set_xlim(0, left.max() * 1.15)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "breakdown.png", dpi=150)
    plt.close(fig)

    # 2. distribuzione della latenza end-to-end
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.boxplot([stat[n]["total_ms"] for n in names], vert=False, tick_labels=names, whis=(1, 99),
               showfliers=False, medianprops={"color": "#eb6834", "linewidth": 2})
    ax.set_xlabel("ms per frame, end-to-end (baffi: p1-p99)")
    ax.set_title("Latenza end-to-end per frame", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "end_to_end.png", dpi=150)
    plt.close(fig)
    print(HERE / "breakdown.png", HERE / "end_to_end.png", sep="\n")

    print(f"\nimread (fuori dalle misure): mediana {env['imread_ms_median']:.2f} ms; temp "
          f"{env['temp_start_c']:.0f} C -> {env[f'temp_round{2}_c']:.0f} C\n")
    print("| configurazione | pre (ms) | inferenza (ms) | post (ms) | end-to-end mediana / p90 / max (ms) | FPS | box medie |")
    print("|---|---|---|---|---|---|---|")
    for n in names[::-1]:
        s = stat[n]
        t = s["total_ms"]
        print(f"| {n} | {np.median(s['pre_ms']):.2f} | {np.median(s['infer_ms']):.2f} | "
              f"{np.median(s['post_ms']):.3f} | {np.median(t):.2f} / {np.percentile(t, 90):.2f} / "
              f"{t.max():.2f} | {1000 / np.median(t):.1f} | {s['ndet'].mean():.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    args = ap.parse_args()
    if not args.report:
        export()
        sample()
        bench()
    report()
