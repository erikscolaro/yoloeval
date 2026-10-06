"""Ablation sul metodo di calibrazione dell'int8 (QDQ di onnxruntime, tutti i nodi in int8 tranne
Softmax e coda della testa: `int8_full`) per il yolo26p potato di experiments/benchmark/pruned_finetuned.
Prima, dalla root: `run.py stage=train|adapt|search|finetune` con gli override di COMMON.
Solo accuratezza, sulla workstation: la calibrazione cambia le scale e non il grafo, quindi la
latenza resta quella di int8_full (22.0 ms sul Pi, experiments/quantization/int8_backend).

    .venv/bin/python experiments/quantization/calibration/run.py           # tutto
    .venv/bin/python experiments/quantization/calibration/run.py --report  # grafici

Metodi (conf/quantization/): int8_full_minmax (MinMax globale), int8_full_ma (MinMax con media mobile,
la piu' vicina al MEAN_MINMAX di NNCF), int8_full_pct (Percentile 99.999), int8_full_entropy
(Entropy/KL). Export dalla pipeline (`stage=export`, ONNX per il Pi), mAP con model.val()
dell'ONNX su CPU, stesse soglie della mAP sul Pi (conf 0.001, iou 0.7, max_det 300).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import ROOT, make_cfg, weights_or_exit  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

COMMON = ["model=yolo26p", "train=scratch_musgd", "strategy=pit_duccio_pico",
          "strategy.n_source=cpu_4", "strategy.n_quantization=int8", "hardware=rpi5",
          "freq_target=max", "compute_target=cpu_4", "backend=onnxruntime"]
QUANTS = {"fp32": "fp32", "int8_full_minmax": "MinMax", "int8_full_ma": "MinMax media mobile",
          "int8_full_pct": "Percentile 99.999", "int8_full_entropy": "Entropy (KL)"}
RESULTS = HERE / "results/results.json"
NNCF = {"map50": 0.883, "map50_95": 0.526}        # OpenVINO int8 (NNCF) sul Pi, stesso modello


def compose(quant: str):
    return make_cfg(*COMMON, f"quantization={quant}")


def onnx_path(quant: str) -> Path:
    from src.cache import export_dir

    return export_dir(compose(quant)) / f"yolo26p_{quant}.onnx"


def export() -> dict:
    weights_or_exit(*COMMON, "quantization=fp32")   # esce se mancano i pesi
    times = {}
    for q in QUANTS:
        if onnx_path(q).exists():
            continue
        t0 = time.time()
        cmd = [str(ROOT / ".venv/bin/python"), "run.py", "stage=export", *COMMON, f"quantization={q}"]
        print(">>>", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        times[q] = round(time.time() - t0)
    return times


def accuracy(times: dict) -> None:
    from ultralytics import YOLO

    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for q in QUANTS:
        if q in res or not onnx_path(q).exists():
            continue
        m = YOLO(str(onnx_path(q)), task="detect").val(
            data=str(compose("fp32").dataset.yaml), imgsz=640, batch=1, conf=0.001, iou=0.7,
            max_det=300, device="cpu", plots=False, verbose=False, workers=4)
        res[q] = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                  "export_s": times.get(q)}
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(res, indent=1))
        print(f"{q}: mAP50 {m.box.map50:.4f}, mAP50-95 {m.box.map:.4f}", flush=True)


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(RESULTS.read_text())
    methods = [q for q in QUANTS if q != "fp32" and q in res]
    quants = [q for q in QUANTS if q in res]
    for key, fname, title in (("map50_95", "map50_95.png", "mAP50-95"), ("map50", "map50.png", "mAP50")):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        y = list(range(len(methods)))[::-1]
        vals = [res[q][key] for q in methods]
        ax.barh(y, vals, color="#2a78d6", height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=9, color="#333333")
        for ref, label, color in ((res["fp32"][key], "fp32", "#333333"),
                                  (NNCF[key], "OpenVINO int8 (NNCF)", "#eb6834")):
            ax.axvline(ref, color=color, linewidth=1.5, linestyle="--", label=f"{label} {ref:.3f}")
        ax.set_yticks(y, [QUANTS[q] for q in methods], fontsize=9)
        lo = min(vals + [NNCF[key]]) * 0.9
        ax.set_xlim(lo, max(vals + [res["fp32"][key]]) * 1.04)
        ax.set_xlabel(title)
        ax.set_title(f"{title} per calibrazione (int8_full, onnxruntime, val Svanstrom)", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)
        print(FIGURES / fname)
    print("\n| calibrazione | mAP50 | mAP50-95 | perdita mAP50-95 rispetto a fp32 | export (s) |\n|---|---|---|---|---|")
    for q in quants:
        r = res[q]
        print(f"| {QUANTS[q]} | {r['map50']:.3f} | {r['map50_95']:.3f} | "
              f"{r['map50_95'] - res['fp32']['map50_95']:+.3f} | {r.get('export_s') or '–'} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    ap.add_argument("--no-export", action="store_true", help="mAP solo degli ONNX gia' esportati")
    ap.add_argument("--after", type=int, help="aspetta la fine del processo con questo PID")
    ap.add_argument("--min-ram-gb", type=float, default=0, help="RAM disponibile minima per partire")
    args = ap.parse_args()
    # se la RAM finisce, il kernel uccide questo processo (e i figli) prima dei training
    Path("/proc/self/oom_score_adj").write_text("1000")
    if args.after:
        while Path(f"/proc/{args.after}").exists():
            time.sleep(60)
    if args.min_ram_gb:
        ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1])
        if ram / 2**20 < args.min_ram_gb:
            sys.exit(f"RAM disponibile {ram / 2**20:.0f} GB < {args.min_ram_gb:.0f} GB")
    if not args.report:
        accuracy({} if args.no_export else export())
    report()
