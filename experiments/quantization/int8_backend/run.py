"""int8 piu' lento di fp32 sul Pi: e' il backend? Stesso modello di experiments/benchmark/pruned_finetuned
(yolo26p potato, finetune MuSGD), onnxruntime int8 e fp32 contro i risultati OpenVINO gia'
misurati (stesso modello, stesso setup): Pi a 2.4 GHz governor performance, 4 core, FIFO 80,
attesa termica fino a 50 °C, mAP sul Pi. Prima, dalla root: `run.py stage=train|adapt|search|finetune`
con gli override di COMMON.

    .venv/bin/python experiments/quantization/int8_backend/run.py           # export + benchmark
    .venv/bin/python experiments/quantization/int8_backend/run.py --report  # solo tabella

- int8: QDQ statico di onnxruntime, quantizzati solo Conv, Gemm e MatMul (le SiLU restano fp32,
  con Quantize/Dequantize attorno a ogni conv);
- int8_full: tutti i nodi in int8 tranne la Softmax e la coda della testa;
- backend openvino: int8 di NNCF via Ultralytics; openvino_qdq: lo stesso QDQ di onnxruntime
  convertito in IR (stessa quantizzazione, cambia solo il backend).
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import ROOT, make_cfg, weights_or_exit  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

COMMON = ["model=yolo26p", "train=scratch_musgd", "strategy=pit_duccio_pico",
          "strategy.n_source=cpu_4", "strategy.n_quantization=int8", "hardware=rpi5",
          "freq_target=max", "compute_target=cpu_4"]
CELLS = [("onnxruntime", "fp32"), ("onnxruntime", "int8"), ("onnxruntime", "int8_full"),
         ("openvino", "fp32"), ("openvino", "int8"),
         ("openvino_qdq", "int8"), ("openvino_qdq", "int8_full")]


def compose(backend: str, quant: str):
    return make_cfg(*COMMON, f"backend={backend}", f"quantization={quant}")


def run(*args: str) -> None:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", *args]
    print(">>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)


def plots(rows: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{b}\n{q}" for b, q in rows]
    colors = {"onnxruntime": "#2a78d6", "openvino": "#eb6834", "openvino_qdq": "#1baf7a"}
    for key, fname, title, xlabel, fmt in (
            ("median_ms", "latency.png", "Latenza mediana sul Pi 5 (4 core, 2.4 GHz)", "ms", "{:.1f}"),
            ("map50_95", "accuracy.png", "mAP50-95 sul Pi 5 (val Svanstrom)", "mAP50-95", "{:.3f}")):
        vals = [rows[k].get(key) for k in rows]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        y = range(len(labels))[::-1]
        ax.barh(list(y), [v or 0 for v in vals], color=[colors[b] for b, _ in rows], height=0.6)
        for yi, v in zip(y, vals):
            if v is not None:
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color="#333333")
        ax.set_yticks(list(y), labels, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, max(v or 0 for v in vals) * 1.15)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)
        print(FIGURES / fname)


def report() -> None:
    from src.cache import cell_key
    from src.jsonio import read_json

    rows = {}
    fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
    print("| backend | precisione | latenza mediana (ms) | media (ms) | max (ms) | FPS | mAP50 | "
          "mAP50-95 | temp. inizio→fine (°C) |\n|---|---|---|---|---|---|---|---|---|")
    for b, q in CELLS:
        r = read_json(ROOT / "results" / f"{cell_key(compose(b, q))}.json") or {}
        lat, acc, rt = r.get("latency") or {}, r.get("accuracy") or {}, r.get("runtime_state") or {}
        med = lat.get("median_ms")
        rows[b, q] = {"median_ms": med, "map50_95": acc.get("map50_95")}
        print(f"| {b} | {q} | {fmt(med)} | {fmt(lat.get('mean_ms'))} | {fmt(lat.get('max_ms'))} | "
              f"{fmt(1000 / med if med else None, '{:.1f}')} | {fmt(acc.get('map50'), '{:.3f}')} | "
              f"{fmt(acc.get('map50_95'), '{:.3f}')} | {fmt(rt.get('temp_start_c'), '{:.0f}')}→"
              f"{fmt(rt.get('temp_end_c'), '{:.0f}')} |")
    if any(r["median_ms"] for r in rows.values()):
        plots(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo la tabella")
    args = ap.parse_args()
    if not args.report:
        weights_or_exit(*COMMON, "backend=onnxruntime", "quantization=fp32")   # esce se mancano
        run("-m", "stage=export", *COMMON, "backend=onnxruntime", "quantization=fp32,int8,int8_full")
        run("-m", "stage=export", *COMMON, "backend=openvino_qdq", "quantization=int8,int8_full")
        run("-m", "stage=benchmark", *COMMON, "backend=onnxruntime", "quantization=int8,fp32,int8_full")
        run("-m", "stage=benchmark", *COMMON, "backend=openvino_qdq", "quantization=int8,int8_full")
    report()
