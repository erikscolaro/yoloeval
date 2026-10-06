"""N ottimo del probe su Raspberry Pi 5 al variare dei core usati e della precisione: probe
`probe_edge` (quello della pipeline per le board) con onnxruntime a 2.4 GHz, per int8, fp16 e
fp32 su cpu_1, cpu_2 e cpu_4.

    .venv/bin/python experiments/rpi5_probe_n_vs_cores/run_experiment.py --after PID  # attende PID
    .venv/bin/python experiments/rpi5_probe_n_vs_cores/run_experiment.py --report     # solo grafici

`stage.sweep.seed=1` (ordine casuale delle misure diverso da quello di default) da' ai probe
un probe_id loro: non sovrascrivono results/probe/rpi5_cpu_1_int8_onnxruntime_134367b1, che e'
quello da cui pit_auto prende N=8 (ed entra nelle chiavi delle search). Il cpu_1 int8 e' quindi
anche una replica di quel probe.
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

QUANTS = ("int8", "fp16", "fp32")
CORES = ("cpu_1", "cpu_2", "cpu_4")
COMMON = ["stage=probe_edge", "hardware=rpi5", "backend=onnxruntime", "freq_target=max",
          "stage.sweep.seed=1"]
QCOLORS = {"int8": "#2a78d6", "fp16": "#eb6834", "fp32": "#1baf7a"}
CCOLORS = {"cpu_1": "#2a78d6", "cpu_2": "#eb6834", "cpu_4": "#1baf7a"}


def compose(q: str, c: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *COMMON,
                                                  f"quantization={q}", f"compute_target={c}"])


def probe() -> None:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", "-m", *COMMON,
           f"quantization={','.join(QUANTS)}", f"compute_target={','.join(CORES)}"]
    print(">>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load() -> dict:
    from src.stages.probe import probe_id

    data = {}
    for q in QUANTS:
        for c in CORES:
            pid = probe_id(compose(q, c))
            base = ROOT / "results/probe" / pid
            per_c = defaultdict(list)
            for line in open(f"{base}.jsonl"):
                r = json.loads(line)
                per_c[int(r["c"])].append(float(r["median_ms"]))
            summary = json.loads(open(f"{base}.summary.json").read())
            cs = sorted(per_c)
            data[q, c] = {"pid": pid, "n": summary.get("n_opt"), "c": cs,
                          "ms": [sorted(per_c[x])[len(per_c[x]) // 2] for x in cs],
                          "noise": summary.get("noise_rel"),
                          "temp": (summary.get("runtime_state") or {}).get("temp_start_c")}
    return data


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load()
    for fname, groups, lines, colors, gtitle, ltitle in (
            ("per_core.png", CORES, QUANTS, QCOLORS, "core", "precisione"),
            ("per_quant.png", QUANTS, CORES, CCOLORS, "precisione", "core")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
        for ax, g in zip(axes, groups):
            for l in lines:
                d = data[(g, l) if g in QUANTS else (l, g)]
                ax.plot(d["c"], d["ms"], color=colors[l], linewidth=2, label=f"{l} (N={d['n']})")
            for x in range(8, 97, 8):
                ax.axvline(x, color="#cccccc", linewidth=0.6, zorder=0)
            ax.set_title(str(g), fontsize=11)
            ax.set_xlabel("canali di uscita C")
            ax.set_xlim(1, 96)
            ax.set_xticks(range(8, 97, 8))
            ax.grid(axis="y", alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False, fontsize=9, title=ltitle, title_fontsize=9, loc="upper left")
        axes[0].set_ylabel("latenza mediana della rete di test (ms)")
        fig.suptitle("Probe onnxruntime su Raspberry Pi 5 (conv 3x3 su 20x20, 4 layer, 2.4 GHz); "
                     "linee verticali ogni 8 canali", fontsize=11)
        fig.tight_layout()
        fig.savefig(HERE / fname, dpi=150)
        plt.close(fig)
        print(HERE / fname)

    print("\n| precisione | " + " | ".join(CORES) + " |\n|---|" + "---|" * len(CORES))
    for q in QUANTS:
        print(f"| {q} | " + " | ".join(str(data[q, c]["n"]) for c in CORES) + " |")
    print("\nlatenza a C=96 (ms):")
    for q in QUANTS:
        print(f"| {q} | " + " | ".join(f"{data[q, c]['ms'][-1]:.3f}" for c in CORES) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo grafici e tabella")
    ap.add_argument("--after", type=int, help="aspetta la fine del processo con questo PID")
    args = ap.parse_args()
    if args.after:
        while Path(f"/proc/{args.after}").exists():
            time.sleep(30)
        print("probe precedente finito, parto", flush=True)
    if not args.report:
        probe()
    report()
