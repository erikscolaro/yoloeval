"""Benchmark su Raspberry Pi 5 del yolo26p potato (search `pit_duccio_pico`) e finetunato
con la recipe `finetune` di default (MuSGD, experiments/finetune/lr_scheduler): OpenVINO in
fp32, fp16 e int8, tutti i core (cpu_4), governor performance a 2.4 GHz (freq_target=max).
I pesi sono quelli della pipeline (`src.cache.final_weights`).

    .venv/bin/python experiments/benchmark/pruned_finetuned/run.py           # tutto
    .venv/bin/python experiments/benchmark/pruned_finetuned/run.py --report  # solo tabella

Prima, dalla root: `run.py stage=train`, `stage=adapt`, `stage=search` e `stage=finetune` con gli
override di COMMON (se i pesi mancano lo script lo dice ed esce).

1. controlla che il finetune sia in cache;
2. export OpenVINO (run.py stage=export);
3. benchmark sul Pi (run.py stage=benchmark), una precisione per volta, campionando la RAM del
   Pi ogni secondo: prima del lancio, durante la latenza (benchmark_app) e durante la mAP
   (eval_map.py). Indicazione grezza: e' la RAM usata da tutto il sistema.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/, per _lib
from _lib import ROOT, PiRam, make_cfg, weights_or_exit  # noqa: E402

HERE = Path(__file__).resolve().parent

QUANTS = ("fp32", "fp16", "int8")
# override della pipeline che identificano il modello (N dal probe rpi5 a 4 core: la search e'
# condivisa con gli altri esperimenti sul potato)
MODEL = ["model=yolo26p", "train=scratch_musgd", "strategy=pit_duccio_pico",
         "strategy.n_source=cpu_4", "strategy.n_quantization=int8", "hardware=rpi5", "quantization=int8"]
COMMON = [*MODEL, "backend=openvino", "freq_target=max", "compute_target=cpu_4"]
RESULTS = HERE / "results"
RAM_LOG = RESULTS / "ram.json"


def compose(quant: str):
    return make_cfg(*COMMON, f"quantization={quant}")


def run(*args: str) -> None:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", *args]
    print(">>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def benchmark() -> None:
    ram = json.loads(RAM_LOG.read_text()) if RAM_LOG.exists() else {}
    for q in QUANTS:
        sampler = PiRam()
        sampler.start()
        time.sleep(5)                      # RAM prima del lancio
        try:
            run("stage=benchmark", *COMMON, f"quantization={q}")
        finally:
            ram[q] = sampler.stop()
            RESULTS.mkdir(exist_ok=True)
            RAM_LOG.write_text(json.dumps(ram, indent=1))
        print(f"RAM del Pi ({q}): {ram[q]}", flush=True)


def report() -> None:
    from src.cache import cell_key
    from src.jsonio import read_json

    ram = json.loads(RAM_LOG.read_text()) if RAM_LOG.exists() else {}
    print("| precisione | latenza mediana (ms) | p90 (ms) | mAP50 | mAP50-95 | "
          "RAM prima / latenza / mAP (MB) | temp. inizio→fine (°C) |\n|---|---|---|---|---|---|---|")
    for q in QUANTS:
        r = read_json(ROOT / "results" / f"{cell_key(compose(q))}.json") or {}
        lat, acc, rt = r.get("latency") or {}, r.get("accuracy") or {}, r.get("runtime_state") or {}
        m = ram.get(q, {})
        fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
        print(f"| {q} | {fmt(lat.get('median_ms'))} | {fmt(lat.get('p90_ms'))} | "
              f"{fmt(acc.get('map50'), '{:.3f}')} | {fmt(acc.get('map50_95'), '{:.3f}')} | "
              f"{m.get('before_mb', '–')} / {m.get('peak_latency_mb', '–')} / "
              f"{m.get('peak_map_mb', '–')} | {fmt(rt.get('temp_start_c'), '{:.0f}')}→"
              f"{fmt(rt.get('temp_end_c'), '{:.0f}')} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo la tabella dei risultati")
    args = ap.parse_args()
    if not args.report:
        print(f"pesi: {weights_or_exit(*COMMON, 'quantization=fp32')}")
        run("-m", "stage=export", *COMMON, f"quantization={','.join(QUANTS)}")
        benchmark()
    report()
