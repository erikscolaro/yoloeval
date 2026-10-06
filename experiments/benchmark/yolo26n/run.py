"""Benchmark su Raspberry Pi 5 di yolo26n allenato su AOD4 (dai pesi COCO, 250 epoche, run della
tesi), come riferimento per experiments/rpi5_finetuned_musgd e rpi5_yolo26p_baseline: stesso
setup (OpenVINO, cpu_4, governor performance a 2.4 GHz), solo fp32.

    .venv/bin/python experiments/rpi5_yolo26n/run_experiment.py           # tutto
    .venv/bin/python experiments/rpi5_yolo26n/run_experiment.py --report  # solo tabella

I pesi vanno in cache con una chiave loro (`+train.source=thesis_baseline` nel gruppo train):
il training non e' stato fatto con conf/train/, e con la chiave di train=default gli export
verrebbero riusati per un yolo26n allenato dalla repo.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rpi5_finetuned_musgd"))

from run_experiment import PiRam  # noqa: E402  stesso campionatore della RAM del Pi

RUN = Path("/home/scolaroe/Desktop/thesis/runs/baseline-yolo26n/train")
QUANTS = ("fp32",)
COMMON = ["model=yolo26n", "train=default", "+train.source=thesis_baseline", "strategy=baseline",
          "hardware=rpi5", "backend=openvino", "freq_target=max", "compute_target=cpu_4"]
RAM_LOG = HERE / "ram.json"


def compose(quant: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *COMMON,
                                                  f"quantization={quant}"])


def register() -> None:
    """best.pt della run della tesi come training della repo, con la sua chiave."""
    from src.cache import read_dataset_hash, training_key, update_index, weights_dir
    from src.jsonio import atomic_write_json

    cfg = compose("fp32")
    out = weights_dir(cfg)
    if (out / "meta.json").exists():
        print(f"pesi gia' registrati: {out.name}")
        return
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(RUN / "results.csv"))]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN / "weights/best.pt", out / "best.pt")
    meta = {
        "training_key": training_key(cfg), "model": "yolo26n", "dataset": "aod4",
        "epochs": 250, "epochs_completed": int(rows[-1]["epoch"]),
        "source": str(RUN), "note": "run della tesi (baseline-yolo26n): pesi COCO, 250 epoche",
        "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
        "metrics": {"map50": max(float(r["metrics/mAP50(B)"]) for r in rows),
                    "map50_95": max(float(r["metrics/mAP50-95(B)"]) for r in rows)},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/weights/index.json", meta["training_key"], meta)
    print(f"pesi registrati: {out.name}")


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
            RAM_LOG.write_text(json.dumps(ram, indent=1))
        print(f"RAM del Pi ({q}): {ram[q]}", flush=True)


def report() -> None:
    from src.cache import cell_key
    from src.jsonio import read_json

    ram = json.loads(RAM_LOG.read_text()) if RAM_LOG.exists() else {}
    print("| precisione | latenza mediana (ms) | media (ms) | max (ms) | FPS medi | mAP50 | mAP50-95 | "
          "RAM prima / latenza (MB) | temp. inizio→fine (°C) |\n|---|---|---|---|---|---|---|---|---|")
    for q in QUANTS:
        r = read_json(ROOT / "results" / f"{cell_key(compose(q))}.json") or {}
        lat, acc, rt = r.get("latency") or {}, r.get("accuracy") or {}, r.get("runtime_state") or {}
        m = ram.get(q, {})
        fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
        print(f"| {q} | {fmt(lat.get('median_ms'))} | {fmt(lat.get('mean_ms'))} | "
              f"{fmt(lat.get('max_ms'))} | {fmt(lat.get('throughput_qps'), '{:.1f}')} | "
              f"{fmt(acc.get('map50'), '{:.3f}')} | {fmt(acc.get('map50_95'), '{:.3f}')} | "
              f"{m.get('before_mb', '–')} / {m.get('peak_latency_mb', '–')} | "
              f"{fmt(rt.get('temp_start_c'), '{:.0f}')}→{fmt(rt.get('temp_end_c'), '{:.0f}')} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo la tabella dei risultati")
    args = ap.parse_args()
    if not args.report:
        register()
        run("-m", "stage=export", *COMMON, f"quantization={','.join(QUANTS)}")
        benchmark()
    report()
