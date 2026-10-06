"""Benchmark su Raspberry Pi 5 del yolo26p potato e finetunato con MuSGD
(experiments/ablation/finetune/lr_scheduler): OpenVINO in fp32, fp16 e int8, tutti i core
(cpu_4), governor performance a 2.4 GHz (freq_target=max).

    .venv/bin/python experiments/rpi5_finetuned_musgd/run_experiment.py           # tutto
    .venv/bin/python experiments/rpi5_finetuned_musgd/run_experiment.py --report  # solo tabella

1. registra il modello nella cache della repo come finetune della strategia
   `pit_duccio_pico_ep58` (search interrotta all'epoca 58, N=8 fisso: nessun probe);
2. export OpenVINO (run.py stage=export);
3. benchmark sul Pi (run.py stage=benchmark), una precisione per volta, campionando la RAM del
   Pi ogni secondo: prima del lancio, durante la latenza (benchmark_app) e durante la mAP
   (eval_map.py). Indicazione grezza: e' la RAM usata da tutto il sistema.
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "outputs/sweep_yolo26p/musgd_test/run"           # finetune MuSGD
QUANTS = ("fp32", "fp16", "int8")
COMMON = ["model=yolo26p", "train=scratch", "finetune=musgd", "strategy=pit_duccio_pico",
          "strategy.name=pit_duccio_pico_ep58", "strategy.search.pit.n=8", "hardware=rpi5",
          "backend=openvino", "freq_target=max", "compute_target=cpu_4"]
RAM_LOG = HERE / "ram.json"


def compose(quant: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *COMMON,
                                                  f"quantization={quant}"])


def register() -> None:
    """Il best.pt del finetune MuSGD come artefatto di finetune della repo."""
    import csv

    from src.cache import (finetune_dir, finetune_key, search_key, training_key, update_index,
                           yolopit_version)
    from src.jsonio import atomic_write_json

    cfg = compose("fp32")
    fkey, out = finetune_key(cfg), finetune_dir(cfg)
    if (out / "meta.json").exists():
        print(f"finetune gia' registrato: {out.name}")
        return
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(SOURCE / "results.csv"))]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "weights/best.pt", out / "best.pt")
    meta = {
        "finetune_key": fkey, "search_key": search_key(cfg), "training_key": training_key(cfg),
        "strategy": "pit_duccio_pico_ep58", "yolopit": yolopit_version(),
        "source": str(SOURCE), "note": "search pit_duccio_pico interrotta all'epoca 58 (N=8), "
                                       "finetune MuSGD 100 epoche: experiments/ablation/finetune",
        "metrics": {"map50": float(rows[-1]["metrics/mAP50(B)"]),
                    "map50_95": float(rows[-1]["metrics/mAP50-95(B)"])},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/finetune/index.json", fkey, meta)
    print(f"finetune registrato: {out.name}")


def run(*args: str) -> None:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", *args]
    print(">>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


class PiRam(threading.Thread):
    """Campiona sul Pi ogni secondo: MB usati, MB disponibili, benchmark_app attivo, eval attivo."""
    LOOP = ("while true; do m=$(free -m | awk 'NR==2{print $3, $7}'); "
            "l=$(pgrep -c benchmark_app); a=$(pgrep -cf '[e]val_map.py'); "
            "echo \"$m $l $a\"; sleep 1; done")

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "rpi5", self.LOOP],
                                     stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 4:
                self.samples.append([int(x) for x in parts])

    def stop(self) -> dict:
        self.proc.terminate()
        used = lambda pick: max((s[0] for s in self.samples if pick(s)), default=None)
        return {"before_mb": self.samples[0][0] if self.samples else None,
                "peak_latency_mb": used(lambda s: s[2] > 0),
                "peak_map_mb": used(lambda s: s[3] > 0),
                "peak_mb": used(lambda s: True),
                "min_available_mb": min((s[1] for s in self.samples), default=None)}


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
        register()
        run("-m", "stage=export", *COMMON, f"quantization={','.join(QUANTS)}")
        benchmark()
    report()
