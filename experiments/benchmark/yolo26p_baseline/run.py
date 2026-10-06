"""Benchmark su Raspberry Pi 5 di yolo26p NON potato, per confronto con
experiments/rpi5_finetuned_musgd: stesso setup (OpenVINO fp32/fp16/int8, cpu_4, governor
performance a 2.4 GHz). Il modello e' il last.pt del training da zero con MuSGD
(experiments/ablation/train/lr_scheduler), preso a meta' training: il confronto riguarda la
latenza, che non dipende dall'epoca.

    .venv/bin/python experiments/rpi5_yolo26p_baseline/run_experiment.py           # tutto
    .venv/bin/python experiments/rpi5_yolo26p_baseline/run_experiment.py --report  # solo tabella
    ... --after PID    # aspetta che il processo PID (l'esperimento sul potato) finisca

Lo snapshot va in cache con una chiave sua (`+train.snapshot=epoch<N>` nel gruppo train): con
la chiave del training vero, gli export di questo modello a meta' verrebbero riusati per quello
finale. Il training in corso non viene toccato.
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

RUN = ROOT / "artifacts/weights/yolo26p_aod4_e250_13fa7d10/run"   # training scratch_musgd
QUANTS = ("fp32", "fp16", "int8")
STATE = HERE / "snapshot.json"        # epoca dello snapshot, fissata al primo lancio
RAM_LOG = HERE / "ram.json"
MIN_RAM_GB = 12                       # export OpenVINO (NNCF) con i training in corso


def common() -> list[str]:
    epoch = json.loads(STATE.read_text())["epoch"]
    return ["model=yolo26p", "train=scratch_musgd", f"+train.snapshot=epoch{epoch}",
            "strategy=baseline", "hardware=rpi5", "backend=openvino", "freq_target=max",
            "compute_target=cpu_4"]


def compose(quant: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *common(),
                                                  f"quantization={quant}"])


def register() -> None:
    """Copia last.pt (verificando che sia leggibile: il training lo riscrive a ogni epoca) e lo
    registra come training della repo, con la chiave dello snapshot."""
    import torch

    from src.cache import read_dataset_hash, training_key, update_index, weights_dir
    from src.jsonio import atomic_write_json

    if STATE.exists():
        out = weights_dir(compose("fp32"))
        if (out / "meta.json").exists():
            print(f"snapshot gia' registrato: {out.name}")
            return
    tmp = HERE / "last.pt.tmp"
    for _ in range(10):
        shutil.copy2(RUN / "weights/last.pt", tmp)
        try:
            ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
            break
        except Exception:      # copiato mentre il training lo scriveva: si riprova
            time.sleep(10)
    else:
        sys.exit("last.pt non leggibile")
    epoch = int(ckpt["epoch"]) + 1
    STATE.write_text(json.dumps({"epoch": epoch, "source": str(RUN / "weights/last.pt")}))
    cfg = compose("fp32")
    out = weights_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp, out / "best.pt")
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(RUN / "results.csv"))]
    row = next((r for r in rows if int(r["epoch"]) == epoch), rows[-1])
    meta = {
        "training_key": training_key(cfg), "model": "yolo26p", "dataset": "aod4",
        "epochs": int(cfg.train.epochs), "epochs_completed": epoch,
        "snapshot_of": "train=scratch_musgd (training in corso)", "source": str(RUN),
        "dataset_hash": read_dataset_hash(cfg.dataset.local_path),
        "metrics": {"map50": float(row["metrics/mAP50(B)"]),
                    "map50_95": float(row["metrics/mAP50-95(B)"])},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/weights/index.json", meta["training_key"], meta)
    print(f"snapshot registrato: {out.name} (epoca {epoch}, mAP50 {meta['metrics']['map50']:.3f})")


def ram_ok() -> None:
    ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1])
    if ram / 2**20 < MIN_RAM_GB:
        sys.exit(f"RAM disponibile {ram / 2**20:.0f} GB < {MIN_RAM_GB} GB: non lancio gli export")


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
            run("stage=benchmark", *common(), f"quantization={q}")
        finally:
            ram[q] = sampler.stop()
            RAM_LOG.write_text(json.dumps(ram, indent=1))
        print(f"RAM del Pi ({q}): {ram[q]}", flush=True)


def report() -> None:
    from src.cache import cell_key
    from src.jsonio import read_json

    ram = json.loads(RAM_LOG.read_text()) if RAM_LOG.exists() else {}
    print(f"snapshot: epoca {json.loads(STATE.read_text())['epoch']}\n")
    print("| precisione | latenza mediana (ms) | mAP50 | mAP50-95 | RAM prima / latenza (MB) | "
          "temp. inizio→fine (°C) |\n|---|---|---|---|---|---|")
    for q in QUANTS:
        r = read_json(ROOT / "results" / f"{cell_key(compose(q))}.json") or {}
        lat, acc, rt = r.get("latency") or {}, r.get("accuracy") or {}, r.get("runtime_state") or {}
        m = ram.get(q, {})
        fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
        print(f"| {q} | {fmt(lat.get('median_ms'))} | {fmt(acc.get('map50'), '{:.3f}')} | "
              f"{fmt(acc.get('map50_95'), '{:.3f}')} | {m.get('before_mb', '–')} / "
              f"{m.get('peak_latency_mb', '–')} | {fmt(rt.get('temp_start_c'), '{:.0f}')}→"
              f"{fmt(rt.get('temp_end_c'), '{:.0f}')} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo la tabella dei risultati")
    ap.add_argument("--after", type=int, help="aspetta la fine del processo con questo PID")
    args = ap.parse_args()
    if args.after:
        while Path(f"/proc/{args.after}").exists():
            time.sleep(30)
        print("esperimento precedente finito, parto", flush=True)
    if not args.report:
        ram_ok()
        register()
        run("-m", "stage=export", *common(), f"quantization={','.join(QUANTS)}")
        benchmark()
    report()
