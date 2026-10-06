"""Benchmark su Raspberry Pi 5 del migliore (mAP50-95) tra finetune e scratch di
experiments/ablation/pruned/scratch_vs_finetune: onnxruntime e OpenVINO, fp32 e fp16, ai tre
punti di frequenza della board (max 2.4 GHz performance, mid 1.5 GHz, low 1.0 GHz), 4 core,
attesa termica fino a 50 °C.

    .venv/bin/python experiments/rpi5_pruned_500ep/run_experiment.py --after PID  # dopo l'ablation
    .venv/bin/python experiments/rpi5_pruned_500ep/run_experiment.py --report     # solo tabella

Il modello va in cache come finetune con strategia `pit_duccio_pico_500ep` e N=8 fisso (quello
della search, dal probe rpi5 int8): con `pit.n: auto` e fp32 la pipeline cercherebbe un probe
fp32 che non c'e'. Export e benchmark passano da `run.py`.
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

ABL = ROOT / "experiments/ablation/pruned/scratch_vs_finetune"
STATE = HERE / "chosen.json"
BACKENDS, QUANTS, FREQS = ("onnxruntime", "openvino"), ("fp32", "fp16"), ("max", "mid", "low")
MAP95 = "metrics/mAP50-95(B)"


def best_of(variant: str) -> dict:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(ABL / variant / "results.csv"))]
    b = max(rows, key=lambda r: float(r[MAP95]))
    return {"variant": variant, "epoch": int(b["epoch"]), "map50": float(b["metrics/mAP50(B)"]),
            "map50_95": float(b[MAP95]), "epochs_done": int(rows[-1]["epoch"])}


def common() -> list[str]:
    v = json.loads(STATE.read_text())["variant"]
    return ["model=yolo26p", "train=scratch_musgd", "+train.extra=close_mosaic15",
            "finetune=musgd", "finetune.epochs=500", "finetune.close_mosaic=50",
            "finetune.lr0=0.01", "finetune.warmup_epochs=3", f"+finetune.init={v}",
            "strategy=pit_duccio_pico", "strategy.name=pit_duccio_pico_500ep",
            "strategy.search.pit.n=8", "hardware=rpi5", "compute_target=cpu_4"]


def compose(*extra: str):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return hydra_compose("config", overrides=[f"project_root={ROOT}", *common(), *extra])


def choose() -> None:
    if STATE.exists():
        return
    cands = [best_of(v) for v in ("finetune", "scratch")]
    for c in cands:
        if c["epochs_done"] != 500:
            sys.exit(f"{c['variant']} all'epoca {c['epochs_done']}, non a 500")
    chosen = max(cands, key=lambda c: c["map50_95"])
    STATE.write_text(json.dumps({**chosen, "candidates": cands}, indent=1))
    print(f"scelto: {chosen['variant']} (mAP50-95 {chosen['map50_95']:.4f}, epoca {chosen['epoch']})",
          flush=True)


def register() -> None:
    from src.cache import (finetune_dir, finetune_key, search_key, training_key, update_index,
                           yolopit_version)
    from src.jsonio import atomic_write_json

    chosen = json.loads(STATE.read_text())
    cfg = compose("quantization=fp32", "backend=openvino")
    fkey, out = finetune_key(cfg), finetune_dir(cfg)
    if (out / "meta.json").exists():
        print(f"finetune gia' registrato: {out.name}", flush=True)
        return
    src = ABL / chosen["variant"] / "weights/best.pt"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / "best.pt")
    meta = {
        "finetune_key": fkey, "search_key": search_key(cfg), "training_key": training_key(cfg),
        "strategy": "pit_duccio_pico_500ep", "yolopit": yolopit_version(), "source": str(src),
        "note": f"{chosen['variant']} di experiments/ablation/pruned/scratch_vs_finetune "
                "(search artifacts/search/yolo26p_aod4_pit_duccio_pico_5b5558e3, N=8)",
        "metrics": {"map50": chosen["map50"], "map50_95": chosen["map50_95"]},
    }
    atomic_write_json(out / "meta.json", meta)
    update_index(ROOT / "artifacts/finetune/index.json", fkey, meta)
    print(f"finetune registrato: {out.name}", flush=True)


def run(*args: str) -> None:
    cmd = [str(ROOT / ".venv/bin/python"), "run.py", *args]
    print(">>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)   # una cella fallita non ferma le altre


def report() -> None:
    from src.cache import cell_key
    from src.jsonio import read_json

    chosen = json.loads(STATE.read_text())
    print(f"modello: {chosen['variant']} (mAP50-95 {chosen['map50_95']:.3f} nel val del training)\n")
    print("| backend | precisione | frequenza | latenza mediana (ms) | FPS | max (ms) | mAP50 | "
          "mAP50-95 | temp. inizio→fine (°C) |\n|---|---|---|---|---|---|---|---|---|")
    fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
    for b in BACKENDS:
        for q in QUANTS:
            for f in FREQS:
                r = read_json(ROOT / "results" / f"{cell_key(compose(f'backend={b}', f'quantization={q}', f'freq_target={f}'))}.json") or {}
                lat, acc, rt = r.get("latency") or {}, r.get("accuracy") or {}, r.get("runtime_state") or {}
                med = lat.get("median_ms")
                print(f"| {b} | {q} | {f} | {fmt(med)} | {fmt(1000 / med if med else None, '{:.1f}')} | "
                      f"{fmt(lat.get('max_ms'))} | {fmt(acc.get('map50'), '{:.3f}')} | "
                      f"{fmt(acc.get('map50_95'), '{:.3f}')} | {fmt(rt.get('temp_start_c'), '{:.0f}')}→"
                      f"{fmt(rt.get('temp_end_c'), '{:.0f}')} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="solo la tabella")
    ap.add_argument("--after", type=int, help="aspetta la fine del processo con questo PID")
    args = ap.parse_args()
    if args.after:
        while Path(f"/proc/{args.after}").exists():
            time.sleep(60)
        print("ablation finita, parto", flush=True)
    if not args.report:
        choose()
        register()
        run("-m", "stage=export", *common(), f"backend={','.join(BACKENDS)}",
            f"quantization={','.join(QUANTS)}")
        run("-m", "stage=benchmark", *common(), f"backend={','.join(BACKENDS)}",
            f"quantization={','.join(QUANTS)}", f"freq_target={','.join(FREQS)}",
            "stage.thermal.threshold_c=50")
    report()
