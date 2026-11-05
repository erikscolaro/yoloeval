"""Separability probe on RPi5: latency of ONE conv vs (c_in, c_out), OpenVINO fp16, 4 cores, max freq.

Same config (stage=probe_edge), same board tuning (src.measure.board.tuned), same benchmark_app
command (backend.build_cmd + _wrap_command) as src/stages/probe.py. dummy.py only builds
c_in == c_out chains, so the nets are built here. Run with ITERS=200 (30 hits the warm-up transient).

  ITERS=200 python run.py pilot 20   # time per point on 20 points, estimate total
  ITERS=200 python run.py run        # full run, resumable (appends to results/raw.csv)
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # experiments/
from _lib import ROOT as REPO  # noqa: E402

OUT = Path(__file__).resolve().parent
ONNX_DIR = OUT / "runs" / "onnx"
DATA = OUT / "results"

FAMILIES = {"A": dict(k=1, hw=40), "B": dict(k=3, hw=20)}
REFS = (32, 64, 128)
GRID = (8, 16, 24, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65, 80, 96, 112, 127, 128)
ID_CIN = (1, 32, 64, 128)          # identity nets: fixed cost of benchmark_app
REPEATS, SEED = 3, 0
TEMP_MAX_C, TEMP_RESUME_C = 75.0, 65.0   # Pi 5 firmware throttles at 80 C: stay well below
FIELDS = ["family", "kind", "c_in", "c_out", "k", "hw", "macs", "repeat", "sequence", "ms",
          "mean_ms", "min_ms", "max_ms", "iters", "timestamp", "temp_c", "throttled",
          "freq_khz", "governor", "n_cores", "infer_precision", "openvino", "remeasured"]


def points() -> list[tuple]:
    """(family, kind, c_in, c_out), unique, in a fixed order."""
    pts = []
    for f in FAMILIES:
        s = set()
        for r in REFS:
            s |= {(r, co) for co in range(1, 129)} | {(ci, r) for ci in range(1, 129)}
        s |= {(ci, co) for ci in GRID for co in GRID}
        pts += [(f, "conv", ci, co) for ci, co in sorted(s)]
        pts += [(f, "identity", ci, ci) for ci in ID_CIN]
    return pts


def macs(f, kind, ci, co) -> int:
    if kind == "identity":
        return 0
    k, hw = FAMILIES[f]["k"], FAMILIES[f]["hw"]
    return co * (ci * k * k + 1) * hw * hw            # bias included, as dummy.costs


def onnx_path(f, kind, ci, co) -> Path:
    return ONNX_DIR / (f"{f}_id_ci{ci}.onnx" if kind == "identity" else f"{f}_ci{ci}_co{co}.onnx")


def build(f, kind, ci, co) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    p = onnx_path(f, kind, ci, co)
    if p.exists():
        return p
    k, hw = FAMILIES[f]["k"], FAMILIES[f]["hw"]
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ci, hw, hw])
    if kind == "identity":
        nodes, inits = [helper.make_node("Identity", ["input"], ["output"], name="output")], []
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ci, hw, hw])
    else:
        rng = np.random.default_rng([ord(f), ci, co])          # deterministic per net
        w = rng.normal(0, 1 / np.sqrt(ci * k * k), (co, ci, k, k)).astype(np.float32)
        b = rng.normal(0, 0.1, co).astype(np.float32)
        inits = [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")]
        nodes = [helper.make_node("Conv", ["input", "w", "b"], ["output"], name="conv",
                                  kernel_shape=[k, k], pads=[k // 2] * 4, strides=[1, 1])]
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, co, hw, hw])
    g = helper.make_graph(nodes, p.stem, [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    p.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(p))
    return p


def make_cfg():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        return compose("config", overrides=[
            "stage=probe_edge", "hardware=rpi5", "backend=openvino", "quantization=fp16",
            "freq_target=max", "compute_target=cpu_4", f"project_root={REPO}",
            "stage.thermal.threshold_c=60"])        # idle Pi is ~53 C: 50 would always time out


STATE = ("; rc=$?; echo \"STATE $(cat /sys/class/thermal/thermal_zone0/temp) "
         "$(sudo -n vcgencmd get_throttled) "
         "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) "
         "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)\"; exit $rc")


def parse_state(stdout: str) -> dict:
    line = next(l for l in stdout.splitlines() if l.startswith("STATE "))
    t, thr, fq, gov = line.split()[1:5]
    return {"temp_c": int(t) / 1000, "throttled": thr.split("=")[-1], "freq_khz": int(fq),
            "governor": gov}


def run(order: list[tuple], csv_path: Path, cfg) -> None:
    from omegaconf import read_write
    from src.backends import get_backend
    from src.measure.board import board_controller, tuned
    from src.measure.compute import resolve_compute
    from src.remote.connection import connection
    from src.remote.sync import _rsync, ensure_support_files
    from src.stages.benchmark import _effective_cores, _ensure_profile, _wrap_command

    with read_write(cfg):
        cfg.backend.benchmark.iters = int(os.environ.get("ITERS", cfg.stage.iters))  # default: as probe.py
    backend = get_backend(cfg.backend.name, cfg)
    precision = cfg.backend.benchmark.infer_precision[cfg.quantization.precision]
    for _, f, kind, ci, co in order:
        build(f, kind, ci, co)
    remote_dir = f"{cfg.hardware.remote.workdir}/sep_probe"
    done = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            done = {(int(r["repeat"]), r["family"], r["kind"], int(r["c_in"]), int(r["c_out"]))
                    for r in csv.DictReader(fh)}
    todo = [o for o in order if o not in done]
    print(f"{len(todo)} misure da fare ({len(done)} gia' fatte)", flush=True)
    new = not csv_path.exists()
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {remote_dir}", hide=True)
        _rsync(f"{ONNX_DIR}/", f"{cfg.hardware.remote.host}:{remote_dir}/")   # once, not per point
        ov = conn.run(f"{cfg.hardware.remote.python} -c 'import openvino;print(openvino.__version__)'",
                      hide=True).stdout.strip()
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        ensure_support_files(live, cfg)
        with tuned(live, cfg, bc) as applied, open(csv_path, "a", newline="") as fh:
            print("tuning:", {k: applied.get(k) for k in ("governor", "freq_actual_khz",
                                                          "cores_online", "temp_start_c")}, flush=True)
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            ct = resolve_compute(cfg)
            n_cores = _effective_cores(live, cfg, bc, ct)
            t0 = time.monotonic()
            for i, (rep, f, kind, ci, co) in enumerate(todo):
                cmd = _wrap_command(live, cfg, backend.build_cmd(
                    cfg, Path(f"{remote_dir}/{onnx_path(f, kind, ci, co).name}")), ct, n_cores)
                remeasured = 0
                while True:
                    r = live.run(cmd + STATE, hide=True, warn=True)
                    if r.failed:
                        raise RuntimeError(f"{f} {kind} {ci}x{co}: {(r.stderr or r.stdout)[-800:]}")
                    st = parse_state(r.stdout)
                    hot = st["temp_c"] >= TEMP_MAX_C or int(st["throttled"], 16) & 0xF
                    if not hot:
                        break
                    remeasured += 1           # too hot or throttling now: cool down and redo
                    print(f"caldo ({st}), attendo", flush=True)
                    while bc.read_temp(live) > TEMP_RESUME_C:
                        time.sleep(10)
                lat = backend.parse(r.stdout)
                k, hw = FAMILIES[f]["k"], FAMILIES[f]["hw"]
                w.writerow({"family": f, "kind": kind, "c_in": ci, "c_out": co, "k": k, "hw": hw,
                            "macs": macs(f, kind, ci, co), "repeat": rep, "sequence": i,
                            "ms": lat.median_ms, "mean_ms": lat.mean_ms, "min_ms": lat.min_ms,
                            "max_ms": lat.max_ms, "iters": lat.iters,
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            **st, "n_cores": n_cores, "infer_precision": precision,
                            "openvino": ov, "remeasured": remeasured})
                fh.flush()
                if i and i % 100 == 0:
                    el = time.monotonic() - t0
                    print(f"{i}/{len(todo)} {el / i:.2f} s/misura, ~{el / i * (len(todo) - i) / 3600:.1f} h"
                          f" alla fine, {st['temp_c']:.1f} C thr={st['throttled']}", flush=True)
            loop = time.monotonic() - t0
            print(f"fine: {len(todo)} misure in {loop / 60:.1f} min", flush=True)
    return loop / max(1, len(todo))


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    DATA.mkdir(parents=True, exist_ok=True)
    pts = points()
    cfg = make_cfg()
    if sys.argv[1] == "pilot":
        n = int(sys.argv[2])
        sample = random.Random(123).sample(pts, n)
        csv_path = DATA / "pilot.csv"
        csv_path.unlink(missing_ok=True)
        per = run([(0, *p) for p in sample], csv_path, cfg)
        total = len(pts) * REPEATS
        print(f"PILOT: {per:.2f} s/misura; {len(pts)} punti x {REPEATS} = {total} misure -> "
              f"stima {per * total / 3600:.2f} h (+ build/rsync iniziale)", flush=True)
        return
    rng = random.Random(SEED)
    order = []
    for rep in range(REPEATS):
        p = pts[:]
        rng.shuffle(p)                 # new order at every repeat, fixed seed
        order += [(rep, *x) for x in p]
    run(order, DATA / "raw.csv", cfg)


if __name__ == "__main__":
    main()
