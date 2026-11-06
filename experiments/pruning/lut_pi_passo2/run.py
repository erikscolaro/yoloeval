"""2D latency LUT, step 2, of the prunable conv families of YOLO26 N (0.25, D126) at 640x640 on the RPi 5.

OpenVINO fp16 (-infer_precision f16), 4 cores, governor performance, 2.4 GHz: same config, board tuning
(src.measure.board.tuned) and benchmark_app command (backend.build_cmd + _wrap_command) as
src/stages/probe.py and pruning/separability. One conv + bias + SiLU per net (as the exported Conv with BN
fused); no identity nets (DECISIONE_IDENTITA.md). Input, output, weights and bias are fp16, the precision used at
inference: no f32<->f16 Convert in the compiled graph (that belongs to the preprocessing). Input and output
tensors are NHWC (PrePostProcessor, saved as IR), the layout the ARM CPU plugin uses inside: no NCHW<->NHWC
Reorder either. At every inference only the Convolution and the SiLU Subgraph run. benchmark_app runs through
bench_dump.py, which also prints the latency of every iteration: saved in runs/latencies_<raw|pilot>.txt, one line per
measure, so the LUT can be re-aggregated later. Each net is built here and copied to the Pi right before its
measurement (70k nets: no bulk copy).

  ITERS=200 python run.py pilot 100   # time per measure on 100 random points, estimate of the total
  ITERS=200 python run.py run         # full run, resumable: skips what is already in results/raw.csv;
                                      # then the unstable points again (see unstable())
bench_dump.py repeats benchmark_app's single warm-up inference for 60 ms (warmup.py, README).
"""
from __future__ import annotations

import csv
import io
import json
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
RES = OUT / "results"

# family -> (kernel, stride, output map, layers as (c_in max, c_out max)); from NEW_MASTERPLAN.md, 4a, "Strati potabili del nano e griglia"
FAMILIES = {
    "k1s1_20": (1, 1, 20, [(128, 64), (128, 128), (256, 128), (256, 256), (384, 256), (512, 256)]),
    "k1s1_40": (1, 1, 40, [(64, 32), (64, 64), (128, 128), (192, 128), (384, 128)]),
    "k1s1_80": (1, 1, 80, [(32, 16), (32, 32), (64, 64), (96, 64), (96, 128), (256, 64)]),
    "k1s1_160": (1, 1, 160, [(32, 32), (48, 64)]),
    "k3s1_20": (3, 1, 20, [(64, 64), (64, 128), (128, 64)]),
    "k3s1_40": (3, 1, 40, [(32, 32)]),
    "k3s1_80": (3, 1, 80, [(16, 16), (16, 32), (32, 16)]),
    "k3s1_160": (3, 1, 160, [(8, 16), (16, 8)]),
    "k3s2_20": (3, 2, 20, [(128, 128), (128, 256)]),
    "k3s2_40": (3, 2, 40, [(64, 64), (128, 128)]),
    "k3s2_80": (3, 2, 80, [(64, 64)]),
    "k3s2_160": (3, 2, 160, [(16, 32)]),
    "k3s2_320": (3, 2, 320, [(3, 16)]),          # stem: c_in is the image, fixed at 3
}
FIXED_CIN = {"k3s2_320"}
N_CONV = 69989
REPEATS, SEED = 3, 0
UNSTABLE_REL, EXTRA_REPEATS = 0.05, 6      # (max-min)/median of the first REPEATS above this: 6 more repeats
TEMP_MAX_C, TEMP_RESUME_C = 75.0, 65.0         # guard: firmware throttles at 80 C; with the fan at max it never trips
CHECKPOINT_S = 6 * 3600
FIELDS = ["family", "kind", "kernel", "stride", "map", "c_in", "c_out", "repeat", "sequence", "ms",
          "mean_ms", "min_ms", "max_ms", "iters", "temp_c", "freq_khz", "throttled", "governor",
          "timestamp", "n_cores", "infer_precision", "openvino", "remeasured"]


def axis(m: int) -> list[int]:
    return sorted({1, m, *range(2, m + 1, 2)})       # 1, even values, the max


def in_map(f: str) -> int:
    _, s, hw, _ = FAMILIES[f]
    return hw * s


def points() -> list[tuple]:
    """(family, "conv", c_in, c_out) in a fixed order. No identity nets: the per-call fixed cost is the same
    for every layer and all layers always exist, so the constant C absorbs it (README, Costo fisso)."""
    pts = []
    for f, (_, _, _, layers) in FAMILIES.items():
        s = set()
        for ci, co in layers:
            s |= {(a, b) for a in ([ci] if f in FIXED_CIN else axis(ci)) for b in axis(co)}
        pts += [(f, "conv", ci, co) for ci, co in sorted(s)]
    assert len(pts) == N_CONV
    return pts


def geometry(f: str, kind: str) -> tuple[int, int, int]:
    """kernel, stride, map: output map for convs, input map for identities."""
    return (0, 1, int(f[2:])) if kind == "identity" else FAMILIES[f][:3]


def build(f, kind, ci, co) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    FP16 = TensorProto.FLOAT16
    if kind == "identity":
        hw = int(f[2:])
        inp = helper.make_tensor_value_info("input", FP16, [1, ci, hw, hw])
        out = helper.make_tensor_value_info("output", FP16, [1, ci, hw, hw])
        nodes, inits = [helper.make_node("Identity", ["input"], ["output"], name="output")], []
    else:
        k, s, hw, _ = FAMILIES[f]
        inp = helper.make_tensor_value_info("input", FP16, [1, ci, hw * s, hw * s])
        out = helper.make_tensor_value_info("output", FP16, [1, co, hw, hw])
        rng = np.random.default_rng([k, s, hw, ci, co])        # deterministic per net
        w = rng.normal(0, 1 / np.sqrt(ci * k * k), (co, ci, k, k)).astype(np.float16)
        b = rng.normal(0, 0.1, co).astype(np.float16)
        inits = [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")]
        nodes = [helper.make_node("Conv", ["input", "w", "b"], ["y"], name="conv", kernel_shape=[k, k],
                                  pads=[k // 2] * 4, strides=[s, s]),
                 helper.make_node("Sigmoid", ["y"], ["sig"], name="sigmoid"),     # SiLU as exported
                 helper.make_node("Mul", ["y", "sig"], ["output"], name="silu")]
    g = helper.make_graph(nodes, f"{f}_{kind}_{ci}_{co}", [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()


def to_ir(onnx_bytes: bytes, nhwc: bool = True) -> tuple[bytes, bytes]:
    """ONNX (NCHW) -> OpenVINO IR, with NHWC input/output tensors if nhwc: (xml, bin).

    Identity nets stay NCHW: with nothing in between the PrePostProcessor Transposes are not absorbed."""
    import tempfile
    import openvino as ov
    from openvino.preprocess import PrePostProcessor
    m = ov.Core().read_model(onnx_bytes, b"")
    if nhwc:
        pp = PrePostProcessor(m)
        for t in (pp.input(), pp.output()):
            t.tensor().set_layout(ov.Layout("NHWC"))
            t.model().set_layout(ov.Layout("NCHW"))
        m = pp.build()
    with tempfile.TemporaryDirectory() as d:
        ov.save_model(m, f"{d}/net.xml", compress_to_fp16=False)   # already fp16
        return Path(f"{d}/net.xml").read_bytes(), Path(f"{d}/net.bin").read_bytes()


def make_cfg():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        return compose("config", overrides=[
            "stage=probe_edge", "hardware=rpi5", "backend=openvino", "quantization=fp16",
            "freq_target=max", "compute_target=cpu_4", f"project_root={REPO}"])  # start threshold 50 C, as in the deploy benchmark


STATE = ("; rc=$?; echo \"STATE $(cat /sys/class/thermal/thermal_zone0/temp) "
         "$(sudo -n vcgencmd get_throttled) "
         "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) "
         "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)\"; exit $rc")


def parse_state(stdout: str) -> dict:
    line = next(l for l in stdout.splitlines() if l.startswith("STATE "))
    t, thr, fq, gov = line.split()[1:5]
    return {"temp_c": int(t) / 1000, "throttled": thr.split("=")[-1], "freq_khz": int(fq), "governor": gov}


def env_info(conn, cfg) -> dict:
    py = cfg.hardware.remote.python
    q = lambda c: conn.run(c, hide=True, warn=True).stdout.strip()       # noqa: E731
    return {"openvino": q(f"{py} -c 'import openvino;print(openvino.__version__)'"),
            "os": q(". /etc/os-release; echo $PRETTY_NAME"), "kernel": q("uname -r"),
            "model": q("tr -d '\\0' < /proc/device-tree/model"), "firmware": q("sudo -n vcgencmd version | tail -1")}


def write_status(csv_path: Path, t0: float, n0: int, n_new: int, total: int, path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    done = len(rows)
    per = (time.monotonic() - t0) / max(1, n_new)
    st = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "done": done, "total": total,
          "s_per_measure_this_session": round(per, 3), "eta_h": round(per * (total - done) / 3600, 2),
          "remeasured": sum(int(r["remeasured"]) for r in rows),
          "throttled_nonzero": sum(r["throttled"] != "0x0" for r in rows),
          "temp_max_c": max((float(r["temp_c"]) for r in rows), default=None),
          "freq_khz": sorted({int(r["freq_khz"]) for r in rows}), "started_this_session_at": n0}
    path.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st


def unstable(csv_path: Path) -> list[tuple]:
    """Points whose first REPEATS measures spread more than UNSTABLE_REL: two regimes per process seen on
    k1s1_160 48x64 (bimodal.py), so a median of 3 can pick either. Uses only repeats < REPEATS, so the set
    does not change when the extra repeats are resumed after a restart."""
    by = {}
    for r in csv.DictReader(open(csv_path)):
        if int(r["repeat"]) < REPEATS:
            by.setdefault((r["family"], r["kind"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    return sorted(k for k, v in by.items()
                  if len(v) == REPEATS and (max(v) - min(v)) / float(np.median(v)) > UNSTABLE_REL)


def run(order: list[tuple], csv_path: Path, cfg, status_path: Path | None = None) -> float:
    from omegaconf import read_write
    from src.backends import get_backend
    from src.measure.board import board_controller, tuned
    from src.measure.compute import resolve_compute
    from src.remote.connection import connection
    from src.remote.sync import ensure_support_files
    from src.stages.benchmark import _effective_cores, _ensure_profile, _wrap_command

    with read_write(cfg):
        cfg.backend.benchmark.iters = int(os.environ.get("ITERS", cfg.stage.iters))
    backend = get_backend(cfg.backend.name, cfg)
    precision = cfg.backend.benchmark.infer_precision[cfg.quantization.precision]
    remote_dir = f"{cfg.hardware.remote.workdir}/lut_pi"
    net = f"{remote_dir}/net.xml"
    done = set()
    if csv_path.exists():
        done = {(int(r["repeat"]), r["family"], r["kind"], int(r["c_in"]), int(r["c_out"]))
                for r in csv.DictReader(open(csv_path))}
    todo = [o for o in order if o not in done]
    print(f"{len(todo)} misure da fare ({len(done)} gia' fatte)", flush=True)
    new = not csv_path.exists()
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {remote_dir}", hide=True)
        env = env_info(conn, cfg)
        (RES / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        ensure_support_files(live, cfg)
        live.put(str(OUT / "bench_dump.py"), f"{remote_dir}/bench_dump.py")
        with tuned(live, cfg, bc) as applied, open(csv_path, "a", newline="") as fh:
            print("tuning:", {k: applied.get(k) for k in ("governor", "freq_actual_khz", "cores_online",
                                                          "temp_start_c")}, env, flush=True)
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            ct = resolve_compute(cfg)
            n_cores = _effective_cores(live, cfg, bc, ct)
            bench = backend.build_cmd(cfg, Path(net))
            tool = bench.split()[0]
            assert tool.endswith("/benchmark_app"), bench
            bench = bench.replace(tool, f"{cfg.hardware.remote.python} {remote_dir}/bench_dump.py", 1)
            cmd = _wrap_command(live, cfg, bench, ct, n_cores) + STATE
            print("comando:", cmd, flush=True)
            lat_path = OUT / "runs" / f"latencies_{csv_path.stem}.txt"   # rep,family,kind,c_in,c_out,<ms per iter>
            lat_path.parent.mkdir(exist_ok=True)
            if new:
                lat_path.unlink(missing_ok=True)
            lat_fh = open(lat_path, "a")
            t0 = last_ck = time.monotonic()
            for i, (rep, f, kind, ci, co) in enumerate(todo):
                xml, bin_ = to_ir(build(f, kind, ci, co), nhwc=kind == "conv")
                live.put(io.BytesIO(xml), net)
                live.put(io.BytesIO(bin_), net[:-4] + ".bin")
                remeasured = 0
                while True:
                    r = live.run(cmd, hide=True, warn=True)
                    if r.failed:
                        raise RuntimeError(f"{f} {kind} {ci}x{co}: {(r.stderr or r.stdout)[-800:]}")
                    st = parse_state(r.stdout)
                    if int(st["throttled"], 16) & 0xF:
                        print(f"THROTTLING attivo: {st}", flush=True)
                    if st["temp_c"] < TEMP_MAX_C and not int(st["throttled"], 16) & 0xF:
                        break
                    remeasured += 1           # too hot or throttling now: cool down and redo
                    print(f"caldo ({st}), attendo", flush=True)
                    while bc.read_temp(live) > TEMP_RESUME_C:
                        time.sleep(10)
                lat = backend.parse(r.stdout)
                iters = next(l for l in r.stdout.splitlines() if l.startswith("LATENCIES_MS ")).split()[1:]
                assert len(iters) == lat.iters, (len(iters), lat.iters)
                lat_fh.write(f"{rep},{f},{kind},{ci},{co}," + " ".join(iters) + "\n")
                lat_fh.flush()          # before raw.csv: on resume a measure is redone, analysis keeps the last line
                k, s, hw = geometry(f, kind)
                w.writerow({"family": f, "kind": kind, "kernel": k, "stride": s, "map": hw, "c_in": ci,
                            "c_out": co, "repeat": rep, "sequence": i, "ms": lat.median_ms,
                            "mean_ms": lat.mean_ms, "min_ms": lat.min_ms, "max_ms": lat.max_ms,
                            "iters": lat.iters, **st,
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "n_cores": n_cores, "infer_precision": precision, "openvino": env["openvino"],
                            "remeasured": remeasured})
                fh.flush()
                if i and i % 500 == 0:
                    el = time.monotonic() - t0
                    print(f"{i}/{len(todo)} {el / i:.2f} s/misura, ~{el / i * (len(todo) - i) / 3600:.1f} h"
                          f" alla fine, {st['temp_c']:.1f} C thr={st['throttled']}", flush=True)
                if status_path and time.monotonic() - last_ck > CHECKPOINT_S:
                    last_ck = time.monotonic()
                    print("CHECKPOINT", write_status(csv_path, t0, len(done), i + 1, len(order), status_path),
                          flush=True)
            loop = time.monotonic() - t0
            print(f"fine: {len(todo)} misure in {loop / 3600:.2f} h", flush=True)
            if status_path:
                write_status(csv_path, t0, len(done), max(1, len(todo)), len(order), status_path)
    return loop / max(1, len(todo))


def main():
    import logging
    import signal
    signal.signal(signal.SIGTERM, signal.default_int_handler)   # systemctl stop: restore the board, as Ctrl-C
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    RES.mkdir(parents=True, exist_ok=True)
    pts = points()
    cfg = make_cfg()
    total = len(pts) * REPEATS
    if sys.argv[1] == "pilot":
        n = int(sys.argv[2])
        csv_path = RES / "pilot.csv"
        csv_path.unlink(missing_ok=True)
        per = run([(0, *p) for p in random.Random(123).sample(pts, n)], csv_path, cfg)
        print(f"PILOT: {per:.2f} s/misura; {len(pts)} punti x {REPEATS} = {total} misure -> "
              f"stima {per * total / 3600:.1f} h", flush=True)
        return
    rng = random.Random(SEED)
    order = []
    for rep in range(REPEATS):
        p = pts[:]
        rng.shuffle(p)                 # all families mixed, new order at every repeat, fixed seed
        order += [(rep, *x) for x in p]
    run(order, RES / "raw.csv", cfg, RES / "status.json")
    bad = unstable(RES / "raw.csv")
    print(f"UNSTABLE: {len(bad)} punti con dispersione > {UNSTABLE_REL:.0%}, {EXTRA_REPEATS} ripetizioni in piu'",
          flush=True)
    extra = []
    for rep in range(REPEATS, REPEATS + EXTRA_REPEATS):
        p = bad[:]
        rng.shuffle(p)
        extra += [(rep, *x) for x in p]
    if extra:
        run(extra, RES / "raw.csv", cfg, RES / "status.json")


if __name__ == "__main__":
    main()
