"""Latency LUT of the Raspberry Pi 5 for the nano (YOLO26 0.25, D126) and B4-R whole-net measures: every parameter of
the procedure, the grid, the nets and the measurement loop (experiments/pruning/lut_pi_passo2, b4r_composizione).

OpenVINO fp16 (-infer_precision f16), 4 cores, governor performance, 2.4 GHz: same config, board tuning
(src.measure.board.tuned) and benchmark_app command (backend.build_cmd + _wrap_command) as
src/stages/probe.py and pruning/separability. One conv + bias + SiLU per net (as the exported Conv with BN
fused); no identity nets (lut_pi_passo2/DECISIONE_IDENTITA.md). Input, output, weights and bias are fp16, the
precision used at inference: no f32<->f16 Convert in the compiled graph (that belongs to the preprocessing). Input
and output tensors are NHWC (PrePostProcessor, saved as IR), the layout the ARM CPU plugin uses inside: no
NCHW<->NHWC Reorder either. At every inference only the Convolution and the SiLU Subgraph run. benchmark_app runs
through bench_dump.py (60 ms warm-up, lut_pi_passo2/warmup.py), which also prints the latency of every iteration:
saved in <dir>/runs/latencies_<stem>.txt, one line per measure, so the LUT can be re-aggregated later. Each LUT net
is built here and copied to the Pi right before its measurement (70k nets: no bulk copy).

  python -m src.lut.pi lut <dir>           # LUT, resumable: skips what is already in <dir>/results/raw.csv,
                                           # then the unstable points again (see unstable())
  python -m src.lut.pi b4r <dir> [net...]  # B4-R nets of <dir>/results/nets.csv (src.lut.b4r) x REPEATS
                                           # -> results/raw.csv; with nets: only those -> results/check.csv
ITERS in the environment overrides the iterations (ITERS_LUT, ITERS_B4R). scripts/build_lut.sh chains everything.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# family -> (kernel, stride, output map, layers as (c_in max, c_out max)); from NEW_MASTERPLAN.md, 4a, "Strati potabili del nano e griglia"
FAMILIES = {
    "k1s1_20": (1, 1, 20, [(128, 64), (128, 128), (256, 128), (256, 256), (384, 256), (512, 256)]),
    "k1s1_40": (1, 1, 40, [(64, 32), (64, 64), (128, 128), (192, 128), (384, 128)]),
    "k1s1_80": (1, 1, 80, [(32, 16), (32, 32), (64, 64), (96, 64), (96, 128), (256, 64), (128, 128)]),
    "k1s1_160": (1, 1, 160, [(32, 32), (48, 64), (64, 64)]),
    "k3s1_20": (3, 1, 20, [(64, 64), (64, 128), (128, 64)]),
    "k3s1_40": (3, 1, 40, [(32, 32), (32, 64), (64, 32)]),
    "k3s1_80": (3, 1, 80, [(16, 16), (16, 32), (32, 16)]),
    "k3s1_160": (3, 1, 160, [(8, 16), (16, 8)]),
    "k3s2_20": (3, 2, 20, [(128, 128), (128, 256)]),
    "k3s2_40": (3, 2, 40, [(64, 64), (128, 128)]),
    "k3s2_80": (3, 2, 80, [(64, 64)]),
    "k3s2_160": (3, 2, 160, [(16, 32)]),
    "k3s2_320": (3, 2, 320, [(3, 16)]),          # stem: c_in is the image, fixed at 3
}
FIXED_CIN = {"k3s2_320"}
# extension of 09/11 (B4-R up to the nano, lut_pi_passo2/README "Coda dopo la LUT principale"): the last layer of
# each family above and the odd 3 of w0.09375; resuming measures only the new points, raw.csv is appended
EXTRA_AXIS = {"k3s1_160": {3}}
N_CONV = 71328
REPEATS, SEED = 3, 0
UNSTABLE_REL, EXTRA_REPEATS = 0.05, 6      # (max-min)/median of the first REPEATS above this: 6 more repeats
TEMP_MAX_C, TEMP_RESUME_C = 75.0, 65.0         # guard: firmware throttles at 80 C; with the fan at max it never trips
CHECKPOINT_S = 6 * 3600
# 200 inferences per LUT net; 50 per B4-R net: a whole net takes tens of ms, so 50 already last well beyond the
# ~20 ms warm-up and keep a measure to a few seconds
ITERS_LUT, ITERS_B4R = 200, 50
# B4-R (src.lut.b4r): every architecture up to the nano (0.375, S and up need a wider LUT), N_NETS random pruned
# nets each plus the unpruned one; 4 classes, yolopit's n = 2 = LUT step
ARCHS = [(w, d) for w in (0.0625, 0.09375, 0.125, 0.1875, 0.25) for d in ("D101", "D126", "D170")]
N_NETS, NC, N_STEP = 50, 4, 2
LEVEL, SIGMA = (0.1, 1.0), 0.5
FIELDS = ["family", "kind", "kernel", "stride", "map", "c_in", "c_out", "repeat", "sequence", "ms",
          "mean_ms", "min_ms", "max_ms", "iters", "temp_c", "freq_khz", "throttled", "governor",
          "timestamp", "n_cores", "infer_precision", "openvino", "remeasured"]
B4R_FIELDS = ["net", "repeat", "sequence", "ms", "mean_ms", "min_ms", "max_ms", "iters", "temp_c", "freq_khz",
              "throttled", "governor", "timestamp", "n_cores", "infer_precision", "openvino", "remeasured"]


def family_label(f: str, sep: str = " ") -> str:
    """'k3s2_40' -> '3×3 s2 @40': kernel, stride and side of the output map at 640, as in the family name."""
    k, s, hw, _ = FAMILIES[f]
    return f"{k}×{k} s{s}{sep}@{hw}"


def axis(m: int, extra=()) -> list[int]:
    return sorted({1, m, *range(2, m + 1, 2), *(e for e in extra if e <= m)})   # 1, even values, the max


def in_map(f: str) -> int:
    _, s, hw, _ = FAMILIES[f]
    return hw * s


def points() -> list[tuple]:
    """(family, "conv", c_in, c_out) in a fixed order. No identity nets: the per-call fixed cost is the same
    for every layer and all layers always exist, so the constant C absorbs it (lut_pi_passo2/README, Costo fisso)."""
    pts = []
    for f, (_, _, _, layers) in FAMILIES.items():
        s = set()
        for ci, co in layers:
            ex = EXTRA_AXIS.get(f, ())
            s |= {(a, b) for a in ([ci] if f in FIXED_CIN else axis(ci, ex)) for b in axis(co, ex)}
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
            "stage=probe_edge", "target=rpi5_640", f"project_root={REPO}"])  # start threshold 50 C, as in the deploy benchmark


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


@contextmanager
def session(cfg, remote_dir: str, res: Path, upload=None):
    """Pi ready to measure: connection, env.json in res, board profile, bench_dump.py copied, upload(live) if
    given, then the board tuned. Yields s with s.live, s.bc, s.backend, s.env, s.n_cores, s.precision and
    s.cmd(xml): benchmark_app through bench_dump.py on that IR, wrapped as the deploy benchmark, + STATE."""
    from src.backends import get_backend
    from src.measure.board import board_controller, tuned
    from src.measure.compute import resolve_compute
    from src.remote.connection import connection
    from src.remote.sync import ensure_support_files
    from src.stages.benchmark import _effective_cores, _ensure_profile, _wrap_command

    backend = get_backend(cfg.backend.name, cfg)
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {remote_dir}", hide=True)
        env = env_info(conn, cfg)
        (res / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        ensure_support_files(live, cfg)
        live.put(str(HERE / "bench_dump.py"), f"{remote_dir}/bench_dump.py")
        if upload:
            upload(live)
        with tuned(live, cfg, bc) as applied:
            print("tuning:", {k: applied.get(k) for k in ("governor", "freq_actual_khz", "cores_online",
                                                          "temp_start_c")}, env, flush=True)
            ct = resolve_compute(cfg)
            n_cores = _effective_cores(live, cfg, bc, ct)

            def cmd(xml: str) -> str:
                bench = backend.build_cmd(cfg, Path(xml))
                tool = bench.split()[0]
                assert tool.endswith("/benchmark_app"), bench
                bench = bench.replace(tool, f"{cfg.hardware.remote.python} {remote_dir}/bench_dump.py", 1)
                return _wrap_command(live, cfg, bench, ct, n_cores) + STATE

            yield SimpleNamespace(live=live, bc=bc, backend=backend, env=env, n_cores=n_cores, cmd=cmd,
                                  precision=cfg.backend.benchmark.infer_precision[cfg.quantization.precision])


def measure(s, cmd: str, what: str) -> tuple:
    """One measure with the thermal guard: if the Pi ended it above TEMP_MAX_C or throttling, wait until it is
    below TEMP_RESUME_C and redo it. Returns (latency, per-iteration ms, state, times redone)."""
    from src.measure.thermal import wait_thermal
    remeasured = 0
    while True:
        r = s.live.run(cmd, hide=True, warn=True)
        if r.failed:
            raise RuntimeError(f"{what}: {(r.stderr or r.stdout)[-800:]}")
        st = parse_state(r.stdout)
        hot = int(st["throttled"], 16) & 0xF
        if hot:
            print(f"THROTTLING attivo: {st}", flush=True)
        if st["temp_c"] < TEMP_MAX_C and not hot:
            break
        remeasured += 1           # too hot or throttling now: cool down and redo
        print(f"caldo ({st}), attendo", flush=True)
        wait_thermal(s.live, s.bc, TEMP_RESUME_C, timeout_s=float("inf"))
    lat = s.backend.parse(r.stdout)
    iters = next(l for l in r.stdout.splitlines() if l.startswith("LATENCIES_MS ")).split()[1:]
    assert len(iters) == lat.iters, (len(iters), lat.iters)
    return lat, iters, st, remeasured


def row(s, lat, st, remeasured) -> dict:
    """The csv columns every measure shares (FIELDS, B4R_FIELDS)."""
    return {"ms": lat.median_ms, "mean_ms": lat.mean_ms, "min_ms": lat.min_ms, "max_ms": lat.max_ms,
            "iters": lat.iters, **st, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "n_cores": s.n_cores, "infer_precision": s.precision, "openvino": s.env["openvino"],
            "remeasured": remeasured}


def set_iters(cfg, default: int) -> None:
    from omegaconf import read_write
    with read_write(cfg):
        cfg.backend.benchmark.iters = int(os.environ.get("ITERS", default))


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
    k1s1_160 48x64 (lut_pi_passo2/bimodal.py), so a median of 3 can pick either. Uses only repeats < REPEATS, so
    the set does not change when the extra repeats are resumed after a restart."""
    by = {}
    for r in csv.DictReader(open(csv_path)):
        if int(r["repeat"]) < REPEATS:
            by.setdefault((r["family"], r["kind"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    return sorted(k for k, v in by.items()
                  if len(v) == REPEATS and (max(v) - min(v)) / float(np.median(v)) > UNSTABLE_REL)


def run(order: list[tuple], csv_path: Path, cfg, status_path: Path | None = None, runs: Path | None = None) -> float:
    """Measure the LUT nets of order, (repeat, family, kind, c_in, c_out), appending to csv_path and skipping
    those already in it. Per-iteration latencies in runs (default: runs/ next to the csv's folder)."""
    set_iters(cfg, ITERS_LUT)
    runs = runs or csv_path.parent.parent / "runs"
    done = set()
    if csv_path.exists():
        done = {(int(r["repeat"]), r["family"], r["kind"], int(r["c_in"]), int(r["c_out"]))
                for r in csv.DictReader(open(csv_path))}
    todo = [o for o in order if o not in done]
    print(f"{len(todo)} misure da fare ({len(done)} gia' fatte)", flush=True)
    if not todo:
        return 0.0
    new = not csv_path.exists()
    remote_dir = f"{cfg.hardware.remote.workdir}/lut_pi"
    net = f"{remote_dir}/net.xml"
    lat_path = runs / f"latencies_{csv_path.stem}.txt"   # rep,family,kind,c_in,c_out,<ms per iter>
    runs.mkdir(parents=True, exist_ok=True)
    if new:
        lat_path.unlink(missing_ok=True)
    with session(cfg, remote_dir, csv_path.parent) as s, open(csv_path, "a", newline="") as fh, \
            open(lat_path, "a") as lat_fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        cmd = s.cmd(net)
        print("comando:", cmd, flush=True)
        t0 = last_ck = time.monotonic()
        for i, (rep, f, kind, ci, co) in enumerate(todo):
            xml, bin_ = to_ir(build(f, kind, ci, co), nhwc=kind == "conv")
            s.live.put(io.BytesIO(xml), net)
            s.live.put(io.BytesIO(bin_), net[:-4] + ".bin")
            lat, iters, st, remeasured = measure(s, cmd, f"{f} {kind} {ci}x{co}")
            lat_fh.write(f"{rep},{f},{kind},{ci},{co}," + " ".join(iters) + "\n")
            lat_fh.flush()          # before raw.csv: on resume a measure is redone, analysis keeps the last line
            k, st_, hw = geometry(f, kind)
            w.writerow({"family": f, "kind": kind, "kernel": k, "stride": st_, "map": hw, "c_in": ci,
                        "c_out": co, "repeat": rep, "sequence": i, **row(s, lat, st, remeasured)})
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


def lut(d: Path) -> None:
    """The whole LUT in d/results/raw.csv: REPEATS shuffled passes, then EXTRA_REPEATS on the unstable points."""
    res = d / "results"
    res.mkdir(parents=True, exist_ok=True)
    pts = points()
    cfg = make_cfg()
    rng = random.Random(SEED)
    order = []
    for rep in range(REPEATS):
        p = pts[:]
        rng.shuffle(p)                 # all families mixed, new order at every repeat, fixed seed
        order += [(rep, *x) for x in p]
    run(order, res / "raw.csv", cfg, res / "status.json", d / "runs")
    bad = unstable(res / "raw.csv")
    print(f"UNSTABLE: {len(bad)} punti con dispersione > {UNSTABLE_REL:.0%}, {EXTRA_REPEATS} ripetizioni in piu'",
          flush=True)
    extra = []
    for rep in range(REPEATS, REPEATS + EXTRA_REPEATS):
        p = bad[:]
        rng.shuffle(p)
        extra += [(rep, *x) for x in p]
    if extra:
        run(extra, res / "raw.csv", cfg, res / "status.json", d / "runs")


def b4r(d: Path, names: list[str]) -> None:
    """Whole-net latency of the B4-R nets (d/runs/nets/<net>, src.lut.b4r) with the LUT's conditions: deploy IR
    of the `export` stage (fp32 image in, one f32->f16 Convert, one-to-one head). Resumable as run()."""
    res, runs = d / "results", d / "runs"
    csv_path = res / ("check.csv" if names else "raw.csv")
    names = names or [r["net"] for r in csv.DictReader(open(res / "nets.csv"))]
    rng = random.Random(SEED)
    order = []
    for rep in range(REPEATS):
        p = names[:]
        rng.shuffle(p)
        order += [(rep, n) for n in p]
    done = set()
    if csv_path.exists():
        done = {(int(r["repeat"]), r["net"]) for r in csv.DictReader(open(csv_path))}
    todo = [o for o in order if o not in done]
    print(f"{len(todo)} misure da fare ({len(done)} gia' fatte)", flush=True)
    if not todo:
        return
    cfg = make_cfg()
    set_iters(cfg, ITERS_B4R)
    remote = f"{cfg.hardware.remote.workdir}/b4r"
    new = not csv_path.exists()

    def upload(live):                                # all IRs up front: a few MB each
        for n in sorted({n for _, n in todo}):
            xml = next((runs / "nets" / n).glob("*.xml"))
            live.run(f"mkdir -p {remote}/{n}", hide=True)
            live.put(str(xml), f"{remote}/{n}/net.xml")
            live.put(str(xml.with_suffix(".bin")), f"{remote}/{n}/net.bin")

    runs.mkdir(parents=True, exist_ok=True)
    with session(cfg, remote, res, upload) as s, open(csv_path, "a", newline="") as fh, \
            open(runs / f"latencies_{csv_path.stem}.txt", "a") as lat_fh:   # repeat,net,<ms per iteration>
        w = csv.DictWriter(fh, B4R_FIELDS)
        if new:
            w.writeheader()
        t0 = time.monotonic()
        for i, (rep, n) in enumerate(todo):
            cmd = s.cmd(f"{remote}/{n}/net.xml")
            if i == 0:
                print("comando:", cmd, flush=True)
            lat, iters, st, remeasured = measure(s, cmd, n)
            lat_fh.write(f"{rep},{n}," + " ".join(iters) + "\n")
            lat_fh.flush()
            w.writerow({"net": n, "repeat": rep, "sequence": i, **row(s, lat, st, remeasured)})
            fh.flush()
            el = time.monotonic() - t0
            print(f"{i + 1}/{len(todo)} {n} rep {rep}: {lat.median_ms:.2f} ms, {st['temp_c']:.1f} C, "
                  f"~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min alla fine", flush=True)


def main():
    import logging
    import signal
    signal.signal(signal.SIGTERM, signal.default_int_handler)   # systemctl stop: restore the board, as Ctrl-C
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    what, d = sys.argv[1], Path(sys.argv[2])
    if what == "lut":
        lut(d)
    elif what == "b4r":
        b4r(d, sys.argv[3:])
    else:
        sys.exit(f"uso: python -m src.lut.pi lut|b4r <dir> [net...]; non {what!r}")


if __name__ == "__main__":
    main()
