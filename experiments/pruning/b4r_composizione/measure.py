"""B4-R: whole-net latency of the generated nanos on the Pi, with the LUT's conditions and protocol.

  ITERS=50 python measure.py            # every net of results/nets.csv x REPEATS, shuffled -> results/raw.csv
  ITERS=50 python measure.py r000 full  # only these nets (quick check)

Same board tuning, command and thermal guard as lut_pi_passo2/run.py: benchmark_app through bench_dump.py
(60 ms warm-up, per-iteration latencies in runs/latencies_<stem>.txt), -infer_precision f16, 4 cores, 2.4 GHz.
The net is the deploy IR of the `export` stage (fp32 image in, one f32->f16 Convert, one-to-one head).
Resumable: measures already in results/raw.csv are skipped. 50 iterations by default: a whole net takes tens
of ms, so 50 inferences already last well beyond the ~20 ms warm-up and keep a measure to a few seconds.
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

OUT = Path(__file__).resolve().parent
LUT_DIR = OUT.parent / "lut_pi_passo2"
sys.path.insert(0, str(LUT_DIR))
import run as lut  # noqa: E402  (adds the repo root to sys.path)

RES, NETS = OUT / "results", OUT / "runs" / "nets"
REPEATS, SEED = 3, 0
FIELDS = ["net", "repeat", "sequence", "ms", "mean_ms", "min_ms", "max_ms", "iters", "temp_c", "freq_khz",
          "throttled", "governor", "timestamp", "n_cores", "infer_precision", "openvino", "remeasured"]


def main():
    from omegaconf import read_write
    from src.backends import get_backend
    from src.measure.board import board_controller, tuned
    from src.measure.compute import resolve_compute
    from src.remote.connection import connection
    from src.remote.sync import ensure_support_files
    from src.stages.benchmark import _effective_cores, _ensure_profile, _wrap_command

    import logging
    import signal
    signal.signal(signal.SIGTERM, signal.default_int_handler)   # systemctl stop: restore the board
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    names = sys.argv[1:] or [r["net"] for r in csv.DictReader(open(RES / "nets.csv"))]
    csv_path = RES / ("raw.csv" if not sys.argv[1:] else "check.csv")
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

    cfg = lut.make_cfg()
    with read_write(cfg):
        cfg.backend.benchmark.iters = int(os.environ.get("ITERS", 50))
    backend = get_backend(cfg.backend.name, cfg)
    precision = cfg.backend.benchmark.infer_precision[cfg.quantization.precision]
    remote = f"{cfg.hardware.remote.workdir}/b4r"
    new = not csv_path.exists()
    with connection(cfg) as conn:
        conn.run(f"mkdir -p {remote}", hide=True)
        env = lut.env_info(conn, cfg)
        (RES / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
        bc = board_controller(cfg)
        live = _ensure_profile(conn, cfg, bc)
        ensure_support_files(live, cfg)
        live.put(str(LUT_DIR / "bench_dump.py"), f"{remote}/bench_dump.py")
        for n in sorted({n for _, n in todo}):        # all IRs up front: a few MB each
            xml = next((NETS / n).glob("*.xml"))
            live.run(f"mkdir -p {remote}/{n}", hide=True)
            live.put(str(xml), f"{remote}/{n}/net.xml")
            live.put(str(xml.with_suffix(".bin")), f"{remote}/{n}/net.bin")
        with tuned(live, cfg, bc) as applied, open(csv_path, "a", newline="") as fh:
            print("tuning:", {k: applied.get(k) for k in ("governor", "freq_actual_khz", "cores_online",
                                                          "temp_start_c")}, env, flush=True)
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            ct = resolve_compute(cfg)
            n_cores = _effective_cores(live, cfg, bc, ct)
            lat_path = OUT / "runs" / f"latencies_{csv_path.stem}.txt"   # repeat,net,<ms per iteration>
            lat_fh = open(lat_path, "a")
            t0 = time.monotonic()
            for i, (rep, n) in enumerate(todo):
                bench = backend.build_cmd(cfg, Path(f"{remote}/{n}/net.xml"))
                tool = bench.split()[0]
                assert tool.endswith("/benchmark_app"), bench
                bench = bench.replace(tool, f"{cfg.hardware.remote.python} {remote}/bench_dump.py", 1)
                cmd = _wrap_command(live, cfg, bench, ct, n_cores) + lut.STATE
                if i == 0:
                    print("comando:", cmd, flush=True)
                remeasured = 0
                while True:                           # thermal guard as lut_pi_passo2/run.py
                    r = live.run(cmd, hide=True, warn=True)
                    if r.failed:
                        raise RuntimeError(f"{n}: {(r.stderr or r.stdout)[-800:]}")
                    st = lut.parse_state(r.stdout)
                    if st["temp_c"] < lut.TEMP_MAX_C and not int(st["throttled"], 16) & 0xF:
                        break
                    remeasured += 1
                    print(f"caldo ({st}), attendo", flush=True)
                    while bc.read_temp(live) > lut.TEMP_RESUME_C:
                        time.sleep(10)
                lat = backend.parse(r.stdout)
                iters = next(l for l in r.stdout.splitlines() if l.startswith("LATENCIES_MS ")).split()[1:]
                lat_fh.write(f"{rep},{n}," + " ".join(iters) + "\n")
                lat_fh.flush()
                w.writerow({"net": n, "repeat": rep, "sequence": i, "ms": lat.median_ms, "mean_ms": lat.mean_ms,
                            "min_ms": lat.min_ms, "max_ms": lat.max_ms, "iters": lat.iters, **st,
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "n_cores": n_cores, "infer_precision": precision, "openvino": env["openvino"],
                            "remeasured": remeasured})
                fh.flush()
                el = time.monotonic() - t0
                print(f"{i + 1}/{len(todo)} {n} rep {rep}: {lat.median_ms:.2f} ms, {st['temp_c']:.1f} C, "
                      f"~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min alla fine", flush=True)


if __name__ == "__main__":
    main()
