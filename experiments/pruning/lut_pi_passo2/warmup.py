"""Is the benchmark_app warm-up a fixed time or a fixed number of iterations, and does an idle pause before
the measure change it?

  python warmup.py measure    # ~20 nets from 0.01 to 3 ms, each run for DURATION_S (-t), not -niter
  python warmup.py analyze    # results/warmup.json (figure: figures.py warmup)
  python warmup.py log        # wandb yolo-bench (job_type lut_pi_warmup)
  python warmup.py check      # same nets, LUT protocol (-niter 200) with bench_dump's WARMUP_MS warm-up,
                              # vs the steady state above: results/warmup_check.csv, warmup_check.json
  python warmup.py pilot      # the pilot's conv points again, 2 repeats, with the warm-up: how much the LUT
                              # gains overall (results/warmup_pilot.csv, warmup_pilot.json)

Nets: identities at every input map (smallest and largest c_in), plus conv nets from results/pilot.csv
and results/transient.csv nearest (results/pilot_0511.csv: the 05/11 pilot, frozen) to log-spaced latency targets. Repeats 0-2 start right after the previous measure; repeats 3-4
idle PAUSE_S on the Pi first. Steady state = median of the last second. Transient = time after which 3
windows in a row of max(WIN iterations, WIN_MS) have their median within TOL of the steady state; the
jitter of those windows in the last second is reported apart (steady_jitter_p90).
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import run  # noqa: E402

DURATION_S, PAUSE_S, WIN, WIN_MS = 2, 10, 20, 10
REPS_NOW, REPS_PAUSED = (0, 1, 2), (3, 4)
TARGETS_MS = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
WARMUPS_MS = (0, 25, 50, 100, 200, 500)
KEEP_ITERS = (100, 200, 500)
CSV = run.RES / "warmup.csv"
LAT = OUT / "runs" / "latencies_warmup.txt"


def nets() -> list[tuple]:
    top = {}                                            # input map -> largest c_in of the layers reading it
    for f, (_, _, _, layers) in run.FAMILIES.items():
        hw = run.in_map(f)
        top[hw] = max(top.get(hw, 0), max(ci for ci, _ in layers))
    out = set()
    for hw, c in top.items():
        lo = c if hw == 640 else 2                      # the stem reads a fixed 3-channel image
        out |= {(f"id{hw}", "identity", lo, lo), (f"id{hw}", "identity", c, c)}
    pilot = [r for p in ("pilot_0511.csv", "transient.csv")          # transient.csv reaches 0.05 and 3 ms
             for r in csv.DictReader(open(run.RES / p)) if r["kind"] == "conv"]
    for t in TARGETS_MS:
        r = min(pilot, key=lambda r: abs(np.log(float(r["ms"]) / t)))
        out.add((r["family"], "conv", int(r["c_in"]), int(r["c_out"])))
    return sorted(out)


def measure():
    rng = random.Random(2)
    order = []
    for rep in REPS_NOW + REPS_PAUSED:
        p = nets()
        rng.shuffle(p)
        order += [(rep, *x) for x in p]
    cfg = run.make_cfg()
    from omegaconf import read_write
    with read_write(cfg):
        cmd = str(cfg.backend.benchmark.cmd)
        assert "-niter {iters}" in cmd, cmd
        cfg.backend.benchmark.cmd = cmd.replace("-niter {iters}", f"-t {DURATION_S}")
    seq = iter(order)             # run.run calls build() once per measure, in order: idle the Pi there
    build = run.build

    def build_paused(*a):
        if next(seq)[0] in REPS_PAUSED:
            time.sleep(PAUSE_S)
        return build(*a)
    run.build = build_paused
    CSV.unlink(missing_ok=True)   # fresh run: the iterator above assumes nothing is skipped
    print(f"{len(order)} misure, {len(nets())} reti", flush=True)
    run.run(order, CSV, cfg)


def check():
    rng = random.Random(3)
    order = []
    for rep in REPS_NOW:
        p = nets()
        rng.shuffle(p)
        order += [(rep, *x) for x in p]
    path = run.RES / "warmup_check.csv"
    path.unlink(missing_ok=True)
    os.environ["ITERS"] = "200"                   # as launch.sh
    run.run(order, path, run.make_cfg())
    steady = {}
    for r in json.loads((run.RES / "warmup.json").read_text())["measures_detail"]:
        steady.setdefault((r["family"], r["c_in"], r["c_out"]), []).append(r["steady_ms"])
    err = {}
    for r in csv.DictReader(open(path)):
        k = (r["family"], int(r["c_in"]), int(r["c_out"]))
        err.setdefault(k, []).append(float(r["ms"]) / np.median(steady[k]) - 1)
    out = {f"{f} {ci}x{co}": {"steady_ms": float(np.median(steady[(f, ci, co)])), "err": [round(e, 4) for e in v]}
           for (f, ci, co), v in sorted(err.items(), key=lambda kv: np.median(steady[kv[0]]))}
    a = np.abs([e for v in err.values() for e in v])
    summ = {"iters": int(next(csv.DictReader(open(path)))["iters"]), "abs_err_median": float(np.median(a)),
            "abs_err_p90": float(np.percentile(a, 90)), "abs_err_max": float(a.max()), "per_net": out}
    (run.RES / "warmup_check.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def pilot():
    path = run.RES / "warmup_pilot.csv"
    if len(sys.argv) < 3 or sys.argv[2] != "analyze":
        pts = [(r["family"], "conv", int(r["c_in"]), int(r["c_out"]))
               for r in csv.DictReader(open(run.RES / "pilot_0511.csv")) if r["kind"] == "conv"]
        rng = random.Random(5)
        order = []
        for rep in (0, 1):
            rng.shuffle(pts)
            order += [(rep, *p) for p in pts]
        os.environ["ITERS"] = "200"
        path.unlink(missing_ok=True)
        run.run(order, path, run.make_cfg())
    old = {(r["family"], int(r["c_in"]), int(r["c_out"])): float(r["ms"])
           for r in csv.DictReader(open(run.RES / "pilot_0511.csv")) if r["kind"] == "conv"}
    new = {}
    for r in csv.DictReader(open(path)):
        new.setdefault((r["family"], int(r["c_in"]), int(r["c_out"])), []).append(float(r["ms"]))
    rows = []
    for k, (a, b) in ((k, v) for k, v in new.items() if len(v) == 2):
        m = (a + b) / 2
        rows.append({"family": k[0], "c_in": k[1], "c_out": k[2], "old_ms": old[k], "new_ms": m,
                     "old_over_new": old[k] / m - 1, "rep_diff": abs(a / b - 1)})
    bias = np.array([r["old_over_new"] for r in rows])
    noise = np.array([r["rep_diff"] for r in rows])
    ms = np.array([r["new_ms"] for r in rows])
    bins = {}
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 100)):
        s = (ms >= lo) & (ms < hi)
        if s.any():
            bins[f"{lo}-{hi}ms"] = {"n": int(s.sum()), "bias_median": float(np.median(bias[s])),
                                    "bias_max": float(bias[s].max()), "rep_diff_median": float(np.median(noise[s]))}
    summ = {"points": len(rows), "old_over_new": {"median": float(np.median(bias)), "mean": float(bias.mean()),
                                                  "p90": float(np.percentile(bias, 90)), "max": float(bias.max()),
                                                  "min": float(bias.min())},
            "rep_diff": {"median": float(np.median(noise)), "p90": float(np.percentile(noise, 90)),
                         "max": float(noise.max())},
            "sum_ms_old_over_new": float(sum(r["old_ms"] for r in rows) / ms.sum() - 1), "bins": bins,
            "points_detail": rows}
    (run.RES / "warmup_pilot.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "points_detail"}, indent=1))


def transient_idx(v: np.ndarray, t: np.ndarray, steady: float, end: int, tol: float) -> int:
    win = max(WIN, int(np.searchsorted(t, WIN_MS)))    # 20 iterations of a 0.01 ms net are pure jitter
    starts = range(0, max(1, end - win + 1), max(1, win // 4))
    ok = [abs(np.median(v[i:i + win]) / steady - 1) <= tol for i in starts]
    # first entry into the band for 3 windows in a row: late excursions are steady-state jitter, kept apart
    return next((s for j, s in enumerate(starts) if all(ok[j:j + 3])), end)


def series():
    """(key, meta row, latencies ms, estimated time since the first inference ms, steady ms, index of the
    last second) for every measure of `measure`; also used by figures.py."""
    meta = {(int(r["repeat"]), r["family"], r["kind"], int(r["c_in"]), int(r["c_out"])): r
            for r in csv.DictReader(open(CSV))}
    for line in open(LAT):
        rep, f, kind, ci, co, vs = line.rstrip("\n").split(",", 5)
        k = (int(rep), f, kind, int(ci), int(co))
        v = np.array(vs.split(), float)
        # shortcut: wall time = latencies + uniform gap so the run lasts DURATION_S (benchmark_app's Duration
        # is not saved); between-inference overhead (~17 us) dominates tiny nets, so cumsum(v) alone is wrong
        t = np.cumsum(v + max(0.0, DURATION_S * 1000 - v.sum()) / len(v))
        last = int(np.searchsorted(t, t[-1] - 1000))      # steady state: last second
        yield k, meta[k], v, t, float(np.median(v[last:])), last


def analyze():
    rows = []
    for k, m, v, t, steady, last in series():
        f, kind = k[1], k[2]
        row = {"rep": k[0], "paused": k[0] in REPS_PAUSED, "cooled": int(m["remeasured"]) > 0,
               "family": f, "kind": kind, "c_in": k[3], "c_out": k[4], "iters": len(v), "steady_ms": steady,
               "temp_c": float(m["temp_c"]), "first10_over_steady": float(v[:10].mean() / steady)}
        win = max(WIN, int(np.searchsorted(t, WIN_MS)))
        late = [np.median(v[i:i + win]) / steady - 1 for i in range(last, len(v) - win + 1, win)]
        row["steady_jitter_p90"] = float(np.percentile(np.abs(late), 90)) if late else 0.0
        for tol in (0.02, 0.05):
            i = transient_idx(v, t, steady, last, tol)
            row[f"transient_iters_{tol}"] = i
            row[f"transient_ms_{tol}"] = float(t[i - 1]) if i else 0.0
        for w in WARMUPS_MS:
            s = int(np.searchsorted(t, w))
            for n in KEEP_ITERS:
                row[f"err_W{w}ms_N{n}"] = float(np.median(v[s:s + n]) / steady - 1)
            row[f"err_W{w}ms_T200ms"] = float(np.median(v[s:np.searchsorted(t, w + 200)]) / steady - 1)
        row["err_niter200"] = float(np.median(v[:200]) / steady - 1)   # what the LUT does now
        rows.append(row)

    def stats(sel, key):
        x = np.array([r[key] for r in rows if sel(r)])
        return {"n": len(x), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                "max": float(x.max())} if len(x) else None

    now = lambda r: not r["paused"] and not r["cooled"]  # noqa: E731
    summary = {"duration_s": DURATION_S, "pause_s": PAUSE_S, "window": WIN, "window_ms": WIN_MS, "measures": len(rows),
               "transient": {f"{w}_{tol}_{g}": stats(sel, f"transient_{w}_{tol}")
                             for w in ("ms", "iters") for tol in (0.02, 0.05)
                             for g, sel in (("now", now), ("paused", lambda r: r["paused"]),
                                            ("cooled", lambda r: r["cooled"]))},
               "abs_err": {}, "per_net": {}}
    for key in [k for k in rows[0] if k.startswith("err_")]:
        x = np.abs([r[key] for r in rows])
        summary["abs_err"][key] = {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                                   "max": float(x.max())}
    for r in rows:
        n = summary["per_net"].setdefault(f"{r['family']} {r['c_in']}x{r['c_out']}", {})
        n.setdefault("steady_ms", []).append(round(r["steady_ms"], 5))
        n.setdefault("transient_ms_0.02", []).append(round(r["transient_ms_0.02"], 1))
        n.setdefault("transient_iters_0.02", []).append(r["transient_iters_0.02"])
        n.setdefault("err_niter200", []).append(round(r["err_niter200"], 4))
        n.setdefault("jitter_p90", []).append(round(r["steady_jitter_p90"], 4))
        n.setdefault("paused", []).append(r["paused"])
    (run.RES / "warmup.json").write_text(json.dumps({**summary, "measures_detail": rows}, indent=1),
                                         encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "per_net"}, indent=1))
    for k, v in sorted(summary["per_net"].items(), key=lambda kv: np.median(kv[1]["steady_ms"])):
        print(k, v)


def log():
    import wandb
    s = json.loads((run.RES / "warmup.json").read_text())
    r = wandb.init(entity="yolo-bench", project="yolo-bench", job_type="lut_pi_warmup",
                   name="lut_pi_warmup_rpi5_openvino_fp16_cpu4_max", tags=["lut", "warmup", "rpi5", "openvino"],
                   config={"board": "rpi5", "backend": "openvino", "infer_precision": "f16", "compute_target": "cpu_4",
                           "freq_target": "max", "duration_s": DURATION_S, "pause_s": PAUSE_S, "fan": False,
                           **json.loads((run.RES / "env.json").read_text())})
    r.summary.update({f"transient/{k}/{m}": x for k, v in s["transient"].items() if v for m, x in v.items()})
    r.summary.update({f"abs_err/{k}/{m}": x for k, v in s["abs_err"].items() for m, x in v.items()})
    for png in sorted((OUT / "figures").glob("warmup*.png")):
        r.log({png.stem: wandb.Image(str(png))})
    art = wandb.Artifact("lut_pi_warmup_rpi5", type="dataset")
    chk = run.RES / "warmup_check.json"
    if chk.exists():
        r.summary.update({f"check/{k}": v for k, v in json.loads(chk.read_text()).items() if k != "per_net"})
    for f in (CSV, run.RES / "warmup.json", LAT, OUT / "warmup.py", OUT / "bench_dump.py",
              *(OUT / "figures").glob("warmup*.p*"), run.RES / "warmup_check.csv", chk,
              run.RES / "warmup_pilot.csv", run.RES / "warmup_pilot.json"):
        if f.exists():
            art.add_file(str(f), name=f.name)
    r.log_artifact(art)
    r.finish()
    print(r.url)


if __name__ == "__main__":
    {"measure": measure, "analyze": analyze, "log": log, "check": check, "pilot": pilot}[sys.argv[1]]()
