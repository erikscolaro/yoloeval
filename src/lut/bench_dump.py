"""benchmark_app with a time-based warm-up, that also prints the latency of every inference (runs on the Pi).

benchmark_app (openvino.tools.benchmark) collects request.latency of each iteration in a list and sorts it
before computing median/min/max. Here the module's `sorted` is wrapped to print that list first, in
execution order, as one line "LATENCIES_MS v1 v2 ...": the measurement itself is untouched.
The single warm-up inference before the measured loop is repeated for WARMUP_MS ("WARMUP <n> <ms>").
"""
import builtins
import sys
import time

from openvino.tools.benchmark import benchmark as _bm
from openvino.tools.benchmark.main import main

WARMUP_MS = 60     # warm-up lasts ~20 ms whatever the net (warmup.py, 06/11): 1 inference is not enough
_first_infer = _bm.Benchmark.first_infer


def _warmup(self, requests):
    """benchmark_app's single warm-up inference, repeated for WARMUP_MS; returns the first latency as before."""
    t0 = time.perf_counter()
    first, n = _first_infer(self, requests), 1
    while (time.perf_counter() - t0) * 1000 < WARMUP_MS:
        _first_infer(self, requests)
        n += 1
    print(f"WARMUP {n} {(time.perf_counter() - t0) * 1000:.1f}", flush=True)
    return first


def _sorted(x, *a, **k):
    print("LATENCIES_MS " + " ".join(f"{v:.5g}" for v in x), flush=True)
    return builtins.sorted(x, *a, **k)


_bm.sorted = _sorted
_bm.Benchmark.first_infer = _warmup
sys.argv[0] = "benchmark_app"
sys.exit(main())
