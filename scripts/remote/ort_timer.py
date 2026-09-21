#!/usr/bin/env python3
"""Timer Python attorno a ONNX Runtime — UNICA eccezione alla regola.

Serve a quantificare l'overhead del binding Python come risultato a se': lo
stesso modello misurato con `onnxruntime_perf_test` e con questo script da' la
differenza fra il tempo del motore e il tempo visto da un'applicazione Python.
Non e' il numero da riportare come latenza del backend.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ep", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.ep == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(args.model, sess_options=opts, providers=providers)

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dtype = np.float16 if "float16" in inp.type else np.float32
    # Input fisso: stesso tensore per ogni iterazione e per ogni cella.
    rng = np.random.default_rng(42)
    x = rng.random(shape, dtype=np.float32).astype(dtype)
    feed = {inp.name: x}

    for _ in range(args.warmup):
        sess.run(None, feed)

    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()

    def pct(p):
        idx = min(len(samples) - 1, int(round(p / 100.0 * (len(samples) - 1))))
        return samples[idx]

    mean = statistics.fmean(samples)
    payload = {
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "throughput_qps": 1000.0 / mean if mean else None,
        "iters": args.iters,
        "warmup_iters": args.warmup,
        "threads": args.threads,
        "providers": sess.get_providers(),
    }
    print("YOLOBENCH_JSON " + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
