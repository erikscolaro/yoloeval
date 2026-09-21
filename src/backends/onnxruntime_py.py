"""ONNX Runtime via binding Python — misura dell'overhead, non della latenza.

E' l'unica eccezione alla regola "mai timer in Python", e serve a quantificare
quanto costa il binding: GIL, conversione numpy<->OrtValue, copie, allocazione
degli oggetti di ritorno. Quel costo e' costante, non proporzionale, quindi su
un modello piccolo e veloce pesa piu' che su uno grande.

I numeri di questo backend non vanno confrontati con quelli degli altri: vanno
confrontati con quelli di `onnxruntime` sullo stesso artefatto.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from ..errors import ParseError
from ..measure.compute import resolve_compute
from . import register
from .base import LatencyResult
from .onnxruntime import EP_BY_DEVICE, OnnxRuntimeBackend

log = logging.getLogger(__name__)

MARKER = "YOLOBENCH_JSON "
HELPER = "scripts/remote/ort_timer.py"


@register
class OnnxRuntimePythonBackend(OnnxRuntimeBackend):
    name = "onnxruntime_py"
    export_format = "onnx"
    builds_on_target = False

    def build_cmd(self, cfg, artifact: Path) -> str:
        from ..remote.connection import is_remote

        ct = resolve_compute(cfg)
        bench = cfg.backend.benchmark
        if is_remote(cfg):
            python = cfg.hardware.remote.get("python") or "python3"
            helper = f"{cfg.hardware.remote.workdir}/tools/ort_timer.py"
        else:
            import sys

            python = shlex.quote(sys.executable)
            helper = str(Path(cfg.project_root) / HELPER)
        return (
            f"{python} {shlex.quote(str(helper))} "
            f"--model {shlex.quote(str(artifact))} "
            f"--iters {int(bench.iters)} --warmup {int(bench.warmup_iters)} "
            f"--threads {int(ct.get('n_cores') or 1)} "
            f"--ep {EP_BY_DEVICE.get(ct.get('device'), 'cpu')}"
        )

    def parse(self, stdout: str) -> LatencyResult:
        for line in reversed((stdout or "").splitlines()):
            if line.startswith(MARKER):
                payload = json.loads(line[len(MARKER):])
                return LatencyResult(
                    mean_ms=payload["mean_ms"],
                    median_ms=payload.get("median_ms"),
                    p90_ms=payload.get("p90_ms"),
                    p95_ms=payload.get("p95_ms"),
                    p99_ms=payload.get("p99_ms"),
                    min_ms=payload.get("min_ms"),
                    max_ms=payload.get("max_ms"),
                    throughput_qps=payload.get("throughput_qps"),
                    iters=payload.get("iters"),
                    warmup_iters=payload.get("warmup_iters"),
                    scope=self.scope,
                    tool="python_timer",
                    raw_stdout=stdout,
                )
        raise ParseError("nessuna riga " + MARKER.strip() + " nell'output")
