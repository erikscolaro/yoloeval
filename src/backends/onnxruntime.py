"""ONNX Runtime — export portabile, misura con `onnxruntime_perf_test`.

L'ONNX e' l'unico artefatto portabile della pipeline: si builda una volta sulla
workstation e si copia sulle board. La misura usa il binario C++ del vendor,
non il binding Python: l'overhead di quest'ultimo e' costante e su un modello
nano quantizzato su CPU ARM sottostima proprio lo speedup della quantizzazione
che si vuole misurare (per quantificarlo esiste il backend `onnxruntime_py`).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..errors import ExportFailed
from ..measure.compute import resolve_compute
from . import register
from .base import Backend, LatencyResult, require, search

log = logging.getLogger(__name__)

#: mappa device del compute target -> execution provider di perf_test
EP_BY_DEVICE = {"cpu": "cpu", "cuda": "cuda", "tensorrt": "tensorrt"}


@register
class OnnxRuntimeBackend(Backend):
    name = "onnxruntime"
    export_format = "onnx"
    builds_on_target = False
    scope = "end_to_end_with_transfers"

    # --- export ----------------------------------------------------------
    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / f"{cfg.model.name}_{cfg.quantization.name}.onnx"
        if target.exists():
            log.info("artefatto gia' presente: %s", target.name)
            return target

        base = self._export_fp32(cfg, Path(src), dst)
        precision = cfg.quantization.precision
        if precision == "fp32":
            if base != target:
                shutil.move(str(base), target)
            return target
        if precision == "fp16":
            return self._to_fp16(base, target)
        if precision == "int8":
            from ..stages.quantize import quantize_onnx_static

            out = quantize_onnx_static(base, target, cfg)
            base.unlink(missing_ok=True)
            return out
        raise ExportFailed(f"precisione non gestita da {self.name}: {precision}")

    def _export_fp32(self, cfg, src: Path, dst: Path) -> Path:
        from ultralytics import YOLO

        model = YOLO(str(src), task="detect")
        produced = model.export(
            format="onnx",
            imgsz=int(cfg.model.imgsz),
            opset=int(cfg.backend.build.opset),
            batch=1,
            simplify=True,
            dynamic=False,
            device="cpu",
        )
        produced = Path(produced)
        base = dst / f"{cfg.model.name}_fp32_base.onnx"
        if produced.resolve() != base.resolve():
            shutil.move(str(produced), base)
        return base

    def _to_fp16(self, base: Path, target: Path) -> Path:
        """Cast a fp16 sul grafo ONNX.

        Si usa la conversione sul grafo e non `half=True` di Ultralytics perche'
        quest'ultima richiede una GPU al momento dell'export: l'artefatto
        dev'essere producibile sulla workstation anche quando la cella di
        destinazione e' una board.
        """
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(base))
        converted = float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False
        )
        onnx.save(converted, str(target))
        base.unlink(missing_ok=True)
        return target

    # --- misura ----------------------------------------------------------
    def build_cmd(self, cfg, artifact: Path) -> str:
        ct = resolve_compute(cfg)
        bench = cfg.backend.benchmark
        ep = EP_BY_DEVICE.get(ct.get("device"), "cpu")
        threads = int(ct.get("n_cores") or 1)
        template = " ".join(str(bench.cmd).split())
        return template.format(
            model=artifact,
            ep=ep,
            iters=int(bench.iters),
            warmup_iters=int(bench.warmup_iters),
            threads=threads,
        )

    def parse(self, stdout: str) -> LatencyResult:
        """`onnxruntime_perf_test` riporta la media in ms e i percentili in s."""
        mean = require(
            search(r"Average inference time cost:\s*([\d.]+)\s*ms", stdout),
            "Average inference time cost", stdout,
        )

        def pct(label):
            value = search(rf"{label} Latency:\s*([\d.eE+-]+)\s*s", stdout)
            return float(value) * 1000.0 if value else None

        qps = search(r"Number of inferences per second:\s*([\d.]+)", stdout)
        iters = search(r"Total inference requests:\s*(\d+)", stdout) or search(
            r"Runs:\s*(\d+)", stdout
        )
        median = pct("P50")
        require(median, "P50 Latency (serve -I in command line)", stdout)
        return LatencyResult(
            mean_ms=float(mean),
            median_ms=median,
            p90_ms=pct("P90"),
            p95_ms=pct("P95"),
            p99_ms=pct("P99"),
            min_ms=pct("Min"),
            max_ms=pct("Max"),
            throughput_qps=float(qps) if qps else None,
            iters=int(iters) if iters else None,
            scope=self.scope,
            tool="onnxruntime_perf_test",
            raw_stdout=stdout,
        )

    # --- ispezione -------------------------------------------------------
    def inspect(self, artifact: Path) -> dict:
        from ..validation.graph import inspect_onnx

        return inspect_onnx(artifact)
