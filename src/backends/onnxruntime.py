"""ONNX Runtime — export portabile, misura con `onnxruntime_perf_test`.

L'ONNX e' l'unico artefatto portabile della pipeline: si builda una volta sulla
workstation e si copia sulle board. La misura usa il binario C++ del vendor,
non il binding Python: l'overhead di quest'ultimo e' costante e su un modello
nano quantizzato su CPU ARM sottostima proprio lo speedup della quantizzazione
che si vuole misurare (per quantificarlo esiste il backend `onnxruntime_py`).
"""

from __future__ import annotations

import logging
import shlex
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

        precision = cfg.quantization.precision
        if precision == "fp32":
            return self._export(cfg, Path(src), target)
        if precision == "fp16":
            return self._export(cfg, Path(src), target, quantize="fp16")
        if precision == "int8":
            from ..stages.quantize import quantize_onnx_static

            base = self._export_fp32(cfg, Path(src), dst)
            out = quantize_onnx_static(base, target, cfg)
            base.unlink(missing_ok=True)
            return out
        raise ExportFailed(f"precisione non gestita da {self.name}: {precision}")

    def _export_fp32(self, cfg, src: Path, dst: Path) -> Path:
        """ONNX FP32 intermedio: base per l'INT8 e per la build TensorRT."""
        return self._export(cfg, src, dst / f"{cfg.model.name}_fp32_base.onnx")

    def _export(self, cfg, src: Path, target: Path,
                quantize: str | None = None) -> Path:
        """Export ONNX con Ultralytics.

        L'FP16 lo fa Ultralytics: su CPU converte il grafo con
        `onnxruntime.transformers.float16`, su GPU esporta direttamente il
        modello in half. `onnxconverter_common` produceva cast sbagliati
        attorno ai `Resize` e un ONNX che ONNX Runtime rifiuta di caricare.
        """
        from ultralytics import YOLO

        model = YOLO(str(src), task="detect")
        produced = Path(model.export(
            format="onnx",
            imgsz=int(cfg.model.imgsz),
            opset=int(cfg.backend.build.opset),
            batch=1,
            simplify=True,
            dynamic=False,
            device="cpu",
            quantize=quantize,
            # Da Ultralytics 8.4 il default nms=None esporta la testa
            # one-to-many: la one-to-one (senza NMS) va chiesta esplicitamente.
            nms=False,
        ))
        if produced.resolve() != target.resolve():
            shutil.move(str(produced), target)
        if quantize == "fp16":
            self._check_fp16(target)
        return target

    @staticmethod
    def _check_fp16(path: Path) -> None:
        """Se la conversione fallisce Ultralytics avvisa e salva l'FP32:
        senza questo controllo un artefatto FP32 verrebbe misurato come FP16."""
        from ..validation.graph import inspect_onnx

        found = inspect_onnx(path).get("precision")
        if found != "fp16":
            path.unlink(missing_ok=True)
            raise ExportFailed(
                f"export FP16 di Ultralytics fallito: {path.name} e' {found}"
            )

    # --- misura ----------------------------------------------------------
    def build_cmd(self, cfg, artifact: Path) -> str:
        ct = resolve_compute(cfg)
        bench = cfg.backend.benchmark
        ep = EP_BY_DEVICE.get(ct.get("device"), "cpu")
        threads = int(ct.get("n_cores") or 1)
        # installato nel venv da scripts/remote/build_ort_perf_test.sh
        template = self.with_venv_tool(cfg, " ".join(str(bench.cmd).split()))
        return template.format(
            model=shlex.quote(str(artifact)),
            ep=ep,
            iters=int(bench.iters),
            threads=threads,
        )

    def parse(self, stdout: str) -> LatencyResult:
        """`onnxruntime_perf_test` riporta la media in ms e i percentili in s. Le versioni
        recenti scrivono `Average inference time cost total:`, stessa media per inferenza."""
        mean = require(
            search(r"Average inference time cost(?: total)?:\s*([\d.]+)\s*ms", stdout),
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
        require(median, "P50 Latency (serve -s in command line)", stdout)
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
