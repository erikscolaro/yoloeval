"""ExecuTorch — `.pte` esportato dal `.pt`, misura con il runner C++ della build.

Il runner non e' distribuito nei wheel: fa parte della build di ExecuTorch
sulla macchina di destinazione. Il config dichiara come si chiama; se non c'e',
la cella fallisce con un messaggio esplicito invece di restituire un numero.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..errors import ExportFailed
from . import register
from .base import Backend, LatencyResult, percentiles, require, search

log = logging.getLogger(__name__)


@register
class ExecuTorchBackend(Backend):
    name = "executorch"
    export_format = "pte"
    builds_on_target = False
    scope = "end_to_end_with_transfers"

    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / f"{cfg.model.name}_{cfg.quantization.name}.pte"
        if target.exists():
            return target
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ExportFailed("ultralytics non disponibile") from exc

        model = YOLO(str(src), task="detect")
        produced = model.export(
            format="executorch",
            imgsz=int(cfg.model.imgsz),
            int8=cfg.quantization.precision == "int8",
            half=cfg.quantization.precision == "fp16",
            batch=1,
        )
        produced = Path(produced)
        if produced.is_dir():
            files = sorted(produced.glob("*.pte"))
            if not files:
                raise ExportFailed(f"nessun .pte prodotto in {produced}")
            produced = files[0]
        produced.replace(target)
        return target

    def build_cmd(self, cfg, artifact: Path) -> str:
        bench = cfg.backend.benchmark
        template = " ".join(str(bench.cmd).split())
        return template.format(
            model=artifact,
            iters=int(bench.iters),
            warmup_iters=int(bench.warmup_iters),
        )

    def parse(self, stdout: str) -> LatencyResult:
        """Il runner stampa una riga per esecuzione, oppure un riepilogo."""
        samples = [
            float(m.group(1))
            for m in re.finditer(r"(?:Inference|Execution) time:\s*([\d.]+)\s*ms",
                                 stdout or "")
        ]
        if samples:
            stats = percentiles(samples)
            return LatencyResult(
                **stats, iters=len(samples), scope=self.scope,
                tool="executor_runner", raw_stdout=stdout,
            )
        mean = require(
            search(r"[Aa]verage(?: inference)? time:\s*([\d.]+)\s*ms", stdout),
            "tempo medio", stdout,
        )
        return LatencyResult(
            mean_ms=float(mean), scope=self.scope, tool="executor_runner",
            raw_stdout=stdout,
        )

    def inspect(self, artifact: Path) -> dict:
        return {
            "actual_e2e": None,
            "head": "unknown",
            "precision": None,
            "input_shape": [],
            "output_names": [],
            "source": "not_inspected",
            "reason": "il formato .pte non e' ispezionabile con gli strumenti "
                      "usati qui: la testa va dedotta dall'ONNX sorgente",
        }
