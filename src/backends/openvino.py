"""OpenVINO — export IR sulla workstation, misura con `benchmark_app`.

L'IR e' portabile come l'ONNX: si esporta sulla workstation e si copia sulla
board. Su ARM (Pi 5, Jetson) gira il plugin CPU per aarch64, molto meno
ottimizzato di quello x86; l'INT8 in particolare puo' ricadere su kernel a
precisione piu' alta, quindi lo speedup va misurato, non assunto.

`benchmark_app` riporta di default mediana, media, minimo e massimo: i
percentili alti non ci sono, e restano `None` invece di essere ricostruiti da
media e deviazione standard. Il perimetro e' allineato agli altri backend con
`-b 1` e uno stream solo (`-nstreams 1`), che e' quello che farebbe
`-hint latency`: l'hint va pero' messo a `none`, perche' benchmark_app rifiuta
`-nthreads` se c'e' un hint, e senza `-nthreads` il numero di core del compute
target non arriverebbe a OpenVINO.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from ..measure.compute import resolve_compute
from . import register
from .base import Backend, LatencyResult, require, search

log = logging.getLogger(__name__)

DEVICE_BY_COMPUTE = {"cpu": "CPU", "cuda": "GPU", "gpu": "GPU"}


@register
class OpenVINOBackend(Backend):
    name = "openvino"
    export_format = "openvino"
    builds_on_target = False
    scope = "end_to_end_with_transfers"

    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        from ultralytics import YOLO

        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / f"{cfg.model.name}_{cfg.quantization.name}_openvino_model"
        if target.exists():
            log.info("artefatto gia' presente: %s", target.name)
            return target

        int8 = cfg.quantization.precision == "int8"
        data = self._calibration_yaml(cfg, dst) if int8 else None
        model = YOLO(str(src), task="detect")
        produced = Path(model.export(
            format="openvino",
            imgsz=int(cfg.model.imgsz),
            quantize=cfg.quantization.precision,   # "fp32" | "fp16" | "int8"
            batch=1,
            nms=False,            # testa one-to-one, vedi onnxruntime.py
            data=data,
        ))
        if produced.resolve() != target.resolve():
            shutil.move(str(produced), target)
        return target

    @staticmethod
    def _calibration_yaml(cfg, dst: Path) -> str:
        """Data yaml ridotto al calibration set condiviso.

        Con il data yaml del dataset Ultralytics passerebbe a NNCF l'intero
        split `val`: migliaia di immagini, e soprattutto un set diverso da
        quello degli altri backend INT8. Qui `val` punta alla stessa lista
        di `calibration_files`, salvata anche nel manifesto.
        """
        import yaml

        from ..stages.quantize import calibration_files, write_manifest

        files = calibration_files(cfg)
        write_manifest(dst, files)
        listing = dst / "calib.txt"
        listing.write_text("".join(f"{f}\n" for f in files), encoding="utf-8")
        spec = yaml.safe_load(Path(cfg.dataset.yaml).read_text(encoding="utf-8"))
        calib = {
            "path": str(cfg.dataset.local_path),
            "train": str(listing),
            "val": str(listing),
            "nc": spec.get("nc", cfg.dataset.nc),
            "names": spec["names"],
        }
        out = dst / "calib_data.yaml"
        out.write_text(yaml.safe_dump(calib, sort_keys=False), encoding="utf-8")
        return str(out)

    def build_cmd(self, cfg, artifact: Path) -> str:
        ct = resolve_compute(cfg)
        bench = cfg.backend.benchmark
        model = artifact
        if Path(artifact).is_dir():
            xml = sorted(Path(artifact).glob("*.xml"))
            model = xml[0] if xml else artifact
        elif Path(artifact).suffix != ".xml":
            # Directory sulla board, non visibile da qui: l'IR ne contiene
            # uno solo, e il glob lo risolve la shell remota.
            model = f"{artifact}/*.xml"
        # benchmark_app e' un entry point del pacchetto openvino: sta nel
        # bin/ del venv.
        template = self.with_venv_tool(cfg, " ".join(str(bench.cmd).split()))
        return template.format(
            model=model,
            device=DEVICE_BY_COMPUTE.get(ct.get("device"), "CPU"),
            iters=int(bench.iters),
            threads=int(ct.get("n_cores") or 1),
        )

    def parse(self, stdout: str) -> LatencyResult:
        def ms(label):
            value = search(rf"{label}:\s*([\d.]+)\s*ms", stdout)
            return float(value) if value else None

        median = require(ms("Median"), "Latency Median", stdout)
        mean = require(ms("Average"), "Latency Average", stdout)
        fps = search(r"Throughput:\s*([\d.]+)\s*FPS", stdout)
        iters = search(r"Count:\s*(\d+)\s*iterations", stdout)
        # Alcune versioni stampano un percentile esplicito se richiesto
        pct = {}
        for m in re.finditer(r"[Pp]ercentile\s*(\d+)[^\d]*([\d.]+)\s*ms", stdout):
            pct[int(m.group(1))] = float(m.group(2))
        return LatencyResult(
            mean_ms=mean,
            median_ms=median,
            p90_ms=pct.get(90),
            p95_ms=pct.get(95),
            p99_ms=pct.get(99),
            min_ms=ms("Min"),
            max_ms=ms("Max"),
            throughput_qps=float(fps) if fps else None,
            iters=int(iters) if iters else None,
            scope=self.scope,
            tool="benchmark_app",
            raw_stdout=stdout,
        )

    def inspect(self, artifact: Path) -> dict:
        """L'IR dichiara forme e precisione nel proprio XML."""
        path = Path(artifact)
        xml = sorted(path.glob("*.xml"))[0] if path.is_dir() else path
        import xml.etree.ElementTree as ET

        from ..validation.graph import classify_head, infer_nc

        root = ET.parse(xml).getroot()
        outputs, inputs, precisions = [], [], set()
        for layer in root.iter("layer"):
            kind = layer.get("type")
            for port in layer.iter("port"):
                dims = tuple(int(d.text) for d in port.iter("dim"))
                if kind == "Parameter":
                    inputs.append(dims)
                elif kind == "Result":
                    outputs.append(dims)
                if port.get("precision"):
                    precisions.add(port.get("precision"))
        precision = (
            "int8" if {"I8", "U8"} & precisions else
            "fp16" if "FP16" in precisions else "fp32"
        )
        head, e2e = classify_head(list(outputs), nms_in_graph=False,
                                 nc=infer_nc(outputs))
        return {
            "actual_e2e": e2e,
            "head": head,
            "nms_in_graph": False,
            "precision": precision,
            "input_shape": list(inputs[0]) if inputs else [],
            "output_names": [],
            "output_shapes": [list(s) for s in outputs],
            "source": "openvino_ir",
        }
