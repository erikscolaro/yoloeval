"""ncnn — export con pnnx (Ultralytics) sulla workstation, misura con `benchncnn`.

fp16: export di Ultralytics con pesi fp16. int8: export fp32, tabella di calibrazione con
`ncnn2table` sullo stesso calibration set degli altri backend, poi `ncnn2int8`. ncnn quantizza
solo conv e depthwise: concat, attivazioni e coda della testa restano float.

Ultralytics esporta ncnn senza la parte one-to-one (`end2end: false`): l'uscita sono le 8400
predizioni e la mAP passa per l'NMS. Il top-k che OpenVINO ha nel grafo qui manca.

`benchncnn` usa pesi vuoti (conta solo la forma), fp16 storage/aritmetica e int8 accesi, e
riporta solo min, max e media: la mediana resta `None`. I tool non sono nel wheel pip: li compila
scripts/remote/build_ncnn.sh.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..measure.compute import resolve_compute
from ..models import deploy_hw
from . import register
from .base import Backend, LatencyResult, require

log = logging.getLogger(__name__)

PARAM = "model.ncnn.param"
BIN = "model.ncnn.bin"


@register
class NcnnBackend(Backend):
    name = "ncnn"
    export_format = "ncnn"
    builds_on_target = False
    scope = "end_to_end_with_transfers"

    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        from ultralytics import YOLO

        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        # Ultralytics carica ncnn solo da cartelle che finiscono in _ncnn_model
        target = dst / f"{cfg.model.name}_{cfg.quantization.name}_ncnn_model"
        if target.exists():
            log.info("artefatto gia' presente: %s", target.name)
            return target

        int8 = cfg.quantization.precision == "int8"
        produced = Path(YOLO(str(src), task="detect").export(
            format="ncnn",
            imgsz=deploy_hw(cfg),
            quantize="fp32" if int8 else cfg.quantization.precision,
            batch=1,
            nms=False,
            max_det=int(cfg.export_max_det),
        ))
        if int8:
            self._quantize(cfg, produced, dst)
        shutil.move(str(produced), target)
        return target

    @staticmethod
    def _quantize(cfg, model: Path, dst: Path) -> None:
        """Sostituisce in `model` param e bin fp32 con quelli int8."""
        from ..stages.quantize import calibration_files, write_manifest

        files = calibration_files(cfg)
        write_manifest(dst, files)
        listing = dst / "calib.txt"
        listing.write_text("".join(f"{f}\n" for f in files), encoding="utf-8")
        h, w = deploy_hw(cfg)
        param, weights, table = model / PARAM, model / BIN, dst / "ncnn.table"
        bin_dir = Path(sys.executable).parent   # gira sulla workstation, nel suo venv (build_ncnn.sh)
        # shortcut: ncnn2table stira le immagini (640x512) a 640x640 invece del letterbox; se la mAP
        # int8 risultasse bassa, passargli immagini gia' con il letterbox
        subprocess.run([str(bin_dir / "ncnn2table"), str(param), str(weights), str(listing), str(table),
                        "mean=[0,0,0]", "norm=[0.003921569,0.003921569,0.003921569]",
                        f"shape=[{w},{h},3]", "pixel=RGB", "thread=8", "method=kl"],
                       check=True, capture_output=True)
        subprocess.run([str(bin_dir / "ncnn2int8"), str(param), str(weights), str(param), str(weights),
                        str(table)], check=True, capture_output=True)

    def build_cmd(self, cfg, artifact: Path) -> str:
        ct = resolve_compute(cfg)
        bench = cfg.backend.benchmark
        h, w = deploy_hw(cfg)
        template = self.with_venv_tool(cfg, " ".join(str(bench.cmd).split()))
        # benchncnn scrive su stderr
        return template.format(
            iters=int(bench.iters),
            threads=int(ct.get("n_cores") or 1),
            param=f"{artifact}/{PARAM}",
            shape=f"[{w},{h},3]",
        ) + " 2>&1"

    def parse(self, stdout: str) -> LatencyResult:
        m = re.search(r"min =\s*([\d.]+)\s+max =\s*([\d.]+)\s+avg =\s*([\d.]+)", stdout or "")
        avg = require(float(m.group(3)) if m else None, "avg di benchncnn", stdout)
        loops = re.search(r"loop_count = (\d+)", stdout or "")
        return LatencyResult(
            mean_ms=avg,
            min_ms=float(m.group(1)),
            max_ms=float(m.group(2)),
            throughput_qps=1000.0 / avg if avg else None,
            iters=int(loops.group(1)) if loops else None,
            scope=self.scope,
            tool="benchncnn",
            raw_stdout=stdout,
        )

    def inspect(self, artifact: Path) -> dict:
        """Il param non dichiara le forme; la precisione si legge dai flag int8 delle conv
        (`8=` int8_scale_term, messo da ncnn2int8) e dal metadata di Ultralytics."""
        path = Path(artifact)
        text = (path / PARAM).read_text(encoding="utf-8")
        meta = (path / "metadata.yaml").read_text(encoding="utf-8") if (path / "metadata.yaml").exists() else ""
        int8 = re.search(r"^Convolution\S*\s.*\s8=\d", text, re.M) is not None
        precision = "int8" if int8 else "fp16" if re.search(r"quantize: 16|half: true", meta) else "fp32"
        return {
            "actual_e2e": False,
            "head": "one_to_many",
            "nms_in_graph": False,
            "precision": precision,
            "input_shape": [],
            "output_names": ["out0"],
            "output_shapes": [],
            "source": "ncnn_param",
        }
