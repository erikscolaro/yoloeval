"""TensorRT — engine compilato sulla board, misura con `trtexec`.

Un engine e' legato a GPU, architettura e versione della libreria: compilarlo
sulla workstation e copiarlo sulla Jetson produce un file che non si carica, o
peggio si carica con prestazioni diverse. La workstation produce solo l'ONNX,
la board produce l'engine.

INT8 per **precisione esplicita**: si parte da un ONNX gia' QDQ invece di
passare `--int8` su un ONNX float e lasciare che `trtexec` calibri da solo con
dati casuali. Cosi' il calibration set e' lo stesso di tutti gli altri
artefatti INT8 della matrice, che e' la condizione per poterli confrontare.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from ..errors import ExportFailed
from ..jsonio import atomic_write_json, read_json
from . import register
from .base import Backend, LatencyResult, require, search
from .onnxruntime import OnnxRuntimeBackend

log = logging.getLogger(__name__)

TRTEXEC_CANDIDATES = ("trtexec", "/usr/src/tensorrt/bin/trtexec")


@register
class TensorRTBackend(Backend):
    name = "tensorrt"
    export_format = "engine"
    builds_on_target = True
    scope = "end_to_end_with_transfers"

    # --- preparazione dell'artefatto portabile --------------------------
    def prepare(self, cfg, src: Path, dst: Path) -> Path:
        """ONNX alla precisione richiesta, prodotto sulla workstation."""
        onnx_backend = OnnxRuntimeBackend(cfg)
        onnx_dir = Path(dst) / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        # Per fp16 l'ONNX resta fp32: e' `trtexec --fp16` a scegliere le
        # precisioni dei layer. Per int8 serve invece il grafo QDQ.
        if cfg.quantization.precision == "int8":
            return onnx_backend.export(None, cfg, Path(src), onnx_dir)
        produced = onnx_backend._export_fp32(cfg, Path(src), onnx_dir)
        return produced

    # --- build on target -------------------------------------------------
    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        """`src` e' l'ONNX gia' sincronizzato sulla board."""
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        marker = dst / "remote.json"
        prev = read_json(marker)
        if prev and conn.run(f"test -f {shlex.quote(prev['remote_path'])}",
                             hide=True, warn=True).ok:
            log.info("engine gia' presente sulla board: %s", prev["remote_path"])
            return Path(prev["remote_path"])

        trtexec = self._trtexec(conn)
        remote_dir = f"{cfg.hardware.remote.workdir}/exports/{dst.name}"
        engine = f"{remote_dir}/{cfg.model.name}_{cfg.quantization.name}.engine"
        conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)

        flags = list(cfg.quantization.backend_args.get("tensorrt", []) or [])
        cmd = (
            f"{trtexec} --onnx={shlex.quote(str(src))} "
            f"--saveEngine={shlex.quote(engine)} "
            f"--memPoolSize=workspace:{int(cfg.backend.build.workspace_mb)} "
            f"{' '.join(flags)} --verbose"
        )
        log.info("build engine su %s", cfg.hardware.board)
        r = conn.run(cmd, hide=True, warn=True,
                     timeout=int(cfg.backend.build.timeout_s))
        (dst / "build.log").write_text(
            (r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8"
        )
        if r.failed:
            raise ExportFailed(
                f"trtexec build fallita ({r.return_code}), log in "
                f"{dst / 'build.log'}"
            )

        atomic_write_json(marker, {
            "remote_path": engine,
            "board": cfg.hardware.board,
            "host": cfg.hardware.remote.host,
            "built_from": str(src),
            "flags": flags,
        })
        self._dump_inspection(conn, cfg, engine, dst)
        return Path(engine)

    def _trtexec(self, conn) -> str:
        for candidate in TRTEXEC_CANDIDATES:
            if conn.run(f"command -v {candidate} || test -x {candidate}",
                        hide=True, warn=True).ok:
                return candidate
        raise ExportFailed(
            "trtexec non trovato sulla board (atteso in PATH o in "
            "/usr/src/tensorrt/bin)"
        )

    def _dump_inspection(self, conn, cfg, engine: str, dst: Path) -> dict:
        """Ispeziona l'engine **sulla board** e porta a casa il risultato.

        Sulla workstation l'engine non e' caricabile, quindi l'ispezione
        viaggia come sidecar JSON accanto al meta dell'export.
        """
        from ..validation.graph import inspect_from_bindings, parse_trtexec_bindings

        trtexec = self._trtexec(conn)
        r = conn.run(
            f"{trtexec} --loadEngine={shlex.quote(engine)} --iterations=1 "
            f"--warmUp=0 --duration=0 --avgRuns=1",
            hide=True, warn=True, timeout=600,
        )
        bindings = parse_trtexec_bindings((r.stdout or "") + (r.stderr or ""))
        info = inspect_from_bindings(bindings, nc=int(cfg.model.nc))
        atomic_write_json(dst / "inspection.json", info)
        return info

    # --- misura ----------------------------------------------------------
    def build_cmd(self, cfg, artifact: Path) -> str:
        bench = cfg.backend.benchmark
        template = " ".join(str(bench.cmd).split())
        cmd = template.format(
            engine=shlex.quote(str(artifact)),
            iters=int(bench.iters),
            warmup_ms=int(bench.warmup_ms),
        )
        # Percentili espliciti: la lista di default cambia fra versioni, e la
        # coda conta piu' del valore centrale.
        if "--percentile" not in cmd:
            cmd += " --percentile=90,95,99"
        return cmd

    def parse(self, stdout: str) -> LatencyResult:
        """Blocco `=== Performance summary ===` di trtexec.

        La riga `Latency` include i trasferimenti host<->device solo grazie a
        `--noDataTransfers=false`; `GPU Compute Time` e' il solo motore e viene
        riportato a parte, non al posto della latenza.
        """
        lat_line = require(
            search(r"Latency:\s*(min\s*=.*)$", stdout, flags=re.MULTILINE),
            "riga Latency", stdout,
        )
        gpu_line = search(
            r"GPU Compute Time:\s*(min\s*=.*)$", stdout, flags=re.MULTILINE
        )

        def field(line, label):
            value = search(rf"{label}\s*=\s*([\d.]+)\s*ms", line or "")
            return float(value) if value else None

        def pctl(line, p):
            value = search(rf"percentile\({p}%\)\s*=\s*([\d.]+)\s*ms", line or "")
            return float(value) if value else None

        mean = require(field(lat_line, "mean"), "Latency mean", stdout)
        qps = search(r"Throughput:\s*([\d.]+)\s*qps", stdout)
        iters = search(r"--iterations=(\d+)", stdout)
        return LatencyResult(
            mean_ms=mean,
            median_ms=field(lat_line, "median"),
            p90_ms=pctl(lat_line, 90),
            p95_ms=pctl(lat_line, 95),
            p99_ms=pctl(lat_line, 99),
            min_ms=field(lat_line, "min"),
            max_ms=field(lat_line, "max"),
            throughput_qps=float(qps) if qps else None,
            gpu_compute_ms=field(gpu_line, "mean"),
            iters=int(iters) if iters else None,
            scope=self.scope,
            tool="trtexec",
            raw_stdout=stdout,
        )

    # --- ispezione -------------------------------------------------------
    def inspect(self, artifact: Path) -> dict:
        """Legge il sidecar prodotto dalla board al momento della build."""
        for candidate in (
            Path(artifact).parent / "inspection.json",
            Path(artifact).with_name("inspection.json"),
        ):
            info = read_json(candidate)
            if info:
                return info
        return {
            "actual_e2e": None,
            "head": "unknown",
            "precision": None,
            "input_shape": [],
            "output_names": [],
            "source": "not_inspected",
            "reason": "inspection.json assente: l'engine e' ispezionabile solo "
                      "sulla board che lo ha compilato",
        }

    def validate(self, conn, cfg, artifact: Path, ref) -> dict:
        """Il confronto avviene sulla board: e' li' che vive l'engine."""
        from ..validation.numerical import compare_with_reference

        return compare_with_reference(conn, cfg, artifact, ref, device="cuda")
