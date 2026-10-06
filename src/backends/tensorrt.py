"""TensorRT — engine compilato sulla board, misura con `trtexec`.

Un engine e' legato a GPU, architettura e versione della libreria: compilarlo
sulla workstation e copiarlo sulla Jetson produce un file che non si carica, o
peggio si carica con prestazioni diverse. La workstation produce solo l'ONNX,
la board produce l'engine.

Due modi per l'INT8, entrambi con il calibration set di tutti gli altri artefatti INT8:

- `tensorrt`, **precisione esplicita**: l'ONNX e' gia' QDQ (lo stesso grafo quantizzato di
  onnxruntime, reso simmetrico) e TensorRT esegue le scale che trova;
- `tensorrt_implicit`, **calibrazione implicita** come Ultralytics su TensorRT 10: ONNX FP32,
  scale calcolate dal calibratore di TensorRT sulla workstation (cache passata a
  `trtexec --calib`), precisione scelta da TensorRT layer per layer.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from ..errors import ExportFailed
from ..jsonio import atomic_write_json, read_json
from ..remote.connection import is_remote, remote_workdir
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
        # precisioni dei layer. Per int8 serve invece il grafo QDQ, e in INT8
        # simmetrico: quello di onnxruntime ha attivazioni UINT8, che TensorRT
        # rifiuta. Si quantizzano solo i nodi di `int8.quantize_op_types`.
        base = onnx_backend._export_fp32(cfg, Path(src), onnx_dir)
        if cfg.quantization.precision != "int8":
            return base
        from ..stages.quantize import quantize_onnx_static

        target = onnx_dir / f"{cfg.model.name}_{cfg.quantization.name}.onnx"
        op_types = (cfg.backend.get("int8") or {}).get("quantize_op_types")
        out = quantize_onnx_static(base, target, cfg, symmetric=True,
                                   op_types=list(op_types) if op_types else None)
        base.unlink(missing_ok=True)
        return out

    def _build_inputs(self, conn, cfg, src: Path, dst: Path) -> tuple[str, list[str]]:
        """(ONNX sulla board, flag di trtexec per la precisione)."""
        flags = list(cfg.quantization.backend_args.get("tensorrt", []) or [])
        if (cfg.quantization.precision == "int8" and (cfg.backend.get("int8") or {}).get("fp16")
                and "--fp16" not in flags):
            flags.append("--fp16")
        return str(src), flags

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
        # Sulla workstation (niente `hardware.remote`) l'engine resta accanto
        # agli altri file dell'export.
        remote_dir = (f"{remote_workdir(cfg)}/exports/{dst.name}"
                      if is_remote(cfg) else str(dst))
        engine = f"{remote_dir}/{cfg.model.name}_{cfg.quantization.name}.engine"
        conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)

        onnx, flags = self._build_inputs(conn, cfg, Path(src), dst)
        cmd = (
            f"{trtexec} --onnx={shlex.quote(onnx)} "
            f"--saveEngine={shlex.quote(engine)} "
            f"--memPoolSize=workspace:{int(cfg.backend.build.workspace_mb)} "
            f"{' '.join(shlex.quote(f) for f in flags)} --verbose"
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
            "host": cfg.hardware.remote.host if is_remote(cfg) else "localhost",
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


# Layer della testa fuori da INT8 nella calibrazione implicita, come Ultralytics: i Sigmoid in
# FP32 (score calibrati), le conv finali (`.2/Conv`) e la DFL in FP16. Solo la testa: ogni SiLU
# contiene un Sigmoid, e vincolarli tutti costerebbe la velocita' INT8 di backbone e neck.
def head_precisions(layer_names_types) -> dict[str, str]:
    """[(nome, tipo)] dei layer TensorRT -> {nome: fp32|fp16} per la testa (l'ultimo /model.N/)."""
    indices = [int(m.group(1)) for n, _ in layer_names_types
               if (m := re.match(r"/model\.(\d+)/", n))]
    head = f"/model.{max(indices)}/" if indices else "/"
    out = {}
    for name, kind in layer_names_types:
        if not name.startswith(head):
            continue
        if kind == "ACTIVATION" and "sigmoid" in name.lower():
            out[name] = "fp32"
        elif kind == "CONVOLUTION" and (name.endswith(".2/Conv") or "/dfl/" in name):
            out[name] = "fp16"
    return out


@register
class TensorRTImplicitBackend(TensorRTBackend):
    name = "tensorrt_implicit"

    def prepare(self, cfg, src: Path, dst: Path) -> Path:
        """ONNX FP32 e, per int8, la cache di calibrazione di TensorRT con i vincoli della testa:
        tutto calcolato sulla workstation, la board riceve ONNX e cache."""
        onnx_dir = Path(dst) / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        base = OnnxRuntimeBackend(cfg)._export_fp32(cfg, Path(src), onnx_dir)
        if cfg.quantization.precision == "int8":
            self._calibrate(cfg, base, onnx_dir)
        return base

    def _calibrate(self, cfg, onnx: Path, onnx_dir: Path) -> None:
        import tensorrt as trt
        import torch

        from ..stages.quantize import calibration_files, preprocess, write_manifest

        files = calibration_files(cfg)
        if not files:
            raise ExportFailed("calibration set vuoto: controllare quantization.calibration")
        opts = cfg.backend.get("int8") or {}
        algo = {"minmax": trt.CalibrationAlgoType.MINMAX_CALIBRATION,
                "entropy2": trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2}[
            str(opts.get("calibration", "minmax"))]
        imgsz = int(cfg.model.imgsz)
        cache_path = onnx_dir / "calibration.cache"

        class Calibrator(trt.IInt8Calibrator):
            def __init__(self):
                trt.IInt8Calibrator.__init__(self)
                self.it, self.batch = iter(files), None

            def get_algorithm(self):
                return algo

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                path = next(self.it, None)
                if path is None:
                    return None
                self.batch = torch.from_numpy(preprocess(path, imgsz)).cuda()
                return [int(self.batch.data_ptr())]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                cache_path.write_bytes(bytes(cache))

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ExportFailed(f"TensorRT non legge {onnx.name}: {errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     int(cfg.backend.build.workspace_mb) << 20)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator()
        log.info("calibrazione implicita di TensorRT su %d immagini (%s)", len(files),
                 opts.get("calibration", "minmax"))
        if builder.build_serialized_network(network, config) is None or not cache_path.exists():
            raise ExportFailed("calibrazione di TensorRT fallita: cache non scritta")
        layers = [(network.get_layer(i).name, network.get_layer(i).type.name)
                  for i in range(network.num_layers)]
        precisions = head_precisions(layers) if opts.get("head_fp32", True) else {}
        atomic_write_json(onnx_dir / "head_precisions.json", precisions)
        write_manifest(onnx_dir, files)

    def _build_inputs(self, conn, cfg, src: Path, dst: Path) -> tuple[str, list[str]]:
        onnx, flags = super()._build_inputs(conn, cfg, src, dst)
        if cfg.quantization.precision != "int8":
            return onnx, flags
        from ..remote.sync import sync_artifact

        onnx_dir = Path(dst) / "onnx"
        cache = sync_artifact(conn, cfg, onnx_dir / "calibration.cache", key=Path(dst).name)
        flags.append(f"--calib={cache}")
        precisions = read_json(onnx_dir / "head_precisions.json") or {}
        if precisions:
            spec = ",".join(f"{name}:{p}" for name, p in precisions.items())
            flags += ["--precisionConstraints=obey", f"--layerPrecisions={spec}",
                      f"--layerOutputTypes={spec}"]
        return onnx, flags
