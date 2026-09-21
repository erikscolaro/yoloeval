"""Axelera Metis — compilazione e misura sulla board, dentro il container.

Tre vincoli che vengono dall'hardware, non da una scelta di progetto:

1. La compilazione richiede la scheda **fisicamente presente**: il Voyager SDK
   quantizza e compila per l'architettura mixed-precision dell'AIPU leggendo il
   device.
2. La quantizzazione e' **forzata a INT8** dall'AIPU: non e' un asse, e le
   celle Axelera non sono confrontabili riga per riga con l'INT8 di TensorRT,
   perche' lo schema di quantizzazione e' diverso.
3. Su Raspberry Pi OS il SDK non e' supportato: tutto gira in un container
   Ubuntu 22.04 con accesso al device PCIe. Il container viene **fermato** e
   non rimosso a fine sweep — ricostruirlo costa, ma lasciarlo acceso durante
   le celle CPU-only falsa la misura, perche' daemon e processi interni
   competono per CPU e memoria.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from ..errors import ExportFailed
from ..jsonio import atomic_write_json, read_json
from . import register
from .base import Backend, LatencyResult, percentiles, require, search

log = logging.getLogger(__name__)

CONTAINER_NAME = "yolo-bench-axelera"
CONTAINER_WORKDIR = "/bench"
DEVICE_GLOB = "/dev/metis*"


def container_enabled(cfg) -> bool:
    return bool(cfg.backend.get("build", {}).get("container")) and bool(
        cfg.hardware.get("provision", {}).get("container")
    )


def to_container_path(cfg, path) -> str:
    """Il workdir della board e' montato in /bench dentro il container."""
    workdir = str(cfg.hardware.remote.workdir).rstrip("/")
    text = str(path)
    if text.startswith(workdir):
        return CONTAINER_WORKDIR + text[len(workdir):]
    return text


def ensure_container(conn, cfg) -> None:
    """Avvia il container se esiste, altrimenti lo crea. Idempotente."""
    name = CONTAINER_NAME
    r = conn.run(f"docker ps -q -f name=^{name}$", hide=True, warn=True)
    if r.ok and r.stdout.strip():
        return
    r = conn.run(f"docker ps -aq -f name=^{name}$", hide=True, warn=True)
    if r.ok and r.stdout.strip():
        conn.run(f"docker start {name}", hide=True, warn=True)
        return
    device = conn.run(f"ls {DEVICE_GLOB} 2>/dev/null | head -1", hide=True, warn=True)
    dev = device.stdout.strip() if device.ok and device.stdout.strip() else None
    if not dev:
        raise ExportFailed(
            "nessun device Metis sotto /dev: verificare metis-dkms e "
            "`modprobe metis` prima di usare il backend axelera"
        )
    image = cfg.backend.build.image
    conn.run(
        f"docker run -d --name {name} --device {dev} "
        f"-v {shlex.quote(str(cfg.hardware.remote.workdir))}:{CONTAINER_WORKDIR} "
        f"--network host {shlex.quote(str(image))} sleep infinity",
        hide=True,
    )


def stop_container(conn, cfg) -> None:
    """Fermato, non rimosso: ricostruirlo a ogni lancio non ha senso."""
    conn.run(f"docker stop {CONTAINER_NAME}", hide=True, warn=True)


def in_container(cfg, cmd: str) -> str:
    return (
        f"docker exec -w {CONTAINER_WORKDIR} {CONTAINER_NAME} "
        f"bash -lc {shlex.quote(cmd)}"
    )


@register
class AxeleraBackend(Backend):
    name = "axelera"
    export_format = "axelera"
    builds_on_target = True
    scope = "end_to_end_with_transfers"

    def _wrap(self, cfg, cmd: str) -> str:
        return in_container(cfg, cmd) if container_enabled(cfg) else cmd

    # --- build on target -------------------------------------------------
    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        """`src` e' il `.pt` gia' sincronizzato sulla board."""
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        marker = dst / "remote.json"
        prev = read_json(marker)
        if prev and conn.run(f"test -e {shlex.quote(prev['remote_path'])}",
                             hide=True, warn=True).ok:
            log.info("modello axelera gia' presente: %s", prev["remote_path"])
            return Path(prev["remote_path"])

        if container_enabled(cfg):
            ensure_container(conn, cfg)
        self._require_device(conn, cfg)

        remote_dir = f"{cfg.hardware.remote.workdir}/exports/{dst.name}"
        conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)

        build = cfg.backend.build
        cmd = (
            f"yolo export model={to_container_path(cfg, src)} format=axelera "
            f"imgsz={int(cfg.model.imgsz)} batch=1 "
            f"int8=True fraction={int(build.calib_fraction)} "
            f"data={to_container_path(cfg, cfg.hardware.remote.workdir)}"
            f"/data/{cfg.dataset.name}/.bench_data.yaml "
            f"project={to_container_path(cfg, remote_dir)} name=export exist_ok=True"
        )
        log.info("compilazione axelera su %s (SDK %s)",
                 cfg.hardware.board, build.sdk_version)
        r = conn.run(self._wrap(cfg, cmd), hide=True, warn=True, timeout=7200)
        (dst / "build.log").write_text(
            (r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8"
        )
        if r.failed:
            raise ExportFailed(
                f"export axelera fallito ({r.return_code}), log in "
                f"{dst / 'build.log'}"
            )

        found = conn.run(
            f"find {shlex.quote(remote_dir)} -name '*.axm' | head -1",
            hide=True, warn=True,
        )
        artifact = found.stdout.strip() if found.ok else ""
        if not artifact:
            raise ExportFailed(f"nessun .axm prodotto sotto {remote_dir}")

        atomic_write_json(marker, {
            "remote_path": artifact,
            "board": cfg.hardware.board,
            "host": cfg.hardware.remote.host,
            "sdk_version": str(build.sdk_version),
            "quantize_bits": int(build.quantize),
            "calib_fraction": int(build.calib_fraction),
        })
        atomic_write_json(dst / "inspection.json", {
            "actual_e2e": None,
            "head": "unknown",
            "precision": "int8",
            "input_shape": [1, 3, int(cfg.model.imgsz), int(cfg.model.imgsz)],
            "output_names": [],
            "source": "voyager_sdk",
            "reason": "il formato .axm non espone il grafo: la testa va letta "
                      "dal log di export",
            "sdk_version": str(build.sdk_version),
        })
        return Path(artifact)

    def _require_device(self, conn, cfg) -> None:
        """`axdevice` prima di ogni sweep che coinvolga l'acceleratore.

        Kernel driver e SDK si aggiornano separatamente e nulla verifica la
        coerenza a install time: un driver sotto la versione minima lascia il
        device inaccessibile con un errore di connessione.
        """
        r = conn.run(self._wrap(cfg, "axdevice"), hide=True, warn=True)
        if r.failed:
            raise ExportFailed(
                "axdevice non vede la scheda Metis: verificare metis-dkms "
                "(>= 1.6.2 per SDK 1.8) e `modprobe metis`"
            )
        log.debug("axdevice: %s", r.stdout.strip())

    # --- misura ----------------------------------------------------------
    def prepare_input(self, conn, cfg) -> None:
        """Il predict di Ultralytics legge da una cartella: va creata."""
        from ..remote.sync import ensure_bench_input

        if container_enabled(cfg):
            ensure_container(conn, cfg)
        ensure_bench_input(conn, cfg, n=int(cfg.backend.benchmark.iters))

    def build_cmd(self, cfg, artifact: Path) -> str:
        source = (
            f"{cfg.hardware.remote.workdir}/data/{cfg.dataset.name}/bench_input"
        )
        cmd = (
            f"yolo predict model={to_container_path(cfg, artifact)} "
            f"source={to_container_path(cfg, source)} "
            f"imgsz={int(cfg.model.imgsz)} "
            # soglie operative, identiche in tutte le celle di latenza
            f"conf={cfg.eval.bench_conf} iou={cfg.eval.bench_iou} "
            f"max_det={cfg.eval.max_det} "
            f"save=False verbose=True stream=False"
        )
        return self._wrap(cfg, cmd)

    def parse(self, stdout: str) -> LatencyResult:
        """Ultralytics stampa una riga per immagine e un riepilogo finale."""
        samples = [
            float(m.group(1))
            for m in re.finditer(r"([\d.]+)ms\s*$", stdout or "", re.MULTILINE)
        ]
        if samples:
            # I percentili vengono dai campioni davvero misurati; con poche
            # immagini restano validi ma poco informativi, e `iters` nel
            # risultato dice quante erano.
            if len(samples) < 30:
                log.warning("solo %d campioni di latenza: i percentili alti "
                            "sono poco significativi", len(samples))
            stats = percentiles(samples)
            return LatencyResult(
                **stats, iters=len(samples), scope=self.scope,
                tool="ultralytics_predict", raw_stdout=stdout,
            )
        mean = require(
            search(r"Speed:.*?([\d.]+)ms inference", stdout),
            "riepilogo Speed di Ultralytics", stdout,
        )
        return LatencyResult(
            mean_ms=float(mean), iters=len(samples) or None, scope=self.scope,
            tool="ultralytics_predict", raw_stdout=stdout,
        )

    def inspect(self, artifact: Path) -> dict:
        info = read_json(Path(artifact).parent / "inspection.json")
        return info or {
            "actual_e2e": None,
            "head": "unknown",
            "precision": "int8",
            "input_shape": [],
            "output_names": [],
            "source": "not_inspected",
        }

    def validate(self, conn, cfg, artifact: Path, ref) -> dict:
        from ..validation.numerical import compare_with_reference

        return compare_with_reference(conn, cfg, artifact, ref, device="metis")
