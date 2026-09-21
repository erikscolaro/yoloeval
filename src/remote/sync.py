"""Sincronizzazione di dataset e artefatti verso la board.

`rsync` e' incrementale e riprendibile: al secondo lancio non trasferisce
nulla. Il dataset viene sincronizzato una sola volta, al primo sweep che lo
richiede, e marcato con un file sentinella.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from .connection import is_local_conn, is_remote

log = logging.getLogger(__name__)


def _rsync(src: str, dst: str, timeout_s: int = 7200) -> None:
    cmd = [
        "rsync", "-az", "--partial", "--info=progress2",
        "--exclude", ".dataset_hash.tmp",
        src, dst,
    ]
    log.info("rsync %s -> %s", src, dst)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"rsync fallito ({r.returncode}):\n{r.stderr}")


def remote_dataset_path(cfg) -> str:
    return f"{cfg.hardware.remote.workdir}/data/{cfg.dataset.name}"


def ensure_dataset(conn, cfg) -> str:
    """Copia il dataset sulla board, una volta sola.

    Se la board monta una microSD l'I/O puo' inquinare le misure: preferire
    NVMe per dataset e workdir dove disponibile.
    """
    if is_local_conn(conn) or not is_remote(cfg):
        return str(cfg.dataset.local_path)

    remote = remote_dataset_path(cfg)
    if conn.run(f"test -f {remote}/.complete", hide=True, warn=True).ok:
        log.debug("dataset gia' presente su %s", cfg.hardware.board)
        return remote

    local_path = Path(cfg.dataset.local_path)
    if not local_path.is_dir():
        raise FileNotFoundError(f"dataset non trovato sulla workstation: {local_path}")

    conn.run(f"mkdir -p {remote}", hide=True)
    _rsync(f"{local_path}/", f"{cfg.hardware.remote.host}:{remote}/")
    conn.run(f"touch {remote}/.complete", hide=True)
    return remote


def sync_artifact(conn, cfg, artifact: str | Path, subdir: str = "artifacts") -> str:
    """Copia un artefatto sulla board e ritorna il path remoto."""
    artifact = Path(artifact)
    if is_local_conn(conn) or not is_remote(cfg):
        return str(artifact)

    remote_dir = f"{cfg.hardware.remote.workdir}/{subdir}"
    conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)
    src = f"{artifact}/" if artifact.is_dir() else str(artifact)
    dst = f"{cfg.hardware.remote.host}:{remote_dir}/{artifact.name}"
    _rsync(src, dst + ("/" if artifact.is_dir() else ""))
    return f"{remote_dir}/{artifact.name}"


def fetch(conn, cfg, remote_path: str, local_path: str | Path) -> Path:
    """Recupera un file dalla board (artefatti compilati on-target, log)."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not is_remote(cfg):
        if Path(remote_path).resolve() != local_path.resolve():
            conn.get(remote_path, str(local_path))
        return local_path
    _rsync(f"{cfg.hardware.remote.host}:{remote_path}", str(local_path))
    return local_path
