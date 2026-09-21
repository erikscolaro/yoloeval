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


#: cosa serve sulla board oltre agli artefatti: gli helper Python che devono
#: girare dove vive il modello, e i file che gli script di provisioning
#: costruiscono (Dockerfile del container Axelera, requirements).
SUPPORT_FILES = (
    ("scripts/remote", "tools"),
    ("scripts/docker", "scripts/docker"),
    ("scripts/requirements", "scripts/requirements"),
)


def ensure_support_files(conn, cfg) -> None:
    """Copia sulla board gli helper e i file di supporto del tool.

    Senza questo passo ogni funzione che invoca uno script sulla board compone
    un path che non esiste: il timer di `onnxruntime_py`, il confronto
    numerico, il Dockerfile che `rpi5_axelera.sh` passa a `docker build`.
    Idempotente e piccolo: si rifa' a ogni sweep, cosi' una modifica agli
    helper arriva senza dover riprovisionare.
    """
    if is_local_conn(conn) or not is_remote(cfg):
        return
    root = Path(cfg.project_root)
    workdir = str(cfg.hardware.remote.workdir).rstrip("/")
    for src_dir, dst_dir in SUPPORT_FILES:
        local_dir = root / src_dir
        if not local_dir.is_dir():
            continue
        remote_dir = f"{workdir}/{dst_dir}"
        conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)
        for f in sorted(local_dir.iterdir()):
            if f.is_file():
                conn.put(str(f), f"{remote_dir}/{f.name}")
    log.debug("file di supporto sincronizzati su %s", cfg.hardware.board)


def sync_artifact(conn, cfg, artifact: str | Path, subdir: str = "artifacts",
                  key: str | None = None) -> str:
    """Copia un artefatto sulla board e ritorna il path remoto.

    `key` e' obbligatorio nella pratica anche se opzionale nella firma: senza,
    ogni `best.pt` finirebbe in `{workdir}/artifacts/best.pt`, lo stesso path
    per yolo26n, yolo26s e per ogni riallenamento. Due export in parallelo
    (che `run.py` consiglia via joblib) si sovrascriverebbero i pesi a vicenda
    e l'engine verrebbe compilato dal modello sbagliato, senza che la
    validazione numerica se ne accorga — confronta contro lo stesso file
    scambiato.
    """
    artifact = Path(artifact)
    if is_local_conn(conn) or not is_remote(cfg):
        return str(artifact)

    remote_dir = f"{cfg.hardware.remote.workdir}/{subdir}"
    if key:
        remote_dir = f"{remote_dir}/{key}"
    conn.run(f"mkdir -p {shlex.quote(remote_dir)}", hide=True)
    src = f"{artifact}/" if artifact.is_dir() else str(artifact)
    dst = f"{cfg.hardware.remote.host}:{remote_dir}/{artifact.name}"
    _rsync(src, dst + ("/" if artifact.is_dir() else ""))
    return f"{remote_dir}/{artifact.name}"


#: quante immagini nel set fisso usato dai backend che misurano su file veri
BENCH_INPUT_N = 200


def ensure_bench_input(conn, cfg, n: int = BENCH_INPUT_N) -> str:
    """Sottoinsieme fisso di immagini per i backend che non generano input.

    `trtexec` e `onnxruntime_perf_test` si costruiscono l'input da soli; il
    predict di Ultralytics no, gli serve una cartella di immagini. Il set e'
    lo stesso in ogni cella — stesse immagini, stesso ordine — altrimenti si
    confronterebbero carichi di post-processing diversi.
    """
    root = ensure_dataset(conn, cfg)
    dest = f"{root}/bench_input"
    if conn.run(f"test -f {shlex.quote(dest)}/.complete", hide=True, warn=True).ok:
        return dest
    conn.run(f"mkdir -p {shlex.quote(dest)}", hide=True)
    # `sort` prima di `head`: l'ordine di find dipende dal filesystem, e senza
    # il set cambierebbe da una board all'altra.
    conn.run(
        f"cd {shlex.quote(root)} && find . -path ./bench_input -prune -o "
        f"-type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \\) "
        f"-print | sort | head -n {int(n)} | "
        f"xargs -I{{}} cp {{}} {shlex.quote(dest)}/",
        hide=True,
    )
    conn.run(f"touch {shlex.quote(dest)}/.complete", hide=True)
    log.info("set di input per la misura preparato in %s (%d immagini)", dest, n)
    return dest


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
