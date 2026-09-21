"""Connessione locale e remota dietro un'unica interfaccia.

`LocalConnection` espone la stessa superficie di `fabric.Connection` usata dal
resto del codice, cosi' `execute_benchmark` non contiene rami locale/remoto.
La presenza o assenza della chiave `remote` nel file della board e' l'unico
discriminante.

I config dichiarano solo l'**alias** SSH, mai IP o credenziali: Fabric legge
`~/.ssh/config`, e `ControlMaster` riusa una sola connessione TCP per l'intero
sweep.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Result:
    """Sottoinsieme di `invoke.Result` effettivamente usato dal tool."""

    stdout: str
    stderr: str
    return_code: int
    command: str = ""

    @property
    def ok(self) -> bool:
        return self.return_code == 0

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def exited(self) -> int:  # alias invoke
        return self.return_code


def shell_env(env: dict | None) -> str:
    """Prefisso `env VAR=val ...` per la command line.

    Le variabili non si passano al kwarg `env` di Fabric: su SSH finirebbero
    in `SendEnv`, che quasi tutti i server rifiutano con `AcceptEnv` non
    configurato, e sparirebbero in silenzio. Con un prefisso esplicito o la
    variabile c'e' o il comando non parte.
    """
    if not env:
        return ""
    parts = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    return f"env {parts} "


class LocalConnection:
    """Stessa interfaccia di `fabric.Connection`, eseguita in locale."""

    is_local = True
    host = "localhost"

    def run(self, cmd, hide=False, warn=False, env=None, pty=False, timeout=None,
            disown=False, **_) -> Result:
        full = shell_env(env) + cmd
        log.debug("local$ %s", full)
        if disown:
            subprocess.Popen(full, shell=True, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return Result("", "", 0, full)
        proc = subprocess.run(
            full, shell=True, capture_output=True, text=True, timeout=timeout
        )
        r = Result(proc.stdout or "", proc.stderr or "", proc.returncode, full)
        if not hide and r.stdout:
            log.debug("stdout:\n%s", r.stdout)
        if r.failed and not warn:
            raise RuntimeError(
                f"comando fallito ({r.return_code}): {full}\n{r.stderr}"
            )
        return r

    def put(self, local, remote, **_):
        dest = Path(remote)
        if dest.is_dir():
            dest = dest / Path(local).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local).resolve() != dest.resolve():
            shutil.copy2(local, dest)
        return dest

    def get(self, remote, local, **_):
        return self.put(remote, local)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def is_remote(cfg) -> bool:
    """La workstation non dichiara `remote`: gira in locale."""
    return "remote" in cfg.hardware and cfg.hardware.remote is not None


def is_local_conn(conn) -> bool:
    """Se il comando gira qui o sulla board lo dice la connessione, non il config.

    Lo stesso config con `remote` puo' essere usato con una `LocalConnection`:
    un ONNX si esporta e si valida sulla workstation anche quando la cella di
    destinazione e' una Jetson. Decidere dal config porterebbe a cercare
    l'artefatto sulla board, dove non e' ancora arrivato.
    """
    return bool(getattr(conn, "is_local", False))


def open_connection(cfg):
    """Apre la connessione descritta dal config (locale o SSH)."""
    if not is_remote(cfg):
        return LocalConnection()
    try:
        from fabric import Connection
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise RuntimeError(
            "fabric non installato: serve per le board remote "
            "(pip install 'fabric>=3.2')"
        ) from exc
    host = cfg.hardware.remote.host
    log.info("connessione a %s", host)
    # inline_ssh_env: le variabili vengono esportate nella shell remota invece
    # di essere passate via SendEnv, che il server rifiuta quasi sempre.
    return Connection(host, inline_ssh_env=True)


@contextmanager
def connection(cfg):
    conn = open_connection(cfg)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # la chiusura non deve mascherare l'errore vero
            log.debug("chiusura connessione fallita", exc_info=True)


def clear_control_socket(host: str) -> None:
    """Rimuove il ControlPath stantio dopo un riavvio.

    Con `ControlMaster` attivo il socket resta aperto verso una board che non
    esiste piu' e i comandi successivi falliscono in modo fuorviante.
    """
    subprocess.run(
        ["ssh", "-O", "exit", host],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    cm_dir = Path(os.path.expanduser("~/.ssh"))
    for sock in cm_dir.glob(f"cm-*@{host}:*"):
        try:
            sock.unlink()
        except OSError:
            log.debug("control socket %s non rimosso", sock)


def remote_workdir(cfg) -> str:
    return str(cfg.hardware.remote.workdir) if is_remote(cfg) else str(cfg.project_root)


def run_logged(conn, cmd, **kwargs):
    """`conn.run` con l'output remoto portato nel logging Python.

    L'output dei comandi remoti non passa dal logging di Python: va catturato
    e rilanciato esplicitamente, altrimenti sparisce.
    """
    kwargs.setdefault("hide", True)
    kwargs.setdefault("warn", True)
    r = conn.run(cmd, **kwargs)
    if getattr(r, "stdout", ""):
        log.debug("stdout:\n%s", r.stdout)
    if r.failed:
        log.error("comando fallito (%s): %s\n%s", r.return_code, cmd, r.stderr)
    return r
