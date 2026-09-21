"""Provisioning delle board e gestione del riavvio.

Il config dichiara *quale* script; lo script contiene il *come*. La logica di
installazione (wheel NVIDIA per una specifica JetPack, repository apt di
Axelera, container Ubuntu) non e' esprimibile in YAML senza inventare un DSL.

Il provisioning e' idempotente, con sentinella basata sull'hash del file dei
requirements: se l'ambiente e' gia' presente e coerente, non fa nulla.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..cache import sha256_file
from ..errors import BoardUnreachable, ProvisionFailed
from .connection import (
    LocalConnection,
    clear_control_socket,
    connection,
    is_remote,
)
from .sync import ensure_support_files

log = logging.getLogger(__name__)


def env_hash(cfg) -> str:
    """Hash del file dei requirements: cambia solo quando cambia l'ambiente."""
    req = Path(cfg.project_root) / cfg.hardware.provision.requirements
    if not req.exists():
        raise ProvisionFailed(f"requirements non trovati: {req}")
    return sha256_file(req)[:12]


def ensure_env(conn, cfg, force: bool = False) -> bool:
    """Provisiona la board se la sentinella non corrisponde.

    Ritorna True se ha eseguito lo script, False se era gia' a posto.
    """
    if not is_remote(cfg):
        return False
    if "provision" not in cfg.hardware:
        log.debug("%s non dichiara provision, salto", cfg.hardware.board)
        return False

    workdir = cfg.hardware.remote.workdir
    conn.run(f"mkdir -p {workdir}", hide=True)
    # Prima della sentinella: gli helper e il Dockerfile devono stare sulla
    # board anche quando l'ambiente e' gia' a posto e il provisioning non gira.
    ensure_support_files(conn, cfg)

    want = env_hash(cfg)
    if not force:
        r = conn.run(f"cat {workdir}/.env_hash", hide=True, warn=True)
        if r.ok and r.stdout.strip() == want:
            log.debug("ambiente su %s gia' coerente (%s)", cfg.hardware.board, want)
            return False

    script = Path(cfg.project_root) / cfg.hardware.provision.script
    req = Path(cfg.project_root) / cfg.hardware.provision.requirements
    if not script.exists():
        raise ProvisionFailed(f"script di provisioning non trovato: {script}")

    log.info("provisioning di %s con %s", cfg.hardware.board, script.name)
    conn.put(str(script), "/tmp/provision.sh")
    conn.put(str(req), "/tmp/requirements.txt")
    r = conn.run(
        f"BENCH_WORKDIR={workdir} BENCH_REQUIREMENTS=/tmp/requirements.txt "
        f"bash /tmp/provision.sh",
        pty=True,
        warn=True,
    )
    if r.failed:
        raise ProvisionFailed(
            f"provisioning di {cfg.hardware.board} fallito "
            f"({r.return_code}):\n{r.stderr or r.stdout}"
        )
    conn.run(f"echo {want} > {workdir}/.env_hash", hide=True)
    return True


def health_check(conn, cfg) -> dict:
    """Controlli rapidi prima di uno sweep.

    Su Axelera e' il controllo che conta: kernel driver e SDK si aggiornano
    separatamente e nulla verifica la coerenza a install time. Un driver sotto
    la versione minima lascia il device inaccessibile con un errore di
    connessione, a meta' sweep invece che all'inizio.
    """
    checks: dict[str, object] = {}
    if cfg.hardware.get("axelera_available"):
        r = conn.run("axdevice", hide=True, warn=True)
        checks["axdevice"] = r.stdout.strip() if r.ok else None
        if r.failed:
            raise ProvisionFailed(
                "axdevice non vede l'acceleratore Metis: verificare "
                "metis-dkms (>= 1.6.2) e `modprobe metis`"
            )
    if cfg.hardware.get("trt_available"):
        r = conn.run(
            "command -v trtexec || test -x /usr/src/tensorrt/bin/trtexec "
            "&& echo /usr/src/tensorrt/bin/trtexec",
            hide=True,
            warn=True,
        )
        checks["trtexec"] = r.stdout.strip() if r.ok else None
    r = conn.run("uptime", hide=True, warn=True)
    checks["uptime"] = r.stdout.strip() if r.ok else None
    return checks


def provision(cfg) -> dict:
    """Stadio `provision`: prepara la board e verifica che sia usabile."""
    with connection(cfg) as conn:
        if isinstance(conn, LocalConnection):
            log.info(
                "%s e' locale: nessun provisioning remoto, verifico soltanto",
                cfg.hardware.board,
            )
            return {"provisioned": False, "checks": health_check(conn, cfg)}
        did = ensure_env(conn, cfg, force=bool(cfg.stage.get("force", False)))
        checks = health_check(conn, cfg)
        log.info("provisioning %s: %s", cfg.hardware.board,
                 "eseguito" if did else "gia' a posto")
        for k, v in checks.items():
            log.info("  %s: %s", k, v)
        return {"provisioned": did, "checks": checks}


def reboot_and_wait(conn, cfg, timeout_s: int = 300, poll_s: int = 5,
                    grace_s: int = 15):
    """Riavvia la board e restituisce una connessione nuova.

    Quattro dettagli senza i quali il riavvio blocca lo sweep invece di
    gestirlo: grace period (per qualche secondo dopo `shutdown -r` la board
    risponde ancora), ControlPath ripulito, connessione ricreata e non riusata,
    stato post-reboot ricontrollato (container fermo, modulo metis scaricato,
    swap riattivo, governor al default).
    """
    from fabric import Connection  # import locale: serve solo qui

    host = cfg.hardware.remote.host
    log.warning("riavvio %s", host)
    conn.run("sudo shutdown -r now", warn=True, disown=True)
    try:
        conn.close()
    except Exception:
        pass
    clear_control_socket(host)
    time.sleep(grace_s)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            c = Connection(
                host,
                connect_timeout=5,
                inline_ssh_env=True,
                connect_kwargs={"banner_timeout": 5},
            )
            c.run("true", hide=True, timeout=5)
            log.info("board tornata dopo %.0fs", time.time() - t0)
            return c
        except Exception:
            time.sleep(poll_s)
    raise BoardUnreachable(f"{host} non tornata entro {timeout_s}s")
