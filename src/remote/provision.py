"""Provisioning delle board e gestione del riavvio.

Il config dichiara *quale* script; lo script contiene il *come*. La logica di
installazione (wheel NVIDIA per una specifica JetPack, repository apt di
Axelera, container Ubuntu) non e' esprimibile in YAML senza inventare un DSL.

Il provisioning e' idempotente, con sentinella basata sull'hash del file dei
requirements: se l'ambiente e' gia' presente e coerente, non fa nulla.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
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


def provision_scripts(cfg) -> list[Path]:
    """`provision.script`: un path o una lista, eseguiti in ordine (rpi5_axelera: la parte
    Axelera, poi quella comune del Pi)."""
    scripts = cfg.hardware.provision.script
    if isinstance(scripts, str):
        scripts = [scripts]
    return [Path(cfg.project_root) / s for s in scripts]


def env_hash(cfg) -> str:
    """Hash di requirements e script: cambia solo quando cambia l'ambiente.

    Gli script contano quanto i requirements: un passo nuovo (come la build di
    onnxruntime_perf_test) su una board gia' provisionata non girerebbe mai.
    """
    files = [Path(cfg.project_root) / cfg.hardware.provision.requirements,
             *provision_scripts(cfg)]
    parts = []
    for f in files:
        if not f.exists():
            raise ProvisionFailed(f"file di provisioning non trovato: {f}")
        parts.append(sha256_file(f))
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:12]


PROXY_SCRIPT = r"""
set -e
P={proxy}
N={no_proxy}
# pip, python, curl, git: /etc/environment, letto a ogni login
sudo sed -i '/^\(http\|https\|no\)_proxy=/Id' /etc/environment
printf 'http_proxy=%s\nhttps_proxy=%s\nno_proxy=%s\nHTTP_PROXY=%s\nHTTPS_PROXY=%s\nNO_PROXY=%s\n' \
  "$P" "$P" "$N" "$P" "$P" "$N" | sudo tee -a /etc/environment >/dev/null
# apt
printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$P" "$P" \
  | sudo tee /etc/apt/apt.conf.d/95proxy >/dev/null
# demone docker (docker pull): vale anche se docker verra' installato dopo
D=/etc/systemd/system/docker.service.d
sudo mkdir -p "$D"
printf '[Service]\nEnvironment="HTTP_PROXY=%s"\nEnvironment="HTTPS_PROXY=%s"\nEnvironment="NO_PROXY=%s"\n' \
  "$P" "$P" "$N" | sudo tee "$D/http-proxy.conf.new" >/dev/null
if sudo cmp -s "$D/http-proxy.conf.new" "$D/http-proxy.conf"; then
  sudo rm "$D/http-proxy.conf.new"
else
  sudo mv "$D/http-proxy.conf.new" "$D/http-proxy.conf"
  sudo systemctl daemon-reload
  if systemctl is-active --quiet docker; then sudo systemctl restart docker; fi
fi
# dentro i container di docker build: ~/.docker/config.json dell'utente di misura
python3 - "$P" "$N" <<'PY'
import json, os, sys
path = os.path.expanduser("~/.docker/config.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    with open(path) as f:
        conf = json.load(f)
except (OSError, ValueError):
    conf = {{}}
conf.setdefault("proxies", {{}})["default"] = {{
    "httpProxy": sys.argv[1], "httpsProxy": sys.argv[1], "noProxy": sys.argv[2]}}
with open(path, "w") as f:
    json.dump(conf, f, indent=2)
PY
"""


def proxy_url(cfg) -> str | None:
    """`stage.proxy`: auto -> il proxy di questo PC, None -> nessuno."""
    proxy = cfg.stage.get("proxy")
    if proxy == "auto":
        proxy = next((os.environ[k] for k in ("https_proxy", "HTTPS_PROXY", "http_proxy",
                                              "HTTP_PROXY") if os.environ.get(k)), None)
    return str(proxy) if proxy else None


def proxy_env(cfg) -> dict:
    """Variabili del proxy per i comandi di questa connessione: /etc/environment vale solo
    dal login successivo."""
    proxy = proxy_url(cfg)
    if not proxy:
        return {}
    no_proxy = ",".join(map(str, cfg.stage.get("no_proxy") or []))
    return {"http_proxy": proxy, "https_proxy": proxy, "no_proxy": no_proxy,
            "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "NO_PROXY": no_proxy}


def configure_proxy(conn, cfg) -> None:
    """Scrive il proxy sulla board per pip, apt, docker e docker build. Idempotente."""
    proxy = proxy_url(cfg)
    if not proxy:
        return
    no_proxy = ",".join(map(str, cfg.stage.get("no_proxy") or []))
    script = PROXY_SCRIPT.format(proxy=shlex.quote(proxy), no_proxy=shlex.quote(no_proxy))
    r = conn.run(f"bash -c {shlex.quote(script)}", hide=True, warn=True)
    if r.failed:
        raise ProvisionFailed(f"proxy su {cfg.hardware.board} non configurato "
                              f"({r.return_code}):\n{r.stderr or r.stdout}")
    log.info("proxy di %s: %s (no_proxy %s) per pip, apt e docker", cfg.hardware.board,
             proxy, no_proxy)


def sync_clock(conn, cfg, max_skew_s: int = 60) -> None:
    """Rimette l'orologio della board su quello di questo PC se e' fuori di piu' di
    `max_skew_s`. Dietro una rete che blocca NTP il Pi (senza batteria per l'RTC) riparte
    dall'ultima data salvata, e con l'orologio indietro pip e apt rifiutano i certificati TLS
    ("certificate is not yet valid")."""
    r = conn.run("date +%s", hide=True, warn=True)
    if r.failed or not r.stdout.strip().isdigit():
        return
    skew = time.time() - int(r.stdout.strip())
    if abs(skew) <= max_skew_s:
        return
    conn.run(f"sudo -n date -u -s @{int(time.time())}", hide=True)
    log.warning("orologio di %s fuori di %.1f ore: rimesso su quello di questo PC "
                "(NTP non sincronizzato)", cfg.hardware.board, skew / 3600)


def retry_minutes(cfg) -> int:
    """Per quanto riprovare ogni download prima di arrendersi (stage.retry_minutes)."""
    return int(cfg.stage.get("retry_minutes", 30))


def check_internet(conn, cfg) -> None:
    """Prova dalla board gli URL di `stage.internet_check` prima dello script di
    provisioning, che scarica da pip, apt e docker: senza rete meglio un errore subito che
    un'installazione a meta'. python3 e non curl, che non e' detto ci sia; urllib rispetta
    http(s)_proxy come pip e apt."""
    urls = list(cfg.stage.get("internet_check") or [])
    probe = ("import sys, urllib.request; "
             "urllib.request.urlopen(sys.argv[1], timeout=15).close()")
    for url in urls:
        # il proxy aziendale cade anche per minuti: si riprova per stage.retry_minutes
        deadline = time.time() + retry_minutes(cfg) * 60
        attempt = 1
        while True:
            r = conn.run(f"python3 -c {shlex.quote(probe)} {shlex.quote(url)}", hide=True,
                         warn=True, env=proxy_env(cfg))
            if r.ok or time.time() >= deadline:
                break
            log.info("internet da %s: %s non risponde (tentativo %d), riprovo fra 30 s",
                     cfg.hardware.board, url, attempt)
            attempt += 1
            time.sleep(30)
        if r.failed:
            err = (r.stderr or r.stdout or "").strip().splitlines()
            raise ProvisionFailed(
                f"{cfg.hardware.board} non raggiunge {url}: niente internet, provisioning "
                f"non avviato (controllare rete, DNS o proxy della board)\n"
                f"{err[-1] if err else ''}")
        log.info("internet da %s: %s raggiungibile", cfg.hardware.board, url)


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

    scripts = provision_scripts(cfg)
    req = Path(cfg.project_root) / cfg.hardware.provision.requirements
    for script in scripts:
        if not script.exists():
            raise ProvisionFailed(f"script di provisioning non trovato: {script}")

    check_internet(conn, cfg)
    conn.put(str(req), "/tmp/requirements.txt")
    # I campi della board che lo script deve conoscere viaggiano come
    # variabili d'ambiente: senza, jetson_jp62.sh userebbe il proprio default
    # e il controllo su /etc/nv_tegra_release confronterebbe la versione
    # sbagliata.
    env = {
        "BENCH_WORKDIR": workdir,
        "BENCH_REQUIREMENTS": "/tmp/requirements.txt",
        "RETRY_MINUTES": retry_minutes(cfg),
        **proxy_env(cfg),
    }
    if cfg.hardware.get("jetpack"):
        env["BENCH_JETPACK"] = cfg.hardware.jetpack
    if cfg.backend.get("build", {}).get("sdk_version"):
        env["BENCH_SDK_VERSION"] = cfg.backend.build.sdk_version
    if cfg.backend.get("build", {}).get("image"):
        env["BENCH_AXELERA_IMAGE"] = cfg.backend.build.image
    prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    for script in scripts:
        log.info("provisioning di %s con %s", cfg.hardware.board, script.name)
        conn.put(str(script), "/tmp/provision.sh")
        r = conn.run(f"{prefix} bash /tmp/provision.sh", pty=True, warn=True)
        if r.failed:
            raise ProvisionFailed(
                f"provisioning di {cfg.hardware.board} fallito in {script.name} "
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
    """Stadio `provision`: prepara la board e verifica che sia usabile. Con
    `stage.bootstrap=admin@host` prima crea utente, sudo, chiave e alias (bootstrap.py)."""
    if cfg.stage.get("bootstrap"):
        if not is_remote(cfg):
            raise ProvisionFailed(f"{cfg.hardware.board} e' locale: niente bootstrap")
        from .bootstrap import bootstrap

        bootstrap(cfg)
    with connection(cfg) as conn:
        if isinstance(conn, LocalConnection):
            log.info(
                "%s e' locale: nessun provisioning remoto, verifico soltanto",
                cfg.hardware.board,
            )
            return {"provisioned": False, "checks": health_check(conn, cfg)}
        sync_clock(conn, cfg)
        configure_proxy(conn, cfg)
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
