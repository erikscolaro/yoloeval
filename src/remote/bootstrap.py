"""Primo accesso a una board: utente di misura, sudo, chiave SSH e alias.

    python run.py stage=provision hardware=rpi5 stage.bootstrap=admin@pi.local

Parte da un utente che sulla board c'e' gia' e ha sudo (admin su Raspberry Pi OS) e prepara
quello che il resto del tool da' per scontato:

- l'utente di misura (`stage.bootstrap_user`, bench) senza password: si entra solo con la
  chiave;
- sudo senza password per lui, in /etc/sudoers.d/, per i comandi di tuning;
- il gruppo docker, se c'e', per il container Axelera;
- la chiave pubblica di questo PC nel suo authorized_keys (creata se manca);
- l'alias `remote.host` in ~/.ssh/config verso l'host e l'utente di misura.

Ogni passo e' idempotente. Usa `ssh` di sistema e non Fabric, cosi' l'eventuale password di
admin la chiede il terminale: va lanciato da terminale, non da un notebook.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from ..errors import ProvisionFailed

log = logging.getLogger(__name__)

KEY_NAMES = ("id_ed25519", "id_ecdsa", "id_rsa")

REMOTE_SCRIPT = r"""
set -e
U={user}
KEY={key}
id -u "$U" >/dev/null 2>&1 || sudo adduser --disabled-password --gecos "" "$U"
H=$(getent passwd "$U" | cut -d: -f6)
F=/etc/sudoers.d/010_$U-nopasswd
echo "$U ALL=(ALL) NOPASSWD: ALL" | sudo tee "$F" >/dev/null
sudo chmod 440 "$F"
sudo visudo -cf "$F" >/dev/null
sudo -u "$U" mkdir -p -m 700 "$H/.ssh"
sudo -u "$U" touch "$H/.ssh/authorized_keys"
sudo -u "$U" chmod 600 "$H/.ssh/authorized_keys"
sudo grep -qxF "$KEY" "$H/.ssh/authorized_keys" \
  || echo "$KEY" | sudo -u "$U" tee -a "$H/.ssh/authorized_keys" >/dev/null
if getent group docker >/dev/null; then sudo usermod -aG docker "$U"; fi
echo "bootstrap: utente $U pronto"
"""


def public_key() -> str:
    """La chiave pubblica di questo PC; se non ce n'e' nessuna ne crea una ed25519."""
    ssh_dir = Path.home() / ".ssh"
    for name in KEY_NAMES:
        pub = ssh_dir / f"{name}.pub"
        if pub.exists():
            return pub.read_text(encoding="utf-8").strip()
    key = ssh_dir / "id_ed25519"
    log.info("nessuna chiave SSH in %s: la creo (%s)", ssh_dir, key)
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(key)],
                   check=True)
    return key.with_name("id_ed25519.pub").read_text(encoding="utf-8").strip()


def ensure_alias(alias: str, hostname: str, user: str) -> None:
    """Aggiunge `Host <alias>` a ~/.ssh/config se non c'e'; se c'e' non lo tocca."""
    config = Path.home() / ".ssh" / "config"
    r = subprocess.run(["ssh", *(["-F", str(config)] if config.exists() else []), "-G", alias],
                       capture_output=True, text=True, check=False)
    opts = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    current = (opts.get("hostname", alias).lower(), opts.get("user"))
    if current[0] != alias.lower():
        if current != (hostname.lower(), user):
            log.warning("alias %s gia' in ~/.ssh/config verso %s@%s, non lo modifico: "
                        "atteso %s@%s", alias, current[1], current[0], user, hostname)
        return
    config.parent.mkdir(mode=0o700, exist_ok=True)
    with config.open("a", encoding="utf-8") as f:
        f.write(f"\n# aggiunto da yolo-bench (stage=provision)\n"
                f"Host {alias}\n    HostName {hostname}\n    User {user}\n")
    config.chmod(0o600)
    log.info("alias %s -> %s@%s aggiunto a %s", alias, user, hostname, config)


def bootstrap(cfg) -> None:
    target = str(cfg.stage.bootstrap)
    user = str(cfg.stage.get("bootstrap_user") or "bench")
    alias = str(cfg.hardware.remote.host)
    hostname = target.rsplit("@", 1)[-1]

    log.info("bootstrap di %s via %s: utente %s, sudo senza password, chiave SSH", alias,
             target, user)
    script = REMOTE_SCRIPT.format(user=shlex.quote(user), key=shlex.quote(public_key()))
    # -t: se sudo di admin chiede la password, la chiede qui sul terminale
    r = subprocess.run(["ssh", "-t", target, script], check=False)
    if r.returncode != 0:
        raise ProvisionFailed(f"bootstrap via {target} fallito ({r.returncode})")

    ensure_alias(alias, hostname, user)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", alias, "sudo", "-n", "true"],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise ProvisionFailed(f"dopo il bootstrap `ssh {alias}` non entra senza password "
                              f"o sudo la chiede:\n{r.stderr.strip()}")
    log.info("bootstrap completato: `ssh %s` entra come %s, sudo senza password", alias,
             user)
