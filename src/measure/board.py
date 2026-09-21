"""Controller di board: stato, tuning temporaneo, termico.

Ogni modifica di sistema e' **temporanea** e ha il suo inverso: governor,
frequenza, profilo di potenza e swap vengono applicati per la singola cella e
ripristinati subito dopo, anche in caso di errore. Tutto passa da sysfs o da
comandi runtime: niente `config.txt`, niente unit systemd, niente parametri
kernel — e' anche il motivo per cui `isolcpus` resta fuori.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager

from .compute import resolve_freq

log = logging.getLogger(__name__)

CPUFREQ = "/sys/devices/system/cpu/cpu{cpu}/cpufreq"


def _out(r) -> str | None:
    return r.stdout.strip() if getattr(r, "ok", False) and r.stdout.strip() else None


class BoardController(ABC):
    """Interfaccia comune, selezionata da `cfg.hardware.controller`."""

    name = "generic"

    def __init__(self, cfg):
        self.cfg = cfg

    # --- stato -----------------------------------------------------------
    @abstractmethod
    def read_state(self, conn) -> dict:
        """Stato corrente, sufficiente a ripristinarlo."""

    @abstractmethod
    def apply_state(self, conn, cfg) -> dict:
        """Applica il profilo richiesto. Ritorna lo stato applicato davvero."""

    @abstractmethod
    def restore_state(self, conn, saved: dict) -> None:
        """Inverso esatto di apply_state."""

    # --- letture ---------------------------------------------------------
    @abstractmethod
    def read_temp(self, conn) -> float | None:
        ...

    @abstractmethod
    def read_throttle(self, conn) -> bool | None:
        ...

    def online_cores(self, conn) -> int | None:
        """Sempre letto a runtime, mai assunto dal config.

        Su Jetson il profilo di potenza porta offline dei core: il conteggio
        dichiarato nel YAML documenta l'intenzione, non il fatto.
        """
        r = conn.run("cat /sys/devices/system/cpu/online", hide=True, warn=True)
        raw = _out(r)
        if raw:
            return _count_cpu_list(raw)
        r = conn.run("nproc", hide=True, warn=True)
        raw = _out(r)
        return int(raw) if raw and raw.isdigit() else None

    def read_freq(self, conn) -> int | None:
        """Frequenza effettiva in kHz. Un valore non supportato viene
        arrotondato in silenzio: si registra richiesta e effettiva."""
        r = conn.run(f"cat {CPUFREQ.format(cpu=0)}/scaling_cur_freq",
                     hide=True, warn=True)
        raw = _out(r)
        return int(raw) if raw and raw.isdigit() else None

    def read_governor(self, conn) -> str | None:
        return _out(conn.run(f"cat {CPUFREQ.format(cpu=0)}/scaling_governor",
                             hide=True, warn=True))

    def read_power(self, conn) -> float | None:
        return None  # default: sensori assenti

    def read_energy(self, conn, duration_s: float | None = None) -> dict | None:
        return None

    def swap_is_off(self, conn) -> bool | None:
        r = conn.run("swapon --show --noheadings", hide=True, warn=True)
        if not getattr(r, "ok", False):
            return None
        return r.stdout.strip() == ""

    # --- azioni comuni ---------------------------------------------------
    def set_governor(self, conn, governor: str) -> None:
        conn.run(
            f"echo {governor} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/"
            f"scaling_governor > /dev/null",
            hide=True, warn=True,
        )

    def set_freq_khz(self, conn, khz: int) -> None:
        conn.run(
            f"echo {khz} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/"
            f"scaling_setspeed > /dev/null",
            hide=True, warn=True,
        )

    def set_swap(self, conn, off: bool) -> None:
        """Meglio un OOM visibile che una cella in swap."""
        conn.run("sudo swapoff -a" if off else "sudo swapon -a",
                 hide=True, warn=True)

    def drop_caches(self, conn) -> None:
        """Cosi' il primo caricamento del modello e' confrontabile fra celle."""
        conn.run("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null",
                 hide=True, warn=True)

    def load_average(self, conn) -> float | None:
        raw = _out(conn.run("cut -d' ' -f1 /proc/loadavg", hide=True, warn=True))
        try:
            return float(raw) if raw else None
        except ValueError:
            return None


def _count_cpu_list(spec: str) -> int:
    """"0-3,6" -> 5."""
    total = 0
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            total += int(hi) - int(lo) + 1
        else:
            total += 1
    return total


def board_controller(cfg) -> BoardController:
    """Sceglie il controller dal config della board."""
    from .jetson import JetsonController
    from .rpi5 import RPi5Controller
    from .x86 import X86Controller

    kind = cfg.hardware.get("controller") or cfg.hardware.board
    table = {
        "jetson": JetsonController,
        "rpi5": RPi5Controller,
        "x86": X86Controller,
    }
    for key, cls in table.items():
        if str(kind).startswith(key):
            return cls(cfg)
    log.warning("controller sconosciuto per %s, uso quello x86", kind)
    return X86Controller(cfg)


@contextmanager
def tuned(conn, cfg, controller: BoardController | None = None):
    """Applica il tuning per la singola cella e lo ripristina sempre.

    Il `finally` copre l'eccezione, non copre `kill -9` ne' una connessione che
    cade: per sweep lunghi non presidiati serve anche un guard lato board che
    ripristini lo stato se non riceve un heartbeat (vedi
    `scripts/prepare_board.sh`).
    """
    from .thermal import wait_thermal

    bc = controller or board_controller(cfg)
    saved = bc.read_state(conn)
    log.debug("stato salvato: %s", saved)
    applied = {}
    try:
        applied = bc.apply_state(conn, cfg)
        thermal = cfg.stage.get("thermal", {}) if "stage" in cfg else {}
        temp_start = wait_thermal(
            conn,
            bc,
            threshold_c=thermal.get("threshold_c", 45),
            timeout_s=thermal.get("timeout_s", 300),
            poll_s=thermal.get("poll_s", 10),
        )
        if cfg.stage.get("scheduling", {}).get("drop_caches", True):
            bc.drop_caches(conn)
        applied["temp_start_c"] = temp_start
        yield applied
    finally:
        try:
            bc.restore_state(conn, saved)
        except Exception:
            log.exception("ripristino dello stato fallito su %s", cfg.hardware.board)
        after = _safe_state(bc, conn)
        applied["state_before"] = saved
        applied["state_after_restore"] = after
        if after and saved and _differs(saved, after):
            # Se una cella lascia la macchina in uno stato diverso da come
            # l'ha trovata, deve essere visibile subito.
            log.warning("stato non ripristinato del tutto: %s -> %s", saved, after)


def _safe_state(bc, conn) -> dict | None:
    try:
        return bc.read_state(conn)
    except Exception:
        log.debug("rilettura stato fallita", exc_info=True)
        return None


def _differs(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    return any(a.get(k) != b.get(k) for k in keys if k not in ("temp_c",))


def target_state(cfg) -> dict:
    """Stato richiesto dal config, per il confronto con quello effettivo."""
    fq = resolve_freq(cfg)
    return {
        "governor": fq.get("governor"),
        "freq_khz": fq.get("khz"),
        "nvpmodel_id": fq.get("nvpmodel_id"),
    }
