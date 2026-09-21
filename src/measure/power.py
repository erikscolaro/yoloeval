"""Energia — dove i sensori esistono.

Su Jetson i rail INA sono esposti da `tegrastats`. Il campionamento parte
prima della misura e si ferma dopo: la potenza media va integrata sulla durata
della misura, non letta una volta a caso.

Sul Pi 5 non c'e' un sensore di potenza accessibile: il blocco `energy` resta
assente, non a zero.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager

log = logging.getLogger(__name__)

#: rail di ingresso, in ordine di preferenza (varia con il modulo e JetPack)
_RAILS = ("VDD_IN", "POM_5V_IN", "VDD_SYS_GPU", "VDD_CPU_GPU_CV")
_RAIL_RE = re.compile(r"(?P<rail>[A-Z0-9_]+)\s+(?P<inst>\d+)mW/(?P<avg>\d+)mW")
_RAIL_RE_LEGACY = re.compile(r"(?P<rail>[A-Z0-9_]+)\s+(?P<inst>\d+)/(?P<avg>\d+)")

LOGFILE = "/tmp/yolo_bench_tegrastats.log"


def parse_tegrastats(stdout: str) -> dict | None:
    """Estrae la potenza media dal log di tegrastats.

    Ritorna None se nessun rail e' riconoscibile: meglio nessun dato che un
    numero inventato.
    """
    samples: list[float] = []
    rail_used = None
    for line in (stdout or "").splitlines():
        rails = {
            m.group("rail"): int(m.group("inst"))
            for m in _RAIL_RE.finditer(line)
        }
        if not rails:
            rails = {
                m.group("rail"): int(m.group("inst"))
                for m in _RAIL_RE_LEGACY.finditer(line)
            }
        if not rails:
            continue
        for rail in _RAILS:
            if rail in rails:
                samples.append(rails[rail] / 1000.0)
                rail_used = rail
                break
    if not samples:
        return None
    return {
        "mean_power_w": round(sum(samples) / len(samples), 3),
        "max_power_w": round(max(samples), 3),
        "n_samples": len(samples),
        "rail": rail_used,
        "source": "tegrastats",
    }


class _NullSampler:
    """Board senza sensori: il blocco energy resta assente."""

    available = False

    def result(self, duration_s: float | None = None) -> None:
        return None


class TegrastatsSampler:
    """Campiona i rail INA per la durata della misura."""

    available = True

    def __init__(self, conn, interval_ms: int = 100, logfile: str = LOGFILE):
        self.conn = conn
        self.interval_ms = interval_ms
        self.logfile = logfile
        self._started = False

    def start(self) -> None:
        self.conn.run(f"sudo tegrastats --stop; rm -f {self.logfile}",
                      hide=True, warn=True)
        r = self.conn.run(
            f"sudo tegrastats --interval {self.interval_ms} "
            f"--logfile {self.logfile} --start",
            hide=True, warn=True,
        )
        self._started = getattr(r, "ok", False)
        if not self._started:
            log.warning("tegrastats non avviato: energia non disponibile")

    def stop(self) -> str:
        if not self._started:
            return ""
        self.conn.run("sudo tegrastats --stop", hide=True, warn=True)
        r = self.conn.run(f"cat {self.logfile}", hide=True, warn=True)
        return r.stdout if getattr(r, "ok", False) else ""

    def result(self, duration_s: float | None = None) -> dict | None:
        payload = parse_tegrastats(self._raw)
        if payload and duration_s:
            payload["energy_j"] = round(payload["mean_power_w"] * duration_s, 3)
            payload["window_s"] = round(duration_s, 3)
        return payload

    _raw = ""


@contextmanager
def energy_sampler(conn, cfg):
    """Campionatore attivo per la durata del blocco. No-op senza sensori."""
    if not cfg.hardware.get("power_sensors"):
        yield _NullSampler()
        return
    sampler = TegrastatsSampler(conn)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler._raw = sampler.stop()
