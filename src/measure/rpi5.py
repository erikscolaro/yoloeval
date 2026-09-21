"""Raspberry Pi 5: cpufreq via sysfs e vcgencmd.

I quattro core condividono il clock: scrivere su `cpu0` li muove tutti, quindi
la frequenza e' proprieta' dell'`hardware`, non del `compute`.

Il firmware applica un proprio throttling **indipendente dal governor**, per
temperatura e per tensione di alimentazione: si puo' avere `userspace` a
1.5 GHz e il firmware che scala comunque. `vcgencmd get_throttled` diverso da
zero significa cella contaminata.
"""

from __future__ import annotations

import logging
import re

from .board import CPUFREQ, BoardController, _out
from .compute import resolve_freq

log = logging.getLogger(__name__)

_TEMP_RE = re.compile(r"temp=([\d.]+)")
_THROTTLED_RE = re.compile(r"throttled=0x([0-9a-fA-F]+)")


def parse_throttled(stdout: str) -> int | None:
    m = _THROTTLED_RE.search(stdout or "")
    return int(m.group(1), 16) if m else None


class RPi5Controller(BoardController):
    name = "rpi5"

    def read_state(self, conn) -> dict:
        return {
            "governor": self.read_governor(conn),
            "freq_khz": self.read_freq(conn),
            "setspeed_khz": self._read_setspeed(conn),
            "swap_off": self.swap_is_off(conn),
        }

    def apply_state(self, conn, cfg) -> dict:
        fq = resolve_freq(cfg)
        governor = fq.get("governor", "performance")
        self._require_governor(conn, governor)
        self.set_governor(conn, governor)
        if governor == "userspace" and fq.get("khz"):
            self.set_freq_khz(conn, int(fq.khz))
        if cfg.stage.get("scheduling", {}).get("swap_off", True):
            self.set_swap(conn, off=True)
        actual = self.read_freq(conn)
        requested = fq.get("khz")
        if requested and actual and abs(actual - int(requested)) > 50_000:
            # Un valore non supportato viene arrotondato in silenzio.
            log.warning("frequenza richiesta %s kHz, effettiva %s kHz",
                        requested, actual)
        return {
            "governor": self.read_governor(conn),
            "freq_requested_khz": requested,
            "freq_actual_khz": actual,
            "cores_online": self.online_cores(conn),
        }

    def restore_state(self, conn, saved: dict) -> None:
        if saved.get("governor"):
            self.set_governor(conn, saved["governor"])
            if saved["governor"] == "userspace" and saved.get("setspeed_khz"):
                self.set_freq_khz(conn, int(saved["setspeed_khz"]))
        if saved.get("swap_off") is False:
            self.set_swap(conn, off=False)

    def read_temp(self, conn) -> float | None:
        raw = _out(conn.run("vcgencmd measure_temp", hide=True, warn=True))
        m = _TEMP_RE.search(raw or "")
        if m:
            return float(m.group(1))
        raw = _out(conn.run("cat /sys/class/thermal/thermal_zone0/temp",
                            hide=True, warn=True))
        return int(raw) / 1000.0 if raw and raw.isdigit() else None

    def read_throttle(self, conn) -> bool | None:
        raw = _out(conn.run("vcgencmd get_throttled", hide=True, warn=True))
        value = parse_throttled(raw or "")
        if value is None:
            return None
        return value != 0

    def _read_setspeed(self, conn) -> int | None:
        raw = _out(conn.run(f"cat {CPUFREQ.format(cpu=0)}/scaling_setspeed",
                            hide=True, warn=True))
        return int(raw) if raw and raw.isdigit() else None

    def _require_governor(self, conn, governor: str) -> None:
        raw = _out(conn.run(
            f"cat {CPUFREQ.format(cpu=0)}/scaling_available_governors",
            hide=True, warn=True))
        if raw and governor not in raw.split():
            # L'alternativa (arm_freq in /boot/firmware/config.txt) richiede
            # reboot ed e' persistente: fuori dalle regole del tool.
            raise RuntimeError(
                f"governor {governor} non disponibile su {self.cfg.hardware.board}: "
                f"{raw}"
            )
