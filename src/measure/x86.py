"""Workstation x86: solo governor.

Nessun profilo di potenza, nessun sensore di energia: il file della board non
dichiara `remote`, quindi i comandi girano in locale attraverso
`LocalConnection`.
"""

from __future__ import annotations

import logging

from .board import BoardController, _out
from .compute import resolve_freq

log = logging.getLogger(__name__)


class X86Controller(BoardController):
    name = "x86"

    def read_state(self, conn) -> dict:
        return {
            "governor": self.read_governor(conn),
            "freq_khz": self.read_freq(conn),
            "swap_off": self.swap_is_off(conn),
        }

    def apply_state(self, conn, cfg) -> dict:
        fq = resolve_freq(cfg)
        if fq.get("governor"):
            self.set_governor(conn, fq.governor)
        if cfg.stage.get("scheduling", {}).get("swap_off", True):
            self.set_swap(conn, off=True)
        return {
            "governor": self.read_governor(conn),
            "freq_requested_khz": fq.get("khz"),
            "freq_actual_khz": self.read_freq(conn),
            "cores_online": self.online_cores(conn),
        }

    def restore_state(self, conn, saved: dict) -> None:
        if saved.get("governor"):
            self.set_governor(conn, saved["governor"])
        if saved.get("swap_off") is False:
            self.set_swap(conn, off=False)

    def read_temp(self, conn) -> float | None:
        raw = _out(conn.run(
            "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null",
            hide=True, warn=True))
        if not raw:
            return None
        values = [int(v) / 1000.0 for v in raw.split() if v.strip().isdigit()]
        values = [v for v in values if -20 < v < 130]
        return max(values) if values else None

    def read_throttle(self, conn) -> bool | None:
        return None  # nessun contatore uniforme su x86
