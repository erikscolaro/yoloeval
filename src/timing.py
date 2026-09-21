"""Blocco `timing` — machine hours.

Presente in ogni artefatto e in ogni cella, anche quando lo stadio fallisce:
una cella che crasha dopo venti minuti di build TensorRT ha comunque occupato
la macchina.

`wall_s` include le attese (cooldown termico, reboot, stabilizzazione
nvpmodel), `compute_s` no. Servono entrambi: la sola wall gonfia il totale, il
solo compute sottostima l'occupazione della macchina.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone


def now_iso() -> str:
    """Timestamp ISO 8601 con offset locale."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# Alias: la specifica usa entrambi i nomi.
iso_now = now_iso


class PhaseTimer:
    """Cronometra le fasi di uno stadio e produce il blocco `timing`.

    Le fasi note sono setup, build, calibration, compute, teardown. `compute_s`
    e' la somma delle fasi che occupano davvero la macchina: il tempo passato
    ad attendere il raffreddamento o un riavvio resta solo in `wall_s`.
    """

    #: fasi che contribuiscono a compute_s. `validation` e `accuracy` ci stanno
    #: perche' sono inferenza vera: una passata di mAP su una CPU ARM dura piu'
    #: della misura di latenza che accompagna, e lasciarla fuori sottostima
    #: l'occupazione della macchina proprio dove pesa di piu'.
    COMPUTE_PHASES = ("build", "calibration", "compute", "validation",
                      "accuracy")

    def __init__(self, device: str):
        self.device = device
        self._t0 = time.time()
        self._started_at = now_iso()
        self._phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        """Cronometra una fase. Rientrante sullo stesso nome: i tempi si sommano."""
        t0 = time.time()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (time.time() - t0)

    def add(self, name: str, seconds: float) -> None:
        """Aggiunge tempo a una fase misurata altrove (es. su un'altra macchina)."""
        self._phases[name] = self._phases.get(name, 0.0) + float(seconds)

    def compute_s(self) -> float:
        return sum(v for k, v in self._phases.items() if k in self.COMPUTE_PHASES)

    def block(self) -> dict:
        """Il blocco `timing` come finisce dentro meta.json / results.json."""
        wall = time.time() - self._t0
        phase_s = {f"{k}_s": round(v, 3) for k, v in self._phases.items()}
        phase_s.setdefault("compute_s", round(self.compute_s(), 3))
        return {
            "wall_s": round(wall, 3),
            "compute_s": round(self.compute_s(), 3),
            "started_at": self._started_at,
            "ended_at": now_iso(),
            "device": self.device,
            "phase_s": phase_s,
        }
