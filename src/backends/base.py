"""Contratto dei backend.

Aggiungere un backend significa: un file qui, piu' un YAML in `conf/backend/`.
Il backend sa *come* si esporta, si misura e si parsa; non sa *dove* gira (lo
sa `remote/`) ne' come si tocca la board (lo sa `measure/`).

Regola sui parser: non devono mai inventare. Se il tool cambia formato di
output fra versioni e il parsing fallisce, la cella deve risultare `failed` con
il raw stdout salvato accanto, non `ok` con numeri a caso.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ParseError

log = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Metriche di latenza normalizzate fra backend.

    I campi che un tool non riporta restano `None`: meglio una colonna vuota
    in analisi che un numero plausibile e falso.
    """

    mean_ms: float
    median_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    throughput_qps: float | None = None
    gpu_compute_ms: float | None = None
    iters: int | None = None
    warmup_ms: int | None = None
    warmup_iters: int | None = None
    scope: str = "end_to_end_with_transfers"
    tool: str | None = None
    raw_stdout: str = field(default="", repr=False)


class Backend(ABC):
    """Interfaccia implementata da ogni backend, e nient'altro."""

    name: str = "base"
    export_format: str = "onnx"
    builds_on_target: bool = False
    #: perimetro della misura, dichiarato: va allineato fra backend
    scope: str = "end_to_end_with_transfers"

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abstractmethod
    def export(self, conn, cfg, src: Path, dst: Path) -> Path:
        """Produce l'artefatto.

        Se `builds_on_target`, `conn` e' la board; altrimenti e' una
        `LocalConnection`. Idempotente: se `dst` esiste ed e' valido, ritorna
        senza rifare.
        """

    def prepare(self, cfg, src: Path, dst: Path) -> Path:
        """Artefatto **portabile** da cui parte la build on-target.

        La workstation produce solo `.pt` e `.onnx`; un engine TensorRT o un
        modello Axelera vanno compilati sulla board. I backend che buildano
        on-target usano questo hook per dire *cosa* va sincronizzato: cosi' lo
        stadio di export resta ignaro del formato intermedio.
        """
        return Path(src)

    @abstractmethod
    def build_cmd(self, cfg, artifact: Path) -> str:
        """Command line del tool nativo di misura, gia' sostituita."""

    @abstractmethod
    def parse(self, stdout: str) -> LatencyResult:
        """Estrae le metriche. Solleva ParseError se l'output non e'
        riconoscibile: mai restituire zeri o valori inventati."""

    @abstractmethod
    def inspect(self, artifact: Path) -> dict:
        """{'actual_e2e': bool, 'precision': str, 'input_shape': tuple,
        'output_names': list}."""

    def validate(self, conn, cfg, artifact: Path, ref) -> dict:
        """Confronto numerico vs FP32 su batch fisso.

        Default: delega a `validation.numerical`, che sa confrontare gli
        artefatti caricabili da Ultralytics/ONNX Runtime. I backend con
        artefatti hardware-specifici sovrascrivono.
        """
        from ..validation.numerical import compare_with_reference

        return compare_with_reference(conn, cfg, artifact, ref)

    # --- utilita' condivise ---------------------------------------------
    def compute_spec(self, cfg) -> dict:
        """Device e numero di core richiesti dalla cella corrente."""
        from ..measure.compute import resolve_compute

        ct = resolve_compute(cfg)
        return {
            "device": ct.get("device"),
            "n_cores": ct.get("n_cores"),
        }

    def bench_cfg(self, cfg):
        return cfg.backend.benchmark


# --- helper di parsing condivisi ----------------------------------------

def _f(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def search(pattern: str, text: str, group: int = 1, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(group) if m else None


def require(value, what: str, stdout: str):
    """Solleva ParseError invece di restituire un valore mancante."""
    if value is None:
        head = "\n".join((stdout or "").splitlines()[:20])
        raise ParseError(f"{what} non trovato nell'output del tool:\n{head}")
    return value


def percentiles(samples: list[float]) -> dict:
    """Percentili da una lista di campioni (nearest-rank)."""
    if not samples:
        raise ParseError("nessun campione di latenza nell'output")
    ordered = sorted(samples)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
        return ordered[idx]

    mean = sum(ordered) / len(ordered)
    return {
        "mean_ms": mean,
        "median_ms": pct(50),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "throughput_qps": 1000.0 / mean if mean else None,
    }
