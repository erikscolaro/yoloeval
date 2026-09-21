"""Celle valide.

Nessuna tabella di compatibilita' hardcoded: la maggior parte delle
combinazioni illegali non e' nemmeno esprimibile, perche' `freq` e `compute`
sono chiavi dentro il file della board e una chiave assente solleva `KeyError`
prima di qualunque validazione. Resta da controllare solo cio' che dipende dal
backend e dal runtime.

La validazione avviene in due momenti: **statica** prima di connettersi, per
non pagare SSH su celle che verranno scartate; **dinamica** dopo la
connessione, solo per il conteggio dei core online.
"""

from __future__ import annotations

import logging

from ..measure.compute import resolve_compute, resolve_freq

log = logging.getLogger(__name__)


def is_valid(cfg, conn=None) -> tuple[bool, str | None]:
    hw, be = cfg.hardware, cfg.backend
    resolve_freq(cfg)                 # KeyError se il profilo non esiste
    ct = resolve_compute(cfg)         # KeyError se il target non esiste

    # capability del backend vs capability della board
    if be.requires.get("trt") and not hw.get("trt_available"):
        return False, "tensorrt non disponibile su questa board"
    if be.requires.get("axelera") and not hw.get("axelera_available"):
        return False, "acceleratore Axelera non presente"
    if hw.arch not in be.requires.arch:
        return False, f"backend non supportato su {hw.arch}"
    # Il sistema operativo richiesto dal backend puo' essere soddisfatto dal
    # container invece che dall'host: e' il caso del Pi, dove Raspberry Pi OS
    # non e' supportato da Axelera e per questo il Voyager SDK gira dentro un
    # Ubuntu 22.04. Il container non e' un dettaglio di installazione, e'
    # quello che rende la cella possibile.
    required_os = be.requires.get("os")
    if required_os:
        in_container = bool(
            be.get("build", {}).get("container")
            and hw.get("provision", {}).get("container")
        )
        if not in_container and hw.get("os") not in required_os:
            return False, (
                f"{be.name} richiede {list(required_os)}, la board dichiara "
                f"{hw.get('os')} e non prevede un container"
            )

    # coerenza backend / device
    device = ct.get("device")
    if device == "metis" and be.name != "axelera":
        return False, f"{be.name} non puo' girare sull'AIPU"
    if device == "cuda" and not be.get("supports_gpu", True):
        return False, f"{be.name} non supporta GPU"
    if device == "cpu" and not be.get("supports_cpu", True):
        return False, f"{be.name} non gira su CPU"
    if device != "metis" and be.name == "axelera":
        return False, "axelera richiede il compute target dell'AIPU"

    # precisione imposta dal backend: il Voyager SDK quantizza e compila
    # sempre a INT8 per l'architettura mixed-precision dell'AIPU, quindi una
    # cella "axelera + fp32" misurerebbe un modello INT8 etichettandolo fp32.
    forced_bits = be.get("build", {}).get("quantize")
    if forced_bits and cfg.quantization.precision != f"int{forced_bits}":
        return False, (
            f"{be.name} quantizza sempre a int{forced_bits}: "
            f"{cfg.quantization.name} non e' rappresentabile su questo backend"
        )

    # unico vincolo interno alla board: core online per profilo
    req = ct.get("requires_freq")
    if req and cfg.freq_target not in req:
        return False, f"{cfg.compute_target} non disponibile in {cfg.freq_target}"

    # dinamico, solo se gia' connessi
    if conn is not None and device == "cpu" and ct.get("n_cores"):
        from ..measure.board import board_controller

        online = board_controller(cfg).online_cores(conn)
        if online is not None and int(ct.n_cores) > online:
            return False, f"richiesti {ct.n_cores} core, online {online}"
    return True, None
