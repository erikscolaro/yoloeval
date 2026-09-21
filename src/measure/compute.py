"""Risoluzione di `freq_target` e `compute_target`.

`freq` e `compute` non sono gruppi Hydra separati: sono dizionari dentro il
file della board. Con gruppi paralleli sarebbe sempre esprimibile un incrocio
illegale (`hardware=rpi5 compute=jetson_orin/gpu`) da intercettare con
controlli scritti a mano; con i dizionari annidati quelle combinazioni non sono
rappresentabili e una chiave assente solleva `KeyError` alla risoluzione, senza
che nessuno abbia dovuto scrivere una regola.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_freq(cfg):
    """KeyError se il profilo non esiste su questa board. Voluto."""
    freq = cfg.hardware.freq
    if cfg.freq_target not in freq:
        raise KeyError(
            f"freq_target={cfg.freq_target!r} non dichiarato da "
            f"{cfg.hardware.board}: disponibili {list(freq.keys())}. "
            f"Il default di config.yaml vale per la workstation; per le board "
            f"il profilo va indicato, per esempio "
            f"freq_target={next(iter(freq.keys()))}"
        )
    return freq[cfg.freq_target]


def resolve_compute(cfg, target=None):
    """KeyError se il target non esiste su questa board. Voluto."""
    target = target if target is not None else cfg.compute_target
    compute = cfg.hardware.compute
    if target is None:
        raise KeyError(
            "compute_target non risolto: usare compute_targets(cfg) per "
            "espandere il default 'tutti quelli dichiarati dalla board'"
        )
    if target not in compute:
        raise KeyError(
            f"compute_target={target!r} non dichiarato da {cfg.hardware.board}: "
            f"disponibili {list(compute.keys())}"
        )
    return compute[target]


def compute_targets(cfg) -> list[str]:
    """`compute_target: null` significa "tutti quelli dichiarati dalla board".

    E' un default, non una validazione: `compute_target=cpu_8` su rpi5 non
    produce una cella saltata, produce un KeyError immediato.
    """
    if cfg.compute_target:
        return [cfg.compute_target]
    return list(cfg.hardware.compute.keys())


def affinity_spec(device: str, n_cores: int | None) -> str | None:
    """Maschera per `taskset`, None quando non serve pinnare."""
    if device != "cpu" or not n_cores or n_cores <= 1:
        return "0" if device == "cpu" and n_cores == 1 else None
    return f"0-{n_cores - 1}"


def thread_env(device: str, n_cores: int | None) -> dict:
    """Variabili di ambiente coerenti con il numero di core richiesto.

    `taskset` limita su quali core il processo *puo'* girare, ma ONNX Runtime e
    PyTorch decidono quanti thread creare leggendo il numero di CPU totali.
    Pinnare su 2 core senza limitare i thread produce 8 thread su 2 core, con
    latenze peggiori del caso monocore.
    """
    if device != "cpu" or not n_cores:
        return {}
    return {
        "OMP_NUM_THREADS": str(n_cores),
        "OPENBLAS_NUM_THREADS": str(n_cores),
        "MKL_NUM_THREADS": str(n_cores),
    }
