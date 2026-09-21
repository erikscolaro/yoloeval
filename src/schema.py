"""Schema di `results.json` — versionato fin dall'inizio.

Quando si aggiunge un campo a meta' lavoro, le run precedenti devono restare
leggibili: `schema_version` e' il modo per saperlo in aggregazione invece di
scoprirlo da una colonna vuota.

Questo modulo costruisce e valida i record. Non scrive su disco.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .timing import now_iso

SCHEMA_VERSION = 1

#: stati ammessi per una cella
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUSES = (STATUS_OK, STATUS_SKIPPED, STATUS_FAILED)

#: perimetri di misura ammessi
SCOPE_E2E = "end_to_end_with_transfers"
SCOPE_COMPUTE = "compute_only"


def axes(cfg: DictConfig, training_key: str | None = None) -> dict:
    """Gli assi della matrice, appiattiti: sono le colonne dell'analisi."""
    return {
        "model": cfg.model.name,
        "training_key": training_key,
        "quantization": cfg.quantization.name,
        "backend": cfg.backend.name,
        "board": cfg.hardware.board,
        "arch": cfg.hardware.arch,
        "freq_target": cfg.freq_target,
        "compute_target": cfg.compute_target,
    }


def base_record(
    cfg: DictConfig,
    cell_id: str,
    training_key: str | None = None,
    env: dict | None = None,
) -> dict:
    """Parte comune a ok / skipped / failed.

    `order_index` viene da Hydra: e' l'indice del job nello sweep, e serve a
    vedere in analisi se la latenza correla con l'ordine di esecuzione — cioe'
    se il termico stava inquinando le misure.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "status": None,
        "timestamp": now_iso(),
        "order_index": order_index(),
        "axes": axes(cfg, training_key),
        "env": env or {},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def order_index() -> int:
    """Indice del job nello sweep Hydra, -1 fuori da un multirun."""
    for var in ("HYDRA_JOB_NUM", "HYDRA_JOB_ID"):
        raw = os.environ.get(var)
        if raw and raw.isdigit():
            return int(raw)
    try:
        from hydra.core.hydra_config import HydraConfig

        return int(HydraConfig.get().job.num)
    except Exception:
        return -1


def skipped(rec: dict, reason: str) -> dict:
    """Cella non applicabile: non e' un errore, ma deve lasciare traccia.

    Senza questo record, in analisi non si distingue una combinazione non
    supportata da una che e' crashata.
    """
    rec["status"] = STATUS_SKIPPED
    rec["reason"] = reason
    return rec


def failed(rec: dict, exc: BaseException) -> dict:
    rec["status"] = STATUS_FAILED
    rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["trace"] = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return rec


def ok(rec: dict, payload: dict) -> dict:
    rec["status"] = STATUS_OK
    rec.update(payload)
    return rec


def latency_block(latency: Any, iters: int, warmup: dict, scope: str) -> dict:
    """Normalizza il risultato del parser nel blocco `latency` dello schema."""
    from dataclasses import asdict, is_dataclass

    data = asdict(latency) if is_dataclass(latency) else dict(latency)
    data.pop("raw_stdout", None)
    data["iters"] = data.get("iters") or iters
    data.update(warmup)
    data["scope"] = scope
    return data


#: campi obbligatori per stato
REQUIRED = {
    STATUS_OK: ("cell_id", "axes", "config", "latency", "timing", "runtime_state"),
    STATUS_SKIPPED: ("cell_id", "axes", "config", "reason"),
    STATUS_FAILED: ("cell_id", "axes", "config", "error", "trace", "timing"),
}


def validate_record(rec: dict) -> list[str]:
    """Ritorna la lista dei problemi. Vuota = record conforme."""
    problems: list[str] = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != {SCHEMA_VERSION}"
        )
    status = rec.get("status")
    if status not in STATUSES:
        problems.append(f"status non valido: {status!r}")
        return problems
    for field in REQUIRED[status]:
        if rec.get(field) in (None, {}, ""):
            problems.append(f"campo obbligatorio mancante: {field}")
    if status == STATUS_OK:
        lat = rec.get("latency") or {}
        for field in ("mean_ms", "median_ms", "p99_ms"):
            if lat.get(field) is None:
                problems.append(f"latency.{field} mancante")
        if lat.get("scope") not in (SCOPE_E2E, SCOPE_COMPUTE):
            problems.append(f"latency.scope non valido: {lat.get('scope')!r}")
    timing = rec.get("timing") or {}
    if status in (STATUS_OK, STATUS_FAILED):
        # Il timing serve anche sulle celle fallite: fanno parte delle
        # machine hours tanto quanto quelle riuscite.
        for field in ("wall_s", "device"):
            if timing.get(field) is None:
                problems.append(f"timing.{field} mancante")
    return problems
