"""Mirror opzionale dei risultati su Weights & Biases.

La fonte di verita' resta il JSON locale: questo modulo copia altrove, non
sostituisce. Se W&B non e' installato o la rete non c'e', la cella resta
valida e viene solo loggato un avviso — un mirror che fa fallire la misura che
sta specchiando sarebbe peggio di nessun mirror.

Esiste perche' `conf/logging/wandb.yaml` dichiara il backend: una voce di
configurazione che non fa niente e' peggio di una voce assente, perche' chi la
attiva crede che i risultati stiano finendo da qualche parte.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_WARNED = False


def _flatten(rec: dict) -> dict:
    """Stessa forma delle colonne di `tools.aggregate`, cosi' i due coincidono."""
    flat = {
        "cell_id": rec.get("cell_id"),
        "status": rec.get("status"),
        "order_index": rec.get("order_index"),
    }
    for block, prefix in (("axes", ""), ("latency", "lat_"), ("accuracy", "acc_"),
                          ("energy", "energy_"), ("runtime_state", "rt_"),
                          ("validation", "val_"), ("timing", "time_")):
        payload = rec.get(block) or {}
        if isinstance(payload, dict):
            for k, v in payload.items():
                if not isinstance(v, (dict, list)):
                    flat[f"{prefix}{k}"] = v
    return flat


def mirror_result(cfg, rec: dict) -> None:
    """Copia una cella su W&B, se il gruppo `logging` lo chiede."""
    global _WARNED

    if cfg.logging.get("backend") != "wandb":
        return
    try:
        import wandb
    except ImportError:
        if not _WARNED:
            log.warning(
                "logging=wandb richiesto ma wandb non e' installato: "
                "i risultati restano solo in results/"
            )
            _WARNED = True
        return

    settings = cfg.logging.get("wandb") or {}
    try:
        run = wandb.init(
            project=settings.get("project", "yolo-bench"),
            entity=settings.get("entity"),
            mode=settings.get("mode", "online"),
            tags=list(settings.get("tags") or []),
            name=rec.get("cell_id"),
            config=rec.get("config"),
            reinit=True,
        )
        run.log(_flatten(rec))
        run.finish()
    except Exception as exc:  # noqa: BLE001 - il mirror non fa fallire la cella
        log.warning("mirror W&B fallito per %s: %s", rec.get("cell_id"), exc)
