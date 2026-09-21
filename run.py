#!/usr/bin/env python3
"""yolo-bench — unico entry point.

    python run.py stage=train model=yolo26n
    python run.py -m stage=benchmark hardware=jetson_orin freq_target=maxn,w15

Il dispatch avviene per `cfg.stage.name`: ogni stadio ha il suo entry point e
gestisce la propria cache in modo indipendente. Qui non c'e' logica di misura.

Parallelismo: `hydra/launcher=joblib` e' utilizzabile per training ed export,
**mai** per la misura di latenza — job concorrenti sulla stessa GPU o sugli
stessi core producono numeri inutilizzabili.
"""

from __future__ import annotations

import logging
import re
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("yolo-bench")

#: flag dichiarati in conf/config.yaml con un default
RUN_FLAGS = (
    "dry_run", "force", "retry_failed", "force_retrain", "force_reexport",
    "allow_reboot", "skip_invalid", "continue_on_error",
)

_PLUS_FLAG = re.compile(rf"^\+(?!\+)({'|'.join(RUN_FLAGS)})=")


def normalize_flag_overrides(argv: list[str]) -> list[str]:
    """`+dry_run=true` -> `++dry_run=true`.

    I flag hanno un default in `config.yaml`, quindi il codice puo' leggerli
    senza `get()`; ma `+chiave=` in Hydra significa "aggiungi una chiave che
    non c'e'" e fallisce proprio perche' c'e'. La forma con il `+` e' quella
    documentata e quella che viene in mente, quindi la si accetta invece di
    rispondere con un errore di sintassi.
    """
    return [_PLUS_FLAG.sub(r"++\1=", a) for a in argv]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage.name
    log.debug("config risolta:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if stage == "provision":
        from src.remote.provision import provision

        provision(cfg)
        return

    if stage == "train":
        from src.stages.train import run_training

        run_training(cfg)
        return

    if stage == "export":
        from src.stages.export import run_export

        run_export(cfg)
        return

    if stage == "benchmark":
        from src.stages.benchmark import run_benchmark

        run_benchmark(cfg)
        return

    raise ValueError(
        f"stadio sconosciuto: {stage!r} "
        f"(stage=train|export|benchmark|provision)"
    )


if __name__ == "__main__":
    sys.argv[1:] = normalize_flag_overrides(sys.argv[1:])
    sys.exit(main())
