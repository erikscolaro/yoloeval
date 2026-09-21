"""Registry dei backend.

`get_backend(name)` e' l'unico punto in cui il resto del codice nomina un
backend: gli stadi ricevono un'istanza che rispetta `Backend` e non sanno
quale sia.
"""

from __future__ import annotations

from .base import Backend, LatencyResult  # noqa: F401

_REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str, cfg=None) -> Backend:
    if not _REGISTRY:
        _load_all()
    if name not in _REGISTRY:
        raise KeyError(
            f"backend {name!r} sconosciuto: disponibili {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](cfg)


def available() -> list[str]:
    if not _REGISTRY:
        _load_all()
    return sorted(_REGISTRY)


def _load_all() -> None:
    from . import (  # noqa: F401
        axelera,
        executorch,
        onnxruntime,
        onnxruntime_py,
        openvino,
        tensorrt,
    )
