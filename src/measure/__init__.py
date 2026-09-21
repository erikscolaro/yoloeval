"""Stato della board, termico, energia, accuratezza, risoluzione freq/compute.

Questo package non conosce i backend: legge e scrive lo stato della macchina
su cui la misura avviene.
"""

from .board import BoardController, board_controller, tuned  # noqa: F401
from .compute import (  # noqa: F401
    affinity_spec,
    compute_targets,
    resolve_compute,
    resolve_freq,
)
