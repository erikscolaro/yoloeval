"""Connessione alle board, provisioning e sincronizzazione.

Questo package sa *dove* gira un comando. Non sa cosa significhi: i backend
compongono la command line, `remote/` la esegue.
"""

from .connection import LocalConnection, Result, connection, shell_env  # noqa: F401
