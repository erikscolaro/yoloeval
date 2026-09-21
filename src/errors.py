"""Eccezioni del tool.

Ogni errore ha un tipo proprio: in fase di analisi il campo `error` di una
cella fallita deve dire *cosa* e' andato storto, non solo che qualcosa e'
andato storto.
"""


class BenchError(Exception):
    """Base di tutte le eccezioni del tool."""


class MissingWeights(BenchError):
    """Nessun artefatto di training corrispondente al training_key."""


class MissingExport(BenchError):
    """Nessun artefatto esportato corrispondente all'export_key."""


class MissingArtifact(BenchError):
    """La directory dell'export esiste ma non contiene il file atteso."""


class ExportFailed(BenchError):
    """L'export o la build on-target e' fallita."""


class BenchmarkFailed(BenchError):
    """Il tool di misura ha restituito un exit code diverso da zero."""


class ParseError(BenchError):
    """L'output del tool di misura non e' riconoscibile.

    Mai sostituire con zeri o valori di default: una cella con numeri
    inventati e' peggio di una cella fallita.
    """


class ProfileChangeRequiresReboot(BenchError):
    """Il cambio di profilo nvpmodel richiede un riavvio non autorizzato."""


class ProfileMismatch(BenchError):
    """Il profilo attivo non corrisponde a quello richiesto."""


class BoardUnreachable(BenchError):
    """La board non e' tornata raggiungibile entro il timeout."""


class ProvisionFailed(BenchError):
    """Lo script di provisioning e' fallito o l'ambiente non e' coerente."""


class DatasetMismatch(BenchError):
    """L'hash del dataset non corrisponde a quello usato per il training."""
