"""I riferimenti in docs/config/ devono restare allineati ai config veri.

Una documentazione che invecchia e' peggio di nessuna documentazione: chi la
legge scrive un campo che non esiste piu' e non se ne accorge finche' non gli
tocca leggere il codice. Qui il controllo e' meccanico: ogni campo usato dai
file di conf/ deve comparire nel riferimento del suo gruppo.

Il contrario non e' un errore: i riferimenti elencano apposta anche i campi
facoltativi che nessun file usa oggi.
"""

from __future__ import annotations

import pytest
import yaml

from tests.conftest import ROOT

DOCS = ROOT / "docs" / "config"
CONF = ROOT / "conf"

#: sezioni le cui chiavi sono nomi scelti da chi scrive il file (un profilo,
#: un compute target, un backend): si confrontano i campi dentro, non i nomi
VARIANTI = {"freq", "compute", "backend_args"}

GRUPPI = ["stage", "model", "train", "dataset", "quantization", "backend",
          "hardware", "eval", "logging"]


def chiavi(node, prefisso: str = "") -> set[str]:
    """Percorsi puntati di un albero YAML, con `*` al posto dei nomi liberi."""
    trovate: set[str] = set()
    if not isinstance(node, dict):
        return trovate
    for k, v in node.items():
        path = f"{prefisso}{k}"
        trovate.add(path)
        if isinstance(v, dict):
            figlio = f"{path}.*." if k in VARIANTI else f"{path}."
            if k in VARIANTI:
                for variante in v.values():
                    trovate |= chiavi(variante, figlio)
            else:
                trovate |= chiavi(v, figlio)
    return trovate


def carica(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("gruppo", GRUPPI)
def test_ogni_campo_usato_e_documentato(gruppo):
    riferimento = chiavi(carica(DOCS / f"{gruppo}.yaml"))
    usate: set[str] = set()
    for f in sorted((CONF / gruppo).glob("*.yaml")):
        usate |= chiavi(carica(f))
    mancanti = usate - riferimento
    assert not mancanti, (
        f"campi usati in conf/{gruppo}/ ma assenti da docs/config/{gruppo}.yaml: "
        f"{sorted(mancanti)}"
    )


def test_config_globale_documentato():
    riferimento = chiavi(carica(DOCS / "config.yaml"))
    usate = chiavi(carica(CONF / "config.yaml"))
    # `defaults` e' una lista, non un dizionario: entra come chiave e basta
    mancanti = {k for k in usate - riferimento if not k.startswith("hydra.")}
    assert not mancanti, (
        f"campi in conf/config.yaml assenti da docs/config/config.yaml: "
        f"{sorted(mancanti)}"
    )


@pytest.mark.parametrize("gruppo", [*GRUPPI, "config"])
def test_il_riferimento_e_yaml_valido(gruppo):
    """Se non parsa non e' copiabile, ed e' tutto quello che deve fare."""
    assert carica(DOCS / f"{gruppo}.yaml")


def test_i_riferimenti_stanno_fuori_da_conf():
    """Dentro un gruppo sarebbero opzioni selezionabili, e finirebbero negli
    sweep scritti con glob(*)."""
    assert not list(CONF.rglob("*example*"))
    assert not list(CONF.rglob("*template*"))
