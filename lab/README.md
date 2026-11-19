# lab

Il launcher della pipeline, in bash puro (`lab/launcher`, dalla root o da qui). Compone righe di comando di
`run.py` e degli script senza modificare nessuno YAML: tutto è un override Hydra.

- **ricette** (`lab/ricette/<nome>.ricetta`): step e override pronti. Prima riga `# descrizione`, poi
  `step a b c` e una riga `set chiave=valore` per override. Si salvano anche dal launcher.
- **step, modello, profilo, parametri**: costruire una catena a mano, con l'help di ogni campo da `docs/config/`.
- **storico**: i run lanciati, con log, esito, rilancio e stop. I log vanno in `runs/`.
- **strumenti**: `scripts/*.sh`, `tools/*.py` e i `rifai.sh` degli esperimenti.
- **utilità**: stato della macchina e del Pi, test, misure per esito e fallite (`tools/results.py`), tabella delle
  misure (`tools/aggregate.py`), report (`tools/report.py`), artefatti (`tools/artifacts.py`), split del dataset,
  validazione di un modello, pulizia dei log di Hydra.

`report.md.j2` è il template di `python -m tools.report`.
