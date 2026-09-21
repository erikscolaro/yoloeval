# Riferimento della configurazione

Un file per gruppo, con ogni campo, i valori che accetta e cosa fa ognuno.
Servono a scrivere un file nuovo senza andare a cercare nel codice quali
stringhe sono ammesse.

| file | documenta |
|---|---|
| [`config.yaml`](config.yaml) | `conf/config.yaml`: flag, path, chiavi di cache |
| [`stage.yaml`](stage.yaml) | `conf/stage/*.yaml`: quale stadio, termico, scheduling |
| [`model.yaml`](model.yaml) | `conf/model/*.yaml` |
| [`train.yaml`](train.yaml) | `conf/train/*.yaml` (gli argomenti sono quelli di Ultralytics) |
| [`dataset.yaml`](dataset.yaml) | `conf/dataset/*.yaml` |
| [`quantization.yaml`](quantization.yaml) | `conf/quantization/*.yaml`: precisione, calibrazione, soglie |
| [`backend.yaml`](backend.yaml) | `conf/backend/*.yaml`: requisiti, build, riga di comando |
| [`hardware.yaml`](hardware.yaml) | `conf/hardware/*.yaml`: una board per file |
| [`eval.yaml`](eval.yaml) | `conf/eval/*.yaml`: le soglie, che sono due coppie diverse |
| [`logging.yaml`](logging.yaml) | `conf/logging/*.yaml` |

Non stanno dentro `conf/` di proposito: Hydra tratta ogni file di un gruppo
come un'opzione selezionabile, quindi una board o un backend di esempio
finirebbero negli sweep scritti con `glob(*)`.

Ogni file elenca piu' campi di quanti ne trovi in un singolo file vero: i
campi facoltativi ci sono tutti, con il loro default. Copiane uno, togli i
commenti e quello che non ti serve, e salvalo nel gruppo giusto.

Sono file YAML validi, e `tests/test_docs_config.py` verifica che ogni campo
usato dai file veri sia documentato qui. Se aggiungi un campo a un gruppo e ti
dimentichi di scriverlo, te lo dice `pytest`.
