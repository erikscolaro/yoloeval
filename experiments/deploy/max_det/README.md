# Ablation: max_det sul Raspberry Pi 5

Latenza e mAP del yolo26p potato e finetunato (`experiments/benchmark/pruned_finetuned`) al variare di
`max_det`, il numero di box in uscita: nella testa one-to-one di YOLO26 e' il k del TopK,
fissato nel grafo all'export. Valori: 1, 2, 4, 8, 16, 32, 64, 128, 256 e 300 (default di
Ultralytics).

- Prima: `run.py stage=train`, `adapt`, `search` e `finetune` con gli override di `COMMON` in `run.py` (pesi di
  `final_weights`); val di Svanstrom.
- Un IR OpenVINO fp32 640x640 per valore, esportato come `src/backends/openvino.py`.
- mAP: `model.val()` dell'IR sul val intero (workstation, CPU, batch 1, conf 0.001), con
  `max_det` uguale a quello del grafo.
- Latenza sul Pi (2.4 GHz governor performance, 4 thread, FIFO 80, attesa termica fino a 50 °C):
  - `benchmark_app`: solo inferenza, due passate in ordine opposto da 300 iterazioni;
  - API Python di OpenVINO: pre-processing + inferenza + post-processing (conf 0.25) su 300
    frame del val gia' 640x640 in memoria (camera a 640), 3 giri alternati.
- Script: `run.py` (export, mAP, misure sul Pi con `bench.sh` e `pi_bench.py`,
  grafici), `--report` per i soli grafici. Dati in `results/`, grafici in `figures/`, IR in `runs/`.

## Criterio

`max_det` taglia le box per punteggio prima della soglia di confidenza, quindi conta solo se
scarta box che la soglia avrebbe tenuto. L'ordine e':

1. si tara prima `conf`, sul punto di lavoro voluto (precision contro recall);
2. poi si tara `max_det`: almeno il numero massimo di box sopra quella soglia atteso in un
   frame reale, con margine. Piu' bassa e' la soglia, piu' box passano e piu' alto deve essere
   `max_det`.

Sotto quel valore si perdono oggetti; sopra non cambia niente nelle detection e costa solo
latenza. La mAP fa eccezione: si calcola a conf 0.001, quindi per riportarla serve un `max_det`
alto (o il default 300).

## Esito storico (AOD4)
Latenza piatta fino a `max_det` 64 (~22.0 ms con benchmark_app, ~27.4 end-to-end), a 300 +0.5 / +1.1 ms; da 16 in su le stesse detection a conf 0.25; la mAP satura verso 64; dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
