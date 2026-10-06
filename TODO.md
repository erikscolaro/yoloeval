# TODO

Cose rimandate, da riprendere quando c'e' tempo.

## Quantizzazione

- **Entropy vero**: con `quantize_static` onnxruntime usa 128/128 bin e Entropy coincide con
  MinMax (`experiments/quantization/calibration`). Serve un calibratore costruito a mano
  (es. 2048 bin come TensorRT) per provarlo davvero. Default attuale: MinMax a media mobile.
- **NNCF parametrico** per `backend=openvino`: ignored scope (Ultralytics esclude tutte le
  Sigmoid e la testa) e correzione del bias.
- **TensorRT**: le build `tensorrt` (QDQ esplicito) e `tensorrt_implicit` non sono mai state
  provate con engine veri.

## Pruning

- **N per il target reale**: sul Pi a 4 core il probe da' N=2 (int8, fp16) e N=1 (fp32), la
  search ha usato N=8 (probe int8 su 1 core). Rifare la search con l'N giusto per il deploy
  (`experiments/pruning/probe_n_vs_cores`).
- **Stabilita' del probe**: su 1 core int8 e fp16 sono al confine tra N=4 e N=8 (soglia del 20%
  di multipli non allineati); valutare piu' passate o una soglia diversa.

## Dataset

- **AOD4 ha leakage tra split** (frame dello stesso video, copie aumentate delle foto di
  uccelli): test finale su dati esterni; eventualmente ricostruire lo split per video
  (dedup con pHash, gruppi per video, `StratifiedGroupKFold`).

## Pipeline

- **Sweep yolo26p** (`outputs/sweep_yolo26p/run.sh`): usa ancora `train=scratch`, il cui
  training e' stato cancellato; passare al vincitore (`scratch_musgd`) e lanciare lo sweep
  completo.
- **Default del finetune** in `conf/config.yaml`: passare a `musgd`?
- **`export_max_det: 64`**: riverificare su Svanstrom con `experiments/deploy/max_det` (la scelta era su AOD4, max 36 oggetti per immagine).
- **`export_key` non include gli argomenti di quantizzazione**: per l'int8 dei backend QDQ
  l'export ora si rifa' da solo se gli argomenti sono cambiati (`_stale_quantization` in
  `src/stages/export.py`); per TensorRT e OpenVINO/NNCF serve ancora `force_reexport`.
- **Chiave delle celle e frequenze**: la chiave include tutto `hardware`, quindi cambiare un
  profilo di frequenza (es. `low`/`mid` del Pi 5, 2026-10-02) cambia le chiavi di tutte le celle
  della board, anche `max`: i `--report` degli esperimenti rpi5 precedenti non trovano piu' i
  risultati (i README hanno i numeri). Far entrare solo il profilo selezionato.
- **Export e risultati orfani** `yolo26p_dd1f1787_*` (training `train=scratch` cancellato): da
  cancellare?
- **Pi 5 riavviato durante onnxruntime fp16** (4 core, FIFO 80, 2.4 GHz) il 2026-10-02 alle 3:50:
  capire la causa (alimentazione? FIFO su tutti i core?) e far scadere la connessione invece di
  restare appesi. Rifatto alle 11:15 senza riavvii: forse un caso, da tenere d'occhio.
- **Sudo sulla workstation** per il governor: regola sudoers da sistemare.
