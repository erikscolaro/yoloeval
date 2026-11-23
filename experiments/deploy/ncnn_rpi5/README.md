# ncnn sul Raspberry Pi 5

**Scopo**: capire se ncnn è più veloce di OpenVINO (il backend di deploy) sul Pi 5, in fp16 e int8, e
quanta mAP perde l'export. Il 23/11 aggiunto anche fp32, per avere il riferimento senza fp16.

**Stato**: chiuso il 16/11/2026; fp32 aggiunto il 23/11/2026.

## Setup

- **Modelli**: P e N standard (D126), gli stessi della griglia: P `59d8534a`, N `9dca8d15`.
- **Board**: Raspberry Pi 5, 4 core a 2,4 GHz, governor performance (`target=rpi5_640`, ingresso 640×640),
  ventola al massimo, misura con `chrt -f 80`.
- **ncnn 20261020**:
  - export con pnnx di Ultralytics;
  - fp16 = pesi fp16, con storage e aritmetica fp16 di ncnn;
  - int8 = `ncnn2table` (KL, le 511 immagini di `images/calib` usate dagli altri backend) e `ncnn2int8`,
    che quantizza solo conv e depthwise. Concat, attivazioni e coda della testa restano float.
- **Latenza**: `benchncnn` con 200 iterazioni dopo 8 di riscaldamento, 4 thread, pesi vuoti (conta solo la forma),
  3 ripetizioni. `benchncnn` dà solo min, max e media, quindi in tabella c'è la **media** anche per OpenVINO
  (`benchmark_app`, 200 iterazioni, `-infer_precision f16`), rimisurato qui nella stessa sessione.
- **mAP**: val completo (6 659 immagini), `conf` 0,001, `iou` 0,7, `max_det` 64, **calcolata sul Pi**
  (`eval.map_on_target=true`), con il runtime che fa anche la latenza.

Comandi (dalla root della repo):

```bash
systemd-run --user --unit=ncnn-rpi5 --working-directory=$PWD experiments/deploy/ncnn_rpi5/rifai.sh
.venv/bin/python experiments/deploy/ncnn_rpi5/run.py      # tabella da results/ -> results/tabella.csv
```

Prima volta: `python run.py stage=provision hardware=rpi5` compila `benchncnn` sul Pi (`scripts/remote/build_ncnn.sh`);
sulla workstation lo stesso script installa `ncnn2table`/`ncnn2int8` nel venv.

## Risultati

Latenza media ± deviazione standard sulle 3 ripetizioni; Δ rispetto a OpenVINO fp16 dello stesso modello.

| modello | backend | precisione | latenza (ms) | Δ latenza | mAP50-95 | Δ mAP50-95 | mAP50 |
|---|---|---|---|---|---|---|---|
| P | OpenVINO | fp16 | 27,74 ± 0,09 | — | 0,569 | — | 0,834 |
| P | **ncnn** | **fp16** | **25,40 ± 0,08** | **−8,4%** | **0,575** | **+0,006** | 0,849 |
| P | ncnn | fp32 | 56,29 ± 0,16 | +102,9% | 0,575 | +0,006 | 0,849 |
| P | ncnn | int8 | 30,61 ± 0,10 | +10,3% | 0,536 | −0,033 | 0,818 |
| N | OpenVINO | fp16 | 56,71 ± 0,16 | — | 0,593 | — | 0,860 |
| N | ncnn | fp16 | 57,65 ± 0,02 | +1,7% | 0,599 | +0,006 | 0,871 |
| N | ncnn | fp32 | 129,02 ± 0,28 | +127,5% | 0,600 | +0,007 | 0,871 |
| N | ncnn | int8 | 63,26 ± 0,68 | +11,5% | 0,566 | −0,027 | 0,847 |

Dati: `results/tabella.csv` (i record per ripetizione sono in `results/*.json`). In una ripetizione di N int8 il
massimo arriva a 124 ms (un singolo picco: media 63,1 ms contro 62,3–62,4 delle altre due).

## Commento

- **ncnn fp16 vince solo su P**: è più veloce dell'8,4% (2,3 ms). Su N è pari a OpenVINO, anzi più lento
  dell'1,7%. Il vantaggio non scala con la rete. Ipotesi, non verificata: su reti piccole pesa il costo fisso per
  nodo, che ncnn ha più basso; su N dominano le conv grandi, dove i kernel si equivalgono.
- **fp32 costa più del doppio dell'fp16** (2,2× su P e su N: 56,3 contro 25,4 ms, 129,0 contro 57,7 ms) e la mAP è
  uguale (0,5748 contro 0,575 su P, 0,5995 contro 0,5993 su N): l'fp16 non perde precisione. `benchncnn` ha fp16
  acceso nel codice, quindi per l'fp32 serve un binario a parte (`build_fp32.sh`, `benchncnn_fp32` con fp16 spento);
  la mAP fp32 è calcolata sulla workstation (x86 fp32), non sul Pi. Le due righe si rifanno con `rifai_fp32.sh`.
- **int8 non conviene neanche con ncnn**: è più lento dell'fp16 del 20% su P e del 10% su N, e perde
  0,03 di mAP. Vale la stessa conclusione della prova X2 con OpenVINO e onnxruntime: l'A76 ha l'fp16 nativo ma
  non i8mm, e ogni conv int8 aggiunge una dequantizzazione prima della SiLU (non profilato strato per strato).
- **La mAP non è confrontabile alla pari fra i due backend.** Ultralytics esporta ncnn senza la parte one-to-one
  (`end2end: false`): il grafo dà le 8 400 predizioni e la mAP passa per l'NMS, mentre OpenVINO ha il top-k
  one-to-one nel grafo. Il +0,006 di ncnn fp16 viene quasi certamente da qui, non dal backend. Per lo stesso motivo
  la latenza ncnn non include il top-k, che però costa poco, e in deploy servirebbe un NMS fuori dal grafo.
- **mAP calcolata sul Pi o sull'host**: per ncnn non cambia. Le prime misure di P l'avevano calcolata sulla
  workstation (`runs/results_map_host`): fp16 0,5749 contro 0,5750 sul Pi, int8 0,5367 contro 0,5362.

## Limiti

- La calibrazione int8 stira le immagini 640×512 a 640×640 invece di usare il letterbox
  (`shortcut:` in `src/backends/ncnn.py`). Potrebbe spiegare una parte della perdita di mAP int8.
- `benchncnn` gira con pesi vuoti: per fp16 e int8 la latenza dipende solo dalla forma, quindi va bene, ma non è
  l'eseguibile usato per la mAP.
- La validazione numerica dell'export risulta "degradata" (scarto massimo 5,6 px sui box grezzi one-to-many, in
  fp16): è un confronto di tensori, non indica errori. La mAP lo conferma.
