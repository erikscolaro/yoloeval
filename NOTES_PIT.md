# Integrazione yolopit — stato dei lavori

Branch `pit-integration`. Aggiornato a ogni commit: se il lavoro si interrompe, si riparte da qui.

## Piano

1. [x] Repo e branch
2. [x] yolopit come pacchetto (repo `erikscolaro/yolopit`)
3. [x] Versioni fissate (torch 2.12.1, ultralytics 8.4.165, numpy 2.2.6, PLiNIO 3d6b5e0)
4. [x] yolopit 0.2.0: config a gruppi, costi del modello intero, standard/duccio, EMA spenta
5. [x] yoloeval: asse `strategy`, stadi `search` e `finetune`
   - [x] chiavi a catena + gruppi `strategy/` e `finetune/` + test sulle chiavi + docs/config
   - [x] stadi `search` e `finetune` + dispatch in run.py
   - [x] export dai pesi finali della strategia, `tools.artifacts` consapevole dei nuovi artefatti
   - [x] README e IMPLEMENTATION.md
   - [x] prova end-to-end su coco8 (train -> search -> finetune -> export ONNX): funziona
6. [x] stadio `probe` per N e `n: auto` (backend onnxruntime / onnxruntime_py)
7. [x] verifica finale: test di yolopit (44) e yoloeval tutti verdi; su coco8 girano
   baseline e pit_duccio fino all'export ONNX, e pit_auto prende N=16 dal probe

## Decisioni prese da solo (da rivedere con Erik)

- La strategia e' un gruppo Hydra `strategy/` (`baseline`, `pit_standard`, `pit_duccio`): il file
  contiene la configurazione della ricerca. Il fine-tuning ha il suo gruppo `finetune/`. Con
  `train/` sono i tre file separati.
- `strategy=baseline` non cambia nessuna chiave esistente: le cache e i risultati gia' fatti
  restano validi.
- Le MAC della ricerca sono calcolate alla risoluzione di deploy (`model.imgsz`); il training
  della ricerca usa `train.imgsz` se la strategia non dice altro.
- La versione di yolopit entra nella chiave della ricerca: aggiornare yolopit rifa' la ricerca.
- Probe, criterio: efficienza = MAC/latenza; N = il piu' PICCOLO fra 1, 2, 4, ... i cui
  multipli sono tutti (tranne il 20%) entro la tolleranza dall'efficienza migliore dei 32
  canali precedenti. N globale = il piu' GRANDE fra le forme concluse (i suoi multipli vanno
  bene per tutte). 3 passate in ordine casuale, mediana per C, tolleranza = max(10%, rumore).
  Il primo criterio che avevo scritto (nessun C intermedio piu' veloce del multiplo) era
  sbagliato: con pochi canali un C non allineato puo' essere piu' veloce del multiplo
  successivo pur essendo molto meno efficiente.
- Prova reale sulla CPU di questa sessione (VM condivisa, ORT Python, 1 thread, fp32):
  conv3x3 -> N=16 (coerente con N=16 trovato a mano sul modello vero), conv1x1 -> non
  concluso (rumore 17%), N globale 16.
- Probe sui layer lineari (C = neuroni): forma `kind: linear`, di default solo informativa
  (`in_global: false`); l'N globale viene dalla conv 3x3. Prova sulla stessa CPU: conv3x3
  N=16; linear N=1 ma al limite (20% dei C sotto l'inviluppo, 11% con N=8, 0% con N=16): i
  lineari sono molto meno sensibili all'allineamento.
- Probe: supporta solo i backend che misurano un ONNX portabile (onnxruntime, onnxruntime_py).
  TensorRT/OpenVINO/Axelera compilano sulla board: da aggiungere. I compute target `metis`
  vengono saltati.
- `pit.n: auto` cerca il probe con stessa board, precisione e backend, sul compute target
  `strategy.n_source` (pit_auto: cpu_1). Se ce ne sono piu' d'uno prende il piu' recente.

## Osservazioni

- Con Ultralytics 8.4.165 il `yolo26n.pt` ufficiale si carica con `end2end=False`: l'export
  ONNX ha la testa one-to-many anche per la BASELINE (warning "motivo non fra quelli noti").
  Non dipende da yolopit; da capire se e' voluto.
- Axelera (risolto): per i modelli potati il pacchetto yolopit installato sulla workstation
  viene copiato sulla board (`{workdir}/python/yolopit-<versione>-<hash>/`) e messo nel
  PYTHONPATH del solo comando che carica il `.pt` (export nel container, confronto numerico).
  Niente pip sulla board: i requisiti fissati di yolopit romperebbero torch/Ultralytics del
  Voyager SDK. Prima dell'export si verifica che `import yolopit.runtime` funzioni nel
  container, altrimenti errore chiaro. Da provare sulla board vera.
- Preesistente, non legato a yolopit: il confronto numerico gira con il python dell'HOST della
  board (`hardware.remote.python`), ma sul Pi i requisiti dell'host non hanno torch.
- `yolopit` nei requisiti e' fissato al commit 64ed4e8 (0.2.0), non al tag: da qui i tag non si
  pubblicano.

## Punti aperti per Erik

1. Merge di `pit-integration` nel branch principale di yoloeval: da fare quando hai
   sincronizzato le tue modifiche locali (non l'ho toccato).
2. Tag `v0.2.0` di yolopit sul commit 64ed4e8 (da qui i tag non si pubblicano).
3. Axelera: meccanismo pronto (copia di yolopit + PYTHONPATH), da provare sulla board vera.
4. Probe: aggiungere TensorRT/OpenVINO (compilano sulla board); prova su RPi5 vera.
5. DUCCIO su GPU e su AOD-4: provato solo su CPU e coco8.
