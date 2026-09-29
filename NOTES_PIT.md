# Integrazione yolopit — stato dei lavori

Branch `pit-integration`. Aggiornato a ogni commit: se il lavoro si interrompe, si riparte da qui.

## Piano

1. [x] Repo e branch
2. [x] yolopit come pacchetto (repo `erikscolaro/yolopit`)
3. [x] Versioni fissate (torch 2.12.1, ultralytics 8.4.165, numpy 2.2.6, PLiNIO 3d6b5e0)
4. [x] yolopit 0.2.0: config a gruppi, costi del modello intero, standard/duccio, EMA spenta
5. [ ] yoloeval: asse `strategy`, stadi `search` e `finetune`
   - [x] chiavi a catena + gruppi `strategy/` e `finetune/` + test sulle chiavi + docs/config
   - [ ] stadi `search` e `finetune` + dispatch in run.py
   - [ ] export dai pesi finali della strategia, `tools.artifacts` consapevole dei nuovi artefatti
   - [ ] README e IMPLEMENTATION.md
   - [ ] prova end-to-end su coco8 (train -> search -> finetune -> export ONNX)
6. [ ] stadio `probe` per N e `n: auto`
7. [ ] verifica finale

## Decisioni prese da solo (da rivedere con Erik)

- La strategia e' un gruppo Hydra `strategy/` (`baseline`, `pit_standard`, `pit_duccio`): il file
  contiene la configurazione della ricerca. Il fine-tuning ha il suo gruppo `finetune/`. Con
  `train/` sono i tre file separati.
- `strategy=baseline` non cambia nessuna chiave esistente: le cache e i risultati gia' fatti
  restano validi.
- Le MAC della ricerca sono calcolate alla risoluzione di deploy (`model.imgsz`); il training
  della ricerca usa `train.imgsz` se la strategia non dice altro.
- La versione di yolopit entra nella chiave della ricerca: aggiornare yolopit rifa' la ricerca.
