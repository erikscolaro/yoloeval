# N del probe su Raspberry Pi 5 al variare di core e precisione

Il probe misura la latenza di una rete di test al variare dei canali C e trova N, il passo di
canali a cui il backend e' piu' efficiente (PIT pota a blocchi di N). Qui: N cambia con il
numero di core usati e con la precisione?

- Probe `probe_edge` della pipeline (conv 3x3 su 20x20, 4 layer, C da 1 a 96, 3 passate, 30
  iterazioni per misura), onnxruntime, 2.4 GHz governor performance, attesa termica fino a
  50 °C; int8, fp16 e fp32 su `cpu_1`, `cpu_2` e `cpu_4`.
- `stage.sweep.seed=1`: probe_id separati, non sovrascrivono il probe int8 su cpu_1 da cui
  pit_auto prende N (n_source cpu_1 di default) (il cpu_1 int8 qui ne e' una replica).
- Script: `run.py` (`--after PID`, `--report` per i soli grafici). Grafici in `figures/`:
  `per_core.png` (un pannello per numero di core, una linea per precisione) e `per_quant.png`
  (un pannello per precisione, una linea per numero di core).

## Esito storico (AOD4)
N ottimo su 1/2/4 core: int8 4/4/2, fp16 8/4/2, fp32 sempre 1; la replica int8 cpu_1 da' N=4 contro 8 (al confine della soglia); dati e grafici nel tag git `aod4-results`.

**Stato**: da rifare con la pipeline AOD4 → Svanström.
