# N del probe su Raspberry Pi 5 al variare di core e precisione

Il probe misura la latenza di una rete di test al variare dei canali C e trova N, il passo di
canali a cui il backend e' piu' efficiente (PIT pota a blocchi di N). Qui: N cambia con il
numero di core usati e con la precisione?

- Probe `probe_edge` della pipeline (conv 3x3 su 20x20, 4 layer, C da 1 a 96, 3 passate, 30
  iterazioni per misura), onnxruntime, 2.4 GHz governor performance, attesa termica fino a
  50 °C; int8, fp16 e fp32 su `cpu_1`, `cpu_2` e `cpu_4`.
- `stage.sweep.seed=1`: probe_id separati, non sovrascrivono il probe int8 su cpu_1 da cui
  pit_auto prende N=8 (il cpu_1 int8 qui ne e' una replica).
- Script: `run_experiment.py` (`--after PID`, `--report` per i soli grafici). Grafici:
  `per_core.png` (un pannello per numero di core, una linea per precisione) e `per_quant.png`
  (un pannello per precisione, una linea per numero di core).

## Risultati

N ottimo (tutti i probe conclusivi, rumore 0.25-0.45%):

| precisione | cpu_1 | cpu_2 | cpu_4 |
|---|---|---|---|
| int8 | 4 | 4 | 2 |
| fp16 | 8 | 4 | 2 |
| fp32 | 1 | 1 | 1 |

Latenza della rete di test a C=96 (ms):

| precisione | cpu_1 | cpu_2 | cpu_4 |
|---|---|---|---|
| int8 | 2.64 | 1.56 | 1.01 |
| fp16 | 6.27 | 4.36 | 3.36 |
| fp32 | 9.28 | 4.87 | 2.77 |

- int8 e fp16 hanno gradini (salti ogni 16 canali, ben visibili su 1 core); con piu' core i
  gradini si attenuano e N scende (8/4 → 4 → 2): il lavoro diviso tra i thread rende meno
  costosi i canali "fuori multiplo". fp32 non ha gradini: N=1 con ogni numero di core.
- La replica di int8 su cpu_1 da' N=4 contro N=8 del probe originale (134367b1): con N=4 i
  multipli non allineati sono il 17% qui e il 21% la' (soglia 20%). Su 1 core int8 e fp16 sono
  al confine tra 4 e 8: la N trovata dipende dal rumore della misura.
- fp16 su onnxruntime e' piu' veloce di fp32 su 1 core, ma scala peggio: su 4 core e' piu'
  lento (3.36 contro 2.77 ms).
- Per il deploy sul Pi (4 core): N=2 per int8 e fp16, N=1 per fp32. La search con N=8 (dal
  probe int8 su 1 core) vincola il pruning piu' del necessario per questo target.

![per numero di core](per_core.png)
![per precisione](per_quant.png)
