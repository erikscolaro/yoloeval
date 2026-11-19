# Costo del tracker di Kalman sul Raspberry Pi 5

Quanto costa sul Pi un passo di `KalmanTracker.step` (`src/tracking.py`): predizione, associazione (Hungarian di
scipy), aggiornamento, NMS fra classi e voto della classe.

- `misura.py`: N oggetti di 40×30 px a velocità costante; per fotogramma N detection forti, N/2 deboli vicine con
  un'altra classe e 1 falso allarme debole. Un core, 2000 fotogrammi dopo 100 di riscaldamento.
- `rifai.sh`: lock del Pi, governor performance, 3 passate, `results/kalman.jsonl`; rimette il governor. Serve
  scipy nel venv del Pi (`scripts/requirements/aarch64-rpi5.txt`, provision del 19/11/2026).

## Risultati (19/11/2026, mediana delle 3 passate; le passate differiscono di meno di 0,03 ms)

| oggetti | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| ms per fotogramma | 0,02 | 0,21 | 0,39 | 0,68 | 1,27 | 2,48 |

Il costo cresce di circa 0,15 ms per oggetto. Con 1–4 oggetti il Kalman costa meno di 0,7 ms, cioè meno del 3% del
budget del modello (28 ms). Il costo non contiene il taglio dei crop e l'inferenza (`../crop_resize_rpi5`).
