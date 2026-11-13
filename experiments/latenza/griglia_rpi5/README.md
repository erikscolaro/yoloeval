# Griglia di latenza del Raspberry Pi 5

Latenza delle reti intere della griglia larghezza × profondità (F..M × D101/D126/D170), pesi casuali, OpenVINO fp16,
4 core, freq max, mediana di 3 passate. È la tabella `A_budget_dettaglio` (NEW_MASTERPLAN fase 2a), da cui si
sceglie il budget `B`.

- `rifai.sh`: train (pesi casuali), export e benchmark, poi la figura in `reports/budget_latenza_rpi5/`. Idempotente.
- Misure: i JSON in `results/` (uno per cella), letti da `tools/figure_sintesi.py`.
- 4/11/2026: righe 480p, 720p, 1080p, 512², 256², 128² (testa one-to-one a k=300).
- 13/11/2026: riga 640 (k=64, come LUT e B4-R), F..SM; M aggiunto dopo. Controlli: N D126 56,5 ms (B4-R 56,1),
  S D126 143,9 ms (pavimento_s 142,7). Il Pi si è riavviato da solo una volta (SM D126, rep 0); la ripresa è andata.
