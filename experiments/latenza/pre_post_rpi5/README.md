# Pre e post-processing sul Raspberry Pi 5 (B1-2)

Quanto costano, a 640×640, le parti dell'end-to-end fuori dal grafo esportato: il pre-processing (uint8 HWC →
float32 NCHW / 255) e il post-processing della testa one-to-one (soglia di confidenza e box sui 64 box del TopK).
Decodifica e TopK sono nel grafo, quindi già nelle latenze del benchmark. Budget del modello = 33,3 ms − pre − post.

- `misura.py`: numpy sul Pi, un core, 2000 iterazioni dopo 200 di riscaldamento; stampa una riga JSON.
- `rifai.sh`: governor performance, 3 passate, `results/pre_post.jsonl`; rimette il governor com'era.
