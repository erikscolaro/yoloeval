# Reti PIT dentro il fit dello stimatore di latenza (Pi 5)

Lo stimatore `latenza = C_arch + Σ_f a_f · LUT_f` era fittato su reti B4-R potate a caso (765 reti, 15 architetture). Sulle
reti prodotte da PIT (canali molto bassi) sottostimava di 0,5–1 ms (livello 0: +0,75 ms, −2,8 %). Qui si generano reti PIT,
si misurano sul Pi e si rifà il fit con B4-R + PIT. **Esito: migliora, il nuovo fit è il default** (`artifacts/lut/rpi5/fit.json`;
il vecchio è `fit_b4r.json` accanto; il candidato `fit_pit.json`, uguale al nuovo default).

    ./rifai.sh              # tutto: gen (CPU, 3 processi), measure (Pi), l0, fit. Idempotente, riprende da dove era
    ./rifai.sh fit          # solo il fit e il confronto (10 s)
    systemd-run --user --unit=fit-pit --working-directory=$PWD ./rifai.sh     # staccato; log in runs/

Codice: `run.py` (`gen`, `measure`, `l0`, `fit`), `fit_eval.py` (valutazione). Riusa `pendenze/livello0/run.py` (`export`),
`src.lut.pi.b4r` (misura) e `src.lut.fit` (`solve`, `cv_lambda`, `predict_held_out`, `errors`); nessun'altra cartella toccata.

## Reti

- **140 reti PIT distinte** (da 42 traiettorie), solo maschere, DUCCIO sulla latenza, CPU, 1500 passi, `discrete_cost`, n = 2,
  con i **pesi e_300 veri** (non lo yaml: vedi sotto). Architetture: nano `w0.25_D126` (108 reti) e vicine `w0.1875_D126`,
  `w0.25_D101`, `w0.25_D170` (10, 10, 12 reti). Stimate da 21 a 57 ms.
- Variano: stimatore (lut/smooth/secant/best), ottimizzatore (AdamW lr 0,01; SGD lr 10 senza momento), target (0,62 e 0,40
  della rete intera, il secondo portato ad almeno 1,05 × minimo raggiungibile: il nano non scende sotto 22,08 ms), seed.
- **Traiettoria**: le maschere si salvano al primo passo che attraversa 46, 40, 35, 31, 27, 24 ms (scalati sull'architettura) e
  alla fine; le reti uguali (stessi canali per layer) si tengono una volta. Con le maschere tutte uguali AdamW salta dalla rete
  intera al minimo in pochi passi (come nel livello 0): `seed > 0` somma rumore (σ = 0,2) alle maschere iniziali e le fa
  attraversare la soglia una dopo l'altra, così lungo la traiettoria ci sono reti diverse. Seed 0 senza rumore, 1 e 2 con.
- **Test (mai nel fit)**: 32 reti del nano con target 0,47 e seed 3 (target e seed assenti dal train), più le 15 reti del
  livello 0 (3 ripetizioni, rimisurate in questa sessione, `l0_remisura/`: dentro ±0,3 ms dalla misura del 13/11 16:00).
  Train: 108 reti PIT.
- Misura sul Pi: `src.lut.pi.b4r` (OpenVINO fp16, 640, cpu_4, frequenza massima, 3 × 50 iterazioni, guardia termica), una
  rete alla volta. 140 reti × 3 = 420 misure, nessun riavvio né blocco del Pi (uptime invariato dalle 15:22 del 13/11).
  Pi occupato in tutto circa 45 minuti (due giri, 17:00–19:45 con pause); libero dalle 19:45.

### Perché i pesi veri e non lo yaml

Il primo tentativo (`results_yaml/`, `runs/nets_yaml/`) costruiva le reti dallo yaml dell'architettura, come B4-R. A parità di
canali per layer l'IR è diverso: con i pesi veri restano le BN come nodi `Add` (nets.csv, `nonconv_elems`: 7,7 M contro 2,3 M)
e la rete misura circa 0,3 ms in più (22,3 contro 22,6–22,7 ms per la stessa rete del livello 0). Con lo yaml il vecchio fit
sottostimava di +0,37 ms (−0,97 %), con i pesi veri di +0,76 ms (−1,97 %), cioè quanto il livello 0: le reti che contano per
la ricerca sono quelle con i pesi veri, quindi il fit usa solo quelle. I numeri del tentativo yaml (stesso verso, stesso
ordine di miglioramento) restano in `results_yaml/valutazione.json`.

## Fit candidato

`fit_eval.py`: ridge verso 1 con a_f ≥ 0 e una C per architettura ai minimi quadrati, come `src.lut.fit` (λ per
cross-validation su architetture intere), sui 750 + 50 B4-R potati + 15 intere + le PIT di train. Varianti: peso w delle PIT
(ripetute w volte): `nuovo` (w = 1), `x3`, `x10`; `C su PIT`: a vecchi, C del nano = media dei residui sulle PIT di train.
Scelto **x10**: 76 reti PIT del nano × 10 pesano quanto B4-R (750). λ = 10. C(w0.25_D126) = 10,99 ms (era 10,55).

## Risultati: vecchio contro nuovo (x10)

Bias = stima/misura − 1 (negativo = sottostima) e media misurata − stimata in ms; p99 dell'errore relativo assoluto.

| set | n | vecchio: bias % / ms / p99 % | nuovo x10: bias % / ms / p99 % |
|---|---|---|---|
| **PIT di test** | 32 | −1,97 / +0,760 / 3,09 | **−0,33 / +0,140 / 1,51** |
| **livello 0** (esterno) | 15 | −2,84 / +0,750 / 4,01 | **−0,49 / +0,136 / 1,64** |
| PIT di train (in-sample) | 108 | −1,83 / +0,654 / 4,03 | −0,19 / +0,078 / 2,44 |
| B4-R leave-one-arch-out, totale | 750 | +0,18 / −0,040 / 4,11 | +0,00 / +0,006 / **3,50** |
| B4-R LOAO, fascia 20–35 ms | 336 | +0,11 / −0,027 / 4,11 | −0,09 / +0,028 / 3,51 |
| B4-R LOAO, nano | 50 | −1,67 / +0,566 / 3,92 | −1,40 / +0,463 / 4,01 |

Le altre varianti (stesse righe, `results/valutazione.json`): `nuovo` (w = 1) PIT test −0,69 % / p99 2,27 %, livello 0 −1,20 % /
2,84 %, B4-R p99 4,14 %; `x3` −0,52 % / 1,95 %, −0,70 % / 2,00 %, 3,86 %; `C su PIT` (solo C) −0,11 % / 1,42 %, −0,16 % / 1,53 %
(B4-R invariato). Fit in-sample sulle 873 reti (B4-R + PIT di train): bias +0,25 %, p99 2,98 % (vecchio, 765 reti: 2,53 %).

- **Decisione: default sì.** Bias e p99 sulle PIT di test e sul livello 0 scendono di 3–6 volte; il p99 su B4-R non peggiora
  (scende da 4,11 a 3,50 %, mai oltre +0,1 punti in nessuna fascia, nano compreso).
- **Nano LOAO**: nel leave-one-out le PIT del nano escono insieme alla sua B4-R, quindi lì il nuovo fit non le vede: la riga
  non cambia (e non deve). Il bias del nano si toglie solo con le PIT del nano nel fit, come nelle prime due righe.
- Resta una sottostima di 0,14 ms (0,3–0,5 %) sulle reti PIT, dentro il rumore di misura (1 %).
- Con il calcolo di C su più reti (report fit-c-multi) si arriva a risultati simili sulle PIT (`C su PIT`), ma solo per
  l'architettura che ha reti PIT; il fit congiunto sistema anche gli a_f e il p99 di B4-R.

## Limiti

- Le PIT sono solo del nano e di tre architetture vicine; le altre 11 architetture hanno ancora la C dalla sola rete intera.
- Reti ottenute senza loss di detection (solo costo): più estreme (layer a 2 canali) di quelle della ricerca vera; la ricerca
  con la loss (livello 1) potrà dare reti meno estreme, comunque dentro lo stesso insieme di forme.
- Il peso x10 è una scelta (pari ai B4-R); non c'è un altro set indipendente su cui sceglierlo oltre il livello 0.
- `artifacts/` è in `.gitignore`: `fit.json` e `fit_pit.json` non sono nel commit (copia in `yolopit/data/rpi5/fit.json`).
