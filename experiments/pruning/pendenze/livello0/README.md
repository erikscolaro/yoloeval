# Livello 0: solo costo di latenza

Primo filtro del piano (`../PIANO.md`): si ottimizzano **solo le maschere di PIT** con il costo di latenza, senza loss di
detection e senza dati, dal nano D126 (`w0.25_D126`, 56,5 ms stimati) al target di 28 ms, una volta per modalità
`estimator` in {`lut`, `smooth`, `secant`, `best`}. Poi le reti si esportano e si misurano sul Pi.

    ./rifai.sh            # tutto: search (~25 min sul PC), poi misura sul Pi e confronto
    ./rifai.sh offline    # solo la ricerca
    ./rifai.sh pi         # aspetta che coda-13nov finisca, poi misura e confronta (riprende da results/raw.csv)
    systemd-run --user --unit=pendenze-l0 --working-directory=$PWD ./rifai.sh pi    # staccato; log in runs/pi.log

Codice: `run.py` (`search`, `measure`, `report`). La misura è `src.lut.pi.b4r`: stesso IR del benchmark (OpenVINO fp16, 640,
testa one-to-one), cpu_4, frequenza massima, 3 ripetizioni da 50 iterazioni, guardia termica.

## Procedura

- `build_pit_model` di yolopit con `pit.latency` (LUT e fit di `artifacts/lut/rpi5`), n = 2, `remainder`, `discrete_cost`;
  DUCCIO con target 28 ms sul modello intero; 3000 passi, un passo = un `backward` del solo termine di DUCCIO (nessun batch).
- I 3000 passi sono divisi in 30 «epoche» finte da 100: la forza di DUCCIO e la D di `secant` seguono il calendario della
  ricerca vera (`search.epochs: 30`). `task_loss` di DUCCIO = 1.
- Due ottimizzatori delle maschere, perché il solo costo non è la ricerca vera:
  - `adamw`: lr 0,01, wd 0 (come `pit_latency`);
  - `sgd`: lr 10, senza momento (variabile `SGD_LR`).
- Per ogni rete: la finale (`final`) e quella al **primo passo con latenza stimata ≤ 28 ms** (`first`; se la rete non ci
  arriva non c'è). Sono i nomi `<modo>_<ottimizzatore>_<final|first>`.

## Definizioni

- **Arriva a 28 ms**: latenza stimata finale (costo discreto, quello dell'export) ≤ 28 ms.
- **Oscillazione**: nell'ultimo 20 % dei passi, `inversions` = quante volte la latenza passa dal lato sopra al lato sotto il
  target (o viceversa); `changes` = quanti passi cambiano la latenza; `range_ms` = max − min.
- **Appena sopra un gradino**: un layer è appena sopra un gradino se togliendo 2 canali di uscita la latenza LUT di quel layer
  (alla sua c_in) cala di più del 5 % (`step_out`; `step_in` lo stesso per i canali di ingresso). È la soglia `STEP_MIN` con
  cui `yolopit.latency` riconosce un gradino (sul Pi: 32 → 34 canali costa +39 %, 62 → 64 sotto il 2 %). Conta
  `above_step_out`; `above_step_any` conta anche `step_in` (in tutte le reti è solo `2.cv2.conv`, con c_in = 6).
- **Layer al minimo**: layer con 2 canali di uscita (un solo blocco).

## Risultati (`results/summary.csv`, `layers.csv`, `trace_*.csv`)

| rete | ≤ 28 ms | stimata ms | 1° passo ≤ 28 | passi che cambiano (ultimo 20 %) | canali di uscita (74 layer; piena 4952) | layer al minimo |
|---|---|---|---|---|---|---|
| `lut_adamw` | sì | 27,81 | 185 | 0 | 2472 | 52 |
| `smooth_adamw` | sì | 22,08 | 50 | 0 | 1350 | 67 |
| `secant_adamw` | sì | 22,22 | 50 | 0 | 1412 | 66 |
| `best_adamw` | sì | 22,08 | 50 | 0 | 1350 | 67 |
| `lut_sgd` | **no** | 28,51 | – | 484 (range 1,09 ms) | 2978 | 49 |
| `smooth_sgd` | sì | 27,94 | 546 | 0 | 3438 | 39 |
| `secant_sgd` | sì | 27,46 | 708 | 0 | 3442 | 41 |
| `best_sgd` | sì | 27,67 | 477 | 0 | 3496 | 36 |

`first` e `final` coincidono ovunque (dopo il target la rete non si muove più): sono la stessa rete e nella tabella ce n'è una sola. Nessun layer è appena sopra un gradino (`above_step_out` = 0 in tutte): nessun
layer ha c_out = 34, 66, 98…; molti stanno al minimo, perché senza loss di detection niente li trattiene.

### Perché con AdamW si scende fino a 22 ms

Non è un errore di yolopit né del confronto soft/discreto (DUCCIO e la latenza finale usano lo stesso costo discreto: la
traccia coincide con `real_costs()`, c'è un `assert`). Il solo costo, con AdamW, fa muovere **tutte** le maschere con
gradiente non nullo della stessa quantità per passo (Adam normalizza il gradiente di ogni maschera), quindi tutte attraversano
la soglia insieme: la latenza va da 56,5 a 22 ms in pochi passi (passo 50 = primo ≤ 28), saltando il target. Poi il
gradiente di DUCCIO è zero, ma la quantità di moto di Adam spinge ancora, e senza loss di detection nulla fa risalire le
maschere. In più Adam cancella la **pendenza**, che è proprio quello che distingue le modalità: `smooth` e `best` danno la
stessa rete, `secant` quasi. Con AdamW il livello 0 non separa le modalità; nella ricerca vera il confronto con la loss di
detection ripristina la scala, qui no. L'unica modalità diversa è `lut`, che usa la pendenza della LUT (non monotona).

`sgd` tiene il modulo del gradiente: le maschere con la pendenza più alta scendono prima, la rete si ferma sul target
(27,5-27,9 ms, sotto di 0,1-0,5 ms) senza superarlo e senza oscillare. È il confronto utile fra modalità.

### Modalità

- `lut`: con SGD non arriva al target e oscilla fino alla fine (484 passi con latenza diversa nell'ultimo 20 %, range 1,1 ms,
  sempre sopra 28 ms: nessuna inversione del segno perché non lo attraversa mai). Con AdamW arriva a 27,81 ms.
- `smooth`, `secant`, `best` con SGD arrivano a 27,5-27,9 ms; `best` è il più rapido (477 passi) e `secant` il più lento
  (708). I canali totali sono quasi uguali (3438-3496): senza detection non c'è differenza di accuratezza da misurare.
- Per il livello 1 non si scarta nessuna modalità da qui, tranne il sospetto su `lut` (oscillazione con SGD).

## Misura sul Pi (`results/raw.csv`, `confronto.csv`)

Le reti di `results/nets.csv` (le 8 della tabella, le loro `first` e `nano_full`) misurate dal Pi: latenza misurata contro stimata, errore
in ms e in %. Stato in `runs/pi.log` e `systemctl --user status pendenze-l0`; `results/confronto.csv` si scrive alla fine.
Misura del 13/11/2026, 16:00 (3 ripetizioni per rete, condizioni della LUT; `first` coincide con `final` tranne dove
indicato). Errore = misurata − stimata.

| rete | stimata (ms) | misurata (ms) | errore |
|---|---|---|---|
| nano intero | 56,55 | 56,19 | −0,36 ms (−0,6 %) |
| lut, AdamW | 27,81 | 28,85 | +1,04 ms (+3,7 %) |
| smooth, AdamW | 22,08 | 22,71 | +0,63 ms (+2,9 %) |
| secant, AdamW | 22,22 | 23,00 | +0,78 ms (+3,5 %) |
| best, AdamW | 22,08 | 22,58 | +0,50 ms (+2,3 %) |
| lut, SGD | 28,51 | 29,36 | +0,86 ms (+3,0 %) |
| smooth, SGD | 27,94 | 28,38 | +0,45 ms (+1,6 %) |
| secant, SGD | 27,46 | 28,44 | +0,98 ms (+3,6 %) |
| best, SGD | 27,67 | 28,27 | +0,60 ms (+2,2 %) |

Le reti `first` misurano da 0 a 0,2 ms in più delle `final` (stessa architettura: è il rumore della misura).

### Cosa dice

- **Lo stimatore regge**: tutte le reti potate stanno entro il 5 % (l'obiettivo dello stimatore), da +1,6 a +3,8 %.
- **Ma sulle reti potate sbaglia sempre nello stesso verso**: le sottostima tutte, di 0,5–1,0 ms, mentre la rete intera
  è stimata giusta (−0,6 %). Sulle reti di prova del fit (B4-R) l'errore non aveva un verso; queste reti sono più
  sbilanciate (molti layer a 2 canali, perché senza loss di detection niente li trattiene), cioè fuori dalla zona del fit.
- **Conseguenza pratica**: con target 28 ms nessuna rete potata sta sotto 28 ms misurati (28,3–29,4 ms con SGD). Una
  ricerca che arriva al target sfora il budget di circa il 2–4 %. Da decidere: un margine sul target (per esempio −4 %),
  oppure la verifica con le reti del livello 1, che con la loss di detection saranno meno estreme.
- **Fra le modalità** l'errore non cambia in modo utile (smooth e best un po' meglio, lut e secant un po' peggio, ma
  dentro 0,5 ms): la scelta della pendenza la fanno il livello 1 e 2, non la precisione della stima.

### Da dove viene la sottostima (verifica del 13/11, senza misure nuove)

- **Ipotesi scartata: un costo dei layer a pochi canali.** Sulle 1515 reti B4-R l'errore non cresce con il numero di
  layer sotto 8 canali (correlazione +0,12; reti con 30–80 layer così: errore medio +0,01 ms). Lo stesso con ≤ 4 canali
  e con canali non multipli di 8.
- **Quello che si vede: sul nano lo stimatore sottostima di più quanto più la rete è leggera.** Reti B4-R del nano
  (w0.25 D126, 101 reti): misurate 40–60 ms errore medio +0,20 ms; 30–40 ms +0,31; 25–30 ms +0,36; sotto 25 ms +0,42.
  Le reti del livello 0, più estreme, stanno a +0,5/+1,0 ms. `C_arch` è fissa e il fit pesa tutto l'intervallo, quindi
  in basso (dove sta il budget) resta un errore con segno.
- **In pratica**: vicino al budget la stima è bassa di circa 0,4 ms (1,5 %) sulle reti normali, fino a 1 ms (3,5 %) su
  quelle estreme. Rimedi: un margine sul target (circa −3 %), oppure rifare il fit pesando di più la zona del budget.
