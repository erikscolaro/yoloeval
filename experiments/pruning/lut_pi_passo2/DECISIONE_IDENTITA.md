# Decisione: niente reti identità nella LUT (06/11/2026)

Questo file spiega perché la LUT di latenza a passo 2 sul Pi 5 non misura più le reti identità e non le
sottrae ai valori degli strati. È scritto per la tesi: contesto, prove, decisione, ipotesi ancora aperte.

## Il problema: il costo fisso per chiamata

benchmark_app misura ogni inferenza dalla chiamata `infer()` al ritorno. Nella LUT ogni misura è uno
strato da solo, quindi il tempo misurato comprende due parti:

- il costo dello strato, cioè conv più SiLU;
- un costo fisso di OpenVINO per chiamata: avvio della richiesta, risveglio e sincronizzazione dei 4
  thread, ritorno del risultato.

Nel nano vero il costo fisso si paga una volta per immagine. Sommando i 102 strati della LUT, invece, lo
si conterebbe 102 volte. In qualche modo va tolto.

## Il piano iniziale: sottrarre l'identità forma per forma

Per ogni forma d'ingresso (c_in, mappa) si misurava una rete identità (ingresso → uscita, nessun calcolo):
622 forme. Nella composizione, a ogni valore della LUT si sottraeva l'identità con lo stesso ingresso:
`LUT_f(c_in, c_out) − id(c_in, mappa d'ingresso)`.

## Perché è stato abbandonato

1. **Non serve.** PIT pota solo canali: i 102 strati del nano ci sono sempre. Il costo fisso entra quindi
   nella somma della LUT sempre 102 volte, qualunque sia la potatura. È una costante, e la costante C
   della formula di composizione (stimata sui ~150 nani potati misurati interi, B4-R) la assorbe
   esattamente. Le identità a 2 canali danno circa 0,013 ms per chiamata, indipendenti dalla mappa:
   0,0128, 0,0131 e 0,0134 ms a 20×20, 40×40 e 80×80 (`warmup.py`, `results/warmup.json`).

2. **Su forme grandi sarebbe sbagliato.** L'identità non misura solo il costo fisso: il nodo `Identity`
   copia tutto l'ingresso. A 160×160×64 costa 0,76 ms, a 80×80×256 0,76 ms, contro 0,013 ms del costo
   fisso puro. La conv quella copia non la fa, perché legge l'ingresso e basta. Sottrarre l'identità
   toglierebbe allo strato un costo che non paga. Questo errore dipende da c_in, quindi cambia con la
   potatura e C non lo compenserebbe.

3. **Con il protocollo di misura le identità piccole venivano sbagliate.** Il warmup di benchmark_app dura
   circa 20 ms a prescindere dalla rete (sezione Warmup del README). Con la mediana delle prime 200
   iterazioni, le identità a 2 canali (circa 0,013 ms) risultavano gonfiate del 47–76%, `id20` 512×512 del
   27%. Sottrarre valori gonfiati avrebbe sottostimato le conv piccole: per esempio una conv da 0,05 ms,
   meno un'identità misurata 0,022 invece di 0,013, dà 0,028 netti invece di 0,037 (−24%).

Prima ancora, nella prova di separabilità, sottrarre l'identità peggiorava il fit: nella famiglia A i punti
entro soglia scendevano dal 71 al 47% (`NEW_MASTERPLAN.md`, 4a; [`separability`](../separability)).

## Decisione

- La LUT misura solo le conv: 69 989 punti × 3 ripetizioni = 209 967 misure (prima 211 833, con le 622
  identità).
- Nessuna sottrazione strato per strato. Il costo fisso per chiamata, contato 102 volte, finisce nella
  costante C del modello `latenza ≈ C + Σ_f a_f · Σ_strati LUT_f(c_in, c_out)`.
- Le identità restano soltanto come strumento diagnostico, in `warmup.py` e `check_fusion.py`.

Codice: `run.py` (`points()` non genera più identità), `analyze.py` (niente `lut_identity.csv`); ora in `src/lut/`.
`warmup.py` costruisce da sé le 11 identità che usa.

## Ipotesi da verificare

- **Il costo per chiamata di una rete conv non dipende dai canali.** Per assorbirlo in C basta questo,
  che sia o no uguale a quello dell'identità: le conv usano I/O NHWC, le identità NCHW. Non è stato
  misurato direttamente. Si vedrà dal fit di B4-R: se C dipendesse dai canali, il residuo crescerebbe con
  la potatura.
- **Il costo fisso si somma, non si sovrappone.** Nella rete vera fra uno strato e l'altro non c'è
  nessuna chiamata `infer()`. Si assume che il costo per chiamata della LUT sia un termine additivo che
  sparisce nella rete intera, e non che interagisca con il costo dello strato.

## Per la tesi

Il punto da far passare: dato che la potatura è solo per canale, l'insieme degli strati non cambia mai.
Così qualunque costo uguale per tutti gli strati diventa una costante della composizione, e non serve
misurarlo né sottrarlo. Sottrarlo per forma, anzi, introduceva un errore che dipendeva da c_in: la copia
dell'ingresso fatta dall'identità.
