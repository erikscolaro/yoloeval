# Tracker di droni su Raspberry Pi 5: configurazione e latenza

## Obiettivo

Il sistema trova e segue droni nei video, in tempo reale, su un Raspberry Pi 5. Distingue 4 classi: drone, aereo, elicottero e uccello. Il requisito è 30 fotogrammi al secondo, cioè circa 33 ms per fotogramma.

## Modelli

Il sistema usa due detector YOLO della stessa famiglia. Il nome ha due parti:

- La lettera indica la larghezza, cioè il numero di canali della rete. P vale 0.125 e S vale 0.5.
- Il segno indica la profondità. "-" è la versione meno profonda, con 101 strati. La versione standard ha 126 strati, e "+" ne ha 170. A pari larghezza, la profondità cambia poco l'accuratezza e costa molta latenza, quindi il sistema usa "-".

I due modelli:

- **P-**: il modello piccolo. A 640×640 fa un'inferenza in circa 26 ms sul Pi ed è il più accurato tra i modelli che entrano nel budget.
- **S-**: un modello 4 volte più largo e più accurato. A 640×640 è troppo lento per il Pi, ma su un ritaglio 256×256 è veloce.

## Funzionamento del tracker

Il tracker ha due stati. In **SCAN**, P- cerca gli oggetti sul fotogramma intero. Quando un oggetto è confermato, il tracker passa in **TRACK**. Un filtro di Kalman predice dove sarà l'oggetto nel fotogramma successivo. S- guarda solo un ritaglio 256×256 intorno a quella posizione, quindi vede l'oggetto più grande e più nitido. Se S- non trova l'oggetto per alcuni fotogrammi, il tracker torna in SCAN. Il ritorno in SCAN ha anche una forma opzionale, il **rescan**: ogni N fotogrammi il tracker fa uno SCAN anche quando sta seguendo un oggetto. Il rescan trova i nuovi oggetti che entrano nella scena, perché in TRACK S- guarda solo il ritaglio. Ma ogni SCAN costa più di un passo TRACK. Il tracker dà a ogni traccia una classe con memoria, così un drone non diventa un uccello per un solo fotogramma sbagliato.

## Configurazione

Un'ottimizzazione automatica (Optuna, 1000 prove su video di validazione) ha scelto i parametri del tracker: soglie di confidenza, dimensione del ritaglio, durata delle tracce. Ci sono due preset, uno per obiettivo:

- **F2**: dà più peso al recall. Il sistema perde meno droni, ma dà qualche falso allarme in più.
- **F1**: dà lo stesso peso a precisione e recall. Il sistema dà meno falsi allarmi.

Per i due preset, un vincolo comune: il tracker non deve chiamare "drone" un aereo o un elicottero più spesso del detector da solo.

Sul Pi:
- **Inferenza**: OpenVINO in mezza precisione (fp16), sui 4 core della CPU, con la CPU alla frequenza massima.
- **Video**: ridotti prima a 640 pixel di lato lungo.
- **Rescan**: spento. Il tracker resta in TRACK fino a quando perde l'oggetto.

## Latenza sul Pi

Tempo per fotogramma, con la decodifica del video (0.7 ms), senza disegno:

| Stato | ms per fotogramma | FPS |
|---|---|---|
| SCAN (P- sul fotogramma intero + Kalman) | 33.2 | 30.1 |
| TRACK (S- sul ritaglio + Kalman) | 27.5 | 36.4 |
| Aggregato, preset F2 (68% dei fotogrammi in TRACK) | 29.3 | 34.1 |
| Aggregato, preset F1 (56% dei fotogrammi in TRACK) | 30.0 | 33.3 |
| Aggregato misurato sul Pi, rescan spento | 28–29 | 35–36 |

Le righe "Aggregato" per F2 e F1 sono una media pesata con la quota di TRACK sui video di validazione, con il rescan di Optuna. L'ultima riga è la misura diretta sulla demo: senza rescan, il tracker sta in TRACK più a lungo.

Il disegno sul Pi aggiunge tempo:

| Visualizzazione | FPS aggregati |
|---|---|
| Nessuna sul Pi: il Pi invia solo le box a un PC, che disegna | 35–36 |
| Finestra sul Pi, alla dimensione del video | 31.7 |
| Finestra sul Pi a schermo intero (+11 ms) | 24 |

Tutti e due gli stati rispettano i 30 FPS. Lo SCAN è al limite, il TRACK ha un margine di circa 6 ms. Il disegno a schermo intero porta il sistema sotto i 30 FPS. Con la finestra alla dimensione del video, oppure con il disegno su un altro computer, il sistema rispetta il requisito.

## Efficacia: tracker contro P- da solo

Il tracker usa lo stesso budget di P- da solo (meno di 33 ms per fotogramma), ma dà risultati migliori. Misure sui droni, per ogni fotogramma, sui video di validazione. Le due fonti sono separate: Svanström (scene con aerei, elicotteri e uccelli) e Anti-UAV (un drone per video). P- da solo gira sul fotogramma intero a 640, alla soglia di confidenza che massimizza lo stesso obiettivo del preset.

**Obiettivo F2** (P- alla soglia 0.02)

| Sistema | Svanström P | Svanström R | Svanström F1 | Anti-UAV P | Anti-UAV R | Anti-UAV F1 |
|---|---|---|---|---|---|---|
| P- da solo | 0.802 | 0.384 | 0.519 | 0.744 | 0.783 | 0.763 |
| Tracker F2 | 0.817 | 0.415 | 0.551 | 0.817 | 0.842 | 0.829 |
| **Differenza** | **+1.5** | **+3.1** | **+3.2** | **+7.3** | **+5.9** | **+6.6** |

**Obiettivo F1** (P- alla soglia 0.10)

| Sistema | Svanström P | Svanström R | Svanström F1 | Anti-UAV P | Anti-UAV R | Anti-UAV F1 |
|---|---|---|---|---|---|---|
| P- da solo | 0.847 | 0.379 | 0.523 | 0.943 | 0.729 | 0.822 |
| Tracker F1 | 0.817 | 0.431 | 0.565 | 0.973 | 0.776 | 0.863 |
| **Differenza** | −3.0 | **+5.2** | **+4.2** | **+3.0** | **+4.7** | **+4.1** |

Differenze in punti percentuali.

Aerei chiamati drone (Svanström): P- da solo 3.2%, tracker F2 0.13%, tracker F1 0.03%. Il tracker riduce i falsi allarmi su aerei di 25–100 volte.

In sintesi, con lo stesso hardware e lo stesso budget:
- Il tracker aumenta F1 di 3–7 punti su tutte e due le fonti e con tutti e due gli obiettivi.
- Il recall aumenta sempre, di 3–6 punti: il TRACK su ritaglio trova droni che P- sul fotogramma intero perde.
- La precisione aumenta in 3 casi su 4. L'unico calo (−3 punti, Svanström con obiettivo F1) è compensato dal recall.

Queste misure usano il rescan scelto da Optuna (ogni 60 fotogrammi per F2, ogni 11 per F1), non il rescan spento del Pi.

## Potenziale massimo: confronto con S- da solo

S- da solo sul fotogramma intero a 640 indica il risultato di un detector più grande. Sul Pi però costa circa 137 ms per fotogramma, cioè 4 volte il budget. La tabella mostra quanto il tracker si avvicina a S-, con il costo di P-. Ogni detector è alla sua soglia ottima per l'obiettivo.

| Obiettivo | Sistema | ms sul Pi | Svanström P | Svanström R | Svanström F1 | Anti-UAV P | Anti-UAV R | Anti-UAV F1 |
|---|---|---|---|---|---|---|---|---|
| F2 | P- da solo (0.02) | 26 | 0.802 | 0.384 | 0.519 | 0.744 | 0.783 | 0.763 |
| F2 | Tracker F2 | 27–33 | 0.817 | 0.415 | 0.551 | 0.817 | **0.842** | **0.829** |
| F2 | S- da solo (0.01) | 137 | **0.840** | **0.489** | **0.618** | **0.867** | 0.725 | 0.790 |
| F1 | P- da solo (0.10) | 26 | 0.847 | 0.379 | 0.523 | 0.943 | 0.729 | 0.822 |
| F1 | Tracker F1 | 27–33 | 0.817 | 0.431 | 0.565 | **0.973** | **0.776** | **0.863** |
| F1 | S- da solo (0.05) | 137 | **0.884** | **0.481** | **0.623** | 0.970 | 0.703 | 0.815 |

In grassetto il valore migliore per obiettivo e colonna.

- **Anti-UAV** (un drone per video): il tracker supera anche S-, con F1 +3.9 punti (F2) e +4.8 punti (F1). Qui il TRACK su ritaglio vale più di un detector 4 volte più largo.
- **Svanström** (oggetti piccoli, più oggetti per scena): il tracker recupera il 32% (F2) e il 42% (F1) del divario in F1 tra P- e S-. Il margine che resta, circa 6 punti di F1, è il potenziale massimo di miglioramento su queste scene.
