# Prevedere i tempi del modello sul Raspberry Pi

*Aggiornamento del 9 novembre 2026*

## In sintesi

- **Obiettivo raggiunto.** Ora sappiamo prevedere quanto tempo impiega un modello sul Raspberry Pi senza doverlo
  provare ogni volta. Su 99 modelli su 100 la previsione sbaglia di meno del 5%, che era la soglia che ci eravamo dati.
- **Molto meglio del metodo usato finora.** Oggi il tempo si stima contando le operazioni del modello. Questo metodo
  sbaglia fino al 30% e sbaglia sempre nella stessa direzione per lo stesso tipo di modello. Il nuovo metodo è
  affidabile in tutti i casi.
- **Già pronto all'uso.** Il metodo è integrato nello strumento che alleggerisce i modelli. Ora possiamo chiedere
  direttamente un modello che stia dentro un certo tempo di risposta, invece di ragionare per approssimazioni.
- **Prossimo passo.** Usarlo per alleggerire i modelli veri e verificare sul dispositivo che rispettino i tempi
  richiesti.

## Il problema

Il modello che riconosce i droni deve girare su un Raspberry Pi 5, un piccolo computer a basso costo, e deve
rispondere abbastanza in fretta da seguire la scena. Per farcelo stare togliamo al modello le parti meno utili, una
tecnica chiamata *potatura*.

Finora la potatura sapeva solo ridurre la quantità di calcoli. Più calcoli non significano però sempre più tempo: due
modelli con gli stessi calcoli possono avere tempi molto diversi, a seconda di come sono fatti. Di conseguenza la
potatura poteva tagliare dove il tempo guadagnato era poco, e per sapere il tempo vero bisognava misurare ogni modello
sul dispositivo.

## Cosa abbiamo fatto

1. **Abbiamo misurato i mattoni.** Un modello è fatto di tanti blocchi ripetuti di pochi tipi. Abbiamo misurato sul
   Raspberry Pi quanto tempo impiega ogni tipo di blocco in tutte le dimensioni che la potatura può produrre, e
   abbiamo raccolto i tempi in una tabella.
2. **Abbiamo costruito il previsore.** Il tempo di un modello si ottiene sommando i tempi dei suoi blocchi presi dalla
   tabella, con una piccola correzione che tiene conto di come i blocchi si comportano quando lavorano insieme.
3. **Lo abbiamo messo alla prova.** Abbiamo generato molti modelli alleggeriti in modi diversi, li abbiamo misurati
   davvero sul dispositivo e abbiamo confrontato i tempi reali con quelli previsti.

## Perché ci si può fidare

- **È stato provato su modelli mai visti.** Ogni modello di prova è stato previsto senza che il metodo avesse visto
  modelli della stessa famiglia: è la situazione reale, quando arriva un modello nuovo.
- **Sbaglia quanto la misura stessa.** Lo stesso modello misurato due volte non dà mai esattamente lo stesso tempo. In
  genere l'errore della previsione è di quell'ordine: con questi strumenti non si potrebbe fare molto meglio.
- **Non ha un verso preferito.** A volte stima un po' di più, a volte un po' di meno, in parti uguali. Il metodo
  basato sui calcoli invece sottostima sempre i modelli più grandi, e porterebbe a scegliere modelli troppo lenti.
- **I tempi sono stabili nel tempo.** La tabella è stata misurata nei giorni precedenti alla prova e la previsione
  regge: il dispositivo, nelle condizioni che abbiamo fissato, non cambia comportamento da un giorno all'altro.

## Dove vale

Il previsore è costruito per il caso che ci interessa: il Raspberry Pi 5 con il software che useremo sul campo e i
modelli fino alla taglia che vogliamo alleggerire. È una scelta voluta: per un altro dispositivo basta ripetere le
misure con la stessa procedura.

## Prossimi passi

- **Alleggerire i modelli con il nuovo obiettivo di tempo** e misurare sul dispositivo quelli ottenuti, per
  confermare che rispettano il tempo richiesto.
- **Scegliere come lo strumento di potatura usa il previsore.** I tempi crescono a scatti e non in modo continuo:
  abbiamo preparato più varianti per gestirlo e le confronteremo sui primi modelli.
- **Confrontare i modelli alleggeriti con quelli piccoli già esistenti**, a parità di tempo di risposta, per capire
  quale strada dà il riconoscimento migliore.

## Aggiornamento del 13 novembre

Sulle reti prodotte dalla potatura vera (molto snelle) il previsore sottostimava di circa 0,7 ms, il 2–3%. L'abbiamo rifatto
misurando sul Raspberry Pi anche un centinaio di queste reti: ora la sottostima è sotto lo 0,5% e, sui modelli di prova della
procedura precedente, l'errore che 99 modelli su 100 non superano scende dal 4,1% al 3,5%.
