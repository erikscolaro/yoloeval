"""Stadi: train, export (con quantizzazione) e benchmark.

Cardinalita' decrescente in costo unitario: il training gira una volta per
modello, l'export una volta per artefatto, il benchmark su tutta la matrice.
Ogni stadio legge dalla cache del precedente e non rigenera mai cio' che
esiste gia', salvo override esplicito.
"""
