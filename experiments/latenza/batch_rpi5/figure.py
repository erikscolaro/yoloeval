"""Figure del batch, nello stile della figura A della griglia (tools/figure_sintesi.figura_a): righe = batch 1, 2, 4, 8
a 256x256, colonne = profondita', caselle = larghezze. Uscite in report/:
    A_batch_dettaglio: latenza dell'inferenza intera (N crop); budget = modello in un fotogramma TRACK a 30 fps
                       (33,3 ms meno crop e Kalman)
    A_crop_dettaglio:  latenza per crop (latenza / N); budget = lo stesso diviso 4 (4 crop per fotogramma)
    latenza_batch.csv: i numeri
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import tools.figure_sintesi as fs  # noqa: E402

B = {"crop256_b1": 1, "crop256_b2": 2, "crop256_b4": 4, "crop256_b8": 8}
TRACK = 1000 / 30 - 2.0 - 0.2   # ms al modello in un fotogramma TRACK: 33,3 meno crop (~2 ms) e Kalman (0,2 ms)


def main():
    lat = {k: v for k, v in fs.lat_misurate(str(HERE.parents[2] / "results"), "rpi5").items()
           if k[1] in B and k[3] == "fp16"}
    fs.RL, fs.PRECS = list(B), ["fp16"]
    fs.RLAB = {r: f"batch {n}" for r, n in B.items()}
    out = HERE / "report"
    fs.BUDGET = TRACK
    fs.figura_a(out, "Raspberry Pi 5, 256², fp16", dettaglio=True, misure=lat, titolo=False)
    (out / "A_budget_dettaglio.png").rename(out / "A_batch_dettaglio.png")
    (out / "A_budget_dettaglio.pdf").rename(out / "A_batch_dettaglio.pdf")
    fs.BUDGET = TRACK / 4
    per_crop = {k: v / B[k[1]] for k, v in lat.items()}
    fs.figura_a(out, "Raspberry Pi 5, 256², fp16", dettaglio=True, misure=per_crop, titolo=False)
    (out / "A_budget_dettaglio.png").rename(out / "A_crop_dettaglio.png")
    (out / "A_budget_dettaglio.pdf").rename(out / "A_crop_dettaglio.pdf")
    with open(out / "latenza_batch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["width", "depth", "batch", "median_ms", "ms_per_crop"])
        for (wd, r, d, _), v in sorted(lat.items()):
            w.writerow([wd, d, B[r], round(v, 2), round(v / B[r], 2)])


if __name__ == "__main__":
    main()
