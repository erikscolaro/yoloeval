"""Un solo export dello SCAN a 640x384 basta anche per Svanstrom (640x512)? mAP di un modello sul val (640x512) in tre modi:
  nativo      640x512, nessun resize (ingresso 640x512)
  schiacciato resize a 640x384 con INTER_AREA, proporzioni cambiate (ingresso 640x384)
  letterbox   resize a 480x384 con INTER_AREA, bordo grigio 114 fino a 640x384 (ingresso 640x384, oggetti x0,75)
mAP50 e mAP50-95 per classe e per fascia di grandezza (lato sqrt(w*h) nell'immagine originale), con il codice di
../crop_zoom_map/run.py (AP COCO a 101 punti), conf 0,001, max 64 detection. Box riportate nelle coordinate 640x512.

    python experiments/tracking/scan_map_per_ingresso/run.py MODELLO.pt
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("czm", HERE.parent / "crop_zoom_map/run.py")
czm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(czm)


def prepara(img, modo):
    """(immagine per il modello, funzione che riporta le box xyxy nelle coordinate 640x512)."""
    if modo == "nativo":
        return img, lambda b: b
    if modo == "schiacciato":
        return cv2.resize(img, (640, 384), interpolation=cv2.INTER_AREA), lambda b: b * [1, 512 / 384, 1, 512 / 384]
    tela = np.full((384, 640, 3), 114, np.uint8)
    tela[:, 80:560] = cv2.resize(img, (480, 384), interpolation=cv2.INTER_AREA)
    return tela, lambda b: (b - [80, 0, 80, 0]) / 0.75


def main():
    model, batch = sys.argv[1], 128
    m, G = YOLO(model), czm.gt()
    stems = list(G)
    res = {"model": model, "bands": list(czm.BANDS)}
    for modo, hw in (("nativo", (512, 640)), ("schiacciato", (384, 640)), ("letterbox", (384, 640))):
        det = {}
        for i in range(0, len(stems), batch):
            lotto = stems[i:i + batch]
            pr = [prepara(cv2.imread(str(czm.VAL / f"images/val/{s}.jpg")), modo) for s in lotto]
            for s, (_, back), r in zip(lotto, pr, m.predict([p[0] for p in pr], imgsz=hw, conf=0.001, max_det=64,
                                                             verbose=False)):
                d = r.boxes.data.cpu().numpy().copy()
                d[:, :4] = back(d[:, :4])
                det[s] = d
        tutto = [(gc, gb, np.zeros(len(gc), bool), det[s], None) for s, (gc, gb) in G.items()]
        out = {"tutto": czm.riassunto(czm.valuta(tutto))}
        for b, (lo, hi) in czm.BANDS.items():
            camp = []
            for s, (gc, gb) in G.items():
                sz = np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]))
                camp.append((gc, gb, ~((sz >= lo) & (sz < hi)), det[s], (lo, hi)))
            out[b] = czm.riassunto(czm.valuta(camp))
        res[modo] = out
        media = np.nanmean([v for v in out["tutto"].values()], axis=0)
        print(modo, "mAP50 %.3f mAP50-95 %.3f" % tuple(media), flush=True)
    (HERE / "results" / f"map_{Path(model).parent.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
