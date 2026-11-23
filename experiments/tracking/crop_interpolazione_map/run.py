"""INTER_AREA contro INTER_LINEAR nei due resize del crop tracker, con P-:
  scan: fotogramma 1080p dei video Anti-UAV di val (data/tracking/val) ridotto a 640 di larghezza, poi il modello a 640.
        mAP50 e mAP50-95 del drone per fascia (lato sqrt(w*h) nel fotogramma originale), un fotogramma ogni --passo.
  crop: lo stesso esperimento di ../crop_zoom_map con INTER_LINEAR anche quando il crop si riduce; il riferimento AREA e'
        ../crop_zoom_map/results (stesso seme, stessi crop).
Usa le funzioni di ../crop_zoom_map/run.py senza modificarlo.

    python experiments/tracking/crop_interpolazione_map/run.py MODELLO.pt scan [--passo 5]
    python experiments/tracking/crop_interpolazione_map/run.py MODELLO.pt crop
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("crop_zoom_map", HERE.parent / "crop_zoom_map/run.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}


class _SoloLinear:
    """cv2 di crop_zoom_map con INTER_AREA = INTER_LINEAR: taglia() riduce con LINEAR, il resto di cv2 e' invariato."""
    INTER_AREA = cv2.INTER_LINEAR

    def __getattr__(self, k):
        return getattr(cv2, k)


def scan(a):
    from ultralytics import YOLO
    m = YOLO(a.model)
    args = dict(conf=0.001, max_det=64, verbose=False)
    seqs = sorted(d for d in (ROOT / "data/tracking/val").iterdir()
                  if json.loads((d / "seq.json").read_text())["fonte"] == "anti_uav")
    frames = [f for d in seqs for f in sorted((d / "images").glob("*.jpg"))[::a.passo]]
    res = {"model": a.model, "passo": a.passo, "frame": len(frames), "sequenze": len(seqs), "bands": list(cm.BANDS)}
    gts = {}
    for f in frames:
        lab = f.parent.parent / "labels" / (f.stem + ".txt")
        g = np.loadtxt(lab, ndmin=2).reshape(-1, 5) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        gts[f] = g
    for nome, interp in INTERP.items():
        det, scale = {}, {}
        for i in range(0, len(frames), a.batch):
            ims = []
            for f in frames[i:i + a.batch]:
                img = cv2.imread(str(f))
                s = 640 / max(img.shape[:2])
                ims.append(cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=interp))
                scale[f] = (s, img.shape[1], img.shape[0])
            # l'immagine e' gia' a 640 di lato: Ultralytics aggiunge solo il padding, nessun altro resize
            for f, r in zip(frames[i:i + a.batch], m.predict(ims, imgsz=640, **args)):
                d = r.boxes.data.cpu().numpy()
                d[:, :4] /= scale[f][0]                       # box nel fotogramma originale
                det[f] = d
        res[nome] = {}
        for b, (lo, hi) in cm.BANDS.items():
            camp = []
            for f, g in gts.items():
                _, W, H = scale[f]
                c, x, y, w, h = g.T
                gb = np.stack([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], 1)
                sz = np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]))
                camp.append((c.astype(int), gb, ~((sz >= lo) & (sz < hi)), det[f], (lo, hi)))
            res[nome][b] = cm.riassunto(cm.valuta(camp))["drone"]
            res.setdefault("n", {})[b] = int(sum(((~c[2]).sum()) for c in camp))
        print(nome, {b: [round(x, 3) for x in v] for b, v in res[nome].items()}, flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"scan_{Path(a.model).parent.name}.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"), p.add_argument("parte", choices=["scan", "crop"])
    p.add_argument("--passo", type=int, default=5, help="scan: un fotogramma ogni passo")
    p.add_argument("--batch", type=int, default=64)
    a, resto = p.parse_known_args()
    if a.parte == "scan":
        scan(a)
    else:
        cm.cv2 = _SoloLinear()
        sys.argv = [sys.argv[0], a.model, "--out", str(HERE / "results/crop_linear")] + resto
        cm.main()


if __name__ == "__main__":
    main()
