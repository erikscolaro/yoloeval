"""Tracker di Kalman con crop adattativo (docs/tracker.md, livello B): il Kalman decide dove far guardare il modello.

- SCAN: il modello gira sul fotogramma intero ridotto con INTER_AREA (scan_imgsz), quando non ci sono track confermati, ogni `rescan`
  fotogrammi, dopo un fotogramma in cui un track confermato non ha trovato detection nel suo crop, o se i track sono
  piu' di `max_crops`.
- TRACK: un crop quadrato per ogni track confermato, centrato sulla box predetta. Lato = box predetta + k sigma
  dell'incertezza (diagonale di P) + `ctx` di contesto, e almeno lato lungo della box / `fill` (l'oggetto occupa al
  massimo `fill` del crop); mai piu' piccolo di crop / `max_up` (niente ingrandimenti oltre `max_up`). Il crop si porta
  a `crop` x `crop` (INTER_AREA se riduce), fuori dal fotogramma si completa con il grigio 114 del letterbox, e le
  detection si riproiettano nel fotogramma. Tutti i crop del fotogramma vanno al modello insieme.
  Se il lato del crop di un track supera `max_side` (oggetto vicino e grande, oltre circa 0,6 * max_side px), quel
  fotogramma e' uno SCAN: il ritaglio costa troppo e nello SCAN l'oggetto grande si vede gia' bene. `max_side` e' un
  parametro di deploy; None lo toglie e il crop si riduce sempre a `crop`.

`detect(images, imgsz, conf)` restituisce per ogni immagine un array (n, 6) xyxy, conf, cls nelle sue coordinate: con
Ultralytics e' `yolo_detect(model)`, nei test una funzione finta.
"""
from __future__ import annotations

import cv2
import numpy as np

from .tracking import KalmanTracker, Params, cxcywh2xyxy


def yolo_detect(model):
    def detect(images, imgsz, conf):
        return [r.boxes.data.cpu().numpy() for r in model.predict(images, imgsz=imgsz, conf=conf, verbose=False)]
    return detect


class CropTracker:
    def __init__(self, detect, nc: int, p: Params | None = None, scan_imgsz=640, crop: int = 256, rescan: int = 30,
                 k: float = 3.0, ctx: float = 0.2, fill: float = 0.6, max_up: float = 1.0, max_crops: int = 4,
                 max_side: float | None = 512):
        self.detect, self.tr = detect, KalmanTracker(nc, p)
        self.scan_imgsz, self.crop, self.rescan, self.k, self.ctx = scan_imgsz, crop, rescan, k, ctx
        self.fill, self.max_up, self.max_crops, self.max_side = fill, max_up, max_crops, max_side
        self.n, self.need_scan = 0, True

    def roi(self, t, dt: float) -> tuple[float, float, float]:
        """(cx, cy, lato) del crop quadrato per il track t."""
        x, P = self.tr.peek(t, dt)
        sx, sy, sw, sh = np.sqrt(np.clip(np.diag(P)[:4], 0, None))
        cx, cy, w, h = x[:4]
        side = max(w + 2 * self.k * sx + self.k * sw, h + 2 * self.k * sy + self.k * sh) * (1 + self.ctx)
        side = max(side, max(w, h) / self.fill, self.crop / self.max_up)
        return cx, cy, side

    def cut(self, frame, cx, cy, side):
        """Crop quadrato (cx, cy, side) portato a crop x crop; restituisce l'immagine, l'origine e la scala."""
        H, W = frame.shape[:2]
        s = int(round(side))
        x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        tela = np.full((s, s, 3), 114, np.uint8)
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
        if fx1 > fx0 and fy1 > fy0:
            tela[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
        interp = cv2.INTER_AREA if s > self.crop else cv2.INTER_LINEAR
        return cv2.resize(tela, (self.crop, self.crop), interpolation=interp), (x0, y0), self.crop / s

    def fit(self, frame):
        """SCAN: fotogramma ridotto con INTER_AREA a lato lungo `scan_imgsz` (Ultralytics userebbe INTER_LINEAR). Restituisce
        l'immagine e la scala; niente ingrandimenti."""
        r = self.scan_imgsz / max(frame.shape[:2])
        if r >= 1:
            return frame, 1.0
        h, w = frame.shape[:2]
        return cv2.resize(frame, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA), r

    def step(self, frame, dt: float):
        """Un fotogramma: (track da mostrare, info) con info = {mode, rois [(x0, y0, lato, id, box predetta xyxy)], dets}."""
        p = self.tr.p
        confirmed = [t for t in self.tr.tracks if t.hits >= p.min_hits]
        scan = (self.need_scan or not confirmed or len(confirmed) > self.max_crops
                or (self.rescan and self.n % self.rescan == 0))
        if not scan:
            geo = [self.roi(t, dt) for t in confirmed]
            scan = self.max_side is not None and any(side > self.max_side for *_, side in geo)
        rois = []
        if scan:
            img, r = self.fit(frame)
            dets = np.array(self.detect([img], self.scan_imgsz, p.lo)[0], float).reshape(-1, 6)
            dets[:, :4] /= r
        else:
            crops, orig = [], []
            for t, g in zip(confirmed, geo):
                img, o, sc = self.cut(frame, *g)
                crops.append(img), orig.append((o, sc))
                rois.append((o[0], o[1], self.crop / sc, t.id, cxcywh2xyxy(self.tr.peek(t, dt)[0])))
            parts = []
            for d, ((x0, y0), sc) in zip(self.detect(crops, self.crop, p.lo), orig):
                d = d.copy()
                d[:, [0, 2]] = d[:, [0, 2]] / sc + x0
                d[:, [1, 3]] = d[:, [1, 3]] / sc + y0
                parts.append(d)
            dets = np.vstack(parts) if parts else np.zeros((0, 6))
        H, W = frame.shape[:2]
        dets = np.asarray(dets, float).reshape(-1, 6)
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, W)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, H)
        dets = dets[(dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])]   # box rimaste solo nel grigio
        shown = self.tr.step(dets, dt)
        vivi = {t.id: t for t in self.tr.tracks}
        # un track confermato senza detection nel suo crop (o cancellato): al prossimo fotogramma si riguarda tutto
        self.need_scan = not scan and any(t.id not in vivi or vivi[t.id].age > 0 for t in confirmed)
        self.n += 1
        return shown, {"mode": "SCAN" if scan else "TRACK", "rois": rois, "dets": dets}
