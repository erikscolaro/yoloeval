"""Predict su un video o una cartella di video, con le box dal vivo e in sovrimpressione la risoluzione del video,
l'ingresso del modello, il tempo di inferenza e gli FPS veri (lettura + inferenza + disegno). Tasti: spazio pausa,
. un fotogramma, n o freccia destra video dopo (anche q), p o freccia sinistra video prima, Esc chiude tutto. Gli stessi
comandi sono pulsanti cliccabili nella banda sotto il video; la barra "fotogramma" salta a un punto del video (il
tracker riparte da zero). I video annotati vanno in OUT/<nome del video>.mp4.

SORGENTE: un video, una cartella di video, una sequenza di immagini di data/tracking (cartella con images/ e seq.json,
fps da seq.json) o una cartella di sequenze (per esempio data/tracking/val).

    python -m tools.predict_video MODELLO.pt SORGENTE [--conf 0.25] [--imgsz 640 | --imgsz 736,1280] [--mode fit|crop]
                                  [--out DIR] [--no-show]

--imgsz: un numero e' il lato lungo (16:9 a 640 entra 640x384); ALTEZZA,LARGHEZZA e' l'ingresso di deploy (480p 480,864,
720p 736,1280, 1080p 1088,1920).
--mode fit (default): il fotogramma si scala per stare nell'ingresso e si completa a multipli di 32 (letterbox).
--mode crop: dal centro del fotogramma si ritaglia un rettangolo grande quanto l'ingresso, a risoluzione nativa e senza
scalare (gli oggetti restano grandi in pixel come nel training); se il fotogramma e' piu' piccolo si completa con il
grigio del letterbox. Sul video si vede il riquadro del ritaglio.
--kalman: tracker con filtro di Kalman (src/tracking.py) sulle detection: classe a voto sulla storia del track, box
filtrata, detection deboli (--lo <= conf < --hi) tenute solo se confermano un track. Le box del tracker sono spesse con
l'ID, quelle grezze del modello sottili e grigie. Con --kalman il modello gira a conf --lo, e --conf non conta.
--kalman-crop: versione con il crop adattativo (src/crop_tracker.py): il fotogramma intero (--imgsz, fit) solo in SCAN,
poi un crop --crop-size per ogni track centrato sulla predizione del Kalman, re-scan ogni --rescan fotogrammi e dopo un
miss, fotogramma intero anche con piu' di --max-crops track. I crop sono i riquadri bianchi; in alto lo stato e il tempo del passo (modello + crop + tracker).
--track-model S.pt (con --kalman-crop): modello per i crop in TRACK; MODELLO.pt resta quello dello SCAN (per esempio
P- per lo SCAN e S- per il TRACK, come in docs/tracker.md). Senza, un solo modello fa tutte e due.
--preset FILE.json: parametri tarati da Optuna, letti dai risultati: experiments/tracking/taratura_kalman/results/
kalman_f1|f2_<studio>_val.json (Kalman) o experiments/tracking/taratura_kalman_crop/results/kalman_crop_f2_val.json (Kalman,
soglie e R di S- in TRACK, crop). Le opzioni del Kalman scritte a mano (--min-hits 2, --hi 0.3, ...) prevalgono sul preset.
Scia (pulsante "scia" o tasto t, con --kalman o --kalman-crop): per ogni track vivo, gli ultimi 30 fotogrammi del
centro predetto dal Kalman prima dell'aggiornamento (bianco, puntini) e di quello aggiornato dalla detection (colore del
track, linea); dove le due si separano la predizione sbaglia.
--allarme N (con --kalman o --kalman-crop): spia rossa "DRONE" in alto a destra quando un track e' mostrato con classe
drone per N fotogrammi di fila (default 15, mezzo secondo a 30 fps; 0 la spegne). Un cambio di classe azzera il conto.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.crop_tracker import CropTracker, yolo_detect
from src.tracking import KalmanTracker, Params

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def cornice(img, righe, allarme, sotto, pulsanti=()):
    """Il fotogramma con una banda nera sopra (stato e spia DRONE) e una sotto (tasti), cosi' le scritte non coprono il
    video e si leggono a ogni risoluzione. allarme: None spia spenta (--allarme 0), [] grigia, [id...] rossa.
    pulsanti: [(etichetta, azione)] disegnati a sinistra nella banda sotto; restituisce anche i loro rettangoli
    [(x0, y0, x1, y1, azione)] nelle coordinate del quadro, per i clic."""
    h, w = img.shape[:2]
    sc = float(np.clip(w / 1280 * 0.8, 0.45, 1.3))
    lh = int(36 * sc)
    alto, basso = len(righe) * lh + lh // 2, lh + lh // 2
    q = np.zeros((h + alto + basso, w, 3), np.uint8)
    q[alto:alto + h] = img
    for i, t in enumerate(righe):
        cv2.putText(q, t, (10, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 255, 255), max(1, round(2 * sc)), cv2.LINE_AA)
    if allarme is not None:
        r = alto // 2 - 6
        c = (0, 0, 255) if allarme else (90, 90, 90)
        cv2.circle(q, (w - r - 10, alto // 2), r, c, -1)
        t = "DRONE #" + " #".join(map(str, allarme)) if allarme else "nessun drone confermato"
        tw = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)[0][0]
        cv2.putText(q, t, (w - 2 * r - 30 - tw, alto // 2 + lh // 3), cv2.FONT_HERSHEY_SIMPLEX, sc, c if allarme else
                    (200, 200, 200), max(1, round(2 * sc)), cv2.LINE_AA)
    x, rett = 10, []
    for etichetta, azione in pulsanti:
        tw = cv2.getTextSize(etichetta, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
        r = (x, alto + h + lh // 6, x + tw + lh // 2, alto + h + basso - lh // 6, azione)
        cv2.rectangle(q, r[:2], r[2:4], (70, 70, 70), -1)
        cv2.rectangle(q, r[:2], r[2:4], (200, 200, 200), 1)
        cv2.putText(q, etichetta, (x + lh // 4, alto + h + lh), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255),
                    max(1, round(1.5 * sc)), cv2.LINE_AA)
        rett.append(r)
        x = r[2] + lh // 3
    cv2.putText(q, sotto, (x + lh // 3, alto + h + lh), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255),
                max(1, round(1.5 * sc)), cv2.LINE_AA)
    return q, rett


KF = ("hi", "lo", "min_hits", "max_age", "show_age", "vote", "gate", "v0", "iou_dup")
ALFA_CROP = 0.35   # opacita' del riquadro del crop e della box predetta: non coprono l'oggetto
COLORI = [(255, 80, 0), (0, 200, 255), (0, 0, 255), (0, 220, 0)]   # BGR per classe: drone, airplane, helicopter, bird


def disegna(img, dets, tracks, names):
    for x1, y1, x2, y2, *_ in dets:                              # grezze: sottili e grigie
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 1)
    for t in tracks:
        x1, y1, x2, y2 = t.box.astype(int)
        c = COLORI[t.cls % len(COLORI)]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        etichetta = f"#{t.id} {names[t.cls]} {t.conf:.2f}" + (" pred" if t.age else "")
        cv2.putText(img, etichetta, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


def disegna_scia(img, scie, tracks):
    """scie: {id: [(centro predetto o None, centro aggiornato o None)]}; solo i track mostrati."""
    for t in tracks:
        c = COLORI[t.cls % len(COLORI)]
        s = scie.get(t.id, [])
        vere = [v for _, v in s if v is not None]
        if len(vere) > 1:
            cv2.polylines(img, [np.int32(vere)], False, c, 2, cv2.LINE_AA)
        for pr, _ in s:
            if pr is not None:
                cv2.circle(img, (int(pr[0]), int(pr[1])), 2, (255, 255, 255), -1)


def preset(path):
    """(Params del Kalman, Params diversi in TRACK, argomenti di CropTracker) da un file di risultati di Optuna."""
    j = json.loads(Path(path).read_text())
    if j.get("migliore"):                                  # taratura_kalman_crop
        b, f = j["migliore"]["parametri"], j["fissi"]
        kal = {k: b[k] for k in ("hi", "lo", "min_hits", "max_age", "show_age", "gate", "v0", "iou_dup", "vote")}
        kal |= {k: f["rq"][k] for k in ("q_pos", "q_size", "r_pos", "r_size")}
        track = {"hi": b["hi_track"], "lo": b["lo_track"], "r_pos": f["rs"]["r_pos"], "r_size": f["rs"]["r_size"]}
        return Params(**kal), track, {k: b[k] for k in ("k", "ctx", "fill", "rescan", "max_side")}
    assert j.get("parametri"), f"{path}: niente parametri (ricerca non finita?)"
    kal = {k: v for k, v in j["parametri"].items() if k != "fps"}      # taratura_kalman
    return Params(**kal), None, {}


PAR_TRACK = {"hi_track": "hi", "lo_track": "lo"}
PAR_CROP = ("k", "ctx", "fill", "rescan", "max_side")


def applica_par(par, track, extra, coppie):
    """--par chiave=valore (anche piu' volte): Kalman (Params), soglie del TRACK (hi_track, lo_track) e crop
    (k, ctx, fill, rescan, max_side); vincono sul preset."""
    track, extra, kal = dict(track or {}), dict(extra), {}
    for c in coppie or []:
        k, v = c.split("=", 1)
        if k in PAR_TRACK:
            track[PAR_TRACK[k]] = float(v)
        elif k in PAR_CROP:
            extra[k] = int(float(v)) if k in ("rescan", "max_side") else float(v)
        else:
            assert hasattr(par, k), f"--par {k}: parametro sconosciuto"
            kal[k] = type(getattr(par, k))(float(v))
    return replace(par, **kal), track or None, extra


def conta_drone(serie, shown, drone):
    """Fotogrammi di fila in cui ogni track e' mostrato con classe drone; gli altri track ripartono da zero."""
    return {t.id: serie.get(t.id, 0) + 1 for t in shown if t.cls == drone}


def spia(serie, n):
    """Gli id dei track drone da almeno n fotogrammi di fila; None se la spia e' spenta (n = 0)."""
    return [i for i, v in serie.items() if v >= n] if n else None


TASTI = "spazio pausa | . passo | t scia | m maschera | n/-> dopo | p/<- prima | Esc esci"
FINESTRA = "predict"


class UI:
    """Clic sui pulsanti e barra dei fotogrammi: le callback di OpenCV scrivono qui, il ciclo legge."""
    rett, azione, salto, pos, barra, misura, scia, maschera = [], None, None, 0, False, False, False, False
    finestra = "predict"   # per accorgersi della X: OpenCV ricrea la finestra al primo imshow dopo

    @classmethod
    def clic(cls, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cls.azione = next((r[4] for r in cls.rett if r[0] <= x <= r[2] and r[1] <= y <= r[3]), cls.azione)

    @classmethod
    def barra_mossa(cls, v):
        if v != cls.pos:          # setTrackbarPos del ciclo richiama la callback: si ignora
            cls.salto = v

    @classmethod
    def prendi(cls):
        a, cls.azione = cls.azione, None
        if a is None and cls.misura and cv2.getWindowProperty(cls.finestra, cv2.WND_PROP_VISIBLE) < 1:
            return ESCI                                  # chiusa con la X
        return a
DOPO, PRIMA, ESCI = 1, -1, "esci"   # ESCI non 0: "tasto() or UI.prendi()" lo perderebbe


class Immagini:
    """Una sequenza di data/tracking (images/*.jpg e seq.json) letta come un video: stessa interfaccia di VideoCapture,
    anche per saltare a un fotogramma."""
    def __init__(self, d: Path):
        self.files, self.i = sorted((d / "images").glob("*.jpg")), 0
        self.fps = float(json.loads((d / "seq.json").read_text()).get("fps", 30))
        h, w = cv2.imread(str(self.files[0])).shape[:2] if self.files else (0, 0)
        self.prop = {3: w, 4: h, cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: len(self.files)}

    def get(self, k):
        return self.i if k == cv2.CAP_PROP_POS_FRAMES else self.prop.get(k, 0)

    def set(self, k, v):
        if k == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(min(max(v, 0), len(self.files)))

    def read(self):
        if self.i >= len(self.files):
            return False, None
        self.i += 1
        return True, cv2.imread(str(self.files[self.i - 1]))

    def release(self):
        self.files = []


def sorgenti(s: Path, fonte: str | None = None):
    """Video o sequenze di immagini da una sorgente (vedi SORGENTE nel docstring); fonte: solo le sequenze di data/tracking
    con quel campo `fonte` in seq.json (svanstrom, anti_uav)."""
    if s.is_file():
        return [s]
    if (s / "seq.json").exists():
        tutte = [s]
    else:
        tutte = sorted(f for f in s.iterdir() if f.suffix.lower() in VIDEO or (f / "seq.json").exists())
    return [f for f in tutte if fonte is None or (f / "seq.json").exists()
            and json.loads((f / "seq.json").read_text()).get("fonte") == fonte]


def tasto(k):
    """Codice di cv2.waitKeyEx -> DOPO, PRIMA, ESCI, "pausa" o None. Frecce: codici GTK (Linux)."""
    if k in (ord("n"), ord("q"), 65363):
        return DOPO
    if k in (ord("p"), 65361):
        return PRIMA
    if k == 27:
        return ESCI
    if k == ord("."):
        return "passo"
    if k == ord("t"):
        return "scia"
    if k == ord("m"):
        return "maschera"
    return "pausa" if k == ord(" ") else None


def un_video(model, src: Path, a) -> int:
    """DOPO a fine video o con n/q, PRIMA con p, ESCI con Esc."""
    cap = Immagini(src) if src.is_dir() else cv2.VideoCapture(str(src))
    w, h, fps_v = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 30
    # ingresso del modello: letterbox di Ultralytics con un .pt (scala per stare in imgsz, poi multipli di 32)
    H, W = a.imgsz if len(a.imgsz) == 2 else a.imgsz * 2
    if a.mode == "crop":
        H, W = (-(-x // 32) * 32 for x in (H, W))
        ch, cw = min(h, H), min(w, W)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        iw, ih = W, H
    else:
        s = min(H / h, W / w)
        iw, ih = (-(-round(x * s) // 32) * 32 for x in (w, h))
    out = None
    if a.out:   # il writer si apre al primo fotogramma, quando la cornice ha la sua misura
        Path(a.out).mkdir(parents=True, exist_ok=True)
    par, track, extra = preset(a.preset) if a.preset else (Params(), None, {})
    par = replace(par, **{k: getattr(a, k) for k in KF if getattr(a, k) is not None})   # a mano vince sul preset
    par, track, extra = applica_par(par, track, extra, a.par)
    extra = {"rescan": 30} | extra | ({} if a.rescan is None else {"rescan": a.rescan})
    conf = par.lo if a.kalman else a.conf
    track_model = YOLO(a.track_model) if a.kalman_crop and a.track_model else model

    def rileva(images, imgsz, conf):                 # i crop (imgsz = crop) al modello del TRACK, il resto allo SCAN
        m = track_model if imgsz == a.crop_size else model
        return yolo_detect(m)(images, imgsz, conf)

    def nuovi():                                     # all'inizio e dopo un salto: il moto non e' piu' continuo
        return (KalmanTracker(len(model.names), par) if a.kalman else None,
                CropTracker(rileva, len(model.names), par, scan_imgsz=a.imgsz if len(a.imgsz) == 2 else a.imgsz[0],
                            crop=a.crop_size, max_crops=a.max_crops, track=track, **extra) if a.kalman_crop else None)

    tr, ct = nuovi()
    scie = {}                                        # id -> [(centro predetto, centro aggiornato)], per la scia
    soglie = None                                    # --maschera-scan: soglia F1 per classe di P-, solo per il disegno
    if a.maschera_scan and ct:
        f = Path(a.model).parent / "soglie_f1_val.json" if a.maschera_scan == "auto" else Path(a.maschera_scan)
        pc = json.loads(f.read_text())["per_classe"]
        soglie = np.array([pc[model.names[i]][0] for i in range(len(model.names))])
        UI.maschera = True
    drone = next((i for i, n in model.names.items() if n == "drone"), None)
    serie, n_fot, pausa, passo = {}, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), False, False
    if a.show:
        if not UI.barra:
            cv2.createTrackbar("fotogramma", FINESTRA, 0, max(n_fot - 1, 1), UI.barra_mossa)
            UI.barra = True
        UI.pos = 0
        cv2.setTrackbarMax("fotogramma", FINESTRA, max(n_fot - 1, 1))
        cv2.setTrackbarPos("fotogramma", FINESTRA, 0)
        UI.salto = None
    fps, t0 = 0.0, time.perf_counter()
    while True:
        while a.show and pausa and not passo and UI.salto is None:   # in pausa: pulsanti, barra e tasti restano vivi
            az = tasto(cv2.waitKeyEx(30)) or UI.prendi()
            if az == "pausa":
                pausa = False
            elif az == "scia":
                UI.scia = not UI.scia
            elif az == "passo":
                passo = True
            elif az in (DOPO, PRIMA, ESCI):
                cap.release(), out and out.release()
                return az
        if UI.salto is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, UI.salto)
            UI.salto, serie, scie = None, {}, {}
            tr, ct = nuovi()
        passo = False
        ok, frame = cap.read()
        if not ok:
            break
        kt = ct.tr if ct else tr                     # predizione prima del passo, per la scia
        pred = {t.id: kt.peek(t, 1 / fps_v)[0][:2] for t in kt.tracks} if kt and UI.scia else {}
        if ct:
            ts = time.perf_counter()
            shown, info = ct.step(frame, 1 / fps_v)
            ms = 1000 * (time.perf_counter() - ts)
            img = frame.copy()
            sopra = img.copy()                           # crop e predetta su un livello a parte, fuso trasparente
            for x0, y0, lato, tid, pb in info["rois"]:   # il crop usato davvero e la box predetta che lo centra
                cv2.rectangle(sopra, (int(x0), int(y0)), (int(x0 + lato), int(y0 + lato)), (255, 255, 255), 2)
                x1, y1, x2, y2 = pb.astype(int)
                for i in range(x1, x2, 8):                                    # predetta: tratteggio bianco
                    cv2.line(sopra, (i, y1), (min(i + 4, x2), y1), (255, 255, 255), 1)
                    cv2.line(sopra, (i, y2), (min(i + 4, x2), y2), (255, 255, 255), 1)
                for j in range(y1, y2, 8):
                    cv2.line(sopra, (x1, j), (x1, min(j + 4, y2)), (255, 255, 255), 1)
                    cv2.line(sopra, (x2, j), (x2, min(j + 4, y2)), (255, 255, 255), 1)
            img = cv2.addWeighted(sopra, ALFA_CROP, img, 1 - ALFA_CROP, 0)
            for x0, y0, lato, tid, pb in info["rois"]:   # le scritte restano piene
                cv2.putText(img, f"crop #{tid} {lato:.0f}px -> {a.crop_size} (x{a.crop_size / lato:.2f})",
                            (int(x0) + 4, int(y0 + lato) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            vis, dvis = shown, info["dets"]
            if soglie is not None and UI.maschera and info["mode"] == "SCAN":   # solo grafica: metriche su shown
                vis = [t for t in shown if t.conf >= soglie[t.cls]]
                dvis = [d for d in dvis if d[4] >= soglie[int(d[5])]]
            disegna(img, dvis, vis, model.names)
            stato = (f"SCAN ingresso {iw}x{ih}" if info["mode"] == "SCAN"
                     else f"TRACK {len(info['rois'])} crop {a.crop_size}x{a.crop_size}")
            righe = [f"video {w}x{h} @ {fps_v:.0f} fps  |  kalman-crop: {stato}",
                     f"passo {ms:.1f} ms  |  {fps:.1f} FPS  |  {len(ct.tr.tracks)} track  hi {par.hi:.2g} lo {par.lo:.2g}"]
        elif a.mode == "crop":
            tela = np.full((H, W, 3), 114, np.uint8)   # gia' alla forma dell'ingresso: il letterbox non scala
            tela[:ch, :cw] = frame[y0:y0 + ch, x0:x0 + cw]
            r = model.predict(tela, imgsz=[H, W], conf=conf, verbose=False)[0]
            d = r.boxes.data.cpu().numpy()
            d = d[(d[:, 0] < cw) & (d[:, 1] < ch)]                  # niente box nel grigio di riempimento
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, cw) + x0
            d[:, [1, 3]] = d[:, [1, 3]].clip(0, ch) + y0
            img = frame.copy()
            if not tr:
                img[y0:y0 + ch, x0:x0 + cw] = r.plot()[:ch, :cw]
            cv2.rectangle(img, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (255, 255, 255), 2)
        else:
            r = model.predict(frame, imgsz=a.imgsz if len(a.imgsz) == 2 else a.imgsz[0], conf=conf, verbose=False)[0]
            d = r.boxes.data.cpu().numpy()
            img = frame.copy() if tr else r.plot()
        if tr:
            shown = tr.step(d, 1 / fps_v)
            disegna(img, d, shown, model.names)
        if kt and UI.scia:
            scie = {t.id: (scie.get(t.id, []) + [(pred.get(t.id), None if t.age else t.x[:2].copy())])[-30:]
                    for t in kt.tracks}
            disegna_scia(img, scie, vis if ct else shown)
        allarme = None
        if tr or ct:
            serie = conta_drone(serie, shown, drone)
            allarme = spia(serie, a.allarme)
        t1 = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (t1 - t0) if fps else 1 / (t1 - t0)   # media mobile: numero leggibile
        t0 = t1
        if not ct:
            righe = [f"video {w}x{h} @ {fps_v:.0f} fps  |  {a.mode}, ingresso {iw}x{ih}",
                     f"inferenza {r.speed['inference']:.1f} ms  |  {fps:.1f} FPS  |  {len(d)} box  "
                     + (f"kalman hi {par.hi:.2g} lo {par.lo:.2g}, {len(tr.tracks)} track" if tr else f"conf {a.conf}")]
        pulsanti = ([("<< prima", PRIMA), ("riprendi" if pausa else "pausa", "pausa"), ("passo", "passo"), ("dopo >>", DOPO)]
                    + ([("scia: si" if UI.scia else "scia: no", "scia")] if kt else [])
                    + ([("maschera: si" if UI.maschera else "maschera: no", "maschera")] if soglie is not None else [])
                    ) if a.show else []
        img, UI.rett = cornice(img, righe, allarme, TASTI + f"  |  {src.name}", pulsanti)
        if a.out:
            if out is None:
                out = cv2.VideoWriter(str(Path(a.out) / f"{src.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_v,
                                      (img.shape[1], img.shape[0]))
            out.write(img)
        if a.show:
            cv2.imshow(FINESTRA, img)
            if not UI.misura:                        # WINDOW_NORMAL nasce piccola: la misura del video, dentro lo schermo
                r = min(1.0, 1800 / img.shape[1], 1000 / img.shape[0])
                cv2.resizeWindow(FINESTRA, int(img.shape[1] * r), int(img.shape[0] * r))
                UI.misura = True
            UI.pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cv2.setTrackbarPos("fotogramma", FINESTRA, UI.pos)
            azione = tasto(cv2.waitKeyEx(1)) or UI.prendi()
            if azione in ("pausa", "passo"):         # passo: un fotogramma e poi fermo
                pausa = azione == "passo" or not pausa
            if azione == "scia":
                UI.scia = not UI.scia
            if azione == "maschera":
                UI.maschera = not UI.maschera
            if azione in (DOPO, PRIMA, ESCI):
                cap.release(), out and out.release()
                return azione
    cap.release()
    if out:
        out.release()
    return DOPO


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"), p.add_argument("source")
    p.add_argument("--conf", type=float, default=0.25), p.add_argument("--imgsz", type=lambda x: [int(v) for v in x.split(",")], default=[640])
    p.add_argument("--mode", choices=["fit", "crop"], default="fit")
    p.add_argument("--out"), p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--kalman", action="store_true")
    p.add_argument("--kalman-crop", action="store_true")
    p.add_argument("--crop-size", type=int, default=256), p.add_argument("--rescan", type=int, help="kalman-crop: re-scan ogni N fotogrammi, 0 mai (default 30, o il preset); prevale sul preset")
    p.add_argument("--max-crops", type=int, default=1, help="kalman-crop (1: S- sta nel budget con un solo crop): con piu' track confermati, fotogramma intero")
    p.add_argument("--track-model", help="kalman-crop: modello per i crop in TRACK (MODELLO.pt resta per lo SCAN)")
    p.add_argument("--maschera-scan", nargs="?", const="auto", metavar="SOGLIE.json",
                   help="kalman-crop, solo grafica: nei fotogrammi SCAN nasconde track e box sotto la soglia F1 per classe del "
                        "modello dello SCAN (soglie_f1_val.json accanto ai pesi; tools/curva_fbeta.py --soglie). Tasto m")
    p.add_argument("--preset", help="risultati di Optuna con i parametri tarati (vedi in testa)")
    p.add_argument("--par", action="append", metavar="CHIAVE=VALORE",
                   help="parametro del tracker (Kalman, hi_track, lo_track, k, ctx, fill, rescan, max_side), vince sul preset")
    p.add_argument("--fonte", choices=["svanstrom", "anti_uav"], help="solo le sequenze di data/tracking di questa fonte")
    p.add_argument("--allarme", type=int, default=15, help="fotogrammi di fila con classe drone per la spia rossa (0: spenta)")
    d = Params()
    for k in KF:
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(getattr(d, k)), default=None,
                       help=f"Kalman (default {getattr(d, k)}, o il preset)")
    a = p.parse_args()
    s = Path(a.source)
    videos = sorgenti(s, a.fonte)
    assert videos, f"nessun video o sequenza in {s}" + (f" con fonte {a.fonte}" if a.fonte else "")
    model = YOLO(a.model)
    if a.show:
        cv2.namedWindow(FINESTRA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(FINESTRA, UI.clic)
    i = 0
    while 0 <= i < len(videos):
        print(videos[i].name, flush=True)
        azione = un_video(model, videos[i], a)
        if azione == ESCI:
            break
        i = max(i + azione, 0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
