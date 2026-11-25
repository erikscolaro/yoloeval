"""Tracker con filtro di Kalman per migliorare la classe e la box delle detection su video (docs/tracker.md).

Un filtro per oggetto, stato [cx, cy, w, h, vx, vy, vw, vh] a velocita' costante, dt in secondi. Associazione a due
soglie come ByteTrack, con la distanza di Mahalanobis fra misura e predizione (come DeepSORT) e non con l'IoU: un
oggetto piccolo e veloce si sposta piu' della sua dimensione fra due fotogrammi e l'IoU con la predizione e' zero. le detection sicure (conf >= hi) fanno nascere e aggiornano i track, quelle deboli
(lo <= conf < hi) aggiornano solo track gia' esistenti. L'associazione ignora la classe, cosi' uno scambio di classe
non spezza il track; la classe del track e' il voto pesato con la confidenza sulla sua storia (EMA). R e Q scalano
con l'altezza della box, come DeepSORT e ByteTrack. Tutti i parametri sono da scegliere sui dati.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Params:
    hi: float = 0.4          # nascita e aggiornamento
    lo: float = 0.1          # solo aggiornamento di track esistenti
    gate: float = 9.49       # distanza di Mahalanobis al quadrato oltre cui la coppia non si associa (chi2, 4 gradi, 95%)
    iou_dup: float = 0.5     # sopra, due detection dello stesso fotogramma sono lo stesso oggetto (NMS senza classe), e
                             # una detection non associata non fa nascere un track nuovo sopra uno esistente
    min_hits: int = 3        # aggiornamenti prima che il track si mostri
    max_age: int = 5         # fotogrammi senza detection prima di cancellarlo
    show_age: int = 99       # un track solo predetto si mostra per al massimo show_age fotogrammi (99: fino a max_age)
    vote: float = 0.9        # memoria del voto di classe: 0 = solo l'ultima detection
    r_pos: float = 1 / 20    # deviazione della misura, in altezze di box
    r_size: float = 1 / 20
    q_pos: float = 4.0       # accelerazione tipica, in altezze di box al s^2 (h/160 per fotogramma^2 a 25 fps = 3,9 h/s^2)
    q_size: float = 2.0
    v0: float = 10.0         # deviazione iniziale della velocita', in altezze di box al s (track nuovo, velocita' ignota)
    fps: float = 25.0        # per il dt quando non si passa


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU fra box xyxy, (n, 4) x (m, 4) -> (n, m)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def xyxy2cxcywh(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], float)


def cxcywh2xyxy(s):
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    id: int
    x: np.ndarray                  # stato (8,)
    P: np.ndarray                  # covarianza (8, 8)
    votes: np.ndarray              # voto di classe (nc,)
    conf: float
    hits: int = 1
    age: int = 0                   # fotogrammi dall'ultima detection

    @property
    def box(self):
        return cxcywh2xyxy(self.x)

    @property
    def cls(self) -> int:
        return int(self.votes.argmax())


class KalmanTracker:
    def __init__(self, nc: int, p: Params | None = None):
        self.nc, self.p, self.tracks, self._next = nc, p or Params(), [], 1
        self.H = np.eye(4, 8)

    # --- filtro -----------------------------------------------------------------------------------------------
    def _FQ(self, h: float, dt: float):
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        sa = np.array([self.p.q_pos, self.p.q_pos, self.p.q_size, self.p.q_size]) * h   # white-noise acceleration
        Q = np.zeros((8, 8))
        for i, s in enumerate(sa):
            Q[i, i], Q[i, i + 4] = s**2 * dt**4 / 4, s**2 * dt**3 / 2
            Q[i + 4, i], Q[i + 4, i + 4] = s**2 * dt**3 / 2, s**2 * dt**2
        return F, Q

    def _R(self, h: float):
        return np.diag(np.array([self.p.r_pos, self.p.r_pos, self.p.r_size, self.p.r_size]) * h) ** 2

    def _predict(self, t: Track, dt: float):
        F, Q = self._FQ(max(t.x[3], 1.0), dt)
        t.x, t.P = F @ t.x, F @ t.P @ F.T + Q
        t.x[2:4] = np.maximum(t.x[2:4], 1.0)

    def peek(self, t: Track, dt: float):
        """Stato e covarianza predetti fra dt secondi, senza toccare il track (per decidere dove guardare)."""
        F, Q = self._FQ(max(t.x[3], 1.0), dt)
        return F @ t.x, F @ t.P @ F.T + Q

    def _update(self, t: Track, det):
        z = xyxy2cxcywh(det[:4])
        S = self.H @ t.P @ self.H.T + self._R(max(z[3], 1.0))
        K = t.P @ self.H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - self.H @ t.x)
        t.P = (np.eye(8) - K @ self.H) @ t.P
        t.votes = self.p.vote * t.votes
        t.votes[int(det[5])] += det[4]
        t.conf, t.age, t.hits = float(det[4]), 0, t.hits + 1

    def _new(self, det):
        z = xyxy2cxcywh(det[:4])
        h = max(z[3], 1.0)
        P = np.diag(np.r_[np.array([self.p.r_pos, self.p.r_pos, self.p.r_size, self.p.r_size]) * h * 2,
                          np.array([1.0, 1.0, 0.5, 0.5]) * self.p.v0 * h]) ** 2   # velocita' iniziale ignota: P0 larga
        v = np.zeros(self.nc)
        v[int(det[5])] = det[4]
        self.tracks.append(Track(self._next, np.r_[z, np.zeros(4)], P, v, float(det[4])))
        self._next += 1

    # --- un fotogramma ------------------------------------------------------------------------------------------
    def _maha(self, tracks, dets):
        """Distanza di Mahalanobis al quadrato fra ogni track (gia' predetto) e ogni detection, sulla misura cx, cy, w, h."""
        z = np.array([xyxy2cxcywh(d[:4]) for d in dets])
        D = np.empty((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            y = z - self.H @ t.x
            Si = np.linalg.inv(self.H @ t.P @ self.H.T + self._R(max(t.x[3], 1.0)))
            D[i] = np.einsum("ni,ij,nj->n", y, Si, y)
        return D

    def _match(self, tracks, dets):
        if not tracks or not len(dets):
            return [], list(range(len(tracks))), list(range(len(dets)))
        D = self._maha(tracks, dets)
        r, c = linear_sum_assignment(np.where(D > self.p.gate, 1e6, D))
        pairs = [(i, j) for i, j in zip(r, c) if D[i, j] <= self.p.gate]
        ti, dj = {i for i, _ in pairs}, {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in ti], [j for j in range(len(dets)) if j not in dj]

    def _nms(self, d):
        """Una sola detection per oggetto, la piu' sicura, qualunque classe: con la testa one-to-one capita che lo stesso
        oggetto esca con due classi."""
        d = d[np.argsort(-d[:, 4])]
        keep = []
        for i in range(len(d)):
            if not keep or iou(d[i:i + 1, :4], d[keep, :4]).max() < self.p.iou_dup:
                keep.append(i)
        return d[keep]

    def step(self, dets: np.ndarray, dt: float | None = None) -> list[Track]:
        """dets: (n, 6) xyxy, conf, cls in pixel. Restituisce i track da mostrare in questo fotogramma."""
        dt = 1.0 / self.p.fps if dt is None else dt
        dets = self._nms(np.asarray(dets, float).reshape(-1, 6))
        for t in self.tracks:
            self._predict(t, dt)
            t.age += 1
        hi, lo = dets[dets[:, 4] >= self.p.hi], dets[(dets[:, 4] >= self.p.lo) & (dets[:, 4] < self.p.hi)]
        pairs, rest, new = self._match(self.tracks, hi)
        for i, j in pairs:
            self._update(self.tracks[i], hi[j])
        left = [self.tracks[i] for i in rest]
        for i, j in self._match(left, lo)[0]:
            self._update(left[i], lo[j])
        boxes = np.array([t.box for t in self.tracks]).reshape(-1, 4)
        for j in new:
            if not len(boxes) or iou(hi[j:j + 1, :4], boxes).max() < self.p.iou_dup:
                self._new(hi[j])
        self.tracks = [t for t in self.tracks if t.age <= self.p.max_age]
        return [t for t in self.tracks if t.hits >= self.p.min_hits and t.age <= self.p.show_age]
