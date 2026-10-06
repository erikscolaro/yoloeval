#!/usr/bin/env python3
"""Costruisce il dataset Svanstrom (Drone-detection-dataset, solo video visibili).

AOD4 resta per il training da zero; questo serve al finetune prima del pruning e
a tutto quello che viene dopo. Passi:

1. scarica `Data/Video_V/*` e l'xlsx con classe e distance bin da GitHub;
2. per ogni video legge le label (.mat MATLAB groundTruth) e tiene solo i frame
   "nuovi" secondo un hash semantico: dHash del ritaglio attorno al bersaglio
   piu' posizione/dimensione del box. L'hash sul frame intero non funziona qui:
   il cielo domina e il bersaglio occupa pochi pixel, tutti i frame sembrano
   uguali. Un frame per cartella video in `<src>/frames/<video>/`; ogni frame
   ha una fascia small/medium/large (soglie COCO sul box piu' grande);
3. split 70/20/10 per video (mai per frame: frame dello stesso video in split
   diversi e' lo stesso leakage di AOD4), stratificato per classe e
   bilanciato sui frame di ogni fascia;
4. `<out>/{all,small,medium,large,background}/images/{train,val,test,calib}` e
   `labels/...` come hardlink ai frame, ognuno col suo `dataset.yaml` e i
   grafici di controllo in `report/`; `<out>/manifest.json` per tutto.

Le label non si leggono con scipy (oggetti MCOS) e `mcos-decoder` sbaglia i
frame con piu' oggetti (2 box letti come vuoto, 3-5 scartati, frame
disallineati): qui c'e' un walker MAT5 minimo. Ogni clip ha la sua verifica:
numero di label == numero di frame del video, altrimenti la clip e' scartata
e finisce nel manifest.

    python tools/svanstrom.py                       # tutto, in data/svanstrom
    python tools/svanstrom.py --hash-thr 6          # selezione piu' fitta
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

REPO = "DroneDetectionThesis/Drone-detection-dataset"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
XLSX = "Data/Video_dataset_description.xlsx"

NAMES = ["airplane", "bird", "drone", "helicopter"]  # stesso ordine di AOD4
SIZES = ["small", "medium", "large"]
SIZE_EDGES = [32, 96]  # sqrt(area) in px, soglie COCO
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # tools.plots anche lanciato come script


# --- download -----------------------------------------------------------------

def _fetch(url: str, tries: int = 10, delay: float = 10) -> bytes:
    """GET con retry: il proxy cade spesso e raw.githubusercontent limita le
    richieste parallele."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def _get(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(_fetch(url))
    tmp.rename(dst)


def download(raw: Path) -> dict[str, str]:
    """Scarica xlsx, video e label; i nomi dei file vengono dall'xlsx."""
    _get(RAW + XLSX, raw / XLSX)
    dist = distance_bins(raw / XLSX)
    paths = [f"Data/Video_V/{v}{s}" for v in dist for s in (".mp4", "_LABELS.mat")]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda p: _get(RAW + p, raw / p), paths))
    print(f"download: {len(paths)} file in {raw}")
    return dist


def distance_bins(xlsx: Path) -> dict[str, str]:
    """{V_DRONE_001: 'medium', ...} dall'xlsx, senza openpyxl."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(xlsx)
    strings = ["".join(si.itertext()) for si in
               ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si")]
    out = {}
    for row in ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        cells = []
        for c in row.iter(ns + "c"):
            v = c.find(ns + "v")
            v = None if v is None else v.text
            cells.append(strings[int(v)] if c.get("t") == "s" and v else v)
        if len(cells) > 3 and cells[0] == "V":
            out[f"V_{cells[1]}_{int(cells[2]):03d}"] = cells[3].strip().lower()
    return out


# --- label MATLAB --------------------------------------------------------------

DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
          7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}


def _elements(buf: bytes):
    """(tipo, payload) degli elementi MAT5 in sequenza."""
    off = 0
    while off + 8 <= len(buf):
        t, n = struct.unpack_from("<II", buf, off)
        if t >> 16:  # small data element: dimensione nei 16 bit alti
            t, n, p, off = t & 0xFFFF, t >> 16, off + 4, off + 8
        else:
            p, off = off + 8, off + 8 + n + (-n % 8)
        if not 0 < t <= 18:
            return
        yield t, buf[p:p + n]


def _matrices(buf: bytes, parent: tuple = ()):
    """(id del padre, sotto-elementi) per ogni miMATRIX, in profondita'."""
    for i, (t, payload) in enumerate(_elements(buf)):
        if t == 14:
            yield parent, list(_elements(payload))
            yield from _matrices(payload, parent + (i,))


def _box_matrix(kids) -> np.ndarray | None:
    """Matrice double kx4 (o vuota) come array (k, 4); None per tutto il resto."""
    if len(kids) < 3 or kids[0][0] != 6 or kids[1][0] != 5:
        return None
    if np.frombuffer(kids[0][1], "<u4")[0] & 0xFF != 6:  # mxDOUBLE_CLASS
        return None
    dims = np.frombuffer(kids[1][1], "<i4")
    if dims.prod() == 0:
        return np.zeros((0, 4))
    if len(dims) != 2 or dims[1] != 4 or len(kids) < 4:
        return None
    t, data = kids[3]  # MATLAB salva double interi anche come int compressi
    return np.frombuffer(data, DTYPES[t]).astype(float).reshape(4, dims[0]).T


def read_labels(mat: Path) -> list[np.ndarray] | None:
    """Per frame un array (k, 5): classe, x, y, w, h in pixel, 1-based (MATLAB).

    Le celle della timetable sono matrici double kx4 sorelle sotto lo stesso
    padre (si prende quello che ne ha di piu'), quattro per frame, una per
    colonna: AIRPLANE, BIRD, DRONE, HELICOPTER, lo stesso ordine di NAMES.
    Le celle accanto che non sono box (es. la stringa col path del video) si
    scartano. None se le celle non sono un multiplo delle colonne.
    """
    import scipy.io

    fw = scipy.io.loadmat(mat)["__function_workspace__"].tobytes()[8:]
    groups = defaultdict(list)
    for parent, kids in _matrices(fw):
        groups[parent].append(_box_matrix(kids))
    best = max(groups.values(), key=lambda g: sum(b is not None and len(b) > 0 for b in g))
    cells = [b for b in best if b is not None]
    if len(cells) % len(NAMES):
        return None
    return [np.vstack([np.column_stack([np.full(len(b), c), b])
                       for c, b in enumerate(cells[f:f + len(NAMES)])])
            for f in range(0, len(cells), len(NAMES))]


# --- selezione frame -------------------------------------------------------------

def _dhash(gray: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def _signature(gray: np.ndarray, boxes: np.ndarray):
    """Hash del ritaglio attorno all'unione dei box + geometria normalizzata."""
    h, w = gray.shape
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    pad = max(x1 - x0, y1 - y0, 8)  # contesto: lo sfondo vicino conta
    crop = gray[int(max(0, y0 - pad)):int(min(h, y1 + pad)),
                int(max(0, x0 - pad)):int(min(w, x1 + pad))]
    geo = np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                    np.log(max(x1 - x0, 1)), np.log(max(y1 - y0, 1))])
    return _dhash(crop), geo, len(boxes)


def _is_new(sig, kept, hash_thr: int, geo_thr: float) -> bool:
    # ponytail: confronto con tutti i frame tenuti, O(n*k) per video; basta
    # finche' i video restano da ~300 frame
    hsh, geo, k = sig
    for h2, g2, k2 in kept:
        if (k == k2 and np.count_nonzero(hsh != h2) <= hash_thr
                and (geo is None or (np.hypot(*(geo[:2] - g2[:2])) <= geo_thr
                                     and np.abs(geo[2:] - g2[2:]).max() <= 0.3))):
            return False
    return True


def _mask(shape, boxes: np.ndarray, grow: float) -> np.ndarray:
    """Maschera dei box (x, y, w, h) allargati di `grow` volte il lato, min 4 px."""
    m = np.zeros(shape, np.uint8)
    for x, y, w, h in boxes:
        g = max(4, grow * max(w, h))
        m[int(max(0, y - g)):int(y + h + g), int(max(0, x - g)):int(x + w + g)] = 255
    return m


def _uniform(gray: np.ndarray, boxes: np.ndarray, std_thr: float) -> bool:
    """Sfondo attorno ai box uniforme (cielo): l'inpainting non lascia tracce."""
    ring = (_mask(gray.shape, boxes, 1.0) > 0) & (_mask(gray.shape, boxes, 0.25) == 0)
    return ring.any() and gray[ring].std() < std_thr


def _size(boxes: np.ndarray) -> str | None:
    """Fascia COCO dell'immagine dal box piu' grande; None senza box.

    Per immagine e non per box: togliere un box dalle label insegnerebbe che
    li' non c'e' niente."""
    if not len(boxes):
        return None
    return SIZES[np.digitize(np.sqrt(boxes[:, 3] * boxes[:, 4]).max(), SIZE_EDGES)]


def process_video(args) -> dict:
    """Seleziona i frame di un video e li scrive in `frames/<video>/`.

    Background: il dataset ne ha solo 82 frame in 4 video. Per averne una quota
    fissa per video (e quindi 70/20/10 come il resto) si prendono frame non
    selezionati con il bersaglio su cielo uniforme e lo si cancella con
    l'inpainting; label vuota, suffisso `_bg`.
    """
    import cv2

    video, mat, dst, hash_thr, geo_thr, bg_frac, bg_std = args
    labels = read_labels(mat)
    if labels is None:
        return {"video": video.stem, "skipped": "celle delle label non multiple di 4"}
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()

    def write(stem, img, text):
        cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dst / "labels" / f"{stem}.txt").write_text(text)

    cap = cv2.VideoCapture(str(video))
    kept, selected, sizes, candidates, n = [], [], [], [], 0
    while (frame := cap.read()[1]) is not None:
        if n < len(labels):  # oltre si contano solo i frame, per la verifica
            boxes = labels[n].copy()
            boxes[:, 1:3] -= 1  # MATLAB 1-based
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sig = _signature(gray, boxes[:, 1:]) if len(boxes) else (_dhash(gray), None, 0)
            if _is_new(sig, kept, hash_thr, geo_thr):
                kept.append(sig)
                write(f"{video.stem}_f{n:05d}", frame,
                      _yolo(boxes, frame.shape[1], frame.shape[0]))
                selected.append(n)
                sizes.append(_size(boxes))
            elif len(boxes) and _uniform(gray, boxes[:, 1:], bg_std):
                candidates.append(n)
        n += 1
    if n != len(labels):
        shutil.rmtree(dst)
        return {"video": video.stem, "skipped": f"{len(labels)} label per {n} frame"}

    # background distribuito nel tempo tra i candidati
    k = min(len(candidates), round(bg_frac * len(selected)))
    background = sorted({candidates[i] for i in np.linspace(0, len(candidates) - 1, k).round().astype(int)}) if k else []
    cap = cv2.VideoCapture(str(video))
    for i in range(background[-1] + 1 if background else 0):
        frame = cap.read()[1]
        if i in background:
            boxes = labels[i][:, 1:].copy()
            boxes[:, :2] -= 1
            mask = _mask(frame.shape[:2], boxes, 0.25)
            write(f"{video.stem}_f{i:05d}_bg", cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA), "")
    return {"video": video.stem, "frames": n, "selected": selected, "sizes": sizes,
            "background": background}


def _yolo(boxes: np.ndarray, W: int, H: int) -> str:
    """(classe, x, y, w, h) in pixel 0-based -> righe YOLO, box tagliati al frame."""
    lines = []
    for c, x, y, w, h in boxes:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"{int(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    return "".join(lines)


# --- split ---------------------------------------------------------------------

def assign_splits(videos: list[dict], seed: int) -> None:
    """Split per video, per classe, bilanciato sui frame di ogni fascia.

    Greedy in ordine casuale: ogni video va allo split piu' indietro rispetto
    al target sul suo mix di fasce (frame small/medium/large, pesati per la
    loro frazione del totale della classe). Non dal piu' pesante: i video
    grossi finirebbero tutti in train e a val/test resterebbero solo le clip
    statiche (pochi frame selezionati).
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for v in videos:
        strata[v["class"]].append(v)
    for group in strata.values():
        rng.shuffle(group)
        counts = [Counter(v["sizes"]) for v in group]
        tot = sum(counts, Counter())
        got = {s: Counter() for s in SPLITS}
        for v, c in zip(group, counts):
            v["split"] = max(SPLITS, key=lambda s: sum(
                k * (SPLITS[s] * tot[b] - got[s][b]) / tot[b] for b, k in c.items()))
            got[v["split"]].update(c)


# --- costruzione ---------------------------------------------------------------

def build(out: Path, frames: Path, videos: list[dict], n_calib: int, seed: int,
          keep=lambda stem: True) -> None:
    """Hardlink ai frame dei video nei loro split, solo quelli con keep(stem)."""
    for d in ("images", "labels"):
        shutil.rmtree(out / d, ignore_errors=True)
    files = defaultdict(list)
    for v in videos:
        src = frames / v["video"]
        for img in sorted((src / "images").glob("*.jpg")):
            if keep(img.stem):
                files[v["split"]].append((img, src / "labels" / f"{img.stem}.txt"))
    # ponytail: calib e' un campione delle immagini di train; disgiunto da
    # val/test, che e' quello che conta per non calibrare sui dati di valutazione
    files["calib"] = random.Random(seed).sample(files["train"],
                                                min(n_calib, len(files["train"])))
    for split, items in files.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img, lab in items:
            os.link(img, out / "images" / split / img.name)
            os.link(lab, out / "labels" / split / lab.name)
    (out / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        "calib: images/calib    # solo calibrazione INT8, campione di train\n\n"
        f"nc: {len(NAMES)}\nnames:\n"
        + "".join(f"    {i}: {n}\n" for i, n in enumerate(NAMES)))


# --- report --------------------------------------------------------------------

def report(out: Path, videos: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.plots import GRAY, HALF, STYLE, WIDTH, color, save

    plt.style.use(STYLE)
    splits = ["train", "val", "test", "calib"]
    rows = []  # (split, classe, x, y, w, h) in pixel, per ogni box
    images, empty = Counter(), Counter()
    W, H = 640, 512  # tutti i video visibili
    for split in splits:
        for lab in (out / "labels" / split).glob("*.txt"):
            lines = lab.read_text().split()
            images[split] += 1
            empty[split] += not lines
            for i in range(0, len(lines), 5):
                c, x, y, w, h = map(float, lines[i:i + 5])
                rows.append((split, NAMES[int(c)], x * W, y * H, w * W, h * H))
    rep = out / "report"
    shutil.rmtree(rep, ignore_errors=True)  # niente grafici di build precedenti
    cats = NAMES + ["background"]  # background: immagini senza oggetti
    x = np.arange(len(cats))

    # 1. immagini per split
    fig, ax = plt.subplots(figsize=(HALF * 1.4, 2.6))
    pos = [images[s] - empty[s] for s in splits]
    ax.bar(splits, pos, color=[color(s) for s in splits], width=0.6)
    ax.bar(splits, [empty[s] for s in splits], bottom=pos, color=GRAY,
           width=0.6, label="senza oggetti")
    for i, s in enumerate(splits):
        ax.text(i, images[s], f"{images[s]}", ha="center", va="bottom")
    ax.set_ylabel("immagini")
    ax.set_title("Immagini per split")
    ax.legend()
    save(fig, rep / "images_per_split")

    # 2. classi per split: assolute e quota di ogni classe per split
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    cnt = Counter((r[0], r[1]) for r in rows)
    cnt.update({(s, "background"): empty[s] for s in splits})
    for i, s in enumerate(splits):
        a1.bar(x + (i - 1.5) * 0.2, [cnt[s, c] for c in cats], 0.18,
               color=color(s), label=s)
    a1.set_xticks(x, cats, rotation=25, ha="right")
    a1.set_ylabel("istanze (background: immagini)")
    a1.set_title("Istanze per classe")
    a1.legend()
    for i, s in enumerate(splits[:3]):
        tot = [sum(cnt[t, c] for t in splits[:3]) or 1 for c in cats]
        a2.bar(x + (i - 1) * 0.25, [100 * cnt[s, c] / t for c, t in zip(cats, tot)],
               0.23, color=color(s), label=s)
        a2.axhline(100 * SPLITS[s], color=color(s), ls="--", lw=1)
    a2.set_xticks(x, cats, rotation=25, ha="right")
    a2.set_ylabel("% della classe nello split")
    a2.set_title("Quota per split (- - target)")
    save(fig, rep / "classes_per_split")
    print(f"report: {rep}")
    for s in splits:
        print(f"  {s:5s} {images[s]:6d} immagini, {empty[s]} senza oggetti")
    if not rows:  # sottoinsieme di solo background: niente box ne' strati
        return

    # 3. box per split, come labels.jpg di Ultralytics: un campione di box
    # centrati, densita' dei centri, densita' di larghezza x altezza (log: i
    # box sono di poche decine di pixel su 640)
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 0.85 * WIDTH))
    wh_bins = np.logspace(np.log10(4), np.log10(W), 40)
    for (a0, a1, a2), s in zip(axes, SPLITS):
        b = np.array([r[2:] for r in rows if r[0] == s])
        for x, y, w, h in b[rng.choice(len(b), min(400, len(b)), replace=False)]:
            a0.add_patch(Rectangle((W / 2 - w / 2, H / 2 - h / 2), w, h, fill=False,
                                   lw=0.4, alpha=0.3, color=color(s)))
        a0.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        a0.set_ylabel(f"{s}\n{len(b)} box")
        a1.hist2d(b[:, 0], b[:, 1], bins=(32, 26), range=((0, W), (0, H)),
                  cmin=1, norm="log")
        a1.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        a2.hist2d(b[:, 2], b[:, 3], bins=(wh_bins, wh_bins), cmin=1, norm="log")
        a2.plot([4, W], [4, W], "--", color=GRAY, lw=0.8)  # box quadrati
        a2.set(xscale="log", yscale="log", aspect="equal")
        for ax in (a0, a1, a2):
            ax.grid(False)
        for ax in (a0, a1):  # bordo intero: e' il frame 640x512
            ax.spines[:].set_visible(True)
    axes[0, 0].set_title("box (campione, centrati) [px]")
    axes[0, 1].set_title("centri [px]")
    axes[0, 2].set_title("larghezza × altezza [px]")
    axes[2, 0].set_xlabel("x")
    axes[2, 1].set_xlabel("x")
    axes[2, 2].set_xlabel("w")
    for r in range(3):
        axes[r, 1].set_ylabel("y")
        axes[r, 2].set_ylabel("h")
    save(fig, rep / "bbox_per_split")

    if not videos:  # sottoinsiemi per fascia: la tabella e' quella di `all`
        return

    # 4. classe x fascia: quota dei frame di ogni cella per split
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    for ax, s in zip(axes, SPLITS):
        m = np.zeros((len(NAMES), len(SIZES)))
        tot = np.zeros_like(m)
        for v in videos:
            i = NAMES.index(v["class"])
            for z in v["sizes"]:
                if z is not None:
                    tot[i, SIZES.index(z)] += 1
                    m[i, SIZES.index(z)] += v["split"] == s
        pct = 100 * m / np.maximum(tot, 1)
        ax.imshow(pct, vmin=0, vmax=100)
        for i in range(len(NAMES)):
            for j in range(len(SIZES)):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n{int(m[i, j])}", ha="center",
                        va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else "#222")
        ax.set_xticks(range(len(SIZES)), SIZES, rotation=30)
        ax.set_yticks(range(len(NAMES)), NAMES if ax is axes[0] else [])
        ax.grid(False)
        ax.set_title(f"{s} (target {SPLITS[s]:.0%})")
    fig.suptitle("Frame di ogni cella classe × fascia per split (quota e numero)",
                 fontsize=10)
    save(fig, rep / "class_size_per_split")


# --- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "svanstrom",
                   help="dataset finale (local_path in conf/dataset/svanstrom.yaml)")
    p.add_argument("--src", type=Path, default=ROOT / "data" / "svanstrom_src",
                   help="video scaricati e frame per video; fuori da --out "
                        "perche' non va sincronizzato sulle board")
    p.add_argument("--hash-thr", type=int, default=8,
                   help="bit di dHash (su 64) sotto cui due frame sono uguali")
    p.add_argument("--geo-thr", type=float, default=0.05,
                   help="spostamento del centro (frazione del frame) sotto cui "
                        "il box e' fermo")
    p.add_argument("--bg-frac", type=float, default=0.1,
                   help="frame di background per video, in proporzione ai selezionati")
    p.add_argument("--bg-std", type=float, default=8.0,
                   help="deviazione standard massima (livelli di grigio) dello "
                        "sfondo attorno al box per cancellarlo con l'inpainting")
    p.add_argument("--n-calib", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = p.parse_args(argv)

    raw = a.src / "raw" / "Data" / "Video_V"
    dist = download(a.src / "raw")
    jobs = [(raw / f"{v}.mp4", raw / f"{v}_LABELS.mat", a.src / "frames" / v,
             a.hash_thr, a.geo_thr, a.bg_frac, a.bg_std) for v in sorted(dist)]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(process_video, jobs))

    skipped = [r for r in results if "skipped" in r]
    videos = [{**r, "class": r["video"].split("_")[1].lower(), "distance": dist[r["video"]]}
              for r in results if "skipped" not in r]
    assign_splits(videos, a.seed)
    # oltre a `all` (tutto), un dataset per fascia (immagini con oggetti) e uno
    # di solo background, da aggiungere a mano: lo split bilancia anche le
    # fasce e il background segue il video, quindi ognuno resta 70/20/10
    size_of = {f"{v['video']}_f{n:05d}": z
               for v in videos for n, z in zip(v["selected"], v["sizes"])}
    subsets = {"all": lambda st: True, "background": lambda st: st.endswith("_bg")} | {
        z: (lambda st, z=z: size_of.get(st) == z) for z in SIZES}
    for name, keep in subsets.items():
        build(a.out / name, a.src / "frames", videos, a.n_calib, a.seed, keep)

    sel = sum(len(v["selected"]) for v in videos)
    bg = sum(len(v["background"]) for v in videos)
    tot = sum(v["frames"] for v in videos)
    (a.out / "manifest.json").write_text(json.dumps({
        "source": f"https://github.com/{REPO}", "sensor": "visible",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "selected_frames": sel, "background_frames": bg, "total_frames": tot,
        "skipped": skipped, "videos": videos}, indent=1))
    print(f"video: {len(videos)} usati, {len(skipped)} scartati; "
          f"frame: {sel} selezionati su {tot} ({sel / max(tot, 1):.1%}), "
          f"{bg} di background con inpainting")
    for s in skipped:
        print(f"  scartato {s['video']}: {s['skipped']}")
    for name in subsets:
        report(a.out / name, videos if name == "all" else [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
