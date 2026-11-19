"""Video completi di val e test per tarare e provare il tracker, senza annotare niente: le clip Svanström e i video
Anti-UAV300 RGB dei gruppi di val e test di VEDETTA (manifest del dataset), quindi niente leakage con il training.

- Svanström: mp4 dal repository GitHub originale, tutti i frame della clip, label rifatte e riviste in CVAT
  (vedetta/data/input/svanstrom/4_box/manuale.tar.gz, una per frame, anche prima della deduplicazione).
- Anti-UAV300: frame dal mirror Hugging Face (licenza MIT), label originali per frame (un drone o nessuno).

Uscita: data/tracking/<split>/<sequenza>/{images,labels}/<frame>.jpg|txt (label YOLO con le classi di yoloeval:
drone 0, airplane 1, helicopter 2, bird 3) e seq.json (fonte, classe, split, gruppo, fps, frame). Idempotente.

    python experiments/tracking/video_val_test/scarica.py [--out data/tracking] [--workers 8]
"""
import argparse
import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VEDETTA = Path.home() / "Desktop/vedetta"
MANIFEST = ROOT / "data/air-detection/all/manifest.csv"
SV_RAW = "https://raw.githubusercontent.com/DroneDetectionThesis/Drone-detection-dataset/master/Data/Video_V/"
SV_LABELS = VEDETTA / "data/input/svanstrom/4_box/manuale.tar.gz"
AU = "https://huggingface.co/datasets/DestinyLove/AntiUAV300/resolve/main/yolo_rgb/"
VEDETTA_A_NOI = {0: 1, 1: 3, 2: 0, 3: 2}   # airplane, bird, drone, helicopter -> drone, airplane, helicopter, bird
FPS = {"svanstrom": 30, "anti_uav": 25}


def get(url, dst=None, tries=8):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
            if dst is None:
                return b
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(b)
            tmp.rename(dst)
            return True
        except OSError:
            if k == tries - 1:
                return False if dst else None
            time.sleep(5 * (k + 1))


def sequenze():
    m = pd.read_csv(MANIFEST)
    m = m[(m.tipo == "reale") & m.fonte.isin(["svanstrom", "anti_uav"]) & m.split.isin(["val", "test"])]
    m = m.assign(clip=m["clip"].astype(str))
    return m.groupby("clip").agg(fonte=("fonte", "first"), split=("split", "first"), gruppo=("gruppo", "first"),
                                 n_split=("split", "nunique")).reset_index()


def svanstrom(seq, out, labels):
    """Una clip (video/sessione): i frame che hanno una label rifatta, con le classi convertite."""
    video, ses = seq.clip.split("/")
    d = out / seq.split / f"{video}_{ses}"
    if (d / "seq.json").exists():
        return d.name, "gia' fatta"
    tutte = labels.get(f"{video}_{ses}", {})
    if not tutte:
        return d.name, "senza label rifatte"
    mp4 = out / "_mp4" / f"{video}.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    if not mp4.exists() and not get(SV_RAW + f"{video}.mp4", mp4):
        return d.name, "download fallito"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    cap, i, n = cv2.VideoCapture(str(mp4)), 0, 0
    while (f := cap.read()[1]) is not None:
        if i in tutte:
            cv2.imwrite(str(d / "images" / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            righe = [r.split() for r in tutte[i].splitlines() if r.strip()]
            (d / "labels" / f"{i:06d}.txt").write_text(
                "".join(f"{VEDETTA_A_NOI[int(r[0])]} {' '.join(r[1:5])}\n" for r in righe))
            n += 1
        i += 1
    meta = {"fonte": "svanstrom", "classe": video.split("_")[1].lower(), "split": seq.split, "gruppo": seq.gruppo,
            "fps": FPS["svanstrom"], "frame": n, "video": video, "sessione": ses}
    (d / "seq.json").write_text(json.dumps(meta, indent=1))
    return d.name, f"{n} frame"


def anti_uav(seq, out, righe, workers):
    video = seq.clip.split("/")[0]          # AUAV_<id>
    d = out / seq.split / video
    if (d / "seq.json").exists():
        return d.name, "gia' fatta"
    mie = righe.get(video[5:], [])
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    jobs = []
    for f, lab in mie:
        i = int(Path(f).stem.rsplit("_", 1)[1])
        (d / "labels" / f"{i:06d}.txt").write_text(f"{lab}\n" if lab.strip() else "")   # classe 0 = drone, come noi
        if not (d / "images" / f"{i:06d}.jpg").exists():
            jobs.append((AU + "images/" + f.replace(".txt", ".jpg"), d / "images" / f"{i:06d}.jpg"))
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda j: get(*j), jobs))
    for (_, img), o in zip(jobs, ok):        # mancante sul mirror: via anche la label
        if not o:
            (d / "labels" / f"{img.stem}.txt").unlink(missing_ok=True)
    n = len(list((d / "images").glob("*.jpg")))
    meta = {"fonte": "anti_uav", "classe": "drone", "split": seq.split, "gruppo": seq.gruppo, "fps": FPS["anti_uav"],
            "frame": n, "video": video, "mancanti": len(ok) - sum(ok)}
    (d / "seq.json").write_text(json.dumps(meta, indent=1))
    return d.name, f"{n} frame, {len(ok) - sum(ok)} mancanti"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data/tracking")), p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    S = sequenze()
    assert (S.n_split == 1).all(), "una clip in due split"
    print(S.groupby(["fonte", "split"]).size().to_string(), flush=True)

    labels = {}
    with tarfile.open(SV_LABELS) as t:
        for mem in t.getmembers():
            if mem.isfile() and mem.name.endswith(".txt"):
                clip, nome = mem.name.split("/")[-2:]
                labels.setdefault(clip, {})[int(nome.rsplit("_f", 1)[1][:-4])] = t.extractfile(mem).read().decode()
    for s in S[S.fonte == "svanstrom"].itertuples():
        print("svanstrom", *svanstrom(s, out, labels), flush=True)

    righe = {}
    for ln in get(AU + "all_labels.txt").decode().splitlines():
        f, lab = ln.split("|||")
        righe.setdefault(Path(f).stem.rsplit("_", 1)[0], []).append((f, lab))
    for s in S[S.fonte == "anti_uav"].drop_duplicates(S.columns[0]).itertuples():
        print("anti_uav", *anti_uav(s, out, righe, a.workers), flush=True)


if __name__ == "__main__":
    main()
