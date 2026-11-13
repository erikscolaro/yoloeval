"""Dati della regola di scelta del modello di partenza (NEW_MASTERPLAN 2c-2d) e la regola stessa.

Latenza delle reti intere a 640 (B4-R, pavimento_s, piu' un csv di misure), mAP dei training e300, pavimento
(rete potata al massimo) stimato con LUT + fit o misurato. Li usano stage=select e tools/figure_guadagno_latenza.py.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/pruning"
LETTERA = {0.0625: "F", 0.09375: "FP", 0.125: "P", 0.1875: "PN", 0.25: "N", 0.375: "NS", 0.5: "S", 0.75: "SM", 1.0: "M"}
SEGNO = {"D101": "-", "D126": "", "D170": "+"}
PROTETTI = ("8.cv2", "9.cv2", "16.cv2", "19.cv2", "22.cv1b", "22.m.0.0.cv2", "22.cv2")  # alimentano C2PSA/PSABlock/Detect
SIZE = {"n": 0.25, "p": 0.125, "s": 0.5}  # yolo26n/p/s_D126: i primi modelli, prima della griglia yolo26w<larghezza>


def arch_di(w, d):
    return f"w{w:g}_{d}"


def sigla(arch):
    w, d = re.fullmatch(r"w([\d.]+)_(D\d+)", arch).groups()
    return LETTERA[float(w)] + SEGNO[d]


def righe(p):
    return list(csv.DictReader(open(p)))


def mediane(p):
    acc = defaultdict(list)
    for r in righe(p):
        acc[r["net"]].append(float(r["ms"]))
    return {k: float(np.median(v)) for k, v in acc.items()}


def map_modelli(weights):
    """{arch: [(mAP50, mAP50-95), ...]} dai meta.json delle ricette e300 (un elemento per seme)."""
    out = defaultdict(list)
    for d in sorted(Path(weights).glob("yolo26*_e300_*")):
        m = re.match(r"yolo26(?:w([\d.]+)|([nps]))_(D\d+)_", d.name)
        met = json.load(open(d / "meta.json")).get("metrics") if (d / "meta.json").exists() else None
        if m and met:
            out[arch_di(float(m[1]) if m[1] else SIZE[m[2]], m[3])].append((met["map50"], met["map50_95"]))
    return out


def pavimenti(layers_csv, nano_csv, fit, lut_dir, archs):
    """{arch: ms} del pavimento stimato con C_arch + sum a_f LUT_f. Lo schema dei canali e' quello di nano_floor (74 conv):
    un conv protetto resta pieno (c_out dell'architettura), gli altri a 2 canali; i nomi delle conv extra di D170
    (m.1, m.2) si leggono come m.0, e un ingresso a piu' pezzi da 2 canali (concat) scala col numero di pezzi."""
    from yolopit.latency import FamilyLUT

    fit = json.load(open(fit))
    a = {f: v for k, v in fit["a"].items() for f in k.split("+")}
    nano = [r for r in righe(nano_csv) if r["net"] in ("nano_floor", "nano_full")]
    fl = {r["layer"]: (int(r["c_in"]), int(r["c_out"])) for r in nano if r["net"] == "nano_floor"}
    fu = {r["layer"]: int(r["c_in"]) for r in nano if r["net"] == "nano_full"}
    lut, tot = {}, {}
    full = defaultdict(list)
    for r in righe(layers_csv):
        if r["net"].endswith("_full") and r["net"][:-5] in archs:
            full[r["net"][:-5]].append(r)
    for arch, rs in full.items():
        w = float(re.match(r"w([\d.]+)_", arch)[1])
        s = w / 0.25
        t = fit["C_ms"][arch]
        for r in rs:
            ref = re.sub(r"(layers\.\d+\.m\.)\d+", r"\g<1>0", r["layer"])
            ci_ref, _ = fl[ref]
            ci, co = int(r["c_in"]), int(r["c_out"])
            prot = r["layer"].endswith(tuple(f"layers.{p}.conv" for p in PROTETTI))
            if ci_ref >= 64:   # ingresso con canali protetti: scalano con la larghezza, i pezzi da 2 restano (258 = 256 + 2)
                ci_f = round((ci_ref - ci_ref % 64) * s) + ci_ref % 64
            elif ci_ref % 2:   # immagine in ingresso
                ci_f = ci_ref
            else:              # numero di pezzi da 2 canali: scala come l'ingresso pieno
                ci_f = round(ci_ref * ci / (fu[ref] * s))
            f = r["family"]
            if f not in lut:
                lut[f] = FamilyLUT(f, Path(lut_dir) / f"lut_{f}.csv")
            t += a[f] * float(lut[f](ci_f, co if prot else 2))
        tot[arch] = t
    return tot




def dati(lut_dir, latency_csv=None, weights=ROOT / "artifacts/weights",
         b4r=EXP / "b4r_composizione/results", pav=EXP / "pavimento_s/results"):
    """(lat, mAP, floor) per le architetture con latenza e training: lat {arch: ms} della rete intera,
    mAP {arch: (mAP50, mAP50-95)} media sui seed, floor {arch: (ms, misurato)}."""
    b4r, pav = Path(b4r), Path(pav)
    lat = {r["arch"]: float(r["measured_ms"]) for r in righe(b4r / "predictions.csv") if r["net"].endswith("_full")}
    nano = mediane(pav / "raw.csv")  # reti di pavimento_s: s_full, s_floor, nano_floor, nano_full
    lat["w0.5_D126"] = nano["s_full"]
    mis = {"w0.25_D126": nano["nano_floor"], "w0.5_D126": nano["s_floor"]}  # pavimenti misurati
    if latency_csv:
        for r in righe(latency_csv):
            a = r.get("arch") or r.get("model") or arch_di(float(r["width"]), r["depth"])   # o width,depth (benchmark)
            lat[re.sub(r"_full$", "", a)] = float(r.get("ms") or r.get("measured_ms") or r["median_ms"])
    stim = pavimenti(b4r / "layers.csv", pav / "layers.csv", Path(lut_dir) / "fit.json", lut_dir, lat)
    mp = map_modelli(weights)
    mAP = {a: tuple(np.mean(mp[a], 0)) for a in lat if a in mp}
    floor = {a: (mis[a], True) if a in mis else (stim[a], False) for a in mAP if a in mis or a in stim}
    return lat, mAP, floor


def scegli(lat, mAP, floor, budget, floor_ratio, tie, seed_noise, start_model=None):
    """Regola 2c-2d sulla mAP50-95 ({arch: valore}). Ritorna un dict con caso, ottimo, tetto e modello di partenza.

    ottimo: la mAP migliore fra le reti intere nel budget; a pari merito (sotto `tie`) la piu' veloce.
    tetto: la mAP migliore senza vincoli. Caso libero se il tetto sta nel budget o perde meno di `tie` dall'ottimo:
    si usa l'ottimo, niente potatura. Caso stretto: si pota il piu' preciso fra quelli con pavimento <= floor_ratio*B;
    a pari merito (sotto `seed_noise`) quello con il pavimento piu' basso. `start_model` forza la partenza."""
    def migliore(cs, poi, soglia):
        if not cs:
            return None
        top = max(mAP[a] for a in cs)
        return min((a for a in cs if top - mAP[a] < soglia), key=poi)

    limite = floor_ratio * budget
    ottimo = migliore([a for a in mAP if lat[a] <= budget], lat.get, tie)
    tetto = max(mAP, key=mAP.get)
    out = {"budget_ms": budget, "limite_pavimento_ms": limite, "ottimo": ottimo, "tetto": tetto}
    if start_model:
        ok = start_model in floor and floor[start_model][0] <= limite
        return out | {"caso": "manuale", "partenza": start_model,
                      "avviso": None if ok else f"{start_model}: pavimento sopra {limite:.1f} ms o non noto"}
    if ottimo and (lat[tetto] <= budget or mAP[tetto] - mAP[ottimo] < tie):
        return out | {"caso": "libero", "partenza": None, "avviso": None}
    passano = [a for a in floor if floor[a][0] <= limite]
    return out | {"caso": "stretto", "partenza": migliore(passano, lambda a: floor[a][0], seed_noise),
                  "avviso": None if passano else "nessun modello passa il test del pavimento"}


if __name__ == "__main__":
    # python -m src.selection results/select/<profilo>.json -> override del modello di partenza, per la catena guidata
    import sys
    r = json.load(open(sys.argv[1]))
    if not r["partenza"]:
        sys.exit(f"select, caso {r['caso']}: nessun modello da potare (ottimo {r['ottimo']}); {r.get('avviso') or ''}")
    w, d = re.fullmatch(r"w([\d.]+)_(D\d+)", r["partenza"]).groups()
    print(f"model=yolo26 model.width={w} model.depth={d}")
