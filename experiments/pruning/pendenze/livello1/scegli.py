"""Classifica delle pendenze del livello 1 dai training di dopo.sh: mAP50-95 dell'epoca migliore, media sui seed.
TAG=ft10 (default) il finetuning corto, TAG=ft100 quello intero, TAG=scratch da zero (300 epoche). La scelta si fa su
scratch: misura l'architettura da sola, senza i pesi usciti dalla ricerca (Liu et al. 2019; DARTS, FBNet, ProxylessNAS
riaddestrano da zero); il finetuning resta come dato secondario.
Regole (README): fuori chi misura sul Pi piu' del target con il margine della ricerca (5% della parte PIT: 28,85 ms);
a meno di 0,4 punti dalla migliore vince smooth.

    .venv/bin/python experiments/pruning/pendenze/livello1/scegli.py            # tabella
    .venv/bin/python experiments/pruning/pendenze/livello1/scegli.py top2       # le due migliori (per il secondo seed)
    .venv/bin/python experiments/pruning/pendenze/livello1/scegli.py vincitore
"""
import csv, json, os, re, sys
from pathlib import Path

D = Path(__file__).parent / "runs"
ROOT = Path(__file__).parents[4]
TAG = os.environ.get("TAG", "ft10")
LIMITE_MS = 28.0 + 0.05 * 17.0     # target + margine di yolopit sulla parte PIT (17 ms)
rows = {}
for log in sorted(D.glob(f"{TAG}_*.log")):
    if not re.fullmatch(TAG + r"(_s\d+)?_[a-z]+", log.stem):
        continue
    e = log.stem.split("_")[-1]
    testo = log.read_text(errors="ignore")
    if not re.search(r"fine-tuning (completato|gia' in cache)", testo):     # ancora in corso o fallito
        continue
    m = re.search(r"(?:artifacts/finetune/|in cache: |completato: )([^\s,/]+)", testo)
    res = ROOT / "artifacts/finetune" / m.group(1) / "run/results.csv" if m else None
    if not res or not res.exists():
        continue
    best = max(float({k.strip(): v for k, v in r.items()}["metrics/mAP50-95(B)"]) for r in csv.DictReader(open(res)))
    pi = D / f"pi_{log.name}"
    rec = re.findall(r"\b([0-9a-f]{12})\b (?:non conforme|gia' completata)", pi.read_text(errors="ignore")) if pi.exists() else []
    ms = json.load(open(ROOT / "results" / f"{rec[-1]}.json"))["latency"]["median_ms"] if rec else None
    rows.setdefault(e, []).append((best, ms))
tab = {e: (sum(a for a, _ in v) / len(v), max((m for _, m in v if m), default=None), len(v)) for e, v in rows.items()}
ok = sorted((e for e, (a, ms, n) in tab.items() if ms is None or ms <= LIMITE_MS), key=lambda e: -tab[e][0])
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "top2":
    print(" ".join(ok[:2]))
elif mode == "vincitore":
    w = ok[0] if ok else "smooth"
    if "smooth" in ok and tab[w][0] - tab["smooth"][0] < 0.004:
        w = "smooth"
    print(w)
else:
    for e in sorted(tab, key=lambda e: -tab[e][0]):
        a, ms, n = tab[e]
        print(f"{e:8s} mAP50-95 {a:.4f}  Pi {ms} ms  seed {n}  {'' if e in ok else 'FUORI BUDGET'}")
