"""Traccia i canali tenuti a ogni epoca delle ricerche del livello 1: legge 'channels' da weights/last.pt (riscritto a ogni
epoca) e accoda una riga a results/maschere_<stima>.jsonl. Serve a vedere quanto cambiano le maschere dopo l'arrivo al target.

    CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/pruning/pendenze/livello1/traccia_maschere.py   # finche' gira pendenze-l1
"""
import json, re, subprocess, time
from pathlib import Path
import torch

D = Path(__file__).parent
seen = {}
while True:
    for e in ("smooth", "best", "lut"):
        log = D / "runs" / f"{e}_sgd.log"
        m = re.search(r"artifacts/search/[^ ,/]+", log.read_text(errors="ignore")) if log.exists() else None
        pt = Path(m.group(0)) / "run/weights/last.pt" if m else None
        if not pt or not pt.exists() or seen.get(e) == pt.stat().st_mtime:
            continue
        try:
            ck = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception:      # file a meta' scrittura: si riprova al giro dopo
            continue
        seen[e] = pt.stat().st_mtime
        row = {"epoch": ck["epoch"] + 1, "latency_ms": ck["costs"]["latency"],
               "channels": [[n, int(c0), int(c)] for n, c0, c, pot in ck["channels"] if pot]}
        with open(D / "results" / f"maschere_{e}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    if subprocess.run(["systemctl", "--user", "is-active", "-q", "pendenze-l1"]).returncode:
        break
    time.sleep(20)
