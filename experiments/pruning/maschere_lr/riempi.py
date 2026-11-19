"""Riempie il budget dopo la ricerca: dalle maschere di un checkpoint della ricerca riaccende i blocchi spenti, dal
piu' vicino alla soglia (|alpha| piu' alto) in giu', finche' la latenza stimata resta entro il target. Scrive pruned.pt
e meta.json in <out>, con search_key <chiave>, cosi' stage=finetune lo trova come una ricerca qualsiasi.

    PYTHONPATH=<yolopit >= 5176375> .venv/bin/python riempi.py <search_dir> <epochN.pt> <target_ms> <out_dir> <chiave>
"""
import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from yolopit import PITYOLO
from yolopit.masks import PITBlockFeaturesMasker

search, ckpt, target, out, chiave = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
pit_args = yaml.safe_load((search / "run/pit_args.yaml").read_text())
meta = json.loads((search / "meta.json").read_text())
model = json.loads((search / "run/pit_summary.json").read_text())["model"]   # pesi non potati di partenza
y = PITYOLO(model, cfg={"pit": {k: v for k, v in pit_args["pit"].items() if k != "warmup_epochs"}})
y.load_search_checkpoint(search / "run/weights" / ckpt)
w = y.model
lat = lambda: float(w.real_costs()["latency"])
l0 = lat()
maskers = [(n, m) for n, m in w.net.seed.named_modules() if isinstance(m, PITBlockFeaturesMasker)]
# candidati: blocchi pieni spenti (non il keep-alive, ultimo pieno; non il resto, legato al minimo dei pieni)
cand = []
for n, m in maskers:
    full = m.block[:-1] if m.ordered_remainder else m.block
    for i in range(len(full) - 1):
        if full[i].abs() <= 0.5:
            cand.append((float(full[i].abs()), n, m, i))
cand.sort(key=lambda c: -c[0])
accesi = []
with torch.no_grad():
    for a, n, m, i in cand:
        old = m.block[i].clone()
        m.block[i] = 0.501 if old >= 0 else -0.501
        l = lat()
        if l > target:
            m.block[i] = old
        else:
            accesi.append({"layer": n, "blocco": i, "alpha": round(a, 4), "latenza_ms": round(l, 3)})
l1 = lat()
print(f"latenza stimata {l0:.2f} -> {l1:.2f} ms (target {target}), blocchi riaccesi {len(accesi)}/{len(cand)}")
assert l1 <= target + 1e-6
out.mkdir(parents=True, exist_ok=True)
y.export_pruned(path=out / "pruned.pt")
meta2 = copy.deepcopy(meta)
meta2["search_key"] = chiave
meta2["riempi"] = {"da": search.name, "checkpoint": ckpt, "target_ms": target, "latenza_prima_ms": l0,
                   "latenza_dopo_ms": l1, "riaccesi": accesi}
(out / "meta.json").write_text(json.dumps(meta2, indent=1))
print("scritto", out)
