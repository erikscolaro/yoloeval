#!/usr/bin/env python3
"""Confronto numerico fra artefatto esportato e riferimento FP32.

Gira dove vive l'artefatto: un engine TensorRT o un modello Axelera non sono
caricabili sulla workstation.

Due modelli, stesso batch fisso di immagini, stesso preprocessing: si
confrontano i tensori grezzi in uscita. Un modello INT8 mal calibrato e'
velocissimo e predice rumore — senza questo controllo comparirebbe in tabella
come il risultato migliore.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#: default di Ultralytics (cfg/default.yaml): conf di `predict` e iou
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


def letterbox(path, imgsz):
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype("float32") / 255.0
    return np.ascontiguousarray(x)


def fixed_batch(images_dir, imgsz, n):
    """Stesse immagini, stesso ordine, in ogni confronto."""
    import numpy as np

    files = sorted(
        p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )[:n]
    if not files:
        raise FileNotFoundError(f"nessuna immagine sotto {images_dir}")
    return np.stack([letterbox(f, imgsz) for f in files]), [str(f) for f in files]


def forward(weights, batch, device, end2end=None):
    import torch
    from ultralytics.nn.autobackend import AutoBackend

    kwargs = {} if end2end is None else {"end2end": end2end}
    model = AutoBackend(weights, device=torch.device(device), fp16=False, **kwargs)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(batch.shape[0]):
            y = model(torch.from_numpy(batch[i:i + 1]).to(device))
            outs.append(y[0] if isinstance(y, (list, tuple)) else y)
    return [o.detach().float().cpu().numpy() for o in outs]


def is_end2end(out) -> bool:
    """Testa one-to-one: [1, max_det, 6] = x1, y1, x2, y2, score, classe."""
    return out.ndim == 3 and out.shape[-1] == 6


def box_iou(a, b):
    import numpy as np

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def compare_detections(ref, tgt, conf, iou):
    """Confronto detection per detection per la testa one-to-one.

    L'output e' una lista di `max_det` detection ordinate per score: sotto la
    soglia ci sono centinaia di candidati quasi a zero, e basta un'inezia per
    cambiarne l'ordine. Un confronto elemento per elemento misurerebbe quel
    rimescolamento, non la quantizzazione. Qui si tengono le detection sopra
    `conf` e si accoppiano per classe e IoU >= `iou`, in ordine di IoU.
    """
    import numpy as np

    n_ref = n_tgt = n_matched = 0
    box_diffs, score_diffs, ious = [], [], []
    for r, t in zip(ref, tgt, strict=True):
        r = r.reshape(-1, 6)
        t = t.reshape(-1, 6)
        r = r[r[:, 4] >= conf]
        t = t[t[:, 4] >= conf]
        n_ref += len(r)
        n_tgt += len(t)
        if not len(r) or not len(t):
            continue
        m = box_iou(r, t)
        m[r[:, 5][:, None] != t[:, 5][None, :]] = 0.0
        used_r, used_t = set(), set()
        for i, j in zip(*np.unravel_index(np.argsort(-m, axis=None), m.shape)):
            if m[i, j] < iou:
                break
            if i in used_r or j in used_t:
                continue
            used_r.add(i)
            used_t.add(j)
            box_diffs.append(float(np.abs(r[i, :4] - t[j, :4]).max()))
            score_diffs.append(float(abs(r[i, 4] - t[j, 4])))
            ious.append(float(m[i, j]))
        n_matched += len(used_r)

    total = max(n_ref, n_tgt)
    return {
        "comparison": "detections",
        "conf": conf,
        "iou": iou,
        "n_ref": n_ref,
        "n_target": n_tgt,
        "n_matched": n_matched,
        # nessuna detection da nessuna delle due parti: concordano
        "match_rate": n_matched / total if total else 1.0,
        "max_box_diff_px": max(box_diffs, default=0.0),
        "max_score_diff": max(score_diffs, default=0.0),
        "mean_iou": float(np.mean(ious)) if ious else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="modello FP32 di riferimento")
    ap.add_argument("--target", required=True, help="artefatto da validare")
    ap.add_argument("--images", required=True, help="directory delle immagini")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ref-device", default=None)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    args = ap.parse_args()

    import numpy as np

    batch, files = fixed_batch(args.images, args.imgsz, args.n)
    tgt = forward(args.target, batch, args.device)
    # Un `.pt` sceglie la testa a runtime e di default usa la one-to-many:
    # va allineato a quella che l'export ha messo nell'artefatto, altrimenti
    # il confronto e' fra due teste diverse e non misura la quantizzazione.
    end2end = None
    if Path(args.ref).suffix == ".pt":
        end2end = tgt[0].ndim == 3 and tgt[0].shape[-1] == 6
    ref = forward(args.ref, batch, args.ref_device or args.device, end2end)

    payload = {
        "n_images": len(files),
        "images": files,
        "ref_shape": list(ref[0].shape),
        "target_shape": list(tgt[0].shape),
    }

    if ref[0].shape != tgt[0].shape:
        # Non e' un errore di calibrazione: e' un'altra testa. Va detto, non
        # nascosto dietro una differenza numerica enorme.
        payload["status"] = "shape_mismatch"
        payload["max_abs_diff_vs_fp32"] = None
        print("YOLOBENCH_JSON " + json.dumps(payload))
        return 0

    if is_end2end(ref[0]):
        payload["status"] = "computed"
        payload.update(compare_detections(ref, tgt, args.conf, args.iou))
        print("YOLOBENCH_JSON " + json.dumps(payload))
        return 0

    diffs = [np.abs(a - b) for a, b in zip(ref, tgt, strict=True)]
    flat_ref = np.concatenate([a.ravel() for a in ref])
    flat_tgt = np.concatenate([b.ravel() for b in tgt])
    denom = float(np.linalg.norm(flat_ref) * np.linalg.norm(flat_tgt)) or 1.0
    payload.update(
        {
            "status": "computed",
            "max_abs_diff_vs_fp32": float(max(d.max() for d in diffs)),
            "mean_abs_diff_vs_fp32": float(np.mean([d.mean() for d in diffs])),
            "cosine_similarity": float(np.dot(flat_ref, flat_tgt) / denom),
            "ref_abs_max": float(np.abs(flat_ref).max()),
        }
    )
    print("YOLOBENCH_JSON " + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
