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


def forward(weights, batch, device):
    import torch
    from ultralytics.nn.autobackend import AutoBackend

    model = AutoBackend(weights, device=torch.device(device), fp16=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(batch.shape[0]):
            y = model(torch.from_numpy(batch[i:i + 1]).to(device))
            outs.append(y[0] if isinstance(y, (list, tuple)) else y)
    return [o.detach().float().cpu().numpy() for o in outs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="modello FP32 di riferimento")
    ap.add_argument("--target", required=True, help="artefatto da validare")
    ap.add_argument("--images", required=True, help="directory delle immagini")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ref-device", default=None)
    args = ap.parse_args()

    import numpy as np

    batch, files = fixed_batch(args.images, args.imgsz, args.n)
    ref = forward(args.ref, batch, args.ref_device or args.device)
    tgt = forward(args.target, batch, args.device)

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
