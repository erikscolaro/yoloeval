"""Log the separability probe to wandb (entity/project from yoloeval conf/logging/wandb.yaml)."""
import json
from pathlib import Path

import wandb

OUT = Path(__file__).resolve().parent
s = json.loads((OUT / "results" / "summary.json").read_text())
run = wandb.init(entity="yolo-bench", project="yolo-bench", job_type="sep_probe",
                 name="sep_probe_rpi5_openvino_fp16_cpu4_max_20261105",
                 tags=["sep_probe", "rpi5", "openvino", "fp16", "cpu_4"],
                 config={"board": "rpi5", "backend": "openvino", "infer_precision": "f16",
                         "compute_target": "cpu_4", "freq_target": "max", "niter": 200,
                         "repeats": 3, "seed": 0, "ref": 64,
                         "families": {"A": "conv1x1 s1 @40x40", "B": "conv3x3 s1 p1 @20x20"}})
for f, d in s.items():
    run.summary.update({f"{f}/{k}": v for k, v in d.items() if not isinstance(v, dict)})
    run.summary.update({f"{f}/minus_identity/{k}": v for k, v in d["minus_identity"].items()})
for png in sorted((OUT / "figures").glob("*.png")):
    run.log({png.stem: wandb.Image(str(png))})
art = wandb.Artifact("sep_probe_rpi5_20261105", type="dataset")
for sub in ("results", "figures"):
    art.add_dir(str(OUT / sub), name=sub)
for py in OUT.glob("*.py"):
    art.add_file(str(py), name=f"scripts/{py.name}")
run.log_artifact(art)
run.finish()
print(run.url)
