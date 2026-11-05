"""Log the step-2 LUT to wandb (entity/project yolo-bench, as pruning/separability run 6lhj71tw)."""
import json
from pathlib import Path

import wandb

OUT = Path(__file__).resolve().parent
s = json.loads((OUT / "results" / "summary.json").read_text())
env = json.loads((OUT / "results" / "env.json").read_text())
run = wandb.init(entity="yolo-bench", project="yolo-bench", job_type="lut_pi",
                 name="lut_pi_passo2_rpi5_openvino_fp16_cpu4_max",
                 tags=["lut", "rpi5", "openvino", "fp16", "cpu_4", "yolo26n"],
                 config={"board": "rpi5", "backend": "openvino", "infer_precision": "f16", "compute_target": "cpu_4",
                         "freq_target": "max", "niter": 200, "repeats": 3, "seed": 0, "step": 2, "fan": False,
                         "net": "conv + bias + SiLU (Sigmoid*Mul), random weights", **env})
run.summary.update({k: v for k, v in s.items() if k != "families"})
for f, d in s["families"].items():
    run.summary.update({f"{f}/{k}": v for k, v in d.items() if k != "anomalous"})
for png in sorted((OUT / "figures").glob("*.png")):
    run.log({png.stem: wandb.Image(str(png))})
art = wandb.Artifact("lut_pi_passo2_rpi5", type="dataset")
art.add_dir(str(OUT / "results"), name="results")
art.add_dir(str(OUT / "figures"), name="figures")
for f in (*OUT.glob("*.py"), OUT / "launch.sh", OUT / "README.md"):
    if f.exists():
        art.add_file(str(f), name=f"scripts/{f.name}")
run.log_artifact(art)
run.finish()
print(run.url)
