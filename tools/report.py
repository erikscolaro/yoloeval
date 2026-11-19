#!/usr/bin/env python3
"""Report dello stato delle misure sul dispositivo del profilo di deploy (default rpi5_640).

    python -m tools.report [--target rpi5_640] [--budget MS] [--no-html]

Una cartella datata `lab/reports/<data_ora>/`, autoconsistente e spostabile (`lab/reports/latest` punta all'ultima):
`report.md` (e `report.html` con pandoc) da `lab/report.md.j2`, le figure in PNG e PDF, le tabelle in CSV e
`data.parquet` con tutte le misure. Sezioni:

1. misure per esito, e le fallite con il motivo;
2. griglia: ogni architettura allenata, latenza fp16 sul dispositivo alla risoluzione di deploy, mAP50-95 del
   training (miglior epoca sul val) e, dove c'e', quella misurata sul dispositivo; chi sta nel budget;
3. reti potate (strategy pit_latency, anche dagli esperimenti): latenza, mAP, differenza con la migliore rete della
   griglia che non e' piu' lenta;
4. altre precisioni e backend rispetto allo stesso modello in fp16;
5. condizioni di misura: throttling, temperatura, ore macchina.

Le mAP che si confrontano sono tutte del training (stesso val, sulla workstation): la griglia non ha la mAP sul
dispositivo per ogni rete.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "lab" / "report.md.j2"
WEIGHTS = ROOT / "artifacts" / "weights"
FINETUNE = ROOT / "artifacts" / "finetune"
#: risultati delle reti potate negli esperimenti (experiments/<area>/<nome>/results)
EXTRA_PRUNED = "experiments/*/*/results"


# ------------------------------------------------------------------ nomi e mAP del training
def arch(model: str) -> tuple[float, str] | None:
    """'yolo26w0.125_D101' -> (0.125, 'D101'); 'yolo26n_D126' -> (0.25, 'D126'); altrimenti None."""
    from src.selection import SIZE
    if m := re.fullmatch(r"yolo26w([\d.]+)_(D\d+)", str(model)):
        return float(m[1]), m[2]
    if (m := re.fullmatch(r"yolo26([nps])_(D\d+)", str(model))) and m[1] in SIZE:
        return SIZE[m[1]], m[2]
    return None


def sigla(model: str) -> str:
    from src.selection import LETTERA, SEGNO
    a = arch(model)
    return LETTERA.get(a[0], str(a[0])) + SEGNO.get(a[1], a[1]) if a else str(model)


def best_map(run: Path) -> float | None:
    """mAP50-95 della miglior epoca di un training Ultralytics (run/results.csv), None se manca."""
    f = run / "results.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    k = next((c for c in rows[0] if c.strip() == "metrics/mAP50-95(B)"), None) if rows else None
    return max(float(r[k]) for r in rows) if k else None


def grid_weights(w: float, d: str) -> Path | None:
    """Il training da 300 epoche dell'architettura (i primi pico, nano e small come in figure_griglia)."""
    from src.selection import SIZE
    old = {(0.125, "D126"): "yolo26p_D126_air_detection_all_e300_59d8534a",
           (0.25, "D126"): "yolo26n_D126_air_detection_all_e300_9dca8d15",
           (0.5, "D126"): "yolo26s_D126_air_detection_all_e300_9ada3b33"}
    if (w, d) in old and (WEIGHTS / old[w, d]).exists():
        return WEIGHTS / old[w, d]
    names = [f"yolo26w{w:g}_{d}"] + [f"yolo26{k}_{d}" for k, v in SIZE.items() if v == w]
    found = [p for n in names for p in sorted(WEIGHTS.glob(f"{n}_air_detection_all_e300_*")) if (p / "meta.json").exists()]
    return found[0] if found else None


# ------------------------------------------------------------------ dati
def load(dirs: list[Path]):
    import pandas as pd

    from .aggregate import load as load_dir
    frames = []
    for d in dirs:
        rows = load_dir(d)
        if rows:
            f = pd.DataFrame(rows)
            f["fonte"] = str(d.relative_to(ROOT) if d.is_absolute() and d.is_relative_to(ROOT) else d)
            frames.append(f)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates("cell_id")


def model_budget(target: str) -> float:
    """target.model_budget_ms del profilo, risolto da Hydra (budget - pre - post)."""
    r = subprocess.run([sys.executable, "run.py", f"target={target}", "--cfg", "job", "--resolve", "-p",
                        "target.model_budget_ms"], cwd=ROOT, capture_output=True, text=True)
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SystemExit(f"budget del profilo {target} non risolto: {r.stderr.strip()[-300:]}") from None


def grid(ok, deploy: str, budget: float):
    """Una riga per architettura: latenza fp16 (mediana delle misure), mAP del training e sul dispositivo."""
    import pandas as pd
    g = ok[(ok["strategy"] == "baseline") & (ok["deploy"] == deploy) & (ok["quantization"] == "fp16")
           & (ok["backend"] == "openvino")]
    g = g[g["model"].map(arch).notna()]
    rows = []
    for (w, d), grp in g.groupby(g["model"].map(arch)):
        wd = grid_weights(w, d)
        m = grp["acc_map50_95"].dropna()
        rows.append({"rete": sigla(grp["model"].iloc[0]), "larghezza": w, "profondita": d,
                     "latenza_ms": round(float(grp["lat_median_ms"].median()), 2),
                     "mAP50_95": round(best_map(wd / "run"), 4) if wd and best_map(wd / "run") else None,
                     "mAP50_95_disp": round(float(m.iloc[-1]), 4) if len(m) else None,
                     "misure": len(grp)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["nel_budget"] = (t["latenza_ms"] <= budget).map({True: "sì", False: ""})
    return t.sort_values(["profondita", "larghezza"]).reset_index(drop=True)


def pruned(ok, grid_t):
    """Reti potate: latenza e mAP, confronto con la migliore della griglia entro la stessa latenza."""
    import pandas as pd
    p = ok[ok["strategy"] == "pit_latency"].sort_values("timestamp").drop_duplicates("weights_key", keep="last")
    rows = []
    for _, r in p.iterrows():
        ft = next(iter(sorted(FINETUNE.glob(f"*_{r['weights_key']}"))), None)
        m_train = best_map(ft / "run") if ft else None
        ref = grid_t[(grid_t["latenza_ms"] <= r["lat_median_ms"]) & grid_t["mAP50_95"].notna()] if len(grid_t) else grid_t
        top = ref.loc[ref["mAP50_95"].idxmax()] if len(ref) else None
        rows.append({"partenza": sigla(r["model"]), "finetune": r["weights_key"],
                     "latenza_ms": round(float(r["lat_median_ms"]), 2),
                     "mAP50_95": round(m_train, 4) if m_train else None,
                     "mAP50_95_disp": round(float(r["acc_map50_95"]), 4) if pd.notna(r.get("acc_map50_95")) else None,
                     "griglia_pari": f"{top['rete']} ({top['latenza_ms']} ms)" if top is not None else None,
                     "delta_mAP": round(m_train - float(top["mAP50_95"]), 4) if m_train and top is not None else None,
                     "fonte": r["fonte"], "data": str(r["timestamp"])[:10]})
    return pd.DataFrame(rows).sort_values("latenza_ms").reset_index(drop=True) if rows else pd.DataFrame()


def variants(ok, deploy: str):
    """Precisioni e backend diversi da fp16/openvino, rispetto allo stesso modello e pesi in fp16 openvino."""
    import pandas as pd
    d = ok[ok["deploy"] == deploy]
    ref = d[(d["quantization"] == "fp16") & (d["backend"] == "openvino")].groupby("weights_key")["lat_median_ms"].median()
    v = d[~((d["quantization"] == "fp16") & (d["backend"] == "openvino"))]
    rows = [{"rete": sigla(r["model"]), "strategia": r["strategy"], "precisione": r["quantization"],
             "backend": r["backend"], "latenza_ms": round(float(r["lat_median_ms"]), 2),
             "fp16_ms": round(float(ref[r["weights_key"]]), 2) if r["weights_key"] in ref else None,
             "mAP50_95_disp": round(float(r["acc_map50_95"]), 4) if pd.notna(r.get("acc_map50_95")) else None,
             "fonte": r["fonte"]} for _, r in v.iterrows()]
    t = pd.DataFrame(rows)
    if not t.empty:
        t["rapporto"] = (t["latenza_ms"] / t["fp16_ms"]).round(2)
    return t


# ------------------------------------------------------------------ figure
def figures(grid_t, pruned_t, budget: float, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    from .plots import GRAY, STYLE, WIDTH, color, save
    plt.style.use(STYLE)
    made = {}
    if grid_t.empty:
        return made
    marks = {"D101": "o", "D126": "s", "D170": "^"}

    # latenza per larghezza, una linea per profondita', con il budget
    fig, ax = plt.subplots(figsize=(WIDTH, 0.55 * WIDTH))
    for j, (d, g) in enumerate(grid_t.groupby("profondita")):
        ax.plot(g["larghezza"], g["latenza_ms"], marker=marks.get(d, "o"), color=color(d, j), label=d)
    ax.axhline(budget, ls="--", color=GRAY, lw=1, label=f"budget {budget:.1f} ms")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(sorted(grid_t["larghezza"].unique()), [f"{w:g}" for w in sorted(grid_t["larghezza"].unique())])
    ax.set_xlabel("larghezza")
    ax.set_ylabel("latenza mediana [ms]")
    ax.legend()
    save(fig, out_dir / "figures" / "latenza_griglia")
    made["latenza_griglia"] = "figures/latenza_griglia.png"

    # latenza / mAP50-95: griglia per profondita', potate a stella, budget e fronte della griglia
    pts = grid_t.dropna(subset=["mAP50_95"])
    if not pts.empty:
        fig, ax = plt.subplots(figsize=(WIDTH, 0.7 * WIDTH))
        for j, (d, g) in enumerate(pts.groupby("profondita")):
            ax.scatter(g["latenza_ms"], g["mAP50_95"], marker=marks.get(d, "o"), color=color(d, j), s=26, label=d)
            for _, r in g.iterrows():
                ax.annotate(r["rete"], (r["latenza_ms"], r["mAP50_95"]), fontsize=6, xytext=(3, 2),
                            textcoords="offset points")
        fr = front(pts)
        ax.plot(fr["latenza_ms"], fr["mAP50_95"], "-", color=GRAY, lw=0.8, label="fronte della griglia")
        pp = pruned_t.dropna(subset=["mAP50_95"]) if not pruned_t.empty else pruned_t
        if not pp.empty:
            ax.scatter(pp["latenza_ms"], pp["mAP50_95"], marker="*", s=70, color="#e34948", label="potate", zorder=3)
        ax.axvline(budget, ls="--", color=GRAY, lw=1, label=f"budget {budget:.1f} ms")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="x", which="minor", labelsize=6)
        ax.set_xlabel("latenza mediana [ms]")
        ax.set_ylabel("mAP50-95 (val)")
        ax.legend(fontsize=7)
        save(fig, out_dir / "figures" / "pareto")
        made["pareto"] = "figures/pareto.png"
    return made


def front(pts):
    """Reti non dominate: nessun'altra piu' veloce e piu' precisa."""
    best, keep = -1.0, []
    for i, r in pts.sort_values("latenza_ms").iterrows():
        if r["mAP50_95"] > best:
            keep.append(i)
            best = r["mAP50_95"]
    return pts.loc[keep]


# ------------------------------------------------------------------ fatti e uscita
def facts(df, grid_t, pruned_t, budget: float, target: str, deploy: str) -> dict:
    from collections import Counter
    f = {"target": target, "deploy": deploy, "budget": round(budget, 2), "n_total": int(len(df)),
         "esiti": dict(Counter(df["status"])), "n_storiche": int((df["deploy"] != deploy).sum())}
    inb = grid_t[(grid_t["nel_budget"] == "sì") & grid_t["mAP50_95"].notna()] if len(grid_t) else grid_t
    if len(inb):
        b = inb.loc[inb["mAP50_95"].idxmax()]
        f["migliore_griglia"] = {"rete": b["rete"], "latenza_ms": b["latenza_ms"], "mAP50_95": b["mAP50_95"]}
    pin = pruned_t[(pruned_t["latenza_ms"] <= budget) & pruned_t["mAP50_95"].notna()] if len(pruned_t) else pruned_t
    if len(pin):
        b = pin.loc[pin["mAP50_95"].idxmax()]
        f["migliore_potata"] = {"partenza": b["partenza"], "finetune": b["finetune"], "latenza_ms": b["latenza_ms"],
                                "mAP50_95": b["mAP50_95"]}
    ok = df[df["status"] == "ok"]
    if "rt_throttled" in ok:
        f["n_throttled"] = int((ok["rt_throttled"] == True).sum())  # noqa: E712
    if "rt_temp_end_c" in ok and ok["rt_temp_end_c"].notna().any():
        f["temp_max"] = round(float(ok["rt_temp_end_c"].max()), 1)
    if "time_wall_s" in df:
        f["ore"] = {str(k): round(v / 3600, 1) for k, v in df.groupby("time_device")["time_wall_s"].sum().items()}
    return f


def failed(df):
    """Fallite e non rifatte: nessuna misura ok dopo, per la stessa rete, risoluzione, precisione e backend."""
    import pandas as pd
    k = ["model", "deploy", "quantization", "backend", "strategy"]
    last_ok = df[df["status"] == "ok"].groupby(k)["timestamp"].max()
    x = df[df["status"] == "failed"]
    x = x[[not (tuple(r[c] for c in k) in last_ok.index and last_ok[tuple(r[c] for c in k)] > r["timestamp"])
           for _, r in x.iterrows()]]
    return pd.DataFrame({"cell_id": x["cell_id"], "rete": x["model"].map(sigla), "precisione": x["quantization"],
                         "backend": x["backend"], "errore": x["error"].astype(str).str.split("\n").str[0].str[:120],
                         "data": x["timestamp"].astype(str).str[:10]})


def render(tables: dict, figs: dict, f: dict, out_dir: Path) -> Path:
    from jinja2 import Template
    t = Template(TEMPLATE.read_text(encoding="utf-8"), keep_trailing_newline=True)
    md = {k: v.astype(object).where(v.notna(), None).to_markdown(index=False, missingval="–")
          for k, v in tables.items() if v is not None and not v.empty}
    path = out_dir / "report.md"
    path.write_text(t.render(generated_at=datetime.now().isoformat(timespec="seconds"), figures=figs, tables=md,
                             facts=f), encoding="utf-8")
    return path


def to_html(md_path: Path) -> Path | None:
    """HTML autoconsistente, immagini incluse in base64."""
    if not shutil.which("pandoc"):
        print("pandoc non installato: salto l'HTML (sudo apt install pandoc)", file=sys.stderr)
        return None
    out = md_path.with_suffix(".html")
    subprocess.run(["pandoc", md_path.name, "-o", out.name, "--embed-resources", "--standalone",
                    "--metadata", "title=Report"], cwd=md_path.parent, check=True)
    return out


def update_latest(reports_dir: Path, run_dir: Path) -> None:
    link = reports_dir / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run_dir.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--extra", default=EXTRA_PRUNED, help="glob delle results/ degli esperimenti (solo reti potate)")
    ap.add_argument("--reports", default=ROOT / "lab" / "reports", type=Path)
    ap.add_argument("--target", default="rpi5_640", help="profilo di deploy (conf/target/) per il budget")
    ap.add_argument("--deploy", default="640", help="risoluzione di deploy delle misure da confrontare")
    ap.add_argument("--budget", type=float, default=None, help="budget del modello in ms; default dal profilo")
    ap.add_argument("--run", default=None, help="nome della cartella; default: data e ora")
    ap.add_argument("--no-html", action="store_true")
    a = ap.parse_args(argv)

    out_dir = a.reports / (a.run or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    df = load([a.results])
    if df is None:
        raise SystemExit(f"nessun risultato in {a.results}")
    extra = load([p for p in sorted(ROOT.glob(a.extra)) if p.is_dir()]) if a.extra else None
    budget = a.budget if a.budget is not None else model_budget(a.target)

    ok = df[df["status"] == "ok"]
    grid_t = grid(ok, a.deploy, budget)
    pool = ok if extra is None else __import__("pandas").concat([ok, extra[extra["status"] == "ok"]]).drop_duplicates("cell_id")
    pruned_t = pruned(pool, grid_t)
    tables = {"griglia": grid_t, "potate": pruned_t, "varianti": variants(ok, a.deploy), "fallite": failed(df)}

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "data.parquet", index=False)
    (out_dir / "tables").mkdir(exist_ok=True)
    for k, v in tables.items():
        if v is not None and not v.empty:
            v.to_csv(out_dir / "tables" / f"{k}.csv", index=False)
    md = render(tables, figures(grid_t, pruned_t, budget, out_dir), facts(df, grid_t, pruned_t, budget, a.target,
                                                                          a.deploy), out_dir)
    if not a.no_html:
        to_html(md)
    update_latest(a.reports, out_dir)
    print(f"report in {out_dir} (latest aggiornato)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
