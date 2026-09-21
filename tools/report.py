#!/usr/bin/env python3
"""Generazione del report.

    python -m tools.report [--results results] [--no-html]

Produce una cartella datata sotto `reports/`, autoconsistente e spostabile:
`report.md` con riferimenti **relativi** alle immagini, le figure in PNG e PDF,
le tabelle in CSV e il dataframe aggregato. Il PDF serve per LaTeX, dove una
figura vettoriale resta nitida a qualsiasi zoom.

Il timestamp in formato `%Y-%m-%d_%H-%M-%S` ordina lessicograficamente come
cronologicamente, senza spazi ne' due punti.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TEMPLATE = "templates/report.md.j2"


def _load(results_dir: Path, out_dir: Path):
    import pandas as pd

    from .aggregate import load

    rows = load(results_dir)
    if not rows:
        raise SystemExit(f"nessun risultato in {results_dir}")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "data.parquet", index=False)
    return df


def _savefig(fig, figures: Path, name: str) -> str:
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f"figures/{name}.png"


def _figures(df, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = out_dir / "figures"
    made = {}
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return made

    # latenza mediana per backend e quantizzazione
    if {"lat_median_ms", "backend", "quantization"} <= set(ok.columns):
        pivot = ok.pivot_table(
            index="backend", columns="quantization", values="lat_median_ms",
            aggfunc="median",
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot.bar(ax=ax)
        ax.set_ylabel("latenza mediana [ms]")
        ax.set_xlabel("")
        ax.set_title("Latenza mediana per backend e quantizzazione")
        made["latency_by_backend"] = _savefig(fig, figures_dir,
                                              "latency_by_backend")

    # fronte di Pareto latenza / accuratezza
    if {"lat_median_ms", "acc_map50"} <= set(ok.columns):
        pts = ok.dropna(subset=["lat_median_ms", "acc_map50"])
        if not pts.empty:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for key, grp in pts.groupby("backend"):
                ax.scatter(grp["lat_median_ms"], grp["acc_map50"], label=key, s=28)
            front = _pareto(pts)
            ax.plot(front["lat_median_ms"], front["acc_map50"], "k--", lw=1,
                    label="fronte di Pareto")
            ax.set_xlabel("latenza mediana [ms]")
            ax.set_ylabel("mAP@50")
            ax.set_title("Latenza / accuratezza")
            ax.legend(fontsize=8)
            made["pareto"] = _savefig(fig, figures_dir, "pareto")

    # latenza vs ordine di esecuzione: se correla, il termico ha inquinato
    if {"order_index", "lat_median_ms"} <= set(ok.columns):
        pts = ok.dropna(subset=["order_index", "lat_median_ms"])
        pts = pts[pts["order_index"] >= 0]
        if len(pts) > 3:
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.scatter(pts["order_index"], pts["lat_median_ms"], s=20)
            ax.set_xlabel("indice di esecuzione nello sweep")
            ax.set_ylabel("latenza mediana [ms]")
            ax.set_title("Latenza rispetto all'ordine di esecuzione")
            made["order_drift"] = _savefig(fig, figures_dir, "order_drift")
    return made


def _pareto(df):
    """Celle non dominate: piu' veloci a parita' di mAP, o piu' accurate."""
    pts = df.sort_values("lat_median_ms")
    best, rows = -1.0, []
    for _, row in pts.iterrows():
        if row["acc_map50"] > best:
            rows.append(row)
            best = row["acc_map50"]
    import pandas as pd

    return pd.DataFrame(rows)


def _tables(df, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    made = {}

    def dump(name, frame):
        if frame is None or frame.empty:
            return
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
        made[name] = frame

    ok = df[df["status"] == "ok"]
    cols = [c for c in ("model", "quantization", "backend", "board",
                        "freq_target", "compute_target", "lat_median_ms",
                        "lat_p99_ms", "acc_map50", "energy_mean_power_w",
                        "rt_throttled") if c in df.columns]
    dump("celle_ok", ok[cols])

    if "status" in df.columns:
        dump("stato_celle", df["status"].value_counts().rename_axis("status")
             .reset_index(name="n"))
    if "reason" in df.columns:
        skipped = df[df["status"] == "skipped"]
        if not skipped.empty:
            dump("celle_saltate", skipped.groupby("reason").size()
                 .rename("n").reset_index())
    if "val_status" in df.columns:
        vcols = [c for c in ("model", "quantization", "backend", "val_status",
                             "val_max_abs_diff_vs_fp32", "val_map50_delta",
                             "val_requested_e2e", "val_actual_e2e",
                             "val_e2e_fallback_reason") if c in df.columns]
        # Le celle saltate non hanno un artefatto: includerle riempirebbe la
        # tabella di righe vuote che sembrano validazioni mancanti.
        dump("artefatti",
             df[df["val_status"].notna()][vcols].drop_duplicates())
    if {"time_wall_s", "time_device"} <= set(df.columns):
        hours = (df.groupby("time_device")["time_wall_s"].sum() / 3600).round(2)
        dump("machine_hours", hours.rename("wall_h").reset_index())
    if "rt_throttled" in df.columns:
        thr = df[df["rt_throttled"] == True]  # noqa: E712
        tcols = [c for c in ("cell_id", "board", "freq_target", "compute_target",
                             "rt_temp_end_c") if c in df.columns]
        dump("celle_throttled", thr[tcols])
    return made


def _facts(df) -> dict:
    ok = df[df["status"] == "ok"]
    facts = {
        "n_total": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_skipped": int((df["status"] == "skipped").sum()),
        "n_failed": int((df["status"] == "failed").sum()),
    }
    if "lat_median_ms" in ok.columns and not ok.empty:
        best = ok.loc[ok["lat_median_ms"].idxmin()] if ok["lat_median_ms"].notna().any() \
            else None
        if best is not None:
            facts["fastest"] = {
                k: best.get(k) for k in ("model", "quantization", "backend",
                                         "board", "compute_target")
            }
            facts["fastest"]["lat_median_ms"] = round(
                float(best["lat_median_ms"]), 3
            )
    # guadagno della quantizzazione, per backend
    gains = {}
    if {"quantization", "backend", "lat_median_ms"} <= set(ok.columns):
        pivot = ok.pivot_table(index="backend", columns="quantization",
                               values="lat_median_ms", aggfunc="median")
        for backend, row in pivot.iterrows():
            if "fp32" in row and row.get("fp32") and row.notna().any():
                for q in ("fp16", "int8"):
                    if q in row and row.get(q):
                        gains[f"{backend}/{q}"] = round(row["fp32"] / row[q], 3)
    facts["speedup_vs_fp32"] = gains
    if "rt_throttled" in df.columns:
        facts["n_throttled"] = int((df["rt_throttled"] == True).sum())  # noqa: E712
    if "val_status" in df.columns:
        facts["n_degraded"] = int((df["val_status"] == "degraded").sum())
    if {"val_requested_e2e", "val_actual_e2e"} <= set(df.columns):
        facts["n_e2e_fallback"] = int(
            ((df["val_requested_e2e"] == True) &  # noqa: E712
             (df["val_actual_e2e"] == False)).sum()  # noqa: E712
        )
    return facts


def render(df, figures: dict, tables: dict, facts: dict, project_root: Path,
           out_dir: Path) -> Path:
    from jinja2 import Template

    template = Template(
        (project_root / TEMPLATE).read_text(encoding="utf-8"),
        keep_trailing_newline=True,
    )
    body = template.render(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        figures=figures,
        tables={k: v.to_markdown(index=False) for k, v in tables.items()},
        facts=facts,
    )
    path = out_dir / "report.md"
    path.write_text(body, encoding="utf-8")
    return path


def to_html(md_path: Path) -> Path | None:
    """HTML autoconsistente, immagini incluse in base64."""
    if not shutil.which("pandoc"):
        print("pandoc non installato: salto l'HTML "
              "(sudo apt install pandoc)", file=sys.stderr)
        return None
    out = md_path.with_suffix(".html")
    subprocess.run(
        ["pandoc", md_path.name, "-o", out.name, "--embed-resources",
         "--standalone"],
        cwd=md_path.parent, check=True,
    )
    return out


def update_latest(reports_dir: Path, run_dir: Path) -> None:
    link = reports_dir / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run_dir.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--reports", default=Path("reports"), type=Path)
    ap.add_argument("--run", default=None,
                    help="timestamp della cartella; default: adesso")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)

    stamp = args.run or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.reports / stamp
    df = _load(args.results, out_dir)

    figures = _figures(df, out_dir)
    tables = _tables(df, out_dir)
    facts = _facts(df)
    md = render(df, figures, tables, facts, Path(__file__).resolve().parents[1],
                out_dir)
    if not args.no_html:
        to_html(md)
    update_latest(args.reports, out_dir)
    print(f"report in {out_dir} (reports/latest aggiornato)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
