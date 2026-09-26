"""Fig. 2 (main result), full ISBI text width. Every number is read from results/; nothing is typed by hand.

(a) membership AUC vs times seen in the proxy runs: the released-head score (no reference), the public backbone with
    the initial-encoder reference (what an outside auditor has), and the backbone against the crossover twin run
    (only a trainer has it). (b) the released RAD-DINO on its CheXpert validation vs test images, with the proxy
    encoders as negative controls. (c) the pre-specified unpaired premium vs times seen (the Holm-tested values), with the
    pre-specified crossover estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# Panel a keeps the as-run mia.csv: it is the only file with E100 scored against the step-0 (outside-auditor)
# reference. Everything else reads the corrected final_v2; its primary/deltas files are byte-identical to as-run.
V1 = RES / "final_v1_asrun"
V2 = RES / "final_v2" / "uzero_P_base"
OUT = ROOT / "paper" / "figures" / "fig2_main"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none",
    "pdf.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.pad": 2,
    "ytick.major.pad": 2, "axes.labelpad": 3, "legend.frameon": False,
})
BLUE, BLUE_SOFT = "#0F4D92", "#9DB6D8"
GREY, GREY_SOFT = "#7A7A7A", "#D5D5D5"
INK, RED, BAND = "#1F1F1F", "#B64342", "#ECECEC"
DOSES = [3, 15, 100, 500]
RADDINO_DOSE = 101


def title(ax, letter: str, text: str) -> None:
    ax.text(-0.04, 1.07, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom")
    ax.text(0.0, 1.07, text, transform=ax.transAxes, fontsize=7.5, fontweight="semibold", ha="left", va="bottom")


def dose_axis(ax, raddino_column: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_xticks(DOSES, [str(d) for d in DOSES])
    ax.minorticks_off()
    ax.set_xlim(2.1, 800)
    ax.set_xlabel("Exposures per image")
    if raddino_column:
        ax.axvspan(RADDINO_DOSE / 1.08, RADDINO_DOSE * 1.08, color=BAND, lw=0, zorder=0)


def curve(ax, d: pd.DataFrame, color: str, soft: str, lw: float, label: str, label_xy: tuple, weight: str,
          ha: str, va: str) -> None:
    g = d.groupby("dose").mia_auc
    lo, hi, mean = g.min().reindex(DOSES), g.max().reindex(DOSES), g.mean().reindex(DOSES)
    ax.fill_between(DOSES, lo - 0.004, hi + 0.004, color=soft, alpha=0.55, lw=0, zorder=1)
    ax.plot(DOSES, mean, color=color, lw=lw, zorder=2, solid_capstyle="round")
    ax.plot(DOSES, mean, "o", ms=4.2, color=color, mec="white", mew=0.8, zorder=3)
    ax.text(*label_xy, label, color=color, fontsize=6.8, fontweight=weight, ha=ha, va=va)


def panel_a(ax) -> None:
    v1 = pd.read_csv(V1 / "mia.csv")
    cor = pd.read_csv(RES / "final_v2" / "mia_corrected.csv")
    head = pd.read_csv(RES / "amendment6" / "proxy_head_mia.csv")
    last = lambda d: d[d.step == d.groupby("run").step.transform("max")]  # noqa: E731
    backbone = last(v1[v1.score == "inv"])
    twin = last(cor[(cor.score == "inv") & (cor.reference == "partner_run")])
    head = head[head.calibration == "none"]
    ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)), zorder=0)
    ax.axhline(0.6, color=GREY_SOFT, lw=0.8, ls=(0, (4, 2)), zorder=0)
    ax.text(2.3, 0.604, "positive-control bar", color=GREY, fontsize=5.8, va="bottom")
    curve(ax, head, BLUE, BLUE_SOFT, 1.8, "head score", (135, head[head.dose == 500].mia_auc.mean() + 0.02),
          "bold", "left", "bottom")  # proxy's own head; kept clear of the RAD-DINO column label
    curve(ax, backbone, GREY, GREY_SOFT, 1.3, "public backbone", (560, 0.565), "normal", "right", "bottom")
    y = twin.mia_auc.mean()
    ax.errorbar(100, y, yerr=[[y - twin.mia_auc.min()], [twin.mia_auc.max() - y]], fmt="D", ms=4.6, color=INK,
                mec="white", mew=0.7, elinewidth=1.0, capsize=0, zorder=4)
    ax.annotate("backbone vs a twin run\nthat never saw the image", xy=(92, y + 0.004), xytext=(62, y + 0.028),
                color=INK, fontsize=5.8, ha="right", va="bottom", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=2))
    ax.text(RADDINO_DOSE, 0.855, "RAD-DINO", color=GREY, fontsize=5.8, ha="center", va="top")
    dose_axis(ax)
    ax.set_ylim(0.46, 0.86)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_ylabel("Membership AUC")
    title(ax, "a", "Seen images leave a trace")


def panel_b(ax) -> None:
    bridge = json.loads((RES / "bridge" / "bridge.json").read_text())
    head = json.loads((RES / "amendment6" / "raddino_head.json").read_text())
    ctl = list(json.loads((RES / "amendment7" / "summary.json").read_text())["proxies"].values())
    rows = [("Released head", head["mia_auc"], head["mia_ci95"], BLUE, True),
            ("Public backbone", bridge["mia_auc_calibrated"], bridge["mia_ci95"], GREY, True),
            ("Proxy run 1", ctl[0]["auc"], ctl[0]["auc_ci95"], BLUE, False),
            ("Proxy run 2", ctl[1]["auc"], ctl[1]["auc_ci95"], BLUE, False)]
    ys = [3.2, 2.2, 0.9, 0.0]
    ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)), zorder=0)
    ax.axhspan(-0.45, 1.35, color=BAND, lw=0, zorder=0)
    for (name, auc, ci, color, filled), y in zip(rows, ys):
        ax.plot(ci, [y, y], color=color, lw=1.6 if filled else 1.1, solid_capstyle="round", zorder=2)
        ax.plot(auc, y, "o", ms=5.2, color=color, mfc=color if filled else "white",
                mec="white" if filled else color, mew=0.8 if filled else 1.0, zorder=3)
        ax.text(ci[1] + 0.012, y, f"{auc:.2f}", color=color, fontsize=6.5, va="center",
                fontweight="bold" if name == "Released head" else "normal")
    ax.text(0.405, 1.22, "never saw either set", color=GREY, fontsize=5.8, style="italic", va="bottom")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 3.75)
    ax.set_xlim(0.40, 0.83)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Membership AUC (CheXpert val vs test)")
    title(ax, "b", "Released RAD-DINO (head: exploratory)")


def panel_c(ax) -> None:
    unp = pd.read_csv(V2 / "deltas_uzero_P_base.csv")  # the pre-specified, Holm-tested unpaired premiums
    unp = unp[(unp.kind == "probe") & (unp.step == unp.groupby("run").step.transform("max"))]
    prim = json.loads((V2 / "primary_uzero_P_base.json").read_text())
    ax.axhspan(-0.5, 0.5, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=GREY, lw=0.7, zorder=1)
    ax.text(2.3, -1.9, "±0.5 margin", color=GREY, fontsize=5.8, ha="left", va="top")
    offset = {1: 0.88, 2: 1.12}
    for run, g in unp.groupby("run"):
        for r in g.itertuples():
            color = RED if r.hi95 < 0 else GREY  # caption: "red, CI below zero"
            x = r.dose * offset[run]
            ax.plot([x, x], [100 * r.lo95, 100 * r.hi95], color=color, lw=1.1, solid_capstyle="round", zorder=2)
            ax.plot(x, 100 * r.delta, "o", ms=3.8, color=color, mec="white", mew=0.6, zorder=3)
    lo, hi = prim["ci90"]
    ax.plot([100, 100], [100 * lo, 100 * hi], color=INK, lw=2.2, solid_capstyle="round", zorder=4)
    ax.plot(100, 100 * prim["delta_x"], "D", ms=5.2, color=INK, mec="white", mew=0.8, zorder=5)
    num = lambda x: f"{100 * x:.2f}".replace("-", "\u2212")  # noqa: E731
    ax.annotate(f"crossover {num(prim['delta_x'])}\n90% CI {num(lo)} to {num(hi)}", xy=(99, 100 * lo - 0.2),
                xytext=(93, -4.2), color=INK,  # connector stays right of the Run-1 dose-100 interval
                fontsize=5.8, ha="right", va="top", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=1, relpos=(1.0, 1.0)))
    ax.text(330, -8.6, "lower after\n500 exposures", color=RED, fontsize=5.8, ha="right", va="center",
            linespacing=1.15)
    dose_axis(ax, raddino_column=False)
    ax.set_ylim(-13.5, 2.6)
    ax.set_yticks([-12, -8, -4, 0])
    ax.set_ylabel("AUC premium (points)")
    title(ax, "c", "Premium by dose")


def main() -> None:
    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.05], wspace=0.6, left=0.07, right=0.985, top=0.84,
                          bottom=0.19)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    panel_c(fig.add_subplot(gs[2]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
