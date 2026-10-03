"""Fig. 1, v2 (study design as one left-to-right flow), full ISBI text width.

Top lane, the randomized experiment: CheXpert patients -> random arms with their dose in each run -> continued DINO
pretraining (two runs, crossover of H_a and E100) -> frozen snapshots -> probe-AUC premium and membership scores.
Bottom lane, the released model: RAD-DINO and its training list -> CheXpert validation (in the list) vs test (not)
-> the same membership scores. Colours follow Fig. 2: grey = backbone, blue = DINO head, ink = twin run; dose cells
use one grey ramp, so RAD-DINO's ~101x reads like the E100 arm.
Arm sizes are read from data/group_summary.csv; the other numbers are those reported in the text.
Text is 4.8-6.5 pt at print size (Fig. 2 uses 5.8-7.5). The figure is 2.04 in tall like v1: the paper is exactly
four pages, and any extra height pushes references onto a paid fifth page.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig1_overview_v2"
CXP = ROOT / "data" / "chexlocalize" / "CheXpert"
SUMMARY = ROOT / "data" / "group_summary.csv"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 5.5})
BLUE, BLUE_SOFT = "#0F4D92", "#DCE6F3"
GREY, GREY_FILL, INK = "#6E6E6E", "#E9E9E9", "#1F1F1F"
LINE, PANEL, EDGE = "#C4C4C4", "#F5F5F5", "#9E9E9E"
DOSE = {0: "#FFFFFF", 3: "#E6E6E6", 10: "#CFCFCF", 15: "#AFAFAF", 100: "#707070", 500: "#2B2B2B"}
FS_HEAD, FS_BODY, FS_NOTE, FS_CELL = 6.5, 5.5, 5.0, 4.8
W, H = 7.0, 2.14
Y0 = 0.1  # the canvas spans y = Y0..H inches, i.e. it is H - Y0 = 2.04 in tall
Y_HEAD = H - 0.065  # top of every lane heading; boxes start at H - 0.01
X_RIGHT = 4.62  # left edge of the measurement column

# arm key in group_summary.csv -> label; order top to bottom; gaps separate test arms, probe sets, background
ARMS = [("E500", "E500"), ("E100", "E100"), ("H_a", r"H$_\mathrm{a}$"), ("H_b", r"H$_\mathrm{b}$"),
        ("E15", "E15"), ("E3", "E3"), ("P_clean", r"P$_\mathrm{clean}$"), ("P_base", r"P$_\mathrm{base}$"),
        ("B", "B")]


# ---------------------------------------------------------------- drawing primitives
def box(ax, x, y, w, h, fc="white", ec=LINE, lw=0.7, r=0.05, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.75, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=6,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, mutation_scale=6, shrinkA=0,
                                shrinkB=0))


def line(ax, xs, ys, color=INK, lw=0.75, ls="-", z=5):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")


def heading(ax, x, text, y=Y_HEAD):
    ax.text(x, y, text, ha="left", va="top", fontsize=FS_HEAD, fontweight="semibold")


def note(ax, x, y, text, **kw):
    ax.text(x, y, text, fontsize=kw.pop("fontsize", FS_NOTE), color=kw.pop("color", GREY), **kw)


def encoder(ax, x, yc, w, h, label="", fc=GREY_FILL, ec=GREY, fs=FS_NOTE):
    """Trapezoid encoder, wide side left (image in), narrow side right (features out)."""
    ax.add_patch(Polygon([(x, yc - h / 2), (x + w, yc - h / 3.2), (x + w, yc + h / 3.2), (x, yc + h / 2)],
                         closed=True, fc=fc, ec=ec, lw=0.7, zorder=3))
    if label:
        ax.text(x + w * 0.45, yc, label, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.05)


def head(ax, x, yc, w, h, text=True):
    box(ax, x, yc - h / 2, w, h, fc=BLUE_SOFT, ec=BLUE, lw=0.7, r=0.02, z=3)
    if text:
        ax.text(x + w / 2, yc, "head", ha="center", va="center", fontsize=FS_CELL, color=BLUE, rotation=90,
                zorder=4)


def dose_cell(ax, x, yc, dose, label=None, w=0.2, h=0.084):
    """One exposure cell: grey ramp by dose, the dose written inside."""
    ax.add_patch(Rectangle((x, yc - h / 2), w, h, fc=DOSE[dose], ec=EDGE, lw=0.45, zorder=3))
    ax.text(x + w / 2, yc, label or f"{dose}×", ha="center", va="center", fontsize=FS_CELL, zorder=4,
            color="white" if dose >= 100 else INK)


def load_cxr(path: Path, keep=0.78) -> np.ndarray:
    img = Image.open(path).convert("L")
    side = int(min(img.size) * keep)  # central crop: removes burned-in corner annotations
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2,
                    (img.height + side) // 2))
    return np.asarray(img.resize((256, 256), Image.LANCZOS))


def picture(ax, arr, x, y, s, ec=EDGE, lw=0.5):
    ax.imshow(arr, cmap="gray", extent=(x, x + s, y, y + s), zorder=3, interpolation="lanczos", vmin=0, vmax=255)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec="white", lw=1.0, zorder=4))
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=ec, lw=lw, zorder=5))


def stack(ax, arrs, x, y, s, step):
    for k, arr in enumerate(arrs):
        picture(ax, arr, x + step * k, y + step * k, s)


def cxr(split: str, patient: int) -> np.ndarray:
    return load_cxr(CXP / split / f"patient{patient}" / "study1" / "view1_frontal.jpg")


def read_arms() -> dict[str, dict[str, int]]:
    with SUMMARY.open() as f:
        return {r["group"]: {k: int(r[k]) for k in ("images", "patients", "dose_run1", "dose_run2")}
                for r in csv.DictReader(f)}


# ---------------------------------------------------------------- top lane
def data_source(ax, arms):
    stack(ax, [cxr("val", p) for p in (64547, 64553, 64541)], 0.08, 1.43, 0.4, 0.05)
    n_img = sum(a["images"] for a in arms.values())
    n_pat = sum(a["patients"] for a in arms.values())
    ax.text(0.06, 1.36, "CheXpert\ntraining split", ha="left", va="top", fontsize=FS_BODY, fontweight="semibold",
            linespacing=1.05)
    note(ax, 0.06, 1.18, f"{n_pat:,} patients\n{n_img:,} frontal images", ha="left", va="top", linespacing=1.15)
    arrow(ax, 0.64, 1.67, 0.8, 1.67)


def arm_table(ax, arms):
    x_name, x_n, x_c1, cw, gap = 0.98, 1.64, 1.68, 0.2, 0.025
    heading(ax, 0.86, "Random arms (by patient)")
    for j, run in enumerate(("Run 1", "Run 2")):
        note(ax, x_c1 + j * (cw + gap) + cw / 2, Y_HEAD - 0.19, run, ha="center", va="bottom")
    note(ax, x_n, Y_HEAD - 0.19, "images", ha="right", va="bottom")
    ys, y = {}, Y_HEAD - 0.161
    for k, (key, label) in enumerate(ARMS):
        y -= 0.094 + (0.02 if k in (6, 8) else 0)
        ys[key] = y
        a = arms[key]
        ax.text(x_name, y, label, ha="left", va="center", fontsize=FS_BODY)
        note(ax, x_n, y, f"{a['images']:,}", ha="right", va="center")
        for j, dose in enumerate((a["dose_run1"], a["dose_run2"])):
            dose_cell(ax, x_c1 + j * (cw + gap), y, dose)
    note(ax, x_name + 0.075, ys["B"], "background", ha="left", va="center", fontsize=FS_CELL)
    for top, bottom, label in (("E500", "E3", "test arms"), ("P_clean", "P_base", "probe")):
        y0, y1 = ys[top] + 0.042, ys[bottom] - 0.042
        line(ax, [0.955, 0.94, 0.94, 0.955], [y0, y0, y1, y1], color=GREY, lw=0.5)
        note(ax, 0.905, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=FS_CELL)
    y0, y1 = ys["E100"] + 0.054, ys["H_a"] - 0.054  # the crossover pair
    box(ax, x_c1 - 0.015, y1, 2 * cw + gap + 0.03, y0 - y1, fc="none", ec=INK, lw=0.75, r=0.02, z=5)
    ax.text(x_c1 + 2 * cw + gap + 0.05, (y0 + y1) / 2, "swap", rotation=90, ha="center", va="center",
            fontsize=FS_NOTE)
    return ys


def dino_panel(ax):
    x0, y0, w, h = 2.32, 0.985, 2.13, H - 0.01 - 0.985
    box(ax, x0 + 0.035, y0 - 0.03, w, h, fc="#FAFAFA", ec="#D6D6D6", lw=0.6, r=0.06, z=0)  # second run behind
    box(ax, x0, y0, w, h, fc=PANEL, ec=LINE, lw=0.7, r=0.06, z=0)
    heading(ax, x0 + 0.07, "Continued DINO pretraining")
    note(ax, x0 + w - 0.07, Y_HEAD - 0.005, "2 runs", ha="right", va="top")
    arrow(ax, 2.2, 1.62, x0, 1.62)
    dino_crops(ax)
    dino_networks(ax)
    snapshots(ax, x0)


def dino_crops(ax):
    img = cxr("val", 64547)
    rng = np.random.default_rng(11)
    for k in range(3):  # local crops, 98 px
        r0, c0 = rng.integers(40, 150, size=2)
        picture(ax, img[r0:r0 + 70, c0:c0 + 70], 2.4 + 0.04 * k, 1.685 + 0.025 * k, 0.12, lw=0.4)
    note(ax, 2.5, 1.873, "8 × 98 px", ha="center", va="bottom")
    for k, (r0, c0) in enumerate(((20, 30), (55, 60))):  # global crops, 224 px
        picture(ax, img[r0:r0 + 180, c0:c0 + 180], 2.4 + 0.06 * k, 1.22 + 0.05 * k, 0.2, lw=0.4)
    note(ax, 2.53, 1.49, "2 × 224 px", ha="center", va="bottom")


def dino_networks(ax):
    xe, we, he = 2.92, 0.36, 0.27
    for yc, who in ((1.74, "student"), (1.37, "teacher")):
        encoder(ax, xe, yc, we, he)
        ax.text(xe + we * 0.45, yc + 0.035, who, ha="center", va="center", fontsize=FS_BODY, zorder=4)
        note(ax, xe + we * 0.45, yc - 0.05, "ViT-S/14", ha="center", va="center", fontsize=FS_CELL, zorder=4)
        head(ax, xe + we + 0.05, yc, 0.1, 0.2)
        line(ax, [xe + we, xe + we + 0.05], [yc, yc])
    arrow(ax, 2.64, 1.8, xe, 1.78)  # local crops -> student
    arrow(ax, 2.7, 1.43, xe, 1.71)  # global crops -> student
    arrow(ax, 2.7, 1.36, xe, 1.36)  # global crops -> teacher
    xm = xe + we * 0.45
    arrow(ax, xm, 1.74 - he / 2 - 0.005, xm, 1.37 + he / 2 + 0.005, color=GREY, ls=(0, (2, 1.2)), lw=0.65)
    note(ax, xm + 0.035, 1.555, "EMA", ha="left", va="center")
    xh = xe + we + 0.15  # head outputs over 4,096 prototypes
    box(ax, xh + 0.09, 1.305, 0.6, 0.13, fc="white", ec=LINE, lw=0.6, r=0.02, z=3)
    ax.text(xh + 0.39, 1.37, "Sinkhorn–Knopp", ha="center", va="center", fontsize=FS_NOTE, zorder=4)
    line(ax, [xh, xh + 0.09], [1.37, 1.37])
    xl, yl, rl = 4.3, 1.555, 0.065
    ax.add_patch(Circle((xl, yl), rl, fc="white", ec=INK, lw=0.7, zorder=4))
    ax.text(xl, yl, "CE", ha="center", va="center", fontsize=FS_CELL, zorder=5)
    line(ax, [xh, xl, xl], [1.74, 1.74, yl + rl])
    line(ax, [xh + 0.69, xl, xl], [1.37, 1.37, yl - rl])
    note(ax, xh + 0.43, 1.755, r"$p_\mathrm{s}$ over 4,096 prototypes", ha="center", va="bottom")
    note(ax, xh + 0.78, 1.355, r"$p_\mathrm{t}$", ha="center", va="top")


def snapshots(ax, x0):
    xs, xe, yt = 3.02, 4.3, 1.1
    note(ax, x0 + 0.07, yt, "frozen snapshots", ha="left", va="center")
    line(ax, [xs, xe], [yt, yt], color=GREY, lw=0.6)
    for frac, lab in ((0.0, "start"), (0.35, "35%"), (1.0, "end")):
        xt = xs + frac * (xe - xs)
        ax.plot(xt, yt, "o", ms=2.6, color=INK if frac == 0.35 else "white", mec=INK, mew=0.6, zorder=6)
        ax.text(xt, yt - 0.03, lab, ha="center", va="top", fontsize=FS_NOTE,
                fontweight="semibold" if frac == 0.35 else "normal")
    note(ax, xs + 0.35 * (xe - xs), yt + 0.03, "stage of RAD-DINO's release", ha="center", va="bottom")


# ---------------------------------------------------------------- right column
def premium_box(ax):
    x0, y0, w = X_RIGHT, 1.5, W - 0.03 - X_RIGHT
    box(ax, x0, y0, w, H - 0.01 - y0)
    heading(ax, x0 + 0.07, "Probe-AUC premium")
    note(ax, x0 + 0.07, Y_HEAD - 0.12, r"logistic probe on P$_\mathrm{base}$ · macro-AUC over 5 labels",
         ha="left", va="top")
    ax.text(x0 + 0.07, y0 + 0.24, r"$\Delta_x = \frac{1}{2}\,[\,(A^{(2)} - A^{(1)})_{\mathrm{H_a}}"
            r" + (A^{(1)} - A^{(2)})_{\mathrm{E100}}\,]$", ha="left", va="center", fontsize=6.3)
    note(ax, x0 + 0.07, y0 + 0.075, "each term: the same images seen vs unseen · margin ±0.5 points",
         ha="left", va="center")
    arrow(ax, 4.49, 1.8, x0, 1.8)


def membership_box(ax):
    x0, y0, w, y1 = X_RIGHT, Y0 + 0.02, W - 0.03 - X_RIGHT, 1.42
    box(ax, x0, y0, w, y1 - y0)
    heading(ax, x0 + 0.07, "Membership: is exposure detectable?", y=y1 - 0.055)
    note(ax, x0 + 0.07, y1 - 0.17, "score per image from 8 augmented views;\n"
         "AUC of seen vs unseen images", ha="left", va="top", linespacing=1.15)
    rows = (("backbone invariance − initial encoder", "what an outside auditor can compute", "init"),
            ("backbone invariance − twin run", "needs a run that never saw the images", "twin"),
            ("DINO head score", "needs only the released head", "head"))
    for k, (what, who, kind) in enumerate(rows):
        membership_row(ax, x0, 0.93 - k * 0.3, what, who, kind)
    arrow(ax, 4.49, 1.2, x0, 1.2)


def membership_row(ax, x0, y, what, who, kind):
    encoder(ax, x0 + 0.1, y, 0.16, 0.15)
    if kind == "head":
        head(ax, x0 + 0.29, y, 0.06, 0.12, text=False)
        line(ax, [x0 + 0.26, x0 + 0.29], [y, y])
    else:
        ax.text(x0 + 0.315, y, "−", ha="center", va="center", fontsize=6.5)
        fc, ec = (GREY_FILL, GREY) if kind == "init" else ("#4A4A4A", INK)
        encoder(ax, x0 + 0.36, y, 0.16, 0.15, fc=fc, ec=ec)
        note(ax, x0 + 0.44, y - 0.085, kind, ha="center", va="top", fontsize=FS_CELL)
    ax.text(x0 + 0.62, y + 0.04, what, ha="left", va="center", fontsize=FS_BODY, color=BLUE if kind == "head" else INK)
    note(ax, x0 + 0.62, y - 0.05, who, ha="left", va="center", style="italic")


# ---------------------------------------------------------------- bottom lane
def raddino_lane(ax):
    heading(ax, 0.06, "Released RAD-DINO", y=0.885)
    yc = 0.53  # flow line of the lane
    encoder(ax, 0.08, yc, 0.38, 0.34, "ViT-B/14")
    head(ax, 0.51, yc, 0.11, 0.22)
    line(ax, [0.46, 0.51], [yc, yc])
    note(ax, 0.06, 0.325, "checkpoint at 35%\nof training", ha="left", va="top", linespacing=1.1)
    manifest(ax, 0.8, 0.29, yc)
    xv, xt, yv, s, step = 2.6, 3.38, 0.355, 0.28, 0.035  # CheXpert validation (in the list) vs test (not)
    stack(ax, [cxr("val", p) for p in (64553, 64541, 64548)], xv, yv, s, step)
    stack(ax, [cxr("test", p) for p in (64741, 64742, 64743)], xt, yv, s, step)
    xmid = s / 2 + step
    for xc, title, dose, cell, n in ((xv, "CheXpert val", 100, "~101×", 202), (xt, "CheXpert test", 0, "0×", 518)):
        ax.text(xc + xmid, yv + s + 2 * step + 0.03, title, ha="center", va="bottom", fontsize=FS_BODY)
        dose_cell(ax, xc + xmid - 0.29, yv - 0.08, dose, cell, w=0.24)
        note(ax, xc + xmid + 0.0, yv - 0.08, f"n = {n}", ha="left", va="center")
    note(ax, (xv + xt + s) / 2 + step, 0.155, "DINOv2-B gate: sets inseparable (AUC 0.52)", ha="center",
         va="center")
    arrow(ax, 2.43, yc, xv - 0.04, yc)
    arrow(ax, 3.8, yc, X_RIGHT, yc)
    note(ax, 4.21, yc + 0.02, "same scores", ha="center", va="bottom")
    note(ax, 4.21, yc - 0.02, "initial = DINOv2-B", ha="center", va="top")


def manifest(ax, x, y, yc):
    """Training list as a document whose highlighted rows are benchmark images."""
    ax.add_patch(Polygon([(x, y), (x + 0.3, y), (x + 0.3, y + 0.36), (x + 0.24, y + 0.42), (x, y + 0.42)],
                         closed=True, fc="white", ec=GREY, lw=0.6, zorder=3))
    for k in range(7):
        yy = y + 0.33 - k * 0.045
        hit = k in (1, 4)
        line(ax, [x + 0.04, x + 0.25], [yy, yy], color=INK if hit else "#C8C8C8", lw=1.0 if hit else 0.8, z=4)
    ax.text(x + 0.38, y + 0.44, "public training list", ha="left", va="top", fontsize=FS_BODY, fontweight="semibold")
    facts = ("882,775 images, each seen ~101×", "incl. all 25,596 NIH ChestX-ray14", "test images",
             "CheXpert: val in, test not")
    for k, fact in enumerate(facts):
        note(ax, x + 0.38, y + 0.33 - k * 0.08, fact, ha="left", va="top")
    line(ax, [0.62, x], [yc, yc], color=GREY, lw=0.6, ls=(0, (2, 1.2)))


def build() -> Figure:
    arms = read_arms()
    fig = plt.figure(figsize=(W, H - Y0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    data_source(ax, arms)
    arm_table(ax, arms)
    dino_panel(ax)
    premium_box(ax)
    membership_box(ax)
    raddino_lane(ax)
    line(ax, [0.04, 4.5], [0.915, 0.915], color="#E3E3E3", lw=0.6, z=0)  # separates the two lanes
    return fig


def main() -> None:
    fig = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
