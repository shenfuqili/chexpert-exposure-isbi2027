"""Fig. 1 (overview schematic), full ISBI text width. Deterministic layout in inches: boxes, straight arrows and
orthogonal connectors only. Colours follow Fig. 2: grey = public backbone, blue = released DINO head, ink = twin run.

(a) benchmark test images end up in a public encoder's pretraining set and are then used to evaluate it;
(b) we randomize, at the patient level, how often each image is seen, and swap two arms in a second run;
(c) two measurements: the score premium, and whether exposure can be detected with what each party holds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig1_overview"
THUMBS = [ROOT / "data" / "chexlocalize" / "CheXpert" / "val" / f"patient{p}" / "study1" / "view1_frontal.jpg"
          for p in (64547, 64553, 64541)]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5})
BLUE, BLUE_SOFT = "#0F4D92", "#DCE6F3"
GREY, GREY_FILL, INK = "#7A7A7A", "#E9E9E9", "#1F1F1F"
LINE, PANEL, EDGE = "#C4C4C4", "#F5F5F5", "#9E9E9E"
DOSE = {"0×": "#FFFFFF", "3×": "#E6E6E6", "10×": "#CFCFCF", "15×": "#AFAFAF", "100×": "#707070", "500×": "#2B2B2B"}
W, H = 7.0, 2.05
TOP = 1.84  # panel title baseline


def box(ax, x, y, w, h, fc="white", ec=LINE, lw=0.8, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0))


def square(ax, x, y, s, shade):
    ax.add_patch(Rectangle((x, y), s, s, fc=shade, ec=EDGE, lw=0.5, zorder=3))


def thumb(ax, path: Path, x, y, s):
    img = Image.open(path).convert("L")
    side = int(min(img.size) * 0.78)  # central crop: removes burned-in corner annotations
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2,
                    (img.height + side) // 2)).resize((200, 200))
    ax.imshow(np.asarray(img), cmap="gray", extent=(x, x + s, y, y + s), zorder=3, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec="white", lw=1.2, zorder=4))
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=EDGE, lw=0.5, zorder=5))


def title(ax, x, letter, text):
    ax.text(x, TOP, letter, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    ax.text(x + 0.16, TOP, text, fontsize=7.5, fontweight="semibold", ha="left", va="bottom")


def panel_a(ax):
    x0 = 0.05
    title(ax, x0, "a", "Benchmark images in pretraining")
    for k, path in enumerate(THUMBS):  # a neat stack of benchmark test images
        thumb(ax, path, x0 + 0.02 + 0.07 * k, 0.86 + 0.07 * k, 0.48)
    ax.text(x0 + 0.33, 0.78, "benchmark\ntest images", ha="center", va="top", fontsize=6.3, linespacing=1.1)
    box(ax, x0 + 0.78, 0.86, 0.62, 0.7, fc=PANEL)
    ax.text(x0 + 1.09, 1.47, "pretraining set", ha="center", va="center", fontsize=6.2)
    rng = np.random.default_rng(7)
    shades = rng.choice(["#D9D9D9", "#CDCDCD", "#E3E3E3", "#C6C6C6"], size=18)
    for k in range(18):  # many training images; three of them are benchmark test images
        r, c = divmod(k, 6)
        xs, ys = x0 + 0.855 + c * 0.08, 1.37 - r * 0.075
        bench = k in (2, 10, 13)
        ax.add_patch(Rectangle((xs, ys - 0.055), 0.065, 0.055, fc="#8A8A8A" if bench else shades[k],
                               ec=INK if bench else "none", lw=0.6, zorder=3))
    ax.text(x0 + 1.09, 1.0, "883k images,\n~101× each", ha="center", va="center", fontsize=5.2,
            color=GREY, linespacing=1.1)
    arrow(ax, x0 + 0.66, 1.21, x0 + 0.78, 1.21)
    xb = x0 + 1.52  # backbone trapezoid, then the DINO head
    ax.add_patch(Polygon([(xb, 0.9), (xb + 0.42, 1.0), (xb + 0.42, 1.42), (xb, 1.52)], closed=True, fc=GREY_FILL,
                         ec=GREY, lw=0.9, zorder=2))
    ax.text(xb + 0.2, 1.21, "back-\nbone", ha="center", va="center", fontsize=6.0, linespacing=1.05)
    box(ax, xb + 0.48, 1.05, 0.2, 0.32, fc=BLUE_SOFT, ec=BLUE, lw=0.9, r=0.03, z=2)
    ax.text(xb + 0.58, 1.21, "head", ha="center", va="center", fontsize=5.4, color=BLUE, rotation=90)
    ax.plot([xb + 0.42, xb + 0.48], [1.21, 1.21], color=INK, lw=0.8, zorder=3)
    arrow(ax, x0 + 1.4, 1.21, xb, 1.21)
    box(ax, xb - 0.06, 0.84, 0.8, 0.74, fc="none", ec="#D0D0D0", lw=0.7, r=0.06, z=0)
    ax.text(xb + 0.34, 1.61, "public encoder", ha="center", va="bottom", fontsize=6.0, color=GREY)
    box(ax, x0 + 1.18, 0.12, 1.0, 0.4, fc="white")
    ax.text(x0 + 1.68, 0.32, "evaluated on the\nbenchmark", ha="center", va="center", fontsize=6.0,
            linespacing=1.1)
    arrow(ax, xb + 0.2, 0.9, xb + 0.2, 0.52)
    ax.plot([x0 + 0.33, x0 + 0.33, x0 + 1.18], [0.58, 0.32, 0.32], color=GREY, lw=0.8, ls=(0, (2.5, 1.5)), zorder=2)
    ax.text(x0 + 0.76, 0.35, "same images", ha="center", va="bottom", fontsize=5.5, color=GREY, style="italic")


def panel_b(ax):
    x0 = 2.47
    title(ax, x0, "b", "We randomize what is seen")
    rng = np.random.default_rng(3)
    arms = ["0×"] * 6 + ["3×"] * 3 + ["10×"] * 10 + ["15×"] * 3 + ["100×"] * 2 + ["500×"]  # ~ image shares
    rng.shuffle(arms)
    s, g = 0.125, 0.028
    for k, arm in enumerate(arms):  # patients, each randomly assigned to one arm
        r, c = divmod(k, 5)
        square(ax, x0 + 0.04 + c * (s + g), 1.6 - r * (s + g) - s, s, DOSE[arm])
    ax.text(x0 + 0.42, 0.8, "patients assigned\nat random", ha="center", va="top", fontsize=6.0, linespacing=1.1)
    ax.text(x0 + 0.04, 0.44, "exposures per image", ha="left", va="center", fontsize=5.5, color=GREY)
    for k, (arm, shade) in enumerate(DOSE.items()):  # dose key as one row under the grid
        xk = x0 + 0.04 + k * 0.19
        square(ax, xk, 0.27, 0.085, shade)
        ax.text(xk + 0.0425, 0.22, arm, ha="center", va="top", fontsize=5.3)
    xr = x0 + 1.18  # two runs; held-out arm A and 100x arm B swap roles in run 2
    for k, (run, a, b) in enumerate((("Run 1", "0×", "100×"), ("Run 2", "100×", "0×"))):
        yb = 1.2 - k * 0.52
        box(ax, xr, yb, 1.05, 0.42, fc=PANEL)
        ax.text(xr + 0.08, yb + 0.28, run, fontsize=6.4, fontweight="semibold", va="center")
        ax.text(xr + 0.08, yb + 0.11, "DINO pretraining", fontsize=5.5, color=GREY, va="center")
        for j, (name, arm) in enumerate((("A", a), ("B", b))):
            xa = xr + 0.74 + j * 0.15
            ax.text(xa + 0.05, yb + 0.33, name, fontsize=5.5, ha="center", va="center", fontweight="semibold")
            square(ax, xa, yb + 0.08, 0.1, DOSE[arm])
    gx = x0 + 0.84
    arrow(ax, gx, 1.21, xr, 1.41)
    arrow(ax, gx, 1.21, xr, 0.89)
    ax.annotate("", xy=(xr + 0.865, 1.21), xytext=(xr + 0.865, 1.1), zorder=4,
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(xr + 0.95, 1.155, "swap", fontsize=5.2, va="center", color=INK)
    ax.text(xr + 0.525, 0.6, "each image is compared\nwith itself across runs", ha="center", va="top",
            fontsize=5.5, color=GREY, linespacing=1.1)


def panel_c(ax):
    x0 = 5.12
    title(ax, x0, "c", "Two measurements")
    box(ax, x0, 1.12, 1.83, 0.58, fc="white")
    ax.text(x0 + 0.1, 1.54, "Score premium", fontsize=6.6, fontweight="semibold", va="center")
    ax.text(x0 + 0.1, 1.3, "probe AUC, seen minus\nunseen test images", fontsize=5.9, va="center", linespacing=1.1)
    for j, (shade, h, lab) in enumerate(((DOSE["0×"], 0.3, "unseen"), (DOSE["100×"], 0.3, "seen"))):
        xb = x0 + 1.24 + j * 0.3
        ax.add_patch(Rectangle((xb, 1.22), 0.13, h, fc=shade, ec=EDGE, lw=0.5, zorder=3))
        ax.text(xb + 0.065, 1.2, lab, fontsize=5.0, ha="center", va="top", color=GREY)
    ax.plot([x0 + 1.19, x0 + 1.72], [1.22, 1.22], color=GREY, lw=0.7, zorder=4)
    ax.text(x0 + 1.455, 1.57, "Δ ?", fontsize=6.0, ha="center", va="bottom", color=INK)
    box(ax, x0, 0.12, 1.83, 0.88, fc="white")
    ax.text(x0 + 0.1, 0.86, "Can exposure be detected?", fontsize=6.6, fontweight="semibold", va="center")
    ax.text(x0 + 1.75, 0.86, "needs", fontsize=5.3, ha="right", va="center", color=GREY)
    rows = [(GREY, "o", "public backbone", "initial weights"), (BLUE, "o", "released DINO head", "no reference"),
            (INK, "D", "twin run that never saw it", "retraining")]
    for k, (color, marker, what, who) in enumerate(rows):
        yr = 0.64 - k * 0.19
        ax.plot(x0 + 0.15, yr, marker, ms=4.4 if marker == "o" else 3.7, color=color, mec="white", mew=0.5, zorder=4)
        ax.text(x0 + 0.27, yr, what, fontsize=6.0, va="center", color=BLUE if color == BLUE else INK)
        ax.text(x0 + 1.75, yr, who, fontsize=5.5, va="center", ha="right", color=GREY, style="italic")


def main() -> None:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_a(ax)
    panel_b(ax)
    panel_c(ax)
    for xs in (2.36, 5.0):  # thin separators instead of boxes around panels
        ax.plot([xs, xs], [0.12, 1.74], color="#E6E6E6", lw=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
