"""
Publication figures: EphrinB3-Fc + EphB2-WT vs EphB2-R79P, PBST, 90-min association / 15 h dissociation (n = 3 runs
each), v3 dimer-tetramer model. Four separate graphs (Kdis1, KD1, Kdis2, KD2) + a combined 4-panel figure.

No new analysis: per-run values are the saved 9-30-26 results (All_runs_v3_results_9-30-26.csv); p-values are the
two-tailed, equal-variance Student's t-tests already reported (recomputed here from the same values and checked).
Each panel: individual runs (dots), mean (column) +- SD (whisker); significance bracket with
* p < 0.05, ** p < 0.01, *** p < 0.001, ns otherwise, and the exact p.

Outputs: figures_EB3_WT_vs_R79P/ (PDF with editable text, SVG, 600-dpi PNG, source-data CSV).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import csv
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = WORK_DIR
SRC = os.path.join(W, "All_runs_v3_results_9-30-26.csv")
OUT = os.path.join(W, "figures_EB3_WT_vs_R79P")
WT_RUNS = ["2026.101", "2026.103", "2026.108"]
R79P_RUNS = ["2026.106", "2026.107", "2026.109"]
REPORTED_P = {"Kdis1": 0.035, "KD1": 0.035, "Kdis2": 0.009, "KD2": 0.015}     # PDF t-test page (3 decimals)

# (key, csv column, scale, y label, panel title)
PANELS = [("Kdis1", "Kdis1 tetramer (1/s)", 1e6, r"$K_{\mathrm{dis1}}$  ($\times10^{-6}$ s$^{-1}$)", "Tetramer dissociation"),
          ("KD1", "KD1 tetramer (nM)", 1.0, r"$K_{\mathrm{D1}}$  (nM)", "Tetramer affinity"),
          ("Kdis2", "Kdis2 dimer (1/s)", 1e4, r"$K_{\mathrm{dis2}}$  ($\times10^{-4}$ s$^{-1}$)", "Dimer dissociation"),
          ("KD2", "KD2 dimer (nM)", 1.0, r"$K_{\mathrm{D2}}$  (nM)", "Dimer affinity")]
GROUPS = [("WT", "EphB2-WT", "#2a78d6"), ("R79P", "EphB2-R79P", "#eb6834")]     # reference palette slots 1, 2
INK, INK2, RULE = "#0b0b0b", "#52514e", "#c9c8c3"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",      # text stays editable in Illustrator
})


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def load():
    rows = {r["Octet Experiment"]: r for r in csv.DictReader(open(SRC))}
    data = {}
    for key, col, scale, _, _ in PANELS:
        wt = np.array([float(rows[r][col]) for r in WT_RUNS])
        mu = np.array([float(rows[r][col]) for r in R79P_RUNS])
        p = stats.ttest_ind(wt, mu, equal_var=True).pvalue
        assert round(p, 3) == REPORTED_P[key], (key, p)                   # same test as the PDF t-test page
        data[key] = dict(WT=wt, R79P=mu, p=float(p), scale=scale)
    for r in WT_RUNS + R79P_RUNS:                                           # the right set of runs
        assert rows[r]["Buffer"] == "PBST" and float(rows[r]["Association (min)"]) > 85, r
        assert float(rows[r]["Dissociation (h)"]) >= 14.9, r
    return data


def panel(ax, key, d, ylabel, title):
    offs = np.array([-0.13, 0.0, 0.13])                                     # fixed offsets (no random jitter)
    top = 0.0
    for x, (g, _, color) in enumerate(GROUPS):
        v = d[g] * d["scale"]
        m, sd = v.mean(), v.std(ddof=1)
        ax.bar(x, m, width=0.56, color=color, alpha=0.22, linewidth=0, zorder=1)
        ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=3.5, capthick=0.8, zorder=3)
        ax.hlines(m, x - 0.2, x + 0.2, color=color, linewidth=1.6, zorder=2)
        order = np.argsort(v)
        ax.scatter(x + offs, v[order], s=22, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        top = max(top, v.max(), m + sd)
    # significance bracket
    y = top * 1.12
    h = top * 0.035
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color=INK, linewidth=0.7, zorder=5)
    ax.text(0.5, y + h * 1.4, stars(d["p"]), ha="center", va="bottom", fontsize=9 if d["p"] < 0.05 else 7, color=INK)
    ax.text(0.5, y + h * 1.4 + top * 0.13, f"p = {d['p']:.3f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_ylim(0, top * 1.55)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], [f"{g}\n(n = {len(d[g])})" for g, _, _ in GROUPS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, min_n_ticks=3))


def save(fig, stem):
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load()
    sub = "EphrinB3-Fc, PBST, 90-min association / 15 h dissociation"
    for key, _, _, ylabel, title in PANELS:                                 # one graph per constant (~45 x 55 mm)
        fig, ax = plt.subplots(figsize=(1.8, 2.2))
        panel(ax, key, data[key], ylabel, title)
        save(fig, f"EB3_WT_vs_R79P_{key}")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.75))   # 183 mm double column
    for ax, (key, _, _, ylabel, title), letter in zip(axes, PANELS, "abcd"):
        panel(ax, key, data[key], ylabel, title)
        ax.text(-0.42, 1.13, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.text(0.5, -0.06, f"{sub}. Dots = individual runs; columns = mean; whiskers = SD. Two-tailed Student's t-test "
             "(equal variance): * p < 0.05, ** p < 0.01.", ha="center", fontsize=6, color=INK2)
    save(fig, "EB3_WT_vs_R79P_4panel")
    with open(os.path.join(OUT, "EB3_WT_vs_R79P_source_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Constant", "Unit", "Group", "Run", "Value", "Group mean", "Group SD", "Student's t-test p (two-tailed)",
                    "Significance"])
        for key, col, _, _, _ in PANELS:
            unit = col[col.index("(") + 1:-1]
            for g, runs in (("WT", WT_RUNS), ("R79P", R79P_RUNS)):
                v = data[key][g]
                for r, x in zip(runs, v):
                    w.writerow([key, unit, f"EphB2-{g}", r, f"{x:.6g}", f"{v.mean():.6g}", f"{v.std(ddof=1):.6g}",
                                f"{data[key]['p']:.4f}", stars(data[key]["p"])])
    for key in data:
        print(f"{key:6s} WT {data[key]['WT'].mean():.4g} ± {data[key]['WT'].std(ddof=1):.3g}  R79P "
              f"{data[key]['R79P'].mean():.4g} ± {data[key]['R79P'].std(ddof=1):.3g}  p = {data[key]['p']:.4f} {stars(data[key]['p'])}")
    print(OUT)


if __name__ == "__main__":
    main()
