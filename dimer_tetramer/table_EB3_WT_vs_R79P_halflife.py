"""
Half-life table in the exact format of the 'EphB2-WT vs EphB2-R79P' t-test page of the all-runs PDF
(all_runs_v3.stats_pages: same page size, fonts, table geometry and colours).
EphrinB3-Fc, PBST, 90-min association / 15 h dissociation; WT = 2026.101, .103, .108; R79P = 2026.106, .107, .109.
Values: the saved 9-30-26 per-run half-lives (All_runs_v3_results_9-30-26.csv; checked = ln 2 / Kdis); two-tailed,
equal-variance Student's t-test on the half-lives (the same tests as figure_EB3_WT_vs_R79P_halflife.py).
Outputs: figures_EB3_WT_vs_R79P/EB3_WT_vs_R79P_halflife_table.pdf (full page) and .png (cropped).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import csv
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_constants_all16_v2 import INK, INK2, GRID  # noqa: E402
from all_runs_v3 import ms  # noqa: E402

W = WORK_DIR
SRC = os.path.join(W, "All_runs_v3_results_9-30-26.csv")
OUT = os.path.join(W, "figures_EB3_WT_vs_R79P", "EB3_WT_vs_R79P_halflife_table")
WT, R79P = ["2026.101", "2026.103", "2026.108"], ["2026.106", "2026.107", "2026.109"]
ROWS = [("Tetramer half-life (days)", "Tetramer half-life (days)", "Kdis1 tetramer (1/s)", 86400),
        ("Dimer half-life (min)", "Dimer half-life (min)", "Kdis2 dimer (1/s)", 60)]


def main():
    rows = {r["Octet Experiment"]: r for r in csv.DictReader(open(SRC))}
    body = []
    for name, col, kcol, unit_s in ROWS:
        a = [float(rows[r][col]) for r in WT]
        b = [float(rows[r][col]) for r in R79P]
        for runs, v in ((WT, a), (R79P, b)):                      # saved half-life = ln 2 / Kdis
            assert np.allclose(v, [np.log(2) / float(rows[r][kcol]) / unit_s for r in runs], rtol=1e-8), name
        p = stats.ttest_ind(a, b, equal_var=True).pvalue
        body.append([name, ms(a, "{:.3g}"), ms(b, "{:.3g}"), f"{np.mean(b) / np.mean(a):.2f}",
                     "<0.001" if p < 0.001 else f"{p:.3f}"])
        print(f"{name:26s} WT {np.mean(a):.4g} ± {np.std(a, ddof=1):.3g}  R79P {np.mean(b):.4g} ± {np.std(b, ddof=1):.3g}  "
              f"ratio {np.mean(b) / np.mean(a):.2f}  p = {p:.4f}")
    # ---- page: same text positions, sizes and table styling as all_runs_v3.stats_pages
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.955, "EphB2-WT vs EphB2-R79P: half-lives", fontsize=13, weight="bold", color=INK)
    fig.text(0.04, 0.935, "Student's t-test, two-tailed, equal variance, on the per-run half-lives (t½ = ln 2 / Kdis), within one "
             "ligand x buffer x protocol (association / dissociation length);\nbuffers and protocols are never pooled. "
             "n = runs.", fontsize=8.2, color=INK2, va="top", linespacing=1.5)
    y0 = 0.86
    fig.text(0.04, y0, f"EphrinB3-Fc, PBST, 90-min association / 15 h dissociation:  WT = {', '.join(WT)};  R79P = "
             f"{', '.join(R79P)}", fontsize=8.6, weight="bold", color=INK)
    ax = fig.add_axes([0.03, y0 - 0.235, 0.94, 0.22]); ax.axis("off")
    t = ax.table(cellText=body, colLabels=["Quantity", "WT mean ± SD", "R79P mean ± SD", "R79P / WT", "Student's p"],
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7.8); t.scale(1, 1.32)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(OUT + ".pdf"); print(OUT + ".png")


if __name__ == "__main__":
    main()
