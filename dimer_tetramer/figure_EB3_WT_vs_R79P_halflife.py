"""
Half-life figures in the exact format of figure_EB3_WT_vs_R79P.py (same styling, panel function and sizes):
EphrinB3-Fc + EphB2-WT vs EphB2-R79P, PBST, 90-min association / 15 h dissociation, n = 3 runs each.
  tetramer half-life = ln 2 / Kdis1 (days),  dimer half-life = ln 2 / Kdis2 (min)
No new analysis: per-run half-lives are the saved 9-30-26 results (All_runs_v3_results_9-30-26.csv, checked against
ln 2 / Kdis). The two-tailed, equal-variance Student's t-tests are run on the per-run half-lives themselves (a
half-life test is not the same test as the rate test, so its p differs from the Kdis1 / Kdis2 p).
Outputs (figures_EB3_WT_vs_R79P/): two separate graphs + a 2-panel figure, PDF (editable text) / SVG / 600-dpi PNG,
and a source-data CSV.
"""
import csv
import os
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_EB3_WT_vs_R79P as F  # noqa: E402   (rcParams, panel(), save(), stars(), runs, colours)

PANELS = [("t12_tet", "Tetramer half-life (days)", "Kdis1 tetramer (1/s)", 86400, "days",
           r"$t_{1/2}$, tetramer  (days)", "Tetramer half-life"),
          ("t12_dim", "Dimer half-life (min)", "Kdis2 dimer (1/s)", 60, "min",
           r"$t_{1/2}$, dimer  (min)", "Dimer half-life")]


def load():
    rows = {r["Octet Experiment"]: r for r in csv.DictReader(open(F.SRC))}
    data = {}
    for key, col, kcol, unit_s, _, _, _ in PANELS:
        vals = {}
        for g, runs in (("WT", F.WT_RUNS), ("R79P", F.R79P_RUNS)):
            t = np.array([float(rows[r][col]) for r in runs])
            k = np.array([float(rows[r][kcol]) for r in runs])
            assert np.allclose(t, np.log(2) / k / unit_s, rtol=1e-8), (key, g)     # saved t1/2 = ln2 / Kdis
            vals[g] = t
        p = stats.ttest_ind(vals["WT"], vals["R79P"], equal_var=True).pvalue
        data[key] = dict(WT=vals["WT"], R79P=vals["R79P"], p=float(p), scale=1.0)
    return data


def main():
    data = load()
    for key, _, _, _, _, ylabel, title in PANELS:                     # one graph per half-life, same size as before
        fig, ax = plt.subplots(figsize=(1.8, 2.2))
        F.panel(ax, key, data[key], ylabel, title)
        F.save(fig, f"EB3_WT_vs_R79P_{key}")
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.3), gridspec_kw=dict(wspace=0.75))   # same panel width as the 4-panel
    for ax, (key, _, _, _, _, ylabel, title), letter in zip(axes, PANELS, "ab"):
        F.panel(ax, key, data[key], ylabel, title)
        ax.text(-0.42, 1.13, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.text(0.5, -0.1, "EphrinB3-Fc, PBST, 90-min association / 15 h dissociation. Dots = individual runs; columns = mean;\n"
             "whiskers = SD. Two-tailed Student's t-test (equal variance) on the half-lives: * p < 0.05, ** p < 0.01, "
             "*** p < 0.001.", ha="center", fontsize=6, color=F.INK2, linespacing=1.4)
    F.save(fig, "EB3_WT_vs_R79P_halflife_2panel")
    with open(os.path.join(F.OUT, "EB3_WT_vs_R79P_halflife_source_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Quantity", "Unit", "Group", "Run", "Value", "Group mean", "Group SD",
                    "Student's t-test p (two-tailed, equal variance)", "Significance"])
        for key, col, _, _, unit, _, _ in PANELS:
            for g, runs in (("WT", F.WT_RUNS), ("R79P", F.R79P_RUNS)):
                v = data[key][g]
                for r, x in zip(runs, v):
                    w.writerow([col.split(" (")[0], unit, f"EphB2-{g}", r, f"{x:.6g}", f"{v.mean():.6g}",
                                f"{v.std(ddof=1):.6g}", f"{data[key]['p']:.4f}", F.stars(data[key]["p"])])
    for key, col, *_ in PANELS:
        d = data[key]
        print(f"{col:26s} WT {d['WT'].mean():.3g} ± {d['WT'].std(ddof=1):.2g}  R79P {d['R79P'].mean():.3g} ± "
              f"{d['R79P'].std(ddof=1):.2g}  p = {d['p']:.4f} {F.stars(d['p'])}")


if __name__ == "__main__":
    main()
