"""
All curve fits as one PDF, in master-table order and in the user's numbering
(1 = tetramer, 2 = dimer). Built from fit_cache.pkl (no refitting); every parameter string
comes from fill_master_tab.row_cells / mean_cells, so the PDF and the Numbers tab can't disagree.

  page 1   model, fitting method and table conventions
  page 2   contents
  page 3   group summary (the PBST mean rows)
  then     one page per experiment:
             association, dissociation (linear time), dissociation (log time),
             k_obs vs [analyte] (dimer Ka2), parameter panel;
             sensors left out of the fit are drawn in grey with the reason
  last     files that could not be fit, and why
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import sys
import glob
import pickle
import textwrap
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_fit import sub, f2a, boundary  # noqa: E402
from parse_octet import parse_octet_csv  # noqa: E402
from plot_all_fits import infer_interaction  # noqa: E402
import fill_master_tab as FT  # noqa: E402
from numbers_parser import Document  # noqa: E402

warnings.filterwarnings("ignore")
W = WORK_DIR
OUT = os.path.join(W, "Dimer-Tetramer_All_Curve_Fits_9-29-26.pdf")
TET_C, DIM_C, GREY = "#0F6E56", "#D85A30", "#9A9A96"
MONO = "DejaVu Sans Mono"
PAGE = (11, 8.5)
FMTS = {"sci": "{:.3E}", "num4": "{:.4f}", "num3": "{:.3f}", "num2": "{:.2f}", "num1": "{:.1f}", "num0": "{:.0f}"}


def cell_text(cells, col):
    v = cells.get(col)
    if v is None:
        return ""
    val, fmt = v
    return FMTS[fmt].format(val) if fmt else str(val)


# ---------------------------------------------------------------- page 1: model
def model_page(pdf, n_fit, n_fail):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.955, "EphrinB-Fc × Eph receptor BLI kinetics: dimer → tetramer model",
             fontsize=17, weight="bold")
    fig.text(0.05, 0.925, f"All curve fits, in master-table order  ·  {n_fit} experiments fitted, "
             f"{n_fail} files could not be fit (last page)  ·  generated 2026-09-29",
             fontsize=9.5, color="#444")

    # scheme
    ax = fig.add_axes([0.05, 0.64, 0.9, 0.25]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    boxes = [(7, "L₂", "free ephrinB-Fc\n(two ephrin arms)"),
             (33, "L₂·A", "one Eph bound\n(1:1 dimer on one arm)"),
             (60, "L₂·A₂", "both arms bound,\nopen"),
             (88, "T", "closed 2:2\nheterotetramer")]
    for x, lab, sub_ in boxes:
        ax.text(x, 20, lab, ha="center", va="center", fontsize=15, weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white",
                          ec=TET_C if lab == "T" else ("#555" if lab == "L₂" else DIM_C), lw=1.6))
        ax.text(x, 8.5, sub_, ha="center", va="center", fontsize=8.5, color="#333")
    arrows = [(13, 27, "2 ka2·[A]", "kdis2", DIM_C), (40, 52.5, "ka2·[A]", "2 kdis2", DIM_C),
              (68, 82, "ka1", "kdis1", TET_C)]
    for x0, x1, fwd, rev, col in arrows:
        ax.annotate("", xy=(x1, 21.3), xytext=(x0, 21.3), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.annotate("", xy=(x0, 18.7), xytext=(x1, 18.7), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.text((x0 + x1) / 2, 24.2, fwd, ha="center", fontsize=9, color=col, family=MONO)
        ax.text((x0 + x1) / 2, 14.2, rev, ha="center", fontsize=9, color=col, family=MONO)
    ax.text(33, 30, "Step 2 = DIMER (orange): each ephrin arm binds one monomeric Eph\n"
            "bimolecular, per-arm ka2 [1/Ms] and kdis2 [1/s]", ha="center", fontsize=8.5, color=DIM_C)
    ax.text(80, 30, "Step 1 = TETRAMER (green): the two 1:1 dimers on one Fc\n"
            "close into a 2:2 tetramer; intramolecular, ka1 and kdis1 [1/s]", ha="center",
            fontsize=8.5, color=TET_C)

    left = (
        "Rate equations (f = fraction of Fc ligands; A = [analyte])\n"
        "  df0/dt = -2 ka2 A f0 + kdis2 f1\n"
        "  df1/dt =  2 ka2 A f0 - kdis2 f1 - ka2 A f1 + 2 kdis2 f2\n"
        "  df2/dt =  ka2 A f1 - 2 kdis2 f2 - ka1 f2 + kdis1 fT\n"
        "  dfT/dt =  ka1 f2 - kdis1 fT\n"
        "  Response R(t) = Rmax ( 0.5 f1 + f2 + fT )\n"
        "  KD2 = kdis2 / ka2   (M, shown in nM)\n"
        "  KD1 = kdis1 / ka1   (dimensionless: closure equilibrium)\n"
        "  half-life t1/2 = ln 2 / kdis\n\n"
        "How the constants are extracted (two-stage empirical fit)\n"
        "  Dissociation, all sensors at once, shared rates:\n"
        "    R_i(t') = a_i exp(-kdis2 t') + b_i exp(-kdis1 t')\n"
        "      fast phase = dimer (Kdis2), slow phase = tetramer (Kdis1)\n"
        "  Association, each sensor:\n"
        "    R_i(t) = c1 (1 - exp(-k1 t)) + c2 (1 - exp(-k2 t))\n"
        "    k_obs,i = ka2 [A]_i + kdis2   ->  Ka2 = slope of k_obs vs [A]\n"
        "    Ka1 = <slow rate at the two top concentrations> - Kdis1\n")
    right = (
        "How to read the numbers (same as the Numbers tab)\n"
        "  1.23E-04     measured / passes QC\n"
        "  < x,  > x    bound: the dissociation window cannot resolve\n"
        "               the tetramer rate (Kdis1 < x, so t1/2 > x)\n"
        "  (x)          not reliable from this run, shown for\n"
        "               reference; the status line says why\n"
        "  —            not determinable from this run\n\n"
        "QC rules\n"
        "  tetramer Kdis1 measured: >= 8 h dissociation and a slow\n"
        "     phase detected; otherwise a lower bound on t1/2\n"
        "  tetramer Ka1: >= 60 min association\n"
        "  dimer Kdis2: >= 8 h dissociation (shorter windows\n"
        "     overestimate it: ~3-5x at 0.7 h ... ~1.1-1.35x at 4 h)\n"
        "  dimer Ka2: k_obs regression R^2 >= 0.9\n"
        "  group means: PBST runs only, QC-passing runs only\n\n"
        "Numbering: 1 = tetramer, 2 = dimer (as in the Numbers tab).\n"
        "The code and CSV files use the opposite (assembly-order)\n"
        "numbering, but always name the species in column headers.")
    fig.text(0.05, 0.60, left, fontsize=8.8, family=MONO, va="top", linespacing=1.4)
    fig.text(0.545, 0.60, right, fontsize=8.8, family=MONO, va="top", linespacing=1.4)
    fig.text(0.05, 0.06, "Grey traces on the fit pages are sensors left out of the fit (controls, near-zero, "
             "drifting, dominant or failed-replicate sensors);\nthe reason is listed on each page. "
             "Full description of the model and every rule: MODEL_DESCRIPTION.md (analysis folder).",
             fontsize=8.5, color="#555", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- contents + summary
def contents_pages(pdf, groups, first_exp_page):
    lines, page = [], first_exp_page
    for g in groups:
        lines.append(("group", g["name"], None))
        for e in g["entries"]:
            cells = FT.row_cells(e)
            s = e["s"]
            buf = {"PBST+casein": "casein"}.get(s["buffer"], s["buffer"])
            lines.append(("exp", f"{page:>3d} {e['r']['experiment'][:24]:24s} {buf:6s} "
                          f"{s['assoc_min']:3.0f}m/{s['dissoc_h']:4.1f}h {cell_text(cells, 5):>8s} "
                          f"{cell_text(cells, 13):>10s}", e))
            page += 1
    per_col, cols = CONTENTS_PER_COL, 2
    head = "  p experiment               buffer  as/dis   t1/2(1) d  KD2(2) nM"
    chunks = [lines[i:i + per_col * cols] for i in range(0, len(lines), per_col * cols)]
    for k, chunk in enumerate(chunks):
        fig = plt.figure(figsize=PAGE)
        fig.text(0.04, 0.955, "Contents" + (" (cont.)" if k else ""), fontsize=15, weight="bold")
        for ci in range(cols):
            x, y = 0.04 + ci * 0.49, 0.905
            fig.text(x, 0.925, head, fontsize=6.4, family=MONO, color="#666")
            for kind, text, _ in chunk[ci * per_col:(ci + 1) * per_col]:
                if kind == "group":
                    fig.text(x, y, text, fontsize=7.4, weight="bold", color="#222")
                else:
                    fig.text(x, y, text, fontsize=6.4, family=MONO, color="#222")
                y -= 0.0185
        pdf.savefig(fig); plt.close(fig)
    return len(chunks)


CONTENTS_PER_COL = 47


def contents_page_count(groups):
    n = sum(1 + len(g["entries"]) for g in groups)
    return int(np.ceil(n / (CONTENTS_PER_COL * 2)))


def summary_page(pdf, groups):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.04, 0.955, "Group summary: mean ± SD over PBST runs that pass QC", fontsize=15, weight="bold")
    fig.text(0.04, 0.927, "Same values as the group-mean rows of the Numbers tab. Bounds (< >): median of the "
             "group's per-run PBST bounds; ( ) = no QC-passing run, approximate only.",
             fontsize=8.5, color="#444")
    cols = [("Interaction", 25), ("PBST runs", 10), ("tet t1/2 (d)", 21), ("tet Kdis1 (1/s)", 28),
            ("tet Ka1 (1/s)", 28), ("dimer Ka2 (1/Ms)", 28), ("dimer t1/2 (min)", 20), ("dimer KD2 (nM)", 20)]
    keys = [None, 2, 5, 4, 3, 7, 9, 13]
    hdr = "".join(f"{h:{w}s}" for h, w in cols)
    fig.text(0.03, 0.885, hdr, fontsize=6.6, family=MONO, weight="bold")
    y = 0.86
    for g in groups:
        mc = FT.mean_cells(g["entries"])
        vals = [g["name"].replace("EphrinB", "EB").replace("-Fc", "")]
        vals += [cell_text(mc, k).replace(" runs in PBST", "") for k in keys[1:]]
        row = "".join(f"{v[:w - 1]:{w}s}" for v, (_, w) in zip(vals, cols))
        fig.text(0.03, y, row, fontsize=6.6, family=MONO)
        st = cell_text(mc, 18)
        if st:
            fig.text(0.05, y - 0.017, textwrap.shorten("tetramer: " + st, 205, placeholder=" ..."),
                     fontsize=5.8, color=TET_C)
        sd = cell_text(mc, 19)
        if sd:
            fig.text(0.05, y - 0.031, textwrap.shorten("dimer: " + sd, 205, placeholder=" ..."),
                     fontsize=5.8, color=DIM_C)
        y -= 0.052
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- experiment page
def draw_experiment(pdf, e, gname, fitnum, page, n_pages):
    r, s, parsed, mask, reasons = e["r"], e["s"], e["parsed"], e["mask"], e["reasons"]
    cells = FT.row_cells(e)
    tb = r["tb"]
    fig = plt.figure(figsize=PAGE)
    fig.text(0.04, 0.965, f"{gname}   —   {r['experiment']}", fontsize=12.5, weight="bold")
    fig.text(0.04, 0.94, f"Global Fit # {fitnum}   ·   {s['buffer']}   ·   association "
             f"{s['assoc_min']:.0f} min   ·   dissociation {s['dissoc_h']:.1f} h   ·   "
             f"{sum(mask)} of {len(mask)} sensors fitted", fontsize=9, color="#444")
    fig.text(0.96, 0.965, f"page {page} / {n_pages}", fontsize=8, color="#777", ha="right")

    gs = GridSpec(2, 3, figure=fig, left=0.06, right=0.98, top=0.9, bottom=0.05,
                  hspace=0.32, wspace=0.26, height_ratios=[1, 0.95])
    axA, axD, axL = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axK = fig.add_subplot(gs[1, 0])
    axP = fig.add_subplot(gs[1, 1:]); axP.axis("off")

    keep_ids = {id(w) for w in r["keep"]}
    concs = sorted({w["conc_nM"] for w in r["keep"]})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(concs), 1)))
    colors = dict(zip(concs, cmap))
    t_end = r["keep"][0]["t"][-1] - tb
    grey_label = False
    for w, m, why in zip(parsed["wells"], mask, reasons):
        if m:
            continue
        ma = w["t"] <= tb
        ta, ya = sub(w["t"][ma], w["y"][ma], 120, 80)
        md = w["t"] >= tb
        td, yd = sub(w["t"][md] - tb, w["y"][md], 150, 120)
        lab = "excluded sensors" if not grey_label else None
        grey_label = True
        axA.plot(ta / 60, ya, ".", ms=1.4, color=GREY, alpha=0.45, label=lab)
        axD.plot(td / 3600, yd, ".", ms=1.4, color=GREY, alpha=0.45)
        axL.plot(np.maximum(td, 1) / 60, yd, ".", ms=1.4, color=GREY, alpha=0.45)
    for w, A1, A2 in zip(r["keep"], r["A1s"], r["A2s"]):
        col = colors[w["conc_nM"]]
        ma = w["t"] <= tb
        ta, ya = sub(w["t"][ma], w["y"][ma], 150, 100)
        axA.plot(ta / 60, ya, ".", ms=1.8, alpha=0.4, color=col)
        if w["conc_nM"] in r["assoc_fits"]:
            tt = np.linspace(0, tb, 300)
            axA.plot(tt / 60, f2a(tt, *r["assoc_fits"][w["conc_nM"]]), "-", lw=1.3, color=col,
                     label=f"{w['conc_nM']:g} nM")
        md = w["t"] >= tb
        td, yd = sub(w["t"][md] - tb, w["y"][md], 200, 150)
        tt = np.geomspace(1, t_end, 400)
        yy = A1 * np.exp(-r["kf"] * tt) + A2 * np.exp(-r["ks"] * tt)
        axD.plot(td / 3600, yd, ".", ms=1.8, alpha=0.4, color=col)
        axD.plot(tt / 3600, yy, "-", lw=1.3, color=col)
        axL.plot(np.maximum(td, 1) / 60, yd, ".", ms=1.8, alpha=0.4, color=col)
        axL.plot(tt / 60, yy, "-", lw=1.3, color=col)
    # scale to the FITTED sensors: an excluded dominant sensor (e.g. 4 nm vs 0.2 nm) must not
    # flatten the curves that were actually fit; excluded traces are simply clipped
    ys = np.concatenate([w["y"] for w in r["keep"]])
    lo, hi = min(0.0, float(np.percentile(ys, 0.1))), float(ys.max())
    pad = 0.06 * (hi - lo)
    for ax in (axA, axD, axL):
        ax.set_ylim(lo - pad, hi + pad)
    axA.set_title("Association  (dots = data, lines = fit)", fontsize=9)
    axA.set_xlabel("time (min)", fontsize=8); axA.set_ylabel("response (nm)", fontsize=8)
    axA.legend(fontsize=5.8, ncol=2, loc="lower right", framealpha=0.85)
    axD.set_title("Dissociation  (global fit, shared rates)", fontsize=9)
    axD.set_xlabel("time (h)", fontsize=8); axD.set_ylabel("response (nm)", fontsize=8)
    axL.set_xscale("log")
    axL.set_title("Dissociation, log time", fontsize=9)
    axL.set_xlabel("time (min, log scale)   fast = dimer, slow = tetramer", fontsize=8)
    if s.get("dimer_phase_resolved", True):
        axL.axvline(np.log(2) / r["kf"] / 60, color=DIM_C, ls=":", lw=1)
        axL.text(np.log(2) / r["kf"] / 60, axL.get_ylim()[1], " t1/2 dimer", color=DIM_C,
                 fontsize=6.5, va="top")
    for ax in (axA, axD, axL, axK):
        ax.tick_params(labelsize=7)

    pts = r["kobs_pts"]
    if pts:
        cx = np.array([p[0] for p in pts]) * 1e9; ky = np.array([p[1] for p in pts])
        axK.plot(cx, ky, "o", ms=4, color=DIM_C, mfc="white", mew=1.3)
        if np.isfinite(r["ka1"]):
            xx = np.linspace(0, cx.max() * 1.05, 50)
            axK.plot(xx, r["ka1"] * xx * 1e-9 + r["ic"], "-", color=DIM_C, lw=1.2)
        axK.text(0.03, 0.95, f"slope = Ka2 = {r['ka1']:.3g} 1/Ms\nR² = {r['r2k']:.3f}",
                 transform=axK.transAxes, fontsize=7, va="top", family=MONO,
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    else:
        axK.text(0.5, 0.5, "no k_obs points", ha="center", transform=axK.transAxes)
    axK.set_title("Dimer Ka2: k_obs vs [analyte]", fontsize=9)
    axK.set_xlabel("[analyte] (nM)", fontsize=8); axK.set_ylabel("k_obs (1/s)", fontsize=8)

    # parameter panel
    def block(x, title, color, rows, status):
        axP.text(x, 1.0, title, fontsize=10, weight="bold", color=color, transform=axP.transAxes, va="top")
        y = 0.9
        for lab, col in rows:
            axP.text(x, y, f"{lab:20s}{cell_text(cells, col)}", fontsize=8.2, family=MONO,
                     transform=axP.transAxes, va="top")
            y -= 0.075
        axP.text(x, y - 0.01, "\n".join(textwrap.wrap(status, 62)), fontsize=6.6, color=color,
                 transform=axP.transAxes, va="top")
    block(0.0, "1 · TETRAMER (2:2 closure)", TET_C,
          [("Ka1   (1/s)", 3), ("Kdis1 (1/s)", 4), ("KD1   (-)", 12), ("t1/2  (days)", 5),
           ("slow phase (% of R)", 14)], cell_text(cells, 18))
    block(0.51, "2 · DIMER (1:1)", DIM_C,
          [("Ka2   (1/Ms)", 7), ("Kdis2 (1/s)", 8), ("KD2   (nM)", 13), ("t1/2  (min)", 9)],
          cell_text(cells, 19))
    qual = (f"R² dissociation   global fit {cell_text(cells, 10)}   per-sensor median {cell_text(cells, 15)}\n"
            f"R² association    per-sensor median {cell_text(cells, 16)}   "
            f"Ka2 regression {cell_text(cells, 17)}")
    axP.text(0.0, 0.23, qual, fontsize=7, family=MONO, transform=axP.transAxes, va="top", linespacing=1.4)
    extra = []
    if cells.get(23):
        extra.append("Tetramer reference (same interaction + buffer, measured 15 h runs): "
                     + cell_text(cells, 23) + " d")
    if cells.get(25) and cell_text(cells, 25):
        extra.append(cell_text(cells, 25))
    if extra:
        axP.text(0.0, 0.1, "\n".join(textwrap.fill(x, 135) for x in extra), fontsize=6.6,
                 color="#444", transform=axP.transAxes, va="top")
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- failures
def diagnose(path, folder):
    try:
        wells = parse_octet_csv(path)
    except Exception as ex:
        return f"CSV could not be parsed ({type(ex).__name__}); likely a different export layout"
    wells = [w for w in wells if np.isfinite(w["conc_M"])]
    if not wells:
        return "no sensor has a readable analyte concentration (different export layout)"
    _, geno = infer_interaction(folder)
    if geno:
        wells = [w for w in wells if w["label"] == geno]
        if not wells:
            return f"no {geno} sensors in this file"
    sig = [w for w in wells if not w["is_control"] and w["y"].max() > 0.02]
    if len(sig) < 3:
        return f"only {len(sig)} sensor(s) with signal (need 3)"
    if boundary(sig) is None:
        t = sig[0]["t"]; Y = np.mean([w["y"] for w in sig], axis=0); i = int(np.argmax(Y))
        return (f"no dissociation step found: the mean signal peaks at {100 * i / len(t):.0f}% of the "
                f"{t[-1] / 60:.0f} min record (a dissociation step must follow the peak)")
    return "unknown"


def failures_page(pdf, fails):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.94, "Files that could not be fit", fontsize=15, weight="bold")
    fig.text(0.05, 0.91, "These are not in the master table. Each needs a re-export or a re-run; the "
             "reason below is the first check the file failed.", fontsize=9, color="#444")
    y = 0.85
    for folder, name, why in fails:
        fig.text(0.05, y, f"{folder} / {name}", fontsize=10, weight="bold")
        fig.text(0.07, y - 0.03, why, fontsize=9, color="#333")
        y -= 0.09
    pdf.savefig(fig); plt.close(fig)


def main():
    cache = pickle.load(open(os.path.join(W, "fit_cache.pkl"), "rb"))
    index = {(e["r"]["interaction"], e["r"]["experiment"]): e for e in cache.values() if "error" not in e}
    t = [x for x in Document(FT.SRC).sheets if x.name == FT.TAB][0].tables[0]
    rows = t.rows(values_only=True)
    groups = FT.table_groups(rows)
    for g in groups:
        g["entries"] = [index[(g["name"], exp)] for _, exp in g["exp_rows"]]
        g["fitnums"] = [rows[i][2] for i, _ in g["exp_rows"]]
    n_exp = sum(len(g["entries"]) for g in groups)
    fails = [(e["folder"], os.path.basename(p), diagnose(p, e["folder"]))
             for p, e in sorted(cache.items()) if "error" in e]

    n_contents = contents_page_count(groups)
    first = 1 + n_contents + 1 + 1
    n_pages = first - 1 + n_exp + 1
    with PdfPages(OUT) as pdf:
        model_page(pdf, n_exp, len(fails))
        assert contents_pages(pdf, groups, first) == n_contents
        summary_page(pdf, groups)
        page = first
        for g in groups:
            for e, fn in zip(g["entries"], g["fitnums"]):
                draw_experiment(pdf, e, g["name"], fn or "—", page, n_pages)
                page += 1
        failures_page(pdf, fails)
        d = pdf.infodict()
        d["Title"] = "Dimer-tetramer model: all BLI curve fits"
        d["Author"] = "Octet dimer-tetramer pipeline (Dulac Lab)"
    print(f"{n_pages} pages ({n_exp} experiments) -> {OUT}")
    for f in fails:
        print("  not fit:", f)


if __name__ == "__main__":
    main()
