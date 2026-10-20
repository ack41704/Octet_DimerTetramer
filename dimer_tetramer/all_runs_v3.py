"""
The v3 dimer-tetramer model (extended_constants_all16_v3.py: fast association phase = DIMER, slow = TETRAMER, point
values) applied to EVERY run listed in 'MASTER TABLE (BEST Colored)' of Global Fit - Master Table on 9-29-26.numbers,
plus 2026.122 (EphrinA5-Fc + EphB2-His, added 2026-09-30). Numbering as in the master table: 1 = tetramer, 2 = dimer.

Per run exactly the v3 analysis (extended_constants_all16_v2.analyze_run + extended_constants_all16_v3.finish_run).
Runs differ in association (13-120 min) and dissociation (0.7-16 h) length; the theory is the same, but what each
number means depends on the windows, so every value carries its source:
  Kdis2  fast rate of the global biexponential dissociation fit. < 8 h windows overestimate it (truncation test of the
         15 h runs, fill_master_tab.dimer_bias_text). '—' if there is no fast (dimer) dissociation phase (2026.40).
  Kdis1  slow rate of that fit: measured with >= 8 h and a detected decay; with a shorter window the fitted rate is an
         APPARENT rate (it overestimates: 13/13 truncated 15 h runs). No decay detected: the pipeline's detection limit
         (as v3 did for 2026.91/.97). No dimer phase: the whole-window decay limit. No decay and no finite limit: '—'.
  Ka1    (k_slow - Kdis1)/[EphB2]_top from the top-concentration fit (v1/v3). With < 60-min associations the slow phase is
         only partly sampled, so Ka1/KD1 are not comparable with the 90-min runs. If Kdis1 is '—' (no decay measurable),
         Ka1 uses Kdis1 = 0.
  Ka2    v3: slope of the fast-phase rate vs [EphB2]; top-concentration fallback (APPROXIMATE) where the series does not
         determine the slope.
  KD1, KD2 = Kdis/Ka (nM).
Not analysable (row kept, reason given): 2026.42 (no dissociation decay at all; data page shown), 2026.59b (file is a raw
export without concentrations), 2026.87b (no data file on this computer).

WT vs R79P Student's t-tests (two-tailed, equal variance) within each ligand x buffer x protocol (association and
dissociation length) set with >= 2 runs per variant; buffers and protocols are never pooled.

Outputs: All_runs_sensorgrams_v3_9-30-26.pdf (analysis folder) and the new tab TAB in a COPY of the master table,
Global Fit - Master Table on 9-30-26.numbers (the 9-29-26 file is not modified; all its tabs are verified unchanged).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import re
import sys
import textwrap
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numbers_parser import Document

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extended_constants_all16_v2 as V  # noqa: E402
import extended_constants_all16_v3 as V3  # noqa: E402
from extended_constants_all16_v2 import INK, INK2, GRID, GREY, EARLY_S, style  # noqa: E402
from extended_constants_all16_v3 import fv, thalf, MIN_FAST_SHARE  # noqa: E402
from plot_all_fits import FLOOR_KF, FLOOR_KS  # noqa: E402
from fill_master_tab import dimer_bias_text  # noqa: E402
from parse_octet import parse_octet_csv  # noqa: E402
from batch_fit import sub  # noqa: E402

W = WORK_DIR
SRC = DATA_DIR + "/Global Fit - Master Table on 9-29-26.numbers"
OUT_NUM = DATA_DIR + "/Global Fit - Master Table on 9-30-26.numbers"
BEST = "MASTER TABLE (BEST Colored)"
TAB = "Dimer-Tetramer v3 (all runs)"
OUT_PDF = os.path.join(W, "All_runs_sensorgrams_v3_9-30-26.pdf")
ADDED = [dict(row=None, interaction="EphrinA5-Fc + EphB2-His", label="2026.122", gf="82")]   # not in the BEST tab
NOT_ANALYSABLE = {
    ("EphrinB2-Fc + EphB1-His", "2026.42"):
        "no dissociation: the signal stays flat or rises for 4 h after the association (ForteBio's fit also put kdis "
        "at its 1e-7 floor), so no dissociation rate, KD1 or KD2 can be formed",
    ("EphrinB2-Fc + EphA4-His", "2026.59b"):
        "data file is a raw export (one time column + 16 sensor columns) with no sample IDs or concentrations; "
        "needs the 'Preprocessed Data' export",
    ("EphrinB1-Fc + EphB2-R79P", "2026.87b"):
        "no data file for 2026.87b on this computer (the 2026.87a file holds EphB2-His only)"}
RAW_PAGE = {"2026.42": DATA_DIR + "/EB2 Interactions - Preprocessed Data FIles/EB2+EphB1-His/2026.42 - Fit 19.csv"}
KEYS = V3.KEYS
TOP_FIT_R2_MIN = 0.95      # top-concentration two-exponential fit: below this its rates are not defined. Across the 79
                           # runs only 2026.40 falls below (0.60; next lowest 0.984, median 0.998)
RUN_NOTES = {"2026.40": "every association trace has a step artifact ~13 min into the 119-min association (the 81 nM "
                        "trace jumps to 0.30 nm and falls back to 0.25 nm), so the top-concentration fit fails"}
NOMINAL_ASSOC_MIN = (13.3, 20, 90, 120)   # protocols in this data set; runs are grouped by these, never pooled across
NAME = {"ka1": "Ka1", "kdis1": "Kdis1", "KD1_nM": "KD1", "ka2": "Ka2", "kdis2": "Kdis2", "KD2_nM": "KD2"}
ROWS_PER_TABLE_PAGE = 27


# ------------------------------------------------------------------ master table -> runs
def label_text(v):
    """master-table experiment cell as text: numbers lose trailing zeros (2026.2 -> '2026.20', 2026.108 kept)"""
    if isinstance(v, (int, float)):
        a, b = f"{round(float(v), 3):.3f}".rstrip("0").split(".")
        return f"{a}.{b.ljust(2, '0')}"
    return str(v).strip()


def run_no(text):
    """'2026.48a (low ...)' -> ('2026.48', 'a')"""
    m = re.match(r"\s*(\d{4}\.\d+)([a-z]?)", text)
    return (m.group(1), m.group(2)) if m else (text, "")


def best_rows():
    t = [s for s in Document(SRC).sheets if s.name == BEST][0].tables[0]
    out, grp = [], None
    for i, r in enumerate(t.rows(values_only=True)[1:], 1):
        if r[0]:
            grp = str(r[0]).strip()
        if r[1] not in (None, ""):
            out.append(dict(row=i, interaction=grp, label=label_text(r[1]), gf=str(r[2]).strip()))
    return out


def match(rows, by):
    """each master-table row -> its cache entry (same interaction and run number; letter from the label, '(a, ...)',
    or the Global Fit #; 'redo' separates 2026.71/.72 and their redos). Every matched entry is used exactly once."""
    per = defaultdict(list)
    for (inter, exp), e in by.items():
        per[inter].append((exp, e))
    used = {}
    for b in rows:
        num, let = run_no(b["label"])
        if not let:
            m = re.search(r"\(([a-z]),", b["label"]); let = m.group(1) if m else ""
        gl = re.match(r"\s*\d+([a-z]?)", b["gf"]).group(1)
        cands = [c for c in per.get(b["interaction"], []) if run_no(c[0])[0] == num]
        for L in (let, gl):
            if len(cands) > 1 and L:
                c2 = [c for c in cands if run_no(c[0])[1] == L]
                cands = c2 or cands
        if len(cands) > 1:
            cands = [c for c in cands if ("redo" in c[0]) == ("redo" in b["label"])]
        key = (b["interaction"], num + let)
        if not cands:
            assert key in NOT_ANALYSABLE or (b["interaction"], num) in NOT_ANALYSABLE, b
            b["e"] = None
            b["reason"] = NOT_ANALYSABLE.get(key) or NOT_ANALYSABLE[(b["interaction"], num)]
            continue
        assert len(cands) == 1, (b, [c[0] for c in cands])
        exp, e = cands[0]
        assert id(e) not in used, (b, used.get(id(e)))
        used[id(e)] = b["label"]; b["e"] = e; b["cache_exp"] = exp
    return rows


# ------------------------------------------------------------------ per-run analysis
def bias(h):
    """short form of fill_master_tab.dimer_bias_text: tested Kdis2 overestimate of a dissociation window of h hours"""
    return ("~3-5x" if h <= 0.8 else "~1.4-5x" if h < 1.5 else "~1.4-2x" if h < 2.6 else "~1.1-2x" if h < 3.5
            else "~1.1-1.35x")


def nominal_assoc(m):
    near = min(NOMINAL_ASSOC_MIN, key=lambda a: abs(a - m))
    return near if abs(near - m) <= 0.1 * near else round(m)


def set_kdis(x, e):
    """window-aware Kdis1/Kdis2 values with their source (see module docstring)"""
    r, s = e["r"], e["s"]; h = s["dissoc_h"]; na = x.setdefault("na_reasons", {})
    x["dimer_phase"] = r["kf"] > FLOOR_KF
    if x["dimer_phase"]:
        x["kdis2"] = r["kf"]
        x["kdis2_src"] = "global dissociation fit" + ("" if h >= 8 else f", {h:.1f} h window (overestimated)")
        x["kdis2_note"] = (f"half-life {thalf(r['kf'], 'min')} min" +
                           ("" if h >= 8 else f"  ({h:.1f} h window: overestimated, {bias(h)} in the truncation test)"))
    else:
        x["kdis2"] = np.nan
        na["kdis2"] = f"no fast (dimer) dissociation phase: signal nearly flat over the {h:.1f} h window"
        x["kdis2_src"] = "not determinable"; x["kdis2_note"] = na["kdis2"]
    ub = s.get("tet_kdis2_upper")
    if not x["dimer_phase"]:
        k, src = (ub, "whole-window decay limit (no dimer phase)") if ub else (np.nan, None)
    elif r["ks"] > FLOOR_KS:
        k, src = r["ks"], ("global dissociation fit" if h >= 8 else f"apparent rate over a {h:.1f} h window (overestimates)")
    elif ub:
        k, src = ub, f"detection limit (no decay detected in {h:.1f} h)"
    else:
        k, src = np.nan, None
    x["kdis1"] = float(k) if k is not None and np.isfinite(k) else np.nan
    if src is None:
        na["kdis1"] = "no slow decay detected and no finite detection limit (late signal flat or rising)"
        src = "not determinable"
    x["kdis1_src"] = src
    x["kdis1_bound"] = src not in ("global dissociation fit", "not determinable")
    suffix = {"apparent": f"  (apparent: a {h:.1f} h window overestimates it)",
              "detection": f"  (no decay detected in {h:.1f} h: detection limit)",
              "whole-window": "  (no dimer phase: whole-window decay limit)"}
    x["kdis1_note"] = (na["kdis1"] if not np.isfinite(x["kdis1"]) else
                       f"half-life {thalf(x['kdis1'], 'd')} days" +
                       next((v for k, v in suffix.items() if src.startswith(k)), ""))


def analyse(b):
    lig, analyte = [p.strip() for p in b["interaction"].split("+", 1)]
    gf = b["gf"].split(" (")[0].strip()
    x = V.analyze_run(b["e"], b["label"], gf, lig, analyte, strict=False)
    set_kdis(x, b["e"])
    no_k1 = not np.isfinite(x["kdis1"])
    if no_k1:
        x["kdis1"] = 0.0                       # Ka1 = k_slow/[A]: no slow decay is measurable, i.e. Kdis1 ~ 0
    V3.finish_run(x, strict=False)
    if no_k1:
        x["kdis1"] = np.nan; x["KD1_nM"] = np.nan
    if x["na_reasons"].get("ka2", "").startswith("fast association") and not np.isfinite(x["kdis2"]):
        x["na_reasons"]["ka2"] = "Kdis2 not determinable, and the slope is not set by the series"
    if x["tt"] is not None and x["r2_top"] < TOP_FIT_R2_MIN:   # top-trace fit failed: its rates are not defined
        why = f"the top-concentration two-exponential fit fails (R² {x['r2_top']:.2f})"
        x["na_reasons"]["ka1"] = why; x["ka1"] = np.nan; x["KD1_nM"] = np.nan
        if x["ka2_from_top"]:
            x["na_reasons"]["ka2"] = why; x["ka2"] = np.nan; x["KD2_nM"] = np.nan
        x["tt"] = None                                          # no star in the rate panel
    x["best"] = b; x["gf_full"] = b["gf"]
    x["notes"] = notes(x)
    x["flags"] = flags(x)
    return x


def notes(x):
    n = [RUN_NOTES[x["exp"]]] if x["exp"] in RUN_NOTES else []
    if x["exp"] in V3.BUFFER_ASSUMED:
        n.append("added 2026-09-30, not in the master table; buffer not stated in the file, PBST assumed")
    for k, why in x["na_reasons"].items():
        n.append(f"{NAME.get(k, k)} not determinable: {why}")
    if x["ka2_from_top"] and np.isfinite(x["ka2"]):
        n.append(f"Ka2 APPROXIMATE, from the top concentration: {V3.ka2_reason(x)} (global slope {x['ka2_slope']:.1e}); "
                 f"Ka2 = (k_fast - Kdis2)/[EphB2]top")
    if x["kdis1_src"] not in ("global dissociation fit", "not determinable"):
        n.append(f"Kdis1 = {x['kdis1_src']}, so KD1 and the tetramer half-life inherit it")
    if x["dimer_phase"] and x["dis_h"] < 8:
        n.append(f"Kdis2 from a {x['dis_h']:.1f} h window: overestimated ({dimer_bias_text(x['dis_h'])}), so KD2 is too")
    if x["dimer_phase"] and not x["ka2_from_top"]:
        if x["kf0"] < 1e-6:
            n.append(f"dimer k_obs intercept ~0 (Kdis2 = {x['kdis2']:.1e}/s)")
        elif not 1 / 3 <= x["kf0"] / x["kdis2"] <= 3:
            n.append(f"dimer k_obs intercept ({x['kf0']:.1e}/s) differs from Kdis2 ({x['kdis2']:.1e}/s) by "
                     f"{max(x['kf0'] / x['kdis2'], x['kdis2'] / x['kf0']):.0f}x")
    if x["assoc_min"] < 60:
        n.append(f"{x['assoc_min']:.0f}-min association: the slow (tetramer) phase is only partly sampled, so Ka1/KD1 are "
                 "not comparable with the 90-min runs")
    if x["fit_issues"]:
        n.append("fit: " + "; ".join(x["fit_issues"]))
    return "; ".join(n)


def flags(x):
    """short provenance codes for the summary table (explained in its footnote)"""
    f = []
    if x["assoc_min"] < 60:
        f.append("A")                       # short association: Ka1/KD1 not comparable with 90-min runs
    if x["dimer_phase"] and x["dis_h"] < 8:
        f.append("D")                       # short dissociation: Kdis2 (and KD2) overestimated
    if x["kdis1_src"].startswith("apparent"):
        f.append("S")                       # Kdis1 = apparent rate over a short window
    if x["kdis1_src"].startswith(("detection", "whole-window")):
        f.append("L")                       # Kdis1 = detection / decay limit
    if x["ka2_from_top"] and np.isfinite(x["ka2"]):
        f.append("T")                       # Ka2 approximate, from the top concentration
    if x["na_reasons"]:
        f.append("N")                       # something not determinable ('—')
    return "".join(f)


FLAG_TEXT = ("A = association < 60 min (Ka1/KD1 not comparable with 90-min runs);  D = dissociation < 8 h (Kdis2, KD2 "
             "overestimated);  S = Kdis1 is the apparent rate of a < 8 h window (overestimates; KD1 too);\nL = Kdis1 is a "
             "detection / decay limit;  T = Ka2 approximate (top concentration);  N = a value is not determinable (—), "
             "reason in the run's notes.  Every value is the model's point value.")


# ------------------------------------------------------------------ PDF
def table_pages(pdf, rows, k0, n_pages):
    head = ["GF", "Run", "Interaction", "Buffer", "Assoc /\nDissoc", "Ka1\n(1/Ms)", "Kdis1\n(1/s)", "KD1\n(nM)",
            "Ka2\n(1/Ms)", "Kdis2\n(1/s)", "KD2\n(nM)", "R² assoc\ntop", "R² assoc\nall", "R²\ndissoc", "Flags"]
    wd = [0.035, 0.1, 0.13, 0.06, 0.06, 0.07, 0.07, 0.055, 0.07, 0.07, 0.05, 0.05, 0.05, 0.045, 0.04]
    body, shade, g = [], [], -1
    prev = None
    for b in rows:
        if b["interaction"] != prev:
            g += 1; prev = b["interaction"]
        inter = b["interaction"].replace("Ephrin", "E").replace("-Fc", "")
        if b.get("x") is None:
            body.append([b["gf"].split(" (")[0], b["label"][:22], inter, "", "", "not analysable", "", "", "",
                         "", "", "", "", "", "N"])
        else:
            x = b["x"]
            body.append([x["gf"], b["label"][:22], inter, x["buffer"].replace("PBST+casein", "+casein"),
                         f"{x['assoc_min']:.0f} m / {x['dis_h']:.1f} h", fv(x["ka1"]), fv(x["kdis1"]),
                         fv(x["KD1_nM"], "{:.3g}"), fv(x["ka2"]), fv(x["kdis2"]), fv(x["KD2_nM"], "{:.3g}"),
                         fv(x["r2_top"], "{:.4f}"), fv(x["r2_all"], "{:.4f}"), fv(x["r2_dis"], "{:.4f}"), x["flags"]])
        shade.append(g % 2)
    pages = [list(range(i, min(i + ROWS_PER_TABLE_PAGE, len(body)))) for i in range(0, len(body), ROWS_PER_TABLE_PAGE)]
    for pi, idx in enumerate(pages):
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.03, 0.955, f"All Octet runs: rate constants and fit quality — v3 (fast = dimer, slow = tetramer)   "
                 f"[{pi + 1}/{len(pages)}]", fontsize=12.5, weight="bold", color=INK)
        fig.text(0.03, 0.93, "1 = tetramer, 2 = dimer. Ka1 = (k_slow − Kdis1)/[EphB2]top (top-concentration slow phase); "
                 "Ka2 = slope of the fast-phase rate vs [EphB2] (global fit); Kdis from the global dissociation fit.\n"
                 "Rows in MASTER TABLE (BEST Colored) order; shading alternates by interaction.", fontsize=7.6,
                 color=INK2, va="top", linespacing=1.5)
        fig.text(0.97, 0.955, f"page {k0 + pi} / {n_pages}", fontsize=8, color="#777", ha="right")
        ax = fig.add_axes([0.01, 0.1, 0.98, 0.8]); ax.axis("off")
        tb = ax.table(cellText=[body[i] for i in idx], colLabels=head, loc="upper center", cellLoc="center",
                      colWidths=[w / sum(wd) for w in wd])
        tb.auto_set_font_size(False); tb.set_fontsize(6.6); tb.scale(1, 1.42)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(GRID)
            if i == 0:
                c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
            elif shade[idx[i - 1]]:
                c.set_facecolor("#f7f7f4")
        fig.text(0.03, 0.035, FLAG_TEXT, fontsize=6.8, color=INK2, linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)
    return len(pages)


def wt_r79p_groups(runs):
    """{(ligand, buffer, assoc_min, dissoc_h): {"WT": [...], "R79P": [...]}} -- same ligand, buffer and protocol only"""
    g = defaultdict(lambda: {"WT": [], "R79P": []})
    for x in runs:
        v = {"EphB2-WT": "WT", "EphB2-R79P": "R79P"}.get(x["analyte"])
        if v:
            g[(x["lig"], x["buffer"], nominal_assoc(x["assoc_min"]), round(x["dis_h"], 1))][v].append(x)
    return g


def ttest_blocks(runs):
    out = []
    for key, d in sorted(wt_r79p_groups(runs).items()):
        rows = []
        for k, nm, f in KEYS:
            a = [x[k] for x in d["WT"] if np.isfinite(x[k])]; b = [x[k] for x in d["R79P"] if np.isfinite(x[k])]
            p = stats.ttest_ind(a, b, equal_var=True).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan
            rows.append(dict(key=k, name=nm, f=f, a=a, b=b, p=p))
        out.append(dict(key=key, wt=[x["exp"] for x in d["WT"]], r79p=[x["exp"] for x in d["R79P"]], rows=rows,
                        tested=any(np.isfinite(r["p"]) for r in rows)))
    return out


def ms(v, f):
    if not v:
        return "—"
    if len(v) == 1:
        return f"{f.format(v[0])} (n=1)"
    return f"{f.format(np.mean(v))} ± {f.format(np.std(v, ddof=1))} (n={len(v)})"


def stats_pages(pdf, runs, k0, n_pages):
    blocks = ttest_blocks(runs)
    tested = [b for b in blocks if b["tested"]]
    per_page = 3
    pages = [tested[i:i + per_page] for i in range(0, len(tested), per_page)] or [[]]
    for pi, chunk in enumerate(pages):
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.955, f"EphB2-WT vs EphB2-R79P   [{pi + 1}/{len(pages)}]", fontsize=13, weight="bold", color=INK)
        fig.text(0.04, 0.935, "Student's t-test, two-tailed, equal variance, within one ligand x buffer x protocol "
                 "(association / dissociation length); buffers and protocols are never pooled.\nn = runs with a value for "
                 "that constant; p only where both variants have n >= 2.", fontsize=8.2, color=INK2, va="top",
                 linespacing=1.5)
        fig.text(0.96, 0.955, f"page {k0 + pi} / {n_pages}", fontsize=8, color="#777", ha="right")
        y0 = 0.86
        for blk in chunk:
            lig, buf, am, dh = blk["key"]
            fig.text(0.04, y0, f"{lig}, {buf}, {am:g}-min association / {dh:g} h dissociation:  WT = "
                     f"{', '.join(blk['wt'])};  R79P = {', '.join(blk['r79p'])}", fontsize=8.6, weight="bold", color=INK)
            ax = fig.add_axes([0.03, y0 - 0.235, 0.94, 0.22]); ax.axis("off")
            body = [[r["name"], ms(r["a"], r["f"]), ms(r["b"], r["f"]),
                     f"{np.mean(r['b']) / np.mean(r['a']):.2f}" if r["a"] and r["b"] else "—",
                     "—" if not np.isfinite(r["p"]) else ("<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")]
                    for r in blk["rows"]]
            t = ax.table(cellText=body, colLabels=["Constant", "WT mean ± SD", "R79P mean ± SD", "R79P / WT", "Student's p"],
                         loc="upper center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(7.8); t.scale(1, 1.32)
            for (i, j), c in t.get_celld().items():
                c.set_edgecolor(GRID)
                if i == 0:
                    c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
            y0 -= 0.275
        if pi == len(pages) - 1:
            untested = [b for b in blocks if not b["tested"]]
            txt = "Not tested (fewer than 2 runs of a variant):  " + ";  ".join(
                f"{b['key'][0].replace('Ephrin', 'E').replace('-Fc', '')} {b['key'][1]} {b['key'][2]:g} min / {b['key'][3]:g} h "
                f"(WT {', '.join(b['wt']) or '—'}; R79P {', '.join(b['r79p']) or '—'})" for b in untested)
            fig.text(0.04, 0.03, "\n".join(textwrap.wrap(txt, 185)), fontsize=6.9, color=INK2, linespacing=1.4)
        pdf.savefig(fig); plt.close(fig)
    return len(pages)


def raw_page(pdf, b, k, n):
    """data-only page for a run the model cannot be applied to (2026.42: no dissociation decay)"""
    path = RAW_PAGE[b["label"]]
    wells = parse_octet_csv(path)
    sig = [w for w in wells if not w["is_control"] and w["y"].max() > 0.02]
    concs = sorted({w["conc_nM"] for w in sig})
    cols = dict(zip(concs, plt.cm.viridis(np.linspace(0.05, 0.9, len(concs)))))
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.955, f"{b['interaction']}   —   {b['label']}   (Global Fit # {b['gf'].split(' (')[0]})", fontsize=13,
             weight="bold", color=INK)
    fig.text(0.05, 0.928, "data only — not fitted · dots = data by concentration (grey = control wells)", fontsize=9,
             color=INK2)
    fig.text(0.95, 0.955, f"page {k} / {n}", fontsize=8, color="#777", ha="right")
    ax = fig.add_axes([0.07, 0.4, 0.86, 0.49])
    sig_ids = {id(w) for w in sig}
    for w in wells:
        t, y = sub(w["t"], w["y"], 300, 300)
        c = GREY if id(w) not in sig_ids else cols[w["conc_nM"]]
        ax.plot(t / 3600, y, ".", ms=1.8, alpha=0.5, color=c,
                label=None if c is GREY else f"{w['conc_nM']:g} nM")
    style(ax, "time (h)", False)
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(list(seen.values()), list(seen.keys()), fontsize=7, frameon=False, loc="lower right", markerscale=4)
    fig.text(0.07, 0.3, "\n".join(textwrap.wrap("Not analysable: " + b["reason"] + ".", 150)), fontsize=9, color=INK,
             linespacing=1.5, va="top")
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------------ Numbers tab
NCOLS = ["Interaction", "Octet Experiment", "Global Fit #", "Buffer", "Association (min)", "Dissociation (h)",
         "Ka1 tetramer (1/Ms)", "Kdis1 tetramer (1/s)", "KD1 tetramer (nM)", "Ka2 dimer (1/Ms)", "Kdis2 dimer (1/s)",
         "KD2 dimer (nM)", "Tetramer half-life (days)", "Dimer half-life (min)", "R² association, top conc",
         "R² association, all curves", "R² dissociation", "Flags", "Ka2 source", "Kdis1 source", "Kdis2 source",
         "Sensors fitted", "Notes"]


def numbers_values(b):
    x = b.get("x")
    base = [b["label"], b["gf"]]
    if x is None:
        return base + [""] * 14 + ["N", "", "", "", "", "not analysable: " + b["reason"]]
    hl1 = np.log(2) / x["kdis1"] / 86400 if np.isfinite(x["kdis1"]) and x["kdis1"] > 0 else np.nan
    hl2 = np.log(2) / x["kdis2"] / 60 if np.isfinite(x["kdis2"]) else np.nan
    src2 = ("—" if not np.isfinite(x["ka2"]) else
            "top concentration, APPROXIMATE" if x["ka2_from_top"] else "slope, all curves")
    return base + [x["buffer"], round(x["assoc_min"], 1), round(x["dis_h"], 2), x["ka1"], x["kdis1"], x["KD1_nM"],
                   x["ka2"], x["kdis2"], x["KD2_nM"], hl1, hl2, x["r2_top"], x["r2_all"], x["r2_dis"], x["flags"], src2,
                   x["kdis1_src"], x["kdis2_src"], f"{x['n_fit']} / {x['n_all']}", x["notes"]]


FMT = {6: ("scientific", 3), 7: ("scientific", 3), 8: ("number", 4), 9: ("scientific", 3), 10: ("scientific", 3),
       11: ("number", 3), 12: ("number", 2), 13: ("number", 1), 14: ("number", 4), 15: ("number", 4),
       16: ("number", 4), 4: ("number", 1), 5: ("number", 2)}


def put(t, r, c, v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            t.write(r, c, "—"); return
        t.write(r, c, float(f"{float(v):.12g}"))
        if c in FMT:
            t.set_cell_formatting(r, c, FMT[c][0], decimal_places=FMT[c][1])
    else:
        t.write(r, c, v)


def write_numbers(rows, runs):
    doc = Document(SRC)
    assert TAB not in [s.name for s in doc.sheets]
    layout, prev = [], None                      # one blank row between interactions, as in the BEST tab
    for b in rows:
        if prev is not None and b["interaction"] != prev:
            layout.append(None)
        layout.append(b); prev = b["interaction"]
    doc.add_sheet(TAB, "Dimer-Tetramer v3 (all runs): point values, fast phase = dimer, slow phase = tetramer",
                  num_rows=len(layout) + 1, num_cols=len(NCOLS))
    sh = [s for s in doc.sheets if s.name == TAB][0]; t = sh.tables[0]
    for c, h in enumerate(NCOLS):
        t.write(0, c, h)
    prev = None; filled = {}
    for i, b in enumerate(layout, 1):
        if b is None:
            prev = None; continue
        if b["interaction"] != prev:
            t.write(i, 0, b["interaction"])
        prev = b["interaction"]
        for c, v in enumerate(numbers_values(b), 1):
            put(t, i, c, v)
        filled[i] = b
    for c, w in ((0, 170), (1, 150), (2, 120), (18, 150), (19, 230), (20, 200), (22, 600)):
        t.col_width(c, w)
    # WT vs R79P statistics (as on the PDF stats pages)
    blocks = [blk for blk in ttest_blocks(runs) if blk["tested"]]
    srows = sum(len(blk["rows"]) for blk in blocks)
    st = sh.add_table("WT vs R79P (Student's t-test, two-tailed, equal variance; same ligand, buffer and protocol only)",
                      num_rows=srows + 1, num_cols=10)
    for c, h in enumerate(["Set (ligand, buffer, association / dissociation)", "WT runs", "R79P runs", "Constant",
                           "WT mean", "WT SD", "R79P mean", "R79P SD", "R79P / WT", "Student's p"]):
        st.write(0, c, h)
    i = 1
    for blk in blocks:
        lig, buf, am, dh = blk["key"]
        for j, r in enumerate(blk["rows"]):
            if j == 0:
                st.write(i, 0, f"{lig}, {buf}, {am:g} min / {dh:g} h")
                st.write(i, 1, ", ".join(blk["wt"])); st.write(i, 2, ", ".join(blk["r79p"]))
            st.write(i, 3, f"{r['name']} (n = {len(r['a'])} / {len(r['b'])})")
            vals = [np.mean(r["a"]) if r["a"] else np.nan, np.std(r["a"], ddof=1) if len(r["a"]) > 1 else np.nan,
                    np.mean(r["b"]) if r["b"] else np.nan, np.std(r["b"], ddof=1) if len(r["b"]) > 1 else np.nan,
                    np.mean(r["b"]) / np.mean(r["a"]) if r["a"] and r["b"] else np.nan, r["p"]]
            for c, v in enumerate(vals, 4):
                if np.isfinite(v):
                    st.write(i, c, float(f"{float(v):.12g}"))
                    kind, dp = ("number", 2) if c == 8 else ("number", 4) if c == 9 else ("scientific", 3)
                    st.set_cell_formatting(i, c, kind, decimal_places=dp)
                else:
                    st.write(i, c, "—")
            i += 1
    st.col_width(0, 260); st.col_width(1, 200); st.col_width(2, 200); st.col_width(3, 140)
    doc.save(OUT_NUM)
    return filled


def verify_numbers(filled):
    a, b = Document(SRC), Document(OUT_NUM)
    names_a, names_b = [s.name for s in a.sheets], [s.name for s in b.sheets]
    assert names_b == names_a + [TAB], names_b
    for sa in a.sheets:
        sb = [s for s in b.sheets if s.name == sa.name][0]
        assert len(sa.tables) == len(sb.tables), sa.name
        for ta, tb in zip(sa.tables, sb.tables):
            ra, rb = ta.rows(values_only=True), tb.rows(values_only=True)
            assert ta.name == tb.name and len(ra) == len(rb), sa.name
            for x, y in zip(ra, rb):
                for u, v in zip(x, y):
                    assert u == v or (isinstance(u, float) and isinstance(v, float) and abs(u - v) <= 1e-12 * max(1, abs(u))), \
                        (sa.name, u, v)
    t = [s for s in b.sheets if s.name == TAB][0].tables[0]
    rows = t.rows(values_only=True)
    n = 0
    for i, bb in filled.items():
        want = numbers_values(bb)
        for c, v in enumerate(want, 1):
            got = rows[i][c]
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                assert isinstance(got, float) and abs(got - v) <= 1e-9 * abs(v), (i, c, got, v)
                n += 1
    return len(names_a), n


# ------------------------------------------------------------------ main
def main():
    rows = best_rows() + ADDED
    by = V.load_cache()
    match(rows, by)
    print(f"{len(rows)} rows ({len(rows) - len(ADDED)} in {BEST} + {len(ADDED)} added): "
          f"{sum(b['e'] is not None for b in rows)} analysable, {sum(b['e'] is None for b in rows)} not")
    for k, b in enumerate(rows, 1):
        if b["e"] is None:
            b["x"] = None
            print(f"  [{k:2d}] {b['interaction']:28s} {b['label']:28s} NOT ANALYSABLE: {b['reason'][:70]}")
            continue
        b["x"] = analyse(b)
        x = b["x"]
        print(f"  [{k:2d}] {b['interaction']:28s} {b['label']:28s} -> {b['cache_exp'][:30]:30s} "
              f"Ka1 {fv(x['ka1'])} Kdis1 {fv(x['kdis1'])} KD1 {fv(x['KD1_nM'], '{:.3g}')} | Ka2 {fv(x['ka2'])} "
              f"Kdis2 {fv(x['kdis2'])} KD2 {fv(x['KD2_nM'], '{:.3g}')} [{x['flags']}]", flush=True)
    runs = [b["x"] for b in rows if b.get("x") is not None]
    n_tab = int(np.ceil(len(rows) / ROWS_PER_TABLE_PAGE))
    n_stat = max(1, int(np.ceil(sum(blk["tested"] for blk in ttest_blocks(runs)) / 3)))
    n_run = sum(1 for b in rows if b.get("x") is not None or b["label"] in RAW_PAGE)
    n = n_tab + n_stat + n_run
    with PdfPages(OUT_PDF) as pdf:
        k = 1
        k += table_pages(pdf, rows, k, n)
        k += stats_pages(pdf, runs, k, n)
        for b in rows:
            if b.get("x") is not None:
                V3.run_page(pdf, b["x"], k, n); k += 1
            elif b["label"] in RAW_PAGE:
                raw_page(pdf, b, k, n); k += 1
        assert k - 1 == n, (k, n)
        pdf.infodict()["Title"] = "All Octet runs, v3 dimer-tetramer model (fast phase = dimer, slow phase = tetramer)"
    filled = write_numbers(rows, runs)
    n_sheets, n_vals = verify_numbers(filled)
    print(f"PDF: {n} pages -> {OUT_PDF}")
    print(f"Numbers: new tab '{TAB}' ({len(filled)} rows, {n_vals} numeric cells verified) -> {OUT_NUM}; "
          f"all {n_sheets} original tabs verified unchanged")
    for blk in ttest_blocks(runs):
        if blk["tested"]:
            print("  t-test", blk["key"], "WT", blk["wt"], "R79P", blk["r79p"],
                  {r["name"]: (None if not np.isfinite(r["p"]) else round(r["p"], 4)) for r in blk["rows"]})
    return rows


if __name__ == "__main__":
    main()
