"""
Fill the user's "Dimer-Tetramer Model" tab (their own layout, columns A-K) from fit_cache.pkl,
plus supporting columns to the right (M onward) with everything else needed to read each row.

User numbering (Sartorius site convention): 1 = TETRAMER (2:2), 2 = DIMER (1:1).
The code internally uses the opposite, assembly-order convention (kdis1 = dimer), so every
value below is mapped explicitly by species, never by index.

Cell conventions (also written into the table title):
  plain number   measured / passes QC
  "< x", "> x"   bound (tetramer rates the dissociation window can't resolve)
  "(x)"          not reliable from this run -- shown for reference; the status columns say why
  "—"            not determinable from this run

Group mean rows use PBST runs only (buffer changes tetramer kdis 2-7x, inconsistently) and only
runs that pass QC for that parameter; where none do, the cell falls back to a bound or a
parenthesized value, so the mean row never presents an unreliable number as a measurement.

Writes a NEW file (the source may be open in Numbers), then verifies every other tab is
unchanged and prints a preview.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import sys
import pickle
from collections import Counter

import numpy as np
from numbers_parser import Document

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_fit import sub, f2a  # noqa: E402

W = WORK_DIR
SRC = DATA_DIR + "/Global Fit - Master Table on 9-25-26 (with Dimer-Tetramer tab).numbers"
OUT = DATA_DIR + "/Global Fit - Master Table on 9-29-26.numbers"
TAB = "Dimer-Tetramer Model"
DASH = "—"
PM = "±"
TET_KA_MIN_ASSOC = 60   # tetramer closure tau ~40-80 min; truncating to 13 min put Ka1 1-60x off
DIMER_MIN_DISSOC = 8    # truncation test: dimer Kdis within ~10% only once dissociation >= 8 h
KA_R2_MIN = 0.9         # pipeline QC on the k_obs-vs-concentration regression
TET_KA_MIN_SAT = 0.25   # top-2 mean theta^2: below this the slow association rate isn't the closure
                        # rate (Ka1 low by >4x); four weak-binding PBSTB 15 h runs 0.01-0.13,
                        # PBST 15 h runs 0.47-0.84

HEADERS = {3: "Ka1 (Tetramer, 1/s)", 4: "KDis 1 (tetramer, 1/s)", 5: "Approx. half-life (tetramer, days)",
           7: "Ka2 (Dimer, 1/Ms)", 8: "KDis2 (Dimer, 1/s)", 9: "Approx. half-life (dimer, min)",
           10: "R^2 (dissociation, global fit)",
           12: "KD1 (tetramer, dimensionless)", 13: "KD2 (dimer, nM)",
           14: "Tetramer fraction of signal at dissociation start (%)",
           15: "R^2 dissociation (median per sensor)", 16: "R^2 association (median per sensor)",
           17: "R^2 Ka2 regression (k_obs vs [analyte])",
           18: "Tetramer result", 19: "Dimer result", 20: "Assoc. (min)", 21: "Dissoc. (h)",
           22: "Buffer", 23: "Tetramer reference: same interaction + buffer, measured t1/2 (d)",
           24: "Sensors used / total", 25: "Notes"}
TITLE = ("MASTER TABLE: Dimer-Tetramer model (1 = 2:2 tetramer, 2 = 1:1 dimer). Plain = measured; "
         "< / > = bound; ( ) = not reliable from this run, see status columns; means = PBST runs only")
FMT = {"sci": ("scientific", 3), "num4": ("number", 4), "num3": ("number", 3),
       "num2": ("number", 2), "num1": ("number", 1), "num0": ("number", 0)}


def finite(v):
    return v is not None and isinstance(v, (int, float, np.floating)) and bool(np.isfinite(v))


def sci(v):
    return f"{v:.3E}"


def dimer_bias_text(h):
    """Tested dimer-Kdis overestimate by window (truncation_test.py, EB3+EphB2 15 h runs)."""
    if h <= 0.8:
        return "truncation test: ~3-5x at 0.7 h"
    if h < 1.5:
        return "between ~3-5x (0.7 h) and ~1.4-2x (2 h) in truncation test"
    if h < 2.6:
        return "truncation test: ~1.4-2x at 2 h"
    if h < 3.5:
        return "between ~1.4-2x (2 h) and ~1.1-1.35x (4 h) in truncation test"
    return "truncation test: ~1.1-1.35x at 4 h"


def per_sensor_r2(r):
    """Median over sensors of each sensor's own R^2 (about its own mean). The pooled global R^2
    also credits the model for between-sensor amplitude differences, so it runs high."""
    tb, r2d, r2a = r["tb"], [], []
    for w, A1, A2 in zip(r["keep"], r["A1s"], r["A2s"]):
        m = w["t"] >= tb
        td, yd = sub(w["t"][m] - tb, w["y"][m], 200, 200)   # the points stage 1 actually fit
        yh = A1 * np.exp(-r["kf"] * td) + A2 * np.exp(-r["ks"] * td)
        r2d.append(1 - np.sum((yd - yh) ** 2) / np.sum((yd - yd.mean()) ** 2))
        if w["conc_nM"] in r["assoc_fits"]:
            ma = w["t"] <= tb
            ya = w["y"][ma]
            yh = f2a(w["t"][ma], *r["assoc_fits"][w["conc_nM"]])
            r2a.append(1 - np.sum((ya - yh) ** 2) / np.sum((ya - ya.mean()) ** 2))
    return (float(np.median(r2d)) if r2d else np.nan), (float(np.median(r2a)) if r2a else np.nan)


def ensure_saturation(e):
    """Mean theta^2 over the two top concentrations, theta = [A]/([A] + KD2). The closure relaxes
    at ka1*theta^2 + kdis1, so reading the slow association rate as ka1 + kdis1 assumes the top
    concentrations saturate the dimer step. Stored on the in-memory summary (never saved)."""
    s = e["s"]
    if "top_sat" not in s:
        kd = s["KD1_nM"]
        top = sorted({w["conc_nM"] for w in e["r"]["keep"]})[-2:]
        s["top_sat"] = (float(np.mean([(c / (c + kd)) ** 2 for c in top]))
                        if finite(kd) and kd > 0 else None)
    return s["top_sat"]


def qc(s):
    ka_valid = finite(s["ka1"]) and s["ka1"] > 0
    phase = s.get("dimer_phase_resolved", True)
    sat = s.get("top_sat")
    return dict(ka_valid=ka_valid, dimer_phase=phase,
                ka_ok=ka_valid and finite(s["r2_kobs"]) and s["r2_kobs"] >= KA_R2_MIN,
                kdis_ok=phase and s["dissoc_h"] >= DIMER_MIN_DISSOC,
                saturated=sat is None or sat >= TET_KA_MIN_SAT,
                # without a dimer phase the two association components can't be assigned
                tka_ok=(phase and finite(s["ka2"]) and s["assoc_min"] >= TET_KA_MIN_ASSOC
                        and (sat is None or sat >= TET_KA_MIN_SAT)))


def row_cells(e):
    """{col: (value, fmt)} for one experiment row."""
    r, s, parsed, mask, reasons = e["r"], e["s"], e["parsed"], e["mask"], e["reasons"]
    ensure_saturation(e)
    q = qc(s)
    c = {}
    cat = s["tet_category"]
    assoc, dis = s["assoc_min"], s["dissoc_h"]

    # ---------------- tetramer ("1") ----------------
    if not finite(s["ka2"]):
        c[3] = (DASH, None)
    elif q["tka_ok"]:
        c[3] = (s["ka2"], "sci")
    else:
        c[3] = (f"({sci(s['ka2'])})", None)
    if cat == "measured":
        c[4] = (s["tet_kdis2"], "sci")
        c[5] = (s["tet_thalf_d"], "num2")
        if finite(s["KD2"]):
            c[12] = (s["KD2"], "sci") if q["tka_ok"] else (f"({sci(s['KD2'])})", None)
        else:
            c[12] = (DASH, None)
    elif cat in ("lower_bound_15h", "lower_bound_short"):
        c[4] = (f"< {sci(s['tet_kdis2_upper'])}", None)
        c[5] = (f"> {s['tet_thalf_lower_d']:.3g}", None)
        if s.get("tet_KD2_upper") is None:
            c[12] = (DASH, None)
        else:   # a KD bound built on an unreliable Ka1 is itself unreliable
            kd = f"< {sci(s['tet_KD2_upper'])}"
            c[12] = (kd if q["tka_ok"] else f"({kd})", None)
    else:
        c[4] = c[5] = c[12] = (DASH, None)

    # ---------------- dimer ("2") ----------------
    if not q["ka_valid"]:
        c[7] = (DASH, None)
    elif q["ka_ok"]:
        c[7] = (s["ka1"], "sci")
    else:
        c[7] = (f"({sci(s['ka1'])})", None)
    if not q["dimer_phase"]:
        c[8] = c[9] = (DASH, None)
    elif q["kdis_ok"]:
        c[8] = (s["kdis1"], "sci")
        c[9] = (s["t12_dimer_min"], "num1")
    else:
        c[8] = (f"({sci(s['kdis1'])})", None)
        c[9] = (f"({s['t12_dimer_min']:.1f})", None)
    if not q["ka_valid"] or not finite(s["KD1_nM"]):
        c[13] = (DASH, None)
    elif q["ka_ok"] and q["kdis_ok"]:
        c[13] = (s["KD1_nM"], "num2")
    else:
        c[13] = (f"({s['KD1_nM']:.3g})", None)

    # ---------------- fit quality ----------------
    c[10] = (s["R2_dissoc"], "num4")
    r2d_ps, r2a_ps = per_sensor_r2(r)
    c[15] = (r2d_ps, "num3") if finite(r2d_ps) else (DASH, None)
    c[16] = (r2a_ps, "num3") if finite(r2a_ps) else (DASH, None)
    c[17] = (s["r2_kobs"], "num3") if finite(s["r2_kobs"]) else (DASH, None)
    A1, A2 = np.sum(r["A1s"]), np.sum(r["A2s"])
    c[14] = (100 * A2 / (A1 + A2), "num1") if A1 + A2 > 0 and q["dimer_phase"] else (DASH, None)

    # ---------------- status ----------------
    tet = {"measured": "measured (15 h dissociation)",
           "lower_bound_15h": "lower bound only: no decay detected in 15 h (sensor-spread 95% bound)",
           "lower_bound_short": f"lower bound only: {dis:.1f} h dissociation too short to resolve",
           "no_bound": "not resolvable from this run (no slow phase, too little late signal)"}[cat]
    if not q["dimer_phase"]:
        tet += "; bound from the whole-window decay (no dimer phase to separate)"
    if not finite(s["ka2"]):
        tet += "; Ka1 not determinable (no slow association component faster than Kdis1)"
    elif not q["dimer_phase"]:
        tet += "; Ka1 not reliable (association phases can't be assigned without a dimer phase)"
    elif assoc < TET_KA_MIN_ASSOC:
        tet += f"; Ka1 not reliable ({assoc:.0f} min association; needs >= {TET_KA_MIN_ASSOC} min)"
    elif not q["saturated"]:
        tet += (f"; Ka1 not reliable: top concentrations reach only {100 * s['top_sat']:.0f}% of "
                f"dimer saturation (theta^2; KD2 = {s['KD1_nM']:.0f} nM), so the slow association "
                f"rate is not the closure rate")
    c[18] = (tet, None)
    dim = []
    if not q["dimer_phase"]:
        dim.append(f"no fast (dimer) dissociation phase: signal nearly flat over the {dis:.1f} h window, "
                   f"so Kdis2, half-life and KD2 are not determinable")
    elif not q["kdis_ok"]:
        dim.append(f"Kdis2/half-life overestimated by the {dis:.1f} h window ({dimer_bias_text(dis)})")
    if not q["ka_valid"]:
        dim.append("Ka2 not determinable (k_obs shows no positive concentration dependence)")
    elif not q["ka_ok"]:
        dim.append(f"Ka2 not reliable (k_obs regression R^2 = {s['r2_kobs']:.2f} < {KA_R2_MIN})")
    c[19] = ("; ".join(dim) if dim else "passes QC", None)

    # ---------------- context ----------------
    c[20] = (assoc, "num0")
    c[21] = (dis, "num1")
    c[22] = (s["buffer"], None)
    if s.get("tet_group_n"):
        sd = s.get("tet_group_thalf_sd_d")
        ref = f"{s['tet_group_thalf_mean_d']:.2f}" + ("" if sd is None else f" {PM} {sd:.2f}")
        c[23] = (f"{ref} (n={s['tet_group_n']}; reference, not this run)", None)
    c[24] = (f"{sum(mask)} / {len(mask)}", None)
    c[25] = (exclusion_note(parsed, reasons) or "", None)
    return c


ROUTINE = ("control well", "near-zero signal", "dissociation phase too short", "drifts upward (not physical)")


def exclusion_note(parsed, reasons):
    """Routine exclusions as counts; sensor-failure exclusions spelled out per sensor."""
    routine = Counter(x for x in reasons if x in ROUTINE)
    parts = [f"{n} {why}" for why, n in routine.items()]
    parts += [f"{w['conc_nM']:g} nM {why}" for w, why in zip(parsed["wells"], reasons)
              if why and why not in ROUTINE]
    return ("excluded: " + "; ".join(parts)) if parts else None


def mean_sd(vals, fmt, approx=False):
    """'m ± sd (n=k)', or '(m ± sd; n=k)' when the values are approximate."""
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return None
    m = float(np.mean(vals))
    core = fmt(m) if len(vals) == 1 else f"{fmt(m)} {PM} {fmt(float(np.std(vals, ddof=1)))}"
    return f"({core}; n={len(vals)})" if approx else f"{core} (n={len(vals)})"


def mean_cells(entries):
    s3 = lambda v: f"{v:.3E}"
    f2 = lambda v: f"{v:.2f}"
    f1 = lambda v: f"{v:.1f}"
    for e in entries:
        ensure_saturation(e)
    pb = [e for e in entries if e["s"]["buffer"] == "PBST"]
    S = [e["s"] for e in pb]
    c = {1: (f"Mean {PM} SD (PBST runs only)", None),
         2: (f"{len(pb)} of {len(entries)} runs in PBST", None), 22: ("PBST", None)}
    if not pb:
        c[18] = ("no PBST run in this group -- see individual rows", None)
        c[19] = ("no PBST run in this group -- see individual rows", None)
        return c

    # ---------------- tetramer ----------------
    tka = [x["ka2"] for x in S if qc(x)["tka_ok"]]
    c[3] = (mean_sd(tka, s3) or DASH, None)
    meas = [x for x in S if x["tet_category"] == "measured"]
    other = Counter(e["s"]["buffer"] for e in entries
                    if e["s"]["tet_category"] == "measured" and e["s"]["buffer"] != "PBST")
    other_txt = ("; measured in " + ", ".join(f"{b} (n={n})" for b, n in other.items())
                 + " -- not pooled, see those rows") if other else ""
    if meas:
        c[4] = (mean_sd([x["tet_kdis2"] for x in meas], s3), None)
        c[5] = (mean_sd([x["tet_thalf_d"] for x in meas], f2), None)
        c[12] = (mean_sd([x["KD2"] for x in meas if qc(x)["tka_ok"]], s3) or DASH, None)
        c[18] = (f"measured in {len(meas)} PBST run(s){other_txt}", None)
    else:
        bnd = [(e, e["s"]) for e in pb if e["s"]["tet_category"] in ("lower_bound_15h", "lower_bound_short")]
        if bnd:
            # Group bound = the ceil(n/2)-th smallest per-run bound: still a valid lower bound as long
            # as a majority of the per-run bounds are, so one noisy short run can't set it.
            bnd.sort(key=lambda z: z[1]["tet_thalf_lower_d"])
            be, bs = bnd[(len(bnd) + 1) // 2 - 1]
            c[4] = (f"< {sci(bs['tet_kdis2_upper'])}", None)
            c[5] = (f"> {bs['tet_thalf_lower_d']:.3g}", None)
            if bs.get("tet_KD2_upper") is None:
                c[12] = (DASH, None)
            else:
                kd = f"< {sci(bs['tet_KD2_upper'])}"
                c[12] = (kd if qc(bs)["tka_ok"] else f"({kd})", None)
            lo, hi = bnd[0][1]["tet_thalf_lower_d"], bnd[-1][1]["tet_thalf_lower_d"]
            how = ("bound from the single PBST run" if len(bnd) == 1 else
                   f"bound = lower of 2 per-run bounds ({lo:.3g}-{hi:.3g} d)" if len(bnd) == 2 else
                   f"bound = median of {len(bnd)} per-run bounds ({lo:.3g}-{hi:.3g} d)")
            c[18] = (f"no measured PBST run; {how}{other_txt}", None)
        else:
            c[4] = c[5] = c[12] = (DASH, None)
            c[18] = (f"no PBST run could be measured or bounded{other_txt}", None)
        if not tka:
            c[18] = (c[18][0] + "; no PBST run with a reliable Ka1 (see rows)", None)

    # ---------------- dimer ----------------
    dim_status = []
    ka_ok = [x for x in S if qc(x)["ka_ok"]]
    c[7] = (mean_sd([x["ka1"] for x in ka_ok], s3) or DASH, None)
    if not ka_ok:
        dim_status.append(f"no PBST run passes Ka2 QC (R^2 >= {KA_R2_MIN})")
    long_ = [x for x in S if qc(x)["kdis_ok"]]
    phased = [x for x in S if qc(x)["dimer_phase"]]
    if long_:
        c[8] = (mean_sd([x["kdis1"] for x in long_], s3), None)
        c[9] = (mean_sd([x["t12_dimer_min"] for x in long_], f1), None)
    elif phased:
        c[8] = (mean_sd([x["kdis1"] for x in phased], s3, approx=True), None)
        c[9] = (mean_sd([x["t12_dimer_min"] for x in phased], f1, approx=True), None)
        dim_status.append(f"Kdis2 approximate: all PBST runs < {DIMER_MIN_DISSOC} h dissociation (overestimated)")
    else:
        c[8] = c[9] = (DASH, None)
    if len(phased) < len(S):
        n_out = len(S) - len(phased)
        dim_status.append(f"{n_out} run{'s' if n_out > 1 else ''} without a dimer phase left out")
    both = [x for x in ka_ok if qc(x)["kdis_ok"]]
    if both:
        c[13] = (mean_sd([x["KD1_nM"] for x in both], f2), None)
    elif ka_ok:
        c[13] = (mean_sd([x["KD1_nM"] for x in ka_ok], f2, approx=True) or DASH, None)
    else:
        c[13] = (DASH, None)
    c[19] = ("; ".join(dim_status) if dim_status else
             f"Ka2 from {len(ka_ok)} run(s), Kdis2 from {len(long_)} run(s) passing QC", None)
    return c


def put(t, row, col, value, fmt):
    if isinstance(value, (float, np.floating)):
        value = float(f"{float(value):.12g}")   # Numbers stores 15 significant digits; avoid the warning
    t.write(row, col, value)
    if fmt:
        kind, dp = FMT[fmt]
        t.set_cell_formatting(row, col, kind, decimal_places=dp)


def table_groups(rows):
    groups = []
    for i, r in enumerate(rows[1:], 1):
        if r[0]:
            groups.append(dict(name=r[0].strip(), exp_rows=[], empty_rows=[]))
        if r[1] not in (None, ""):
            groups[-1]["exp_rows"].append((i, str(r[1]).strip()))
        else:
            groups[-1]["empty_rows"].append(i)
    return groups


def verify(src, out):
    a, b = Document(src), Document(out)
    assert [s.name for s in a.sheets] == [s.name for s in b.sheets], "sheet list changed"
    for sa, sb in zip(a.sheets, b.sheets):
        for ta, tb in zip(sa.tables, sb.tables):
            ra, rb = ta.rows(values_only=True), tb.rows(values_only=True)
            if sa.name != TAB:
                assert ta.name == tb.name and len(ra) == len(rb), f"{sa.name}: table changed"
                for x, y in zip(ra, rb):
                    for u, v in zip(x, y):
                        same = (u == v) or (isinstance(u, float) and isinstance(v, float)
                                            and abs(u - v) <= 1e-12 * max(1, abs(u)))
                        assert same, f"{sa.name}: {u!r} -> {v!r}"
            else:
                for x, y in zip(ra[1:], rb[1:]):   # mean rows were empty, so only non-empty cells must match
                    assert all(u in (None, "") or u == v for u, v in zip(x[:3], y[:3])), \
                        f"{TAB}: identity columns changed: {x[:3]} -> {y[:3]}"
    return len(a.sheets)


def main():
    cache = pickle.load(open(os.path.join(W, "fit_cache.pkl"), "rb"))
    index = {(e["r"]["interaction"], e["r"]["experiment"]): e
             for e in cache.values() if "error" not in e}

    doc = Document(SRC)
    t = [s for s in doc.sheets if s.name == TAB][0].tables[0]
    groups = table_groups(t.rows(values_only=True))
    for col, text in HEADERS.items():
        t.write(0, col, text)
    t.name = TITLE

    n_rows = 0
    for g in groups:
        entries = []
        for i, exp in g["exp_rows"]:
            e = index[(g["name"], exp)]
            entries.append(e)
            for col, (v, fmt) in row_cells(e).items():
                put(t, i, col, v, fmt)
            n_rows += 1
        for col, (v, fmt) in mean_cells(entries).items():
            put(t, g["empty_rows"][0], col, v, fmt)
    doc.save(OUT)
    n_sheets = verify(SRC, OUT)
    print(f"filled {n_rows} experiment rows + {len(groups)} group-mean rows -> {OUT}")
    print(f"verified: {n_sheets} sheets present, all {n_sheets - 1} other tabs' values unchanged "
          f"(floats to 1e-12), identity columns of '{TAB}' unchanged")


if __name__ == "__main__":
    main()
