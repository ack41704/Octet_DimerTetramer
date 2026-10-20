"""Diagnostic plots: data vs fit, for every experiment the batch pipeline fit.

One page per experiment in a single multi-page PDF, ordered by interaction then
experiment. Each page has two panels:
  left  = association phase: data (dots) + per-concentration two-exponential fit (lines)
  right = dissociation phase: data (dots) + shared two-exponential fit (lines),
          same kdis1/kdis2 across concentrations, only amplitudes differ

Reuses the exact fitting logic from batch_fit.py (boundary detection, well filtering,
stage1/stage2) so what's plotted is exactly what produced batch_results.csv / the
Numbers tab -- not a re-derivation.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, os, glob, csv
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_octet import parse_octet_csv
from batch_fit import LIG, split_name, boundary, sub, f2a

W = WORK_DIR


def stage1_full(curves):
    """same as batch_fit.stage1 but also returns per-curve amplitudes for plotting.
    Light multistart (4 starting points spanning the kdis1/kdis2 scale) picks the
    lowest-cost local optimum -- can only match or improve the old single-start
    result, since we keep whichever converges to lower SSE."""
    n = len(curves)
    lb = [-5, -8] + [0] * n + [0] * n
    ub = [-0.5, -3.5] + [3] * n + [3] * n

    def r(p):
        kf, ks = 10 ** p[0], 10 ** p[1]
        A1, A2 = p[2:2 + n], p[2 + n:2 + 2 * n]
        return np.concatenate([a1 * np.exp(-kf * c[0]) + a2 * np.exp(-ks * c[0]) - c[1]
                               for c, a1, a2 in zip(curves, A1, A2)])

    amp0 = [float(np.clip(c[1][0] * .3, 0, 3)) for c in curves]
    amp1 = [float(np.clip(c[1][0] * .7, 0, 3)) for c in curves]
    best = None
    for kf0, ks0 in ((8e-4, 1.5e-6), (5e-3, 5e-6), (1e-4, 5e-7), (2e-3, 5e-5)):
        p0 = [np.log10(kf0), np.log10(ks0)] + amp0 + amp1
        res = least_squares(r, p0, bounds=(lb, ub), method="trf",
                            xtol=1e-14, ftol=1e-14, max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    res = best
    kf, ks = 10 ** res.x[0], 10 ** res.x[1]
    A1, A2 = res.x[2:2 + n], res.x[2 + n:2 + 2 * n]
    ya = np.concatenate([c[1] for c in curves])
    R2 = 1 - np.sum(res.fun ** 2) / np.sum((ya - ya.mean()) ** 2)
    return kf, ks, A1, A2, R2


def stage2_full(wells, tb):
    """same as batch_fit.stage2 but keeps per-well fit params + all k_obs points"""
    fits, pts, slow = {}, [], []
    for w in wells:
        m = w["t"] <= tb
        ta, ya = w["t"][m], w["y"][m]
        if len(ta) < 40:
            continue
        i = np.unique(np.geomspace(1, len(ta) - 1, 300).astype(int))
        ta_s, ya_s = ta[i], ya[i]
        span = max(ya_s[-1], 1e-6)
        try:
            p, _ = curve_fit(f2a, ta_s, ya_s, p0=[span * .5, 5e-3, span * .5, 4e-4],
                             bounds=([0, 1e-5, 0, 1e-6], [3, 1, 3, 1e-1]), maxfev=40000)
        except Exception:
            continue
        A1, k1, A2, k2 = p
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1
        fits[w["conc_nM"]] = (A1, k1, A2, k2)
        tot = A1 + A2
        if tot <= 0:
            continue
        cand = [(A, k) for A, k in ((A1, k1), (A2, k2)) if A / tot >= .15 and k < 5e-2]
        if cand:
            A, k = max(cand, key=lambda z: z[0])
            pts.append((w["conc_M"], k))
    if len(pts) >= 3:
        c = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
        sl, ic, rr, pv, se = stats.linregress(c, k)
    else:
        sl = ic = rr = np.nan
    return fits, pts, sl, ic, (rr ** 2 if np.isfinite(rr) else np.nan)


def infer_interaction(folder):
    """(interaction label, genotype-or-None). Falls back to the raw folder name
    for anything outside the EBn+Partner convention -- still usable, just without
    a pretty ligand name or genotype-filtering."""
    lig = LIG.get(folder.split("+")[0])
    if lig:
        partner = folder.split("+", 1)[1]
        geno = "R79P" if partner.endswith("R79P") else ("WT" if partner.endswith("-WT") else None)
        return f"{lig} + {partner}", geno
    return folder, None


def parse_for_gui(path, folder):
    """Parse once and apply only the STRUCTURAL filters (genotype, which row this
    file belongs to) -- deliberately keeps controls and near-zero wells in the
    returned list so a checklist UI can show and toggle every sensor, not just the
    ones a heuristic already approved. Returns None only when there's truly nothing
    to fit (boundary can't be found). No curve-fitting happens here."""
    interaction, geno = infer_interaction(folder)
    wells = parse_octet_csv(path)
    wells = [w for w in wells if np.isfinite(w["conc_M"])]
    if geno:
        wells = [w for w in wells if w["label"] == geno]
    if not wells:
        return None
    # boundary detection needs real signal, matching the historical filter order
    # (controls out, then near-zero out) so it's identical to the old pipeline
    signal_wells = [w for w in wells if not w["is_control"] and w["y"].max() > 0.02]
    if len(signal_wells) < 3:
        return None
    tb = boundary(signal_wells)
    if tb is None:
        return None
    return dict(interaction=interaction, wells=wells, tb=tb)


SENSOR_RATIO = 3.0   # physically implausible peak-response ratio between sensors (see below)


def inconsistent_sensors(wells, ok_idx):
    """Two sensor failures the other filters miss, applied to the sensors that passed them:
      dominant   peak >3x the LARGEST peak of every other sensor. Binding saturates, so no
                 concentration step can do that; in 2026.26/2026.27 a 182 nM sensor read 12-18x
                 its 182.3 nM replicate (likely a mislabelled control) and, carrying ~99% of the
                 squared signal, dominated the unweighted global fit (dimer t1/2 404 min vs 2 min).
      failed     at a concentration measured by >=2 sensors, one peaks <1/3 of its replicate.
                 A sensor can fail to bind (unloaded / reference tip) but can't bind 3x more
                 than an identical one, so the LOW one is dropped."""
    peaks = {i: float(wells[i]["y"].max()) for i in ok_idx}
    flagged = {}
    for i in ok_idx:
        others = [peaks[j] for j in ok_idx if j != i]
        if others and peaks[i] > SENSOR_RATIO * max(others):
            flagged[i] = f"dominant sensor ({peaks[i] / max(others):.2g}x every other sensor's peak)"
    rest = sorted((i for i in ok_idx if i not in flagged), key=lambda i: wells[i]["conc_nM"])
    groups = []
    for i in rest:
        c0 = wells[groups[-1][0]]["conc_nM"] if groups else None
        if c0 is not None and abs(wells[i]["conc_nM"] - c0) <= 0.02 * c0:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        top = max(peaks[i] for i in g)
        for i in g:
            if peaks[i] * SENSOR_RATIO < top:
                flagged[i] = f"failed replicate ({peaks[i] / top:.2g}x its same-concentration replicate)"
    return flagged


def default_mask(parsed):
    """Which wells the automatic pipeline keeps, plus a reason string for every excluded one --
    this IS the pre-checked state of the GUI's checklist. Exclusion order: control -> near-zero
    -> too-short-dissociation -> upward-drift (batch_fit.analyze()'s original rules), then
    dominant / failed-replicate sensors among the survivors (inconsistent_sensors)."""
    tb = parsed["tb"]
    mask, reasons = [], []
    for w in parsed["wells"]:
        reason = None
        if w["is_control"]:
            reason = "control well"
        elif w["y"].max() <= 0.02:
            reason = "near-zero signal"
        else:
            yd = w["y"][w["t"] >= tb]
            if len(yd) < 100:
                reason = "dissociation phase too short"
            elif yd[-100:].mean() > yd[:100].mean():
                reason = "drifts upward (not physical)"
        mask.append(reason is None)
        reasons.append(reason)
    for i, why in inconsistent_sensors(parsed["wells"], [i for i, m in enumerate(mask) if m]).items():
        mask[i], reasons[i] = False, why
    return mask, reasons


def fit_wells(parsed, keep_idx, experiment=None, fit=None):
    """Run stage1/stage2 on exactly the wells at keep_idx. This is the only
    function that actually fits -- parsing and mask selection above are both
    fitting-free, so toggling a checkbox re-fits in milliseconds without
    re-reading the CSV."""
    wells = [parsed["wells"][i] for i in keep_idx]
    if len(wells) < 3:
        return None
    tb = parsed["tb"]
    curves = []
    for w in wells:
        m = w["t"] >= tb
        td, yd = sub(w["t"][m] - tb, w["y"][m], 200, 200)
        curves.append((td, yd))
    kf, ks, A1s, A2s, R2d = stage1_full(curves)
    assoc_fits, kobs_pts, ka1, ic, r2k = stage2_full(wells, tb)
    return dict(interaction=parsed["interaction"], experiment=experiment, fit=fit,
                tb=tb, keep=wells, kf=kf, ks=ks, A1s=A1s, A2s=A2s, R2d=R2d,
                assoc_fits=assoc_fits, kobs_pts=kobs_pts, ka1=ka1, ic=ic, r2k=r2k)


def analyze_for_plot(path, folder):
    """Thin wrapper preserving the exact old one-call signature (used by main()
    below): parse -> default mask -> fit. Bit-for-bit equivalent to the previous
    monolithic implementation, verified by regenerating All_Fits_Diagnostic.pdf
    (identical page count and byte size) after this refactor."""
    parsed = parse_for_gui(path, folder)
    if parsed is None:
        return None
    mask, _ = default_mask(parsed)
    keep_idx = [i for i, m in enumerate(mask) if m]
    if len(keep_idx) < 3:
        return None
    experiment, fit = split_name(path)
    return fit_wells(parsed, keep_idx, experiment, fit)


def buffer_of(exp):
    e = exp.lower()
    if "pbstb" in e:
        return "PBSTB"
    if "casein" in e:
        return "PBST+casein"
    return "PBST"


def compute_ka2(assoc_fits, kdis2):
    """same 'slow, concentration-independent component at the top concentrations'
    logic as batch_fit.stage2, re-derived from the per-well fits analyze_for_plot
    already computed -- no re-fitting, just re-reading."""
    items = []
    for conc_nM, (A1, k1, A2, k2) in assoc_fits.items():
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1
        tot = A1 + A2
        if tot <= 0:
            continue
        if abs(k1 - k2) / max(k1, k2) > 1e-3 and A2 / tot >= .10 and k2 < 5e-2:
            items.append((conc_nM, k2))
    if not items:
        return np.nan
    items.sort(key=lambda z: z[0])
    top2 = items[-2:] if len(items) >= 2 else items
    ka2 = float(np.mean([k for _, k in top2])) - kdis2
    return ka2 if ka2 > 0 else np.nan    # slow association no faster than kdis2: not determinable


# --- tetramer resolution rules (evidence: tetramer_window_test.py, tetramer_bounds_check.py) ---
# A stage1 kdis2 at or below this sits on the fit's 1e-8 lower bound: no slow decay was detected.
FLOOR_KS = 2e-8
# A stage1 kdis1 at or below this sits on the fit's 1e-5 lower bound: no fast (dimer) phase was
# found, so both exponentials are describing the slow decay (2026.40: flat after 2 h association).
FLOOR_KF = 1.05e-5
# Truncating the 13 measured 15 h runs showed a short-window kdis2 OVERestimates the true rate
# (13/13 runs at <=3 h, 12/13 at 4 h, smallest ratio 0.92). Dividing by 0.9 makes
# "kdis2 < apparent/0.9" hold in 13/13 runs at every window length present in the data.
BOUND_FACTOR = 0.9


def sensor_bound_kdis2(r, whole_window=False):
    """95% upper bound on kdis2 from sensor-to-sensor spread of the late-phase decay rate
    (-slope/level, after the dimer has decayed). Used only when stage1 found no slow phase.
    Sensors are independent measurements, unlike the ~2000 autocorrelated points within a
    curve -- a within-curve profile likelihood gave bounds (>300-400 d) far too tight.
    whole_window=True uses the entire dissociation (for runs with no dimer phase): any
    faster component only raises the apparent rate, so the bound stays conservative."""
    tb = r["tb"]; W = r["keep"][0]["t"][-1] - tb
    start = 0.0 if whole_window else max(5 * np.log(2) / r["kf"], 0.4 * W)
    ks = []
    for w in r["keep"]:
        m = (w["t"] >= tb + start) & (w["t"] <= tb + W)
        td, yd = w["t"][m] - tb, w["y"][m]
        if len(td) < 50 or yd.mean() < 0.05:
            continue
        b, _ = np.polyfit(td, yd, 1)
        ks.append(-b / yd.mean())
    if len(ks) < 2:
        return None, len(ks)
    ks = np.array(ks); n = len(ks)
    return float(ks.mean() + stats.t.ppf(0.975, n - 1) * ks.std(ddof=1) / np.sqrt(n)), n


def tetramer_resolution(r, ks, dis_h, ka2):
    """Every run gets a defensible tetramer statement -- a measurement or a bound, never
    a short-window number presented as a measurement.
      measured              >=8 h window and a slow phase was actually detected
      lower_bound_15h       >=8 h window but no decay detected (sensor-spread bound)
      lower_bound_short     <8 h window: own apparent rate is an upper bound on kdis2
      no_bound              no slow phase detected and too little late signal to bound it
    If stage 1 found no dimer phase (kdis1 on its floor) the biexponential split is not a
    dimer/tetramer split, so ks is not used; the bound comes from the whole-window decay."""
    ln2 = np.log(2)
    if r["kf"] <= FLOOR_KF:
        ub, n = sensor_bound_kdis2(r, whole_window=True)
        if ub is None or ub <= 0:
            return dict(tet_category="no_bound", tet_kdis2=None, tet_thalf_d=None,
                        tet_kdis2_upper=None, tet_thalf_lower_d=None, tet_KD2_upper=None,
                        tet_bound_method="no dimer phase resolved and no usable whole-window decay")
        return dict(tet_category="lower_bound_15h" if dis_h >= 8 else "lower_bound_short",
                    tet_kdis2=None, tet_thalf_d=None, tet_kdis2_upper=ub,
                    tet_thalf_lower_d=ln2 / ub / 86400,
                    tet_KD2_upper=(ub / ka2 if np.isfinite(ka2) and ka2 > 0 else None),
                    tet_bound_method=(f"no dimer phase resolved: sensor-to-sensor 95% bound on the "
                                      f"whole-window decay rate (n={n} sensors)"))
    if ks > FLOOR_KS and dis_h >= 8:
        return dict(tet_category="measured", tet_kdis2=ks, tet_thalf_d=ln2 / ks / 86400,
                    tet_kdis2_upper=None, tet_thalf_lower_d=None,
                    tet_KD2_upper=None, tet_bound_method="")
    if ks > FLOOR_KS:
        ub, method = ks / BOUND_FACTOR, ("own-window apparent rate / 0.9 (short windows overestimate "
                                         "kdis2: 13/13 validated runs)")
    else:
        ub, n = sensor_bound_kdis2(r)
        method = f"sensor-to-sensor 95% bound on late-phase rate (n={n} sensors)"
        if ub is None or ub <= 0:
            return dict(tet_category="no_bound", tet_kdis2=None, tet_thalf_d=None,
                        tet_kdis2_upper=None, tet_thalf_lower_d=None, tet_KD2_upper=None,
                        tet_bound_method=("no slow phase detected and too few sensors with late "
                                          "signal to bound it" if ub is None else
                                          "late-phase signal flat or rising -- no finite bound"))
    return dict(tet_category="lower_bound_15h" if dis_h >= 8 else "lower_bound_short",
                tet_kdis2=None, tet_thalf_d=None, tet_kdis2_upper=ub,
                tet_thalf_lower_d=ln2 / ub / 86400,
                tet_KD2_upper=(ub / ka2 if np.isfinite(ka2) and ka2 > 0 else None),
                tet_bound_method=method)


def group_references(entries):
    """(interaction, buffer) -> mean/SD of MEASURED tetramer t1/2 and kdis2 across runs.
    Attached to non-measured runs as a reference only: it is the group's value, not that
    run's measurement, and must never be counted as an additional replicate."""
    by = {}
    for r, s in entries:
        if s.get("tet_category") == "measured":
            by.setdefault((r["interaction"], s["buffer"]), []).append((s["tet_thalf_d"], s["tet_kdis2"]))
    out = {}
    for k, v in by.items():
        th = np.array([a for a, _ in v]); kd = np.array([b for _, b in v])
        out[k] = dict(n=len(v), thalf_mean_d=float(th.mean()),
                      thalf_sd_d=float(th.std(ddof=1)) if len(v) > 1 else None,
                      kdis2_mean=float(kd.mean()))
    return out


def summary_from_result(r):
    """KD1/KD2/QC flags -- the exact same thresholds as batch_fit.analyze() and
    build_tab.py (dissoc >=8h, ka1 regression r2>0.9), computed from the fit
    analyze_for_plot() already produced. Single source of truth for all three
    consumers (batch CSV, PDF, GUI). tetramerOK now also requires a slow phase to
    have been detected: a 15 h run whose kdis2 sits on the fit floor is NOT resolved."""
    kf, ks, ka1, r2k = r["kf"], r["ks"], r["ka1"], r["r2k"]
    keep = r["keep"]
    t_dissoc = keep[0]["t"][-1] - r["tb"]
    ka2 = compute_ka2(r["assoc_fits"], ks)
    dimer_phase = kf > FLOOR_KF
    KD1 = kf / ka1 * 1e9 if np.isfinite(ka1) and ka1 > 0 and dimer_phase else np.nan
    KD2 = ks / ka2 if np.isfinite(ka2) and ka2 > 0 and dimer_phase else np.nan
    ka1qc = (r2k or 0) > 0.9 and (ka1 or 0) > 0
    dis_h = t_dissoc / 3600
    s = dict(KD1_nM=KD1, KD2=KD2, kdis1=kf, kdis2=ks, ka1=ka1, ka2=ka2,
             R2_dissoc=r["R2d"], r2_kobs=r2k, buffer=buffer_of(r["experiment"]),
             assoc_min=r["tb"] / 60, dissoc_h=dis_h, dimer_phase_resolved=bool(dimer_phase),
             dimerOK=bool(dis_h >= 8 and ka1qc and dimer_phase),
             tetramerOK=bool(dis_h >= 8 and ks > FLOOR_KS and dimer_phase),
             t12_dimer_min=np.log(2) / kf / 60, t12_tetramer_d=np.log(2) / ks / 86400,
             decay_frac=1 - np.exp(-ks * t_dissoc), n_sensors=len(keep))
    s.update(tetramer_resolution(r, ks, dis_h, ka2))
    return s


def draw_fit_panels(axes, r):
    """Draw data+fit into a pre-existing pair of matplotlib axes. Shared by the
    PDF generator (make_page, below) and the GUI (gui_app.py) -- same function,
    same pixels, so what you see live matches what's in the PDF and the tab."""
    keep = r["keep"]; tb = r["tb"]
    for ax in axes:
        ax.cla()
    concs = sorted({w["conc_nM"] for w in keep})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(concs)))
    colors = dict(zip(concs, cmap))

    for w in keep:
        c = colors[w["conc_nM"]]
        m = w["t"] <= tb
        ta, ya = sub(w["t"][m], w["y"][m], 150, 100)
        axes[0].plot(ta / 60, ya, '.', ms=2, alpha=.4, color=c)
        if w["conc_nM"] in r["assoc_fits"]:
            A1, k1, A2, k2 = r["assoc_fits"][w["conc_nM"]]
            tt = np.linspace(0, tb, 300)
            axes[0].plot(tt / 60, f2a(tt, A1, k1, A2, k2), '-', lw=1.4, color=c,
                        label=f"{w['conc_nM']:g} nM")
    axes[0].set_xlabel("association time (min)"); axes[0].set_ylabel("response (nm)")
    axes[0].set_title("Association (dots=data, lines=per-conc fit)", fontsize=10)
    axes[0].legend(fontsize=6.5, ncol=2)

    t_dissoc_end = keep[0]["t"][-1] - tb
    for w, A1, A2 in zip(keep, r["A1s"], r["A2s"]):
        c = colors[w["conc_nM"]]
        m = w["t"] >= tb
        td, yd = sub(w["t"][m] - tb, w["y"][m], 200, 150)
        axes[1].plot(td / 3600, yd, '.', ms=2, alpha=.4, color=c)
        tt = np.geomspace(1, t_dissoc_end, 300)
        axes[1].plot(tt / 3600, A1 * np.exp(-r["kf"] * tt) + A2 * np.exp(-r["ks"] * tt),
                    '-', lw=1.4, color=c)
    axes[1].set_xlabel("dissociation time (h)"); axes[1].set_ylabel("response (nm)")
    axes[1].set_title("Dissociation (dots=data, lines=shared global fit)", fontsize=10)


def page_title(r, s):
    return (f"{r['interaction']}   |   {r['experiment']}  (fit {r['fit']})   |   {s['buffer']}\n"
            f"assoc={s['assoc_min']:.0f} min  dissoc={s['dissoc_h']:.1f} h   "
            f"R2_dissoc={s['R2_dissoc']:.3f}  r2_kobs={s['r2_kobs']:.3f}\n"
            f"kdis1={s['kdis1']:.2e} s-1  kdis2={s['kdis2']:.2e} s-1  ka1={s['ka1']:.2e} M-1s-1  "
            f"KD1={s['KD1_nM']:.2f} nM   dimerOK={'Y' if s['dimerOK'] else 'N'}   "
            f"tetramer: {tetramer_label(s)}")


def tetramer_label(s):
    cat = s.get("tet_category")
    if cat == "measured":
        return f"measured, t1/2 = {s['tet_thalf_d']:.2f} d"
    if cat in ("lower_bound_15h", "lower_bound_short"):
        why = "no decay detected in 15 h" if cat == "lower_bound_15h" else "short window"
        return f"t1/2 > {s['tet_thalf_lower_d']:.2g} d (lower bound: {why})"
    if cat == "no_bound":
        return "cannot be bounded from this run"
    return "measured" if s.get("tetramerOK") else "not resolved"


def make_page(pdf, r):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_fit_panels(axes, r)
    s = summary_from_result(r)
    fig.suptitle(page_title(r, s), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    files = sorted(glob.glob(DATA_DIR + "/EB*Interactions - Preprocessed Data FIles/*/*.csv"))
    results = []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        try:
            r = analyze_for_plot(f, folder)
        except Exception as e:
            print(f"PLOT-FAIL {os.path.basename(f)}: {type(e).__name__}: {e}")
            r = None
        if r is not None:
            results.append(r)
            print(f"plotted  {r['interaction']:32s} {r['experiment'][:28]:28s}")
    results.sort(key=lambda r: (r["interaction"], r["experiment"]))

    out = os.path.join(W, "All_Fits_Diagnostic.pdf")
    with PdfPages(out) as pdf:
        for r in results:
            make_page(pdf, r)
    print(f"\n{len(results)} pages -> {out}")


if __name__ == "__main__":
    main()
