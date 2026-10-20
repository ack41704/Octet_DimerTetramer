"""
Full batch run across all Octet files under the EB1/EB2/EB3 folders: fits every
experiment with the same pipeline the GUI uses by default (parse_for_gui ->
default_mask -> fit_wells, with stage1_full's 4-point multistart), and exports:

  1. All_78_Fits.pdf            one page per file, data (dots) + fit (lines)
  2. comprehensive_summary.csv  one row per file -- dimer AND tetramer constants,
                                 named explicitly (not "1"/"2") to avoid the
                                 numbering-convention ambiguity entirely. Tetramer
                                 columns are split into a validated value (blank
                                 unless dissoc >= 8h) and an "unvalidated" value
                                 (always populated) so short-window runs can't be
                                 mistaken for real measurements of tetramer koff.
  3. all_data_points.csv        every well in every file, every data point AND
                                 the matching model-predicted value (blank for
                                 wells the auto-mask excluded, since those were
                                 never fit) -- both association and dissociation.

Wells the auto-mask would exclude (controls, near-zero, drift, too-short
dissociation, dominant sensor, failed replicate) are POINT-EXPORTED but not fit,
matching the GUI's default checklist state. Nothing here overrides that checklist
per-file; run the GUI and toggle a box if a specific file needs a curated exclusion.
`amplitude_outlier_wells` (>5x or <0.2x the median peak) stays as an informational
column only -- it mostly marks the lowest concentration, which is expected; the real
sensor failures are now caught by default_mask's dominant/failed-replicate rules.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys
import os
import glob
import csv
import pickle
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_all_fits import (parse_for_gui, default_mask, fit_wells, draw_fit_panels,  # noqa: E402
                           summary_from_result, page_title, f2a, sub, buffer_of)
from batch_fit import split_name  # noqa: E402

DEFAULT_GLOB = DATA_DIR + "/EB*Interactions - Preprocessed Data FIles/*/*.csv"
W = WORK_DIR


def amplitude_outlier_flags(parsed, mask):
    """Wells whose peak response is a large outlier vs the rest of the INCLUDED
    series -- informational only. This is what catches the 182 nM well in 2026.27
    that no other filter (control/near-zero/drift) notices."""
    included = [w for w, m in zip(parsed["wells"], mask) if m]
    if len(included) < 3:
        return {}
    peaks = np.array([w["y"].max() for w in included])
    med = np.median(peaks)
    flags = {}
    for w, p in zip(included, peaks):
        if med > 0 and (p / med > 5 or p / med < 0.2):
            flags[w["conc_nM"]] = p / med
    return flags


def process_one(path, pdf, summary_rows, data_rows, cache, max_assoc_pts=120, max_dissoc_pts=150):
    folder = os.path.basename(os.path.dirname(path))
    experiment, fitnum = split_name(path)
    base = dict(file=os.path.basename(path), folder=folder, experiment=experiment, fit=fitnum)

    parsed = parse_for_gui(path, folder)
    if parsed is None:
        summary_rows.append({**base, "interaction": folder, "status": "failed: no usable boundary/sensors"})
        cache[path] = dict(folder=folder, error="no usable fit (boundary/sensors)")
        return False

    mask, reasons = default_mask(parsed)
    keep_idx = [i for i, m in enumerate(mask) if m]
    if len(keep_idx) < 3:
        summary_rows.append({**base, "interaction": parsed["interaction"],
                             "status": f"failed: only {len(keep_idx)} usable sensors"})
        cache[path] = dict(folder=folder, error=f"only {len(keep_idx)} usable sensors")
        return False

    r = fit_wells(parsed, keep_idx, experiment, fitnum)
    s = summary_from_result(r)
    outliers = amplitude_outlier_flags(parsed, mask)

    summary_rows.append({
        **base, "interaction": r["interaction"], "buffer": s["buffer"],
        "assoc_min": round(s["assoc_min"], 1), "dissoc_h": round(s["dissoc_h"], 2),
        "n_sensors_used": s["n_sensors"], "n_sensors_total": len(parsed["wells"]),
        "dimer_ka_M1s1": s["ka1"], "dimer_kdis_s1": s["kdis1"], "dimer_KD_nM": s["KD1_nM"],
        "dimer_thalf_min": s["t12_dimer_min"], "dimer_ka_regression_r2": s["r2_kobs"],
        "dimer_resolved": s["dimerOK"],
        "tetramer_ka_s1": s["ka2"],
        "tetramer_kdis_s1_VALIDATED": s["kdis2"] if s["tetramerOK"] else "",
        "tetramer_KD_VALIDATED": s["KD2"] if s["tetramerOK"] else "",
        "tetramer_thalf_days_VALIDATED": s["t12_tetramer_d"] if s["tetramerOK"] else "",
        "tetramer_kdis_s1_unvalidated": s["kdis2"],
        "tetramer_thalf_days_unvalidated": s["t12_tetramer_d"],
        "tetramer_resolved": s["tetramerOK"],
        "tetramer_note": ("resolved -- dissociation window long enough to trust kdis/KD/t1/2"
                          if s["tetramerOK"] else
                          f"dissociation window only {s['dissoc_h']:.1f} h (<8h) -- the "
                          f"'_unvalidated' tetramer columns are a LOWER-BOUND stability "
                          f"estimate only, not a measurement"),
        "R2_dissociation_biexponential": s["R2_dissoc"],
        "amplitude_outlier_wells": ("; ".join(f"{c:g} nM ({v:.1f}x median)" for c, v in outliers.items())
                                    if outliers else ""),
        "status": "ok"})

    # ---- every data point, every well (kept or not), both phases ----
    tb = r["tb"]
    kept_amp = {orig_i: (r["A1s"][j], r["A2s"][j]) for j, orig_i in enumerate(keep_idx)}
    for i, w in enumerate(parsed["wells"]):
        included = mask[i]
        reason = reasons[i] or ""
        ma = w["t"] <= tb
        ta, ya = sub(w["t"][ma], w["y"][ma], max_assoc_pts, max_assoc_pts // 2)
        assoc_fit = r["assoc_fits"].get(w["conc_nM"]) if included else None
        for t_, y_ in zip(ta, ya):
            ym = f2a(t_, *assoc_fit) if assoc_fit is not None else ""
            data_rows.append((os.path.basename(path), r["interaction"], experiment, w["conc_nM"],
                             included, reason, "association", round(float(t_), 2),
                             round(float(y_), 6), round(float(ym), 6) if ym != "" else ""))
        md = w["t"] >= tb
        td, yd = sub(w["t"][md] - tb, w["y"][md], max_dissoc_pts, max_dissoc_pts)
        amp = kept_amp.get(i)
        for t_, y_ in zip(td, yd):
            ym = (amp[0] * np.exp(-r["kf"] * t_) + amp[1] * np.exp(-r["ks"] * t_)) if amp else ""
            data_rows.append((os.path.basename(path), r["interaction"], experiment, w["conc_nM"],
                             included, reason, "dissociation", round(float(t_), 2),
                             round(float(y_), 6), round(float(ym), 6) if ym != "" else ""))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_fit_panels(axes, r)
    fig.suptitle(page_title(r, s), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    pdf.savefig(fig)
    plt.close(fig)

    cache[path] = dict(folder=folder, parsed=parsed, mask=mask, reasons=reasons, r=r, s=s)
    return True


def main():
    files = sorted(glob.glob(DEFAULT_GLOB))
    print(f"found {len(files)} files")
    summary_rows, data_rows, cache = [], [], {}
    n_ok = n_fail = 0
    with PdfPages(os.path.join(W, "All_78_Fits.pdf")) as pdf:
        for path in files:
            try:
                ok = process_one(path, pdf, summary_rows, data_rows, cache)
            except Exception as e:
                ok = False
                cache[path] = dict(folder=os.path.basename(os.path.dirname(path)),
                                   error=f"{type(e).__name__}: {e}")
                summary_rows.append(dict(file=os.path.basename(path),
                                         folder=os.path.basename(os.path.dirname(path)),
                                         interaction=os.path.basename(os.path.dirname(path)),
                                         status=f"error: {type(e).__name__}: {e}"))
                print(f"FAIL {os.path.basename(path)}: {type(e).__name__}: {e}")
                print(traceback.format_exc()[-800:])
            if ok:
                n_ok += 1
                r_ = summary_rows[-1]
                print(f"OK   {r_['interaction']:32s} {r_['experiment'][:26]:26s} "
                      f"dimerKD={r_['dimer_KD_nM']:.2f}nM  tetramerOK={r_['tetramer_resolved']}  "
                      f"outliers={r_['amplitude_outlier_wells'] or 'none'}")
            else:
                n_fail += 1
                print(f"SKIP {os.path.basename(path)}: {summary_rows[-1]['status']}")

    all_keys = []
    for row in summary_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(os.path.join(W, "comprehensive_summary.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=all_keys, restval="")
        wtr.writeheader()
        wtr.writerows(summary_rows)

    with open(os.path.join(W, "all_data_points.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["file", "interaction", "experiment", "conc_nM", "included_in_fit",
                     "exclusion_reason", "phase", "time_s", "response_data", "response_model"])
        wtr.writerows(data_rows)

    cache_path = os.path.join(W, "fit_cache.pkl")
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)

    import resolve_tetramer  # categorize tetramer results + same-buffer references
    resolve_tetramer.main()

    print(f"\n{n_ok} fitted, {n_fail} failed")
    print(f"All_78_Fits.pdf, comprehensive_summary.csv ({len(summary_rows)} rows), "
          f"all_data_points.csv ({len(data_rows)} rows), {cache_path} ({len(cache)} entries)")


if __name__ == "__main__":
    main()
