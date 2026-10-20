"""
Give every fitted run a defensible tetramer result, using the cached fits (no refitting).

Categories (rules and evidence live in plot_all_fits.tetramer_resolution):
  measured           >=8 h dissociation and a slow phase actually detected -> point estimate
  lower_bound_15h    >=8 h but no decay detected -> t1/2 lower bound from sensor spread
  lower_bound_short  <8 h window -> t1/2 lower bound from the run's own (overestimated) rate
  no_bound           nothing to bound from (no slow phase, too little late signal)

For every non-measured run whose interaction+buffer group has measured 15 h replicates, the
group's measured mean is attached as a REFERENCE (separate columns): the value expected for
that interaction, not a measurement of that run, and never an extra replicate.

Updates fit_cache.pkl (backup kept as fit_cache.pre_resolution.pkl), writes
tetramer_resolution.csv, and adds the new columns to comprehensive_summary.csv.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import sys
import csv
import shutil
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_all_fits import summary_from_result, group_references  # noqa: E402

W = WORK_DIR
CACHE = os.path.join(W, "fit_cache.pkl")


def apply_group_refs(s, r, refs):
    ref = refs.get((r["interaction"], s["buffer"])) if s["tet_category"] != "measured" else None
    s["tet_group_n"] = ref["n"] if ref else None
    s["tet_group_thalf_mean_d"] = ref["thalf_mean_d"] if ref else None
    s["tet_group_thalf_sd_d"] = ref["thalf_sd_d"] if ref else None
    s["tet_group_kdis2_mean"] = ref["kdis2_mean"] if ref else None


def main():
    cache = pickle.load(open(CACHE, "rb"))
    backup = os.path.join(W, "fit_cache.pre_resolution.pkl")
    if not os.path.exists(backup):
        shutil.copy2(CACHE, backup)

    for p, e in cache.items():
        if "error" not in e:
            e["s"] = summary_from_result(e["r"])
    ok = [(p, e) for p, e in cache.items() if "error" not in e]
    refs = group_references([(e["r"], e["s"]) for _, e in ok])
    for _, e in ok:
        apply_group_refs(e["s"], e["r"], refs)

    with open(CACHE, "wb") as fh:
        pickle.dump(cache, fh)

    cols = ["file", "interaction", "buffer", "dissoc_h", "tet_category",
            "tet_kdis2", "tet_thalf_d", "tet_kdis2_upper", "tet_thalf_lower_d", "tet_KD2_upper",
            "tet_bound_method", "tet_group_n", "tet_group_thalf_mean_d", "tet_group_thalf_sd_d",
            "tet_group_kdis2_mean"]
    rows = []
    for p, e in sorted(ok, key=lambda z: (z[1]["r"]["interaction"], z[1]["s"]["buffer"], z[0])):
        s = e["s"]
        rows.append({"file": os.path.basename(p), "interaction": e["r"]["interaction"],
                     "buffer": s["buffer"], "dissoc_h": round(s["dissoc_h"], 2),
                     **{c: s.get(c) for c in cols[4:]}})
    with open(os.path.join(W, "tetramer_resolution.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # merge into comprehensive_summary.csv, and correct its tetramer_resolved flag
    summ_path = os.path.join(W, "comprehensive_summary.csv")
    summ = list(csv.DictReader(open(summ_path)))
    by_file = {os.path.basename(p): e["s"] for p, e in ok}
    new_cols = cols[4:]
    for row in summ:
        s = by_file.get(row["file"])
        if s is None:
            continue
        row["tetramer_resolved"] = s["tetramerOK"]
        if s["tetramerOK"]:
            row["tetramer_kdis_s1_VALIDATED"] = s["kdis2"]
            row["tetramer_KD_VALIDATED"] = s["KD2"]
            row["tetramer_thalf_days_VALIDATED"] = s["t12_tetramer_d"]
        else:
            row["tetramer_kdis_s1_VALIDATED"] = ""
            row["tetramer_KD_VALIDATED"] = ""
            row["tetramer_thalf_days_VALIDATED"] = ""
        for c in new_cols:
            v = s.get(c)
            row[c] = "" if v is None else v
    fields = list(summ[0].keys())
    for c in new_cols:
        if c not in fields:
            fields.append(c)
    with open(summ_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(summ)

    cats = Counter(e["s"]["tet_category"] for _, e in ok)
    with_ref = sum(1 for _, e in ok if e["s"]["tet_group_n"])
    print("tetramer categories:", dict(cats))
    print(f"non-measured runs with a same-buffer measured reference: {with_ref}")
    print("wrote tetramer_resolution.csv, updated comprehensive_summary.csv and fit_cache.pkl")


if __name__ == "__main__":
    main()
