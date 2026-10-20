"""
Fit Octet files that are NOT in the EB* interaction folders with the standard pipeline (plot_all_fits: parse_for_gui ->
default_mask -> fit_wells -> summary_from_result, exactly as export_full_dataset.process_one) and store them in
fit_cache_extra.pkl, same entry format as fit_cache.pkl. The shared fit_cache.pkl is left untouched, so the master
table / master PDF scripts that iterate it are unaffected. extended_constants_all16_v2/_v3 read both caches.

EXTRA = (path, folder) pairs; the folder name gives the interaction the same way as the EB* folders do
("EA5+EphB2-His" -> "EphrinA5-Fc + EphB2-His"; batch_fit.LIG maps the ligand prefix).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import re
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_all_fits import parse_for_gui, default_mask, fit_wells, summary_from_result  # noqa: E402
from batch_fit import split_name  # noqa: E402

W = WORK_DIR
OUT = os.path.join(W, "fit_cache_extra.pkl")
EXTRA = [(DATA_DIR + "/2026.122 - Preprocessed Data - Fit 82.csv", "EA5+EphB2-His"),
         # master-table runs whose files were never in the EB* folders (found in ~/Downloads, 2026-09-30)
         (DATA_DIR + "/2026.85a - Preprocessed Data.csv", "EB1+EphB2-WT"),     # Global Fit 51
         (DATA_DIR + "/2026.87a - Preprocessed Data.csv", "EB1+EphB2-His"),    # Global Fit 54
         # added 2026-10-02 (ligand and buffer from the user: EphrinA5-Fc, PBST)
         (DATA_DIR + "/2026.123 - Preprocessed Data - Fit 83.csv", "EA5+EphB2-WT"),
         (DATA_DIR + "/2026.124 - Preprocessed Data - Fit 84.csv", "EA5+EphB2-R79P")]


def main():
    """fits only the files not yet in fit_cache_extra.pkl; entries already there are kept exactly as saved"""
    cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    for path, folder in EXTRA:
        if path in cache:
            continue
        experiment, fitnum = split_name(path)
        experiment = re.sub(r"\s*-\s*Preprocessed Data.*$", "", experiment, flags=re.I)   # "2026.122 - Preprocessed Data"
        parsed = parse_for_gui(path, folder)
        assert parsed is not None, path
        mask, reasons = default_mask(parsed)
        keep_idx = [i for i, m in enumerate(mask) if m]
        assert len(keep_idx) >= 3, path
        r = fit_wells(parsed, keep_idx, experiment, fitnum)
        s = summary_from_result(r)
        cache[path] = dict(folder=folder, parsed=parsed, mask=mask, reasons=reasons, r=r, s=s)
        print(f"{r['interaction']}  {experiment} (fit {fitnum}): {len(keep_idx)}/{len(mask)} sensors, "
              f"assoc {s['assoc_min']:.0f} min, dissoc {s['dissoc_h']:.1f} h, Kdis2 {r['kf']:.3e}, Kdis1 {r['ks']:.3e} "
              f"({s['tet_category']})")
    pickle.dump(cache, open(OUT, "wb"))
    print(OUT)


if __name__ == "__main__":
    main()
