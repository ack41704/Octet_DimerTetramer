"""Add a 'Dimer-Tetramer Model' tab to the master table, matching its layout.

Columns 0-9 reproduce the existing 1:1 global-fit values (carried over by matching
interaction + global fit number) so the two models sit side by side.
Columns 11-23 hold the two-step model's constants. Columns 25+ are appended diagnostics.

NOTE ON NUMBERING: Sartorius' 2:1 model numbers its sites so that site 1 is the
high-affinity one (the one that pins at kdis1 = 1e-7). This model numbers by assembly
order: step 1 is the low-affinity 1:1 DIMER, step 2 is the TETRAMER. Headers say which.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import csv, re, os, sys
import numpy as np
from numbers_parser import Document

W = WORK_DIR
SRC = DATA_DIR + "/Global Fit - Master Table on 9-25-26.numbers"
OUT = DATA_DIR + "/Global Fit - Master Table on 9-25-26 (with Dimer-Tetramer tab).numbers"
SHEET = "Dimer-Tetramer Model"

HDR = ["Interaction", "Octet Experiment", "Global Fit #",
       "KD", "KD Error", "ka (1/Ms)", "ka Error", "kdis (1/s)", "kdis Error", "R2", None,
       "KD1 (dimer, M)", "KD2 (tetramer, dimensionless)", "KD1 Error", "KD2 Error",
       "ka1 (1/Ms)", "ka2 (1/s)", "ka1 Error", "ka2 Error",
       "kdis1 (dimer, 1/s)", "kdis2 (tetramer, 1/s)", "kdis1 Error", "kdis2 Error", "R2",
       None, "t1/2 dimer (min)", "t1/2 tetramer (d)", "tetramer fraction",
       "buffer", "assoc window (min)", "dissoc window (h)",
       "slow phase decay observed", "ka1 QC (r2>0.9)", "dimer OK? (dissoc>=8h)",
       "tetramer OK? (dissoc>=8h)", "kdis2 resolved?",
       "n sensors", "r2 (kobs vs conc)"]

def buffer_of(exp):
    e = exp.lower()
    if "pbstb" in e:
        return "PBSTB"
    if "casein" in e:
        return "PBST + casein"
    return "PBST"


def fitkey(fit):
    if fit is None:
        return None
    m = re.match(r"\s*(\d+)\s*([a-c])?", str(fit))
    return (m.group(1) + (m.group(2) or "")) if m else None


def build(csv_path=None, src_path=None, out_path=None, sheet_name=SHEET):
    """Callable core (used by both the CLI entry point and the GUI's export button).
    Reads csv_path (defaults to batch_results.csv), adds `sheet_name` to src_path
    (defaults to SRC), writes to out_path (defaults to OUT). Returns (out_path, n_rows, csv_out).
    If sheet_name already exists in src_path (re-exporting into a file that already
    has our tab), a numeric suffix is appended -- numbers_parser has no way to
    remove/replace a sheet, so we never silently collide with one."""
    csv_path = csv_path or os.path.join(W, "batch_results.csv")
    src_path = src_path or SRC
    out_path = out_path or OUT

    doc = Document(src_path)
    existing_names = {s.name for s in doc.sheets}
    if sheet_name in existing_names:
        i = 2
        while f"{sheet_name} ({i})" in existing_names:
            i += 1
        sheet_name = f"{sheet_name} ({i})"
    src_rows = doc.sheets[0].tables[0].rows(values_only=True)
    theirs, order, cur = {}, [], None
    for r in src_rows[1:]:
        if r[0]:
            cur = r[0].strip()
            if cur not in order:
                order.append(cur)
        k = fitkey(r[2])
        if cur and k and r[3] is not None:
            theirs[(cur, k)] = list(r[3:10])

    mine = list(csv.DictReader(open(csv_path)))
    for m in mine:
        for f in ("KD1", "KD1_err", "KD2", "KD2_err", "ka1", "ka1_err", "ka2",
                  "kdis1", "kdis1_err", "kdis2", "kdis2_err", "R2_dissoc", "r2_kobs",
                  "t12_dimer_min", "t12_tetramer_d", "tetramer_frac",
                  "t_assoc_s", "t_dissoc_s", "slow_decay_observed", "n_sensors"):
            try:
                v = float(m[f])
                m[f] = None if not np.isfinite(v) else v
            except (TypeError, ValueError):
                m[f] = None
        m["rel"] = (m["kdis2_reliable"] == "True")
        # convergence thresholds measured by truncating the validated 15 h runs:
        # kdis1 is within ~10% by 8 h and ~1.1-1.35x biased at 4 h; kdis2 needs >=8 h.
        m["ka1qc"] = (m["r2_kobs"] or 0) > 0.9 and (m["ka1"] or 0) > 0
        m["ok_dim"] = (m["t_dissoc_s"] or 0) >= 28800 and m["ka1qc"]
        m["ok_tet"] = (m["t_dissoc_s"] or 0) >= 28800
        m["ok"] = m["ok_dim"]
        m["buf"] = buffer_of(m["experiment"])

    groups = {}
    for m in mine:
        groups.setdefault(m["interaction"], []).append(m)
    ordered = [i for i in order if i in groups] + [i for i in sorted(groups) if i not in order]

    out = [HDR]
    for inter in ordered:
        rows = sorted(groups[inter], key=lambda r: r["experiment"])
        for j, m in enumerate(rows):
            ks = [fitkey(x) for x in re.split(r"&|,", m["fit"])] if m["fit"] else []
            one = next((theirs[(inter, k)] for k in ks if k and (inter, k) in theirs), [None] * 7)
            out.append([inter if j == 0 else None, m["experiment"], m["fit"]] + one + [None,
                        m["KD1"], m["KD2"] if m["rel"] else None, m["KD1_err"],
                        m["KD2_err"] if m["rel"] else None,
                        m["ka1"], m["ka2"], m["ka1_err"], None,
                        m["kdis1"], m["kdis2"] if m["rel"] else None,
                        m["kdis1_err"], m["kdis2_err"] if m["rel"] else None,
                        m["R2_dissoc"], None,
                        m["t12_dimer_min"], m["t12_tetramer_d"] if m["rel"] else None,
                        m["tetramer_frac"], m["buf"], m["t_assoc_s"] / 60.0,
                        m["t_dissoc_s"] / 3600.0, m["slow_decay_observed"],
                        "yes" if m["ok"] else "no", "yes" if m["rel"] else "no",
                        m["n_sensors"], m["r2_kobs"]])

        def mean(field, only_reliable=False, tet=False):
            v = [r[field] for r in rows
                 if r[field] is not None and r["buf"] == "PBST"
                 and (r["ok_tet"] if tet else r["ok_dim"])
                 and (r["rel"] or not only_reliable)]
            r_ = float(np.mean(v)) if v else None
            return r_ if (r_ is None or np.isfinite(r_)) else None
        one_mean = []
        for c in range(7):
            v = [theirs[(inter, k)][c] for r in rows
                 for k in ([fitkey(x) for x in re.split(r"&|,", r["fit"])] if r["fit"] else [])
                 if k and (inter, k) in theirs and isinstance(theirs[(inter, k)][c], (int, float))]
            mv = float(np.mean(v)) if v else None
            one_mean.append(mv if (mv is None or np.isfinite(mv)) else None)
        out.append([None, None, None] + one_mean + [None,
                    mean("KD1"), mean("KD2", True, True), mean("KD1_err"), mean("KD2_err", True, True),
                    mean("ka1"), mean("ka2", tet=True), mean("ka1_err"), None,
                    mean("kdis1"), mean("kdis2", True, True), mean("kdis1_err"), mean("kdis2_err", True, True),
                    mean("R2_dissoc"), None,
                    mean("t12_dimer_min"), mean("t12_tetramer_d", True, True), mean("tetramer_frac"),
                    "PBST only", None, None, None,
                    None, f"dimer n={sum(1 for r in rows if r['ok_dim'] and r['buf']=='PBST')}",
                    f"tetramer n={sum(1 for r in rows if r['ok_tet'] and r['buf']=='PBST')}",
                    f"{sum(1 for r in rows if r['rel'])}/{len(rows)}",
                    None, mean("r2_kobs")])
        out.append([None] * len(HDR))

    doc.add_sheet(sheet_name, "MASTER TABLE: Dimer-Tetramer (two-step) model. "
                         "Step 1 = 1:1 dimer, Step 2 = 2:2 tetramer (intramolecular closure). "
                         "The limiting factor is the DISSOCIATION window. Truncating the validated 15 h runs "
                  "reproduces the short-run values with both off-rates biased fast (kdis1 3-5x at 0.7 h, "
                  "1.1-1.35x at 4 h, within ~10% by 8 h; kdis2 needs >=8 h). Association length is NOT "
                  "limiting for ka1: a 13 min association gives ka1 within 2-13% of the 90 min value. "
                  "MEAN rows use PBST runs with dissoc >=8 h that also pass the ka1 regression QC.",
                  num_rows=len(out), num_cols=len(HDR))
    t = doc.sheets[-1].tables[0]
    for i, row in enumerate(out):
        for j, v in enumerate(row):
            if v is None:
                continue
            if isinstance(v, float) and not np.isfinite(v):
                continue
            t.write(i, j, v)
    doc.save(out_path)

    csv_out = os.path.join(W, "dimer_tetramer_tab.csv")
    with open(csv_out, "w", newline="") as fh:
        csv.writer(fh).writerows([["" if v is None else v for v in r] for r in out])
    return out_path, len(out), csv_out


def main():
    out_path, n, csv_out = build()
    print(f"wrote {n} rows -> {out_path}")
    print(f"also wrote {csv_out}")


if __name__ == "__main__":
    main()
