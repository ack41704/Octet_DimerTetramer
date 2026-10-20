"""Controlled test: take the validated 15 h runs and TRUNCATE the dissociation window.
Same sensors, same association, same everything -- only the observation window changes.
If truncation alone reproduces the short-run numbers, the discrepancy is a window
artefact rather than a real difference between experiments."""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, os
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_octet import parse_octet_csv
import warnings; warnings.filterwarnings("ignore")
TA = 5400.0


def stage1(curves):
    n = len(curves)
    lb = [-5, -8] + [0]*n + [0]*n
    ub = [-0.5, -3.5] + [3]*n + [3]*n
    p0 = ([np.log10(8e-4), np.log10(1.5e-6)]
          + [float(np.clip(c[1][0]*.3, 0, 3)) for c in curves]
          + [float(np.clip(c[1][0]*.7, 0, 3)) for c in curves])
    def r(p):
        kf, ks = 10**p[0], 10**p[1]
        A1, A2 = p[2:2+n], p[2+n:2+2*n]
        return np.concatenate([a1*np.exp(-kf*c[0]) + a2*np.exp(-ks*c[0]) - c[1]
                               for c, a1, a2 in zip(curves, A1, A2)])
    res = least_squares(r, p0, bounds=(lb, ub), method="trf", xtol=1e-14, ftol=1e-14, max_nfev=20000)
    kf, ks = 10**res.x[0], 10**res.x[1]
    A1, A2 = res.x[2:2+n], res.x[2+n:2+2*n]
    ya = np.concatenate([c[1] for c in curves])
    R2 = 1 - np.sum(res.fun**2)/np.sum((ya-ya.mean())**2)
    tet = np.sum(A2)/max(np.sum(A1)+np.sum(A2), 1e-12)
    return kf, ks, R2, tet


RUNS = {"2026.101": "WT", "2026.103": "WT", "2026.108": "WT",
        "2026.106": "R79P", "2026.107": "R79P", "2026.109": "R79P"}
WINDOWS = [0.67, 2.0, 4.0, 8.0, 15.0]
print(f"{'run':10s} {'gt':5s} {'window':>7s} {'kdis1':>10s} {'kdis2':>10s} "
      f"{'t12 dimer':>10s} {'t12 tet(d)':>11s} {'tet frac':>8s} {'R2':>6s}")
for run, gt in RUNS.items():
    wells = [w for w in parse_octet_csv(f"{DATA_DIR}/{run} - Preprocessed Data.csv")
             if not w["is_control"] and w["y"].max() > 0.15]
    for W_h in WINDOWS:
        cur = []
        for w in wells:
            m = (w["t"] >= TA) & (w["t"] <= TA + W_h*3600)
            td, yd = w["t"][m]-TA, w["y"][m]
            if len(td) < 200:
                continue
            i = np.unique(np.concatenate([np.geomspace(1, len(td)-1, 200).astype(int),
                                          np.linspace(0, len(td)-1, 200).astype(int)]))
            cur.append((td[i], yd[i]))
        if len(cur) < 3:
            continue
        kf, ks, R2, tet = stage1(cur)
        print(f"{run:10s} {gt:5s} {W_h:6.2f}h {kf:10.2e} {ks:10.2e} "
              f"{np.log(2)/kf/60:8.1f}m {np.log(2)/ks/86400:11.2f} {tet:8.2f} {R2:6.3f}")
    print()
