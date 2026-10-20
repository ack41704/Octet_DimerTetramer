"""Does truncating the ASSOCIATION window bias ka1?
Same validated runs, full 15 h dissociation (so kdis1 is the converged value);
only the association window fed to the k_obs analysis is shortened."""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, os
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_octet import parse_octet_csv
import warnings; warnings.filterwarnings("ignore")
TA = 5400.0
KD1 = {"2026.101": 5.39e-4, "2026.103": 3.90e-4, "2026.108": 4.36e-4,
       "2026.106": 1.00e-3, "2026.107": 1.55e-3, "2026.109": 1.09e-3}


def f2a(t, A1, k1, A2, k2):
    return A1*(1-np.exp(-k1*t)) + A2*(1-np.exp(-k2*t))


def ka1_for(wells, win, kdis1):
    pts, slow = [], []
    for w in wells:
        m = w["t"] <= min(win, TA)
        ta, ya = w["t"][m], w["y"][m]
        if len(ta) < 40:
            continue
        i = np.unique(np.geomspace(1, len(ta)-1, 300).astype(int))
        ta, ya = ta[i], ya[i]
        sp = max(ya[-1], 1e-6)
        try:
            p, _ = curve_fit(f2a, ta, ya, p0=[sp*.5, 5e-3, sp*.5, 4e-4],
                             bounds=([0, 1e-5, 0, 1e-6], [3, 1, 3, 1e-1]), maxfev=40000)
        except Exception:
            continue
        A1, k1, A2, k2 = p
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1
        tot = A1+A2
        cand = [(A, k) for A, k in ((A1, k1), (A2, k2)) if A/tot >= .15 and k < 5e-2]
        if not cand:
            continue
        A, k = max(cand, key=lambda z: z[0])
        pts.append((w["conc_M"], k))
        if abs(k1-k2)/max(k1, k2) > 1e-3 and A2/tot >= .10 and k2 < 5e-2:
            slow.append(k2)
    if len(pts) < 3:
        return np.nan, np.nan, np.nan
    c = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
    sl, ic, rr, pv, se = stats.linregress(c, k)
    ka2 = float(np.mean(sorted(slow)[-2:])) if len(slow) >= 2 else np.nan
    return sl, rr**2, ka2


print(f"{'run':10s} {'assoc win':>9s} {'ka1':>10s} {'KD1 (nM)':>9s} {'ka2':>10s} {'r2':>6s}")
for run in KD1:
    wells = [w for w in parse_octet_csv(f"{DATA_DIR}/{run} - Preprocessed Data.csv")
             if not w["is_control"] and w["y"].max() > 0.15]
    for win in (800, 1800, 3600, 5400):
        ka1, r2, ka2 = ka1_for(wells, win, KD1[run])
        kd = KD1[run]/ka1*1e9 if np.isfinite(ka1) and ka1 > 0 else np.nan
        print(f"{run:10s} {win/60:8.0f}m {ka1:10.2e} {kd:9.2f} {ka2:10.2e} {r2:6.3f}")
    print()
