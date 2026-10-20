"""Controlled test: can kdis2 be recovered from a SHORT dissociation window?
Ground truth = the 13 runs whose 15 h dissociation gives a genuinely resolved
(non-boundary) kdis2. Each is truncated to a shorter window and re-estimated two ways:
  A. global biexponential (current method) on the truncated curves
  B. terminal slope: once the dimer has decayed (>=5 dimer half-lives), what's left is
     tetramer, so kdis2 = -slope/level over the late part of the window (your manual method)
Reports estimate/truth, so a method only counts as 'resolving' a window if it lands close."""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, os, pickle, numpy as np
sys.path.insert(0, ".")
from plot_all_fits import stage1_full, sub

cache = pickle.load(open(WORK_DIR + "/fit_cache.pkl", "rb"))
truth = []
for p, e in cache.items():
    if "error" in e: continue
    s = e["s"]
    if s["tetramerOK"] and s["kdis2"] > 2e-8:          # genuinely resolved, not floor-pinned
        truth.append((p, e))
print(f"ground-truth runs: {len(truth)}")

def curves_for(e, W_s):
    r = e["r"]; tb = r["tb"]; out = []
    for w in r["keep"]:
        m = (w["t"] >= tb) & (w["t"] <= tb + W_s)
        td, yd = w["t"][m] - tb, w["y"][m]
        if len(td) > 50: out.append((td, yd))
    return out

def terminal_slope(e, W_s):
    r = e["r"]; tb = r["tb"]
    t_dimer = np.log(2) / r["kf"]
    start = max(5 * t_dimer, 0.4 * W_s)                # dimer gone, and at least the last 60%
    if start >= W_s * 0.9: return np.nan
    ks = []
    lv = []
    for w in r["keep"]:
        m = (w["t"] >= tb + start) & (w["t"] <= tb + W_s)
        td, yd = w["t"][m] - tb, w["y"][m]
        if len(td) < 50: continue
        b, a = np.polyfit(td, yd, 1)
        level = yd.mean()
        if level > 0.05 and b < 0:
            ks.append(-b / level); lv.append(level)
    if not ks: return np.nan
    order = np.argsort(lv)[::-1]
    top = [ks[i] for i in order[:max(1, (len(ks) + 1) // 2)]]   # upper half by signal level
    return float(np.median(top))

WINDOWS_H = [1, 2, 3, 4, 8, 15]
resA = {W: [] for W in WINDOWS_H}; resB = {W: [] for W in WINDOWS_H}
for p, e in truth:
    k_true = e["s"]["kdis2"]
    for W in WINDOWS_H:
        cur = curves_for(e, W * 3600)
        if len(cur) >= 2:
            kf, ks, *_ = stage1_full(cur)
            resA[W].append(ks / k_true)
        kb = terminal_slope(e, W * 3600)
        if np.isfinite(kb): resB[W].append(kb / k_true)

print(f"\n{'window':>7s} | {'A: biexponential  median [IQR]  within 2x':>44s} | {'B: terminal slope  median [IQR]  within 2x':>44s}")
for W in WINDOWS_H:
    def fmt(v):
        if not v: return "n/a".rjust(44)
        v = np.array(v); q = np.percentile(v, [25, 50, 75])
        w2 = np.mean((v > 0.5) & (v < 2.0)) * 100
        return f"{q[1]:6.2f}x [{q[0]:5.2f}-{q[2]:5.2f}]  {w2:3.0f}% (n={len(v)})".rjust(44)
    print(f"{W:5d} h | {fmt(resA[W])} | {fmt(resB[W])}")
