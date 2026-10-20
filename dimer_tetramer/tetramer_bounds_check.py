"""(1) Is a short-window kdis2 always an OVERestimate (=> valid lower bound on t1/2)?
(2) Profile-likelihood 95% upper bound on kdis2 for 15 h runs whose fit pins at the floor."""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, pickle, numpy as np
sys.path.insert(0, ".")
from scipy.optimize import least_squares
from plot_all_fits import stage1_full, sub

cache = pickle.load(open(WORK_DIR + "/fit_cache.pkl", "rb"))
ok = {p: e for p, e in cache.items() if "error" not in e}
truth = [(p, e) for p, e in ok.items() if e["s"]["tetramerOK"] and e["s"]["kdis2"] > 2e-8]
pinned = [(p, e) for p, e in ok.items() if e["s"]["tetramerOK"] and e["s"]["kdis2"] <= 2e-8]

def curves(e, W_s=None):
    r = e["r"]; tb = r["tb"]; out = []
    for w in r["keep"]:
        m = w["t"] >= tb
        if W_s is not None: m &= w["t"] <= tb + W_s
        td, yd = sub(w["t"][m] - tb, w["y"][m], 200, 200)
        if len(td) > 50: out.append((td, yd))
    return out

print("(1) short-window kdis2 / 15h truth, per ground-truth run (biexponential, same downsampling as production)")
for W in (1, 2, 3, 4, 8):
    ratios = []
    for p, e in truth:
        kf, ks, *_ = stage1_full(curves(e, W * 3600))
        ratios.append(ks / e["s"]["kdis2"])
    ratios = np.array(ratios)
    print(f"   {W} h:  min={ratios.min():.2f}x  median={np.median(ratios):.2f}x  "
          f"overestimates in {np.sum(ratios > 1)}/{len(ratios)} runs")

def profile_upper(e, level=3.84):
    cur = curves(e); n = len(cur)
    def resid(p, lks):
        kf, ks = 10 ** p[0], 10 ** lks
        A1, A2 = p[1:1 + n], p[1 + n:]
        return np.concatenate([a1*np.exp(-kf*c[0]) + a2*np.exp(-ks*c[0]) - c[1] for c, a1, a2 in zip(cur, A1, A2)])
    lb = [-5] + [0]*2*n; ub = [-0.5] + [3]*2*n
    p0 = [np.log10(e["r"]["kf"])] + [float(np.clip(c[1][0]*.3, 0, 3)) for c in cur] + [float(np.clip(c[1][0]*.7, 0, 3)) for c in cur]
    base = least_squares(lambda p: resid(p, -8), p0, bounds=(lb, ub), method="trf", max_nfev=20000)
    sse0 = np.sum(base.fun**2); npts = len(base.fun); sigma2 = sse0 / (npts - len(p0) - 1)
    x = base.x
    for lks in np.arange(-8, -4, 0.1):
        res = least_squares(lambda p: resid(p, lks), x, bounds=(lb, ub), method="trf", max_nfev=20000)
        x = res.x
        if (np.sum(res.fun**2) - sse0) / sigma2 >= level:
            return 10 ** lks
    return None

print("\n(2) floor-pinned 15 h runs: profile-likelihood 95% upper bound on kdis2 -> lower bound on t1/2")
for p, e in pinned:
    ku = profile_upper(e)
    lab = p.split("/")[-1]
    if ku: print(f"   {e['r']['interaction']:28s} {lab:30s} kdis2 < {ku:.2e} s-1  =>  t1/2 > {np.log(2)/ku/86400:.1f} d")
    else:  print(f"   {e['r']['interaction']:28s} {lab:30s} no upper bound found below 1e-4")
