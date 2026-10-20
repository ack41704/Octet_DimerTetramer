"""Apply the bivalent-ligand dimer->tetramer model to every Octet dataset.

Pipeline per (interaction folder, file) pair:
  1. parse; drop Max-Control wells, reference sensors (no signal), upward-drifting sensors
  2. if the folder names a genotype (WT / R79P), keep only that genotype's wells
  3. detect the association/dissociation boundary as the peak of the mean signal
  4. STAGE 1  global biexponential on dissociation, shared k_fast/k_slow, decay to zero
                -> kdis1 (dimer), kdis2 (tetramer)
  5. STAGE 2  k_obs vs [analyte]: slope -> ka1 ; slow saturating component -> ka2
  6. KD1 = kdis1/ka1 (M) ; KD2 = kdis2/ka2 (dimensionless)

Standard errors come from the fit Jacobians and are propagated to KD.
kdis2 is only resolvable when the dissociation window samples a real fraction of the slow
phase; the observed decay fraction is reported so unreliable rows can be filtered.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys, os, glob, re, csv, warnings
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_octet import parse_octet_csv
warnings.filterwarnings("ignore")

LIG = {"EB1": "EphrinB1-Fc", "EB2": "EphrinB2-Fc", "EB3": "EphrinB3-Fc", "EA5": "EphrinA5-Fc"}


def split_name(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    m = re.split(r"\s*-\s*Fits?\s*", base, flags=re.I)
    return (m[0].strip(), m[1].strip()) if len(m) > 1 else (base.strip(), "")


def boundary(wells):
    t = wells[0]["t"]
    Y = np.mean([w["y"] for w in wells], axis=0)
    i = int(np.argmax(Y))
    # need >= 5 min of association before the peak and a dissociation step after it (a percentage floor
    # rejected 2026.90: 20 min association + 16 h dissociation puts the peak at 2% of the record)
    if not (t[i] - t[0] >= 300 and i < 0.9 * len(t)):
        return None
    return t[i]


def sub(t, y, n_log, n_lin):
    idx = np.unique(np.concatenate([
        np.geomspace(1, len(t) - 1, n_log).astype(int),
        np.linspace(0, len(t) - 1, n_lin).astype(int)]))
    return t[idx], y[idx]


def se_from(res, n_par):
    """standard errors from the least_squares Jacobian"""
    try:
        J = res.jac
        dof = max(len(res.fun) - J.shape[1], 1)
        s2 = np.sum(res.fun ** 2) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        return np.sqrt(np.abs(np.diag(cov)))[:n_par]
    except Exception:
        return np.full(n_par, np.nan)


def stage1(curves):
    """global biexponential on dissociation; returns kdis1,kdis2,errors,R2,tet amplitude"""
    n = len(curves)
    lb = [-5, -8] + [0] * n + [0] * n
    ub = [-0.5, -3.5] + [3] * n + [3] * n
    p0 = ([np.log10(8e-4), np.log10(1.5e-6)]
          + [float(np.clip(c[1][0] * .3, 0, 3)) for c in curves]
          + [float(np.clip(c[1][0] * .7, 0, 3)) for c in curves])

    def r(p):
        kf, ks = 10 ** p[0], 10 ** p[1]
        A1, A2 = p[2:2 + n], p[2 + n:2 + 2 * n]
        return np.concatenate([a1 * np.exp(-kf * c[0]) + a2 * np.exp(-ks * c[0]) - c[1]
                               for c, a1, a2 in zip(curves, A1, A2)])
    res = least_squares(r, p0, bounds=(lb, ub), method="trf", xtol=1e-14, ftol=1e-14, max_nfev=20000)
    kf, ks = 10 ** res.x[0], 10 ** res.x[1]
    se = se_from(res, 2)
    ekf, eks = kf * np.log(10) * se[0], ks * np.log(10) * se[1]
    A1, A2 = res.x[2:2 + n], res.x[2 + n:2 + 2 * n]
    ya = np.concatenate([c[1] for c in curves])
    R2 = 1 - np.sum(res.fun ** 2) / np.sum((ya - ya.mean()) ** 2)
    tet = np.sum(A2) / max(np.sum(A1) + np.sum(A2), 1e-12)
    pinned = (res.x[1] <= -7.9) or (res.x[1] >= -3.55)
    return kf, ks, ekf, eks, R2, tet, pinned


def f2a(t, A1, k1, A2, k2):
    return A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t))


def stage2(wells, tb, kdis2):
    """k_obs vs [A] -> ka1 (slope); slow saturating component -> ka2"""
    pts, slow = [], []
    for w in wells:
        m = w["t"] <= tb
        ta, ya = w["t"][m], w["y"][m]
        if len(ta) < 40:
            continue
        i = np.unique(np.geomspace(1, len(ta) - 1, 300).astype(int))
        ta, ya = ta[i], ya[i]
        span = max(ya[-1], 1e-6)
        try:
            p, _ = curve_fit(f2a, ta, ya, p0=[span * .5, 5e-3, span * .5, 4e-4],
                             bounds=([0, 1e-5, 0, 1e-6], [3, 1, 3, 1e-1]), maxfev=40000)
        except Exception:
            continue
        A1, k1, A2, k2 = p
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1
        tot = A1 + A2
        if tot <= 0:
            continue
        cand = [(A, k) for A, k in ((A1, k1), (A2, k2)) if A / tot >= .15 and k < 5e-2]
        if not cand:
            continue
        A, k = max(cand, key=lambda z: z[0])
        pts.append((w["conc_M"], k))
        if abs(k1 - k2) / max(k1, k2) > 1e-3 and A2 / tot >= .10 and k2 < 5e-2:
            slow.append((w["conc_M"], k2))
    if len(pts) < 3:
        return (np.nan,) * 4
    c = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
    sl, ic, rr, pv, se = stats.linregress(c, k)
    if len(slow):
        sc = np.array([s[0] for s in slow]); sk = np.array([s[1] for s in slow])
        hi = sc >= np.sort(sc)[-2] if len(sc) >= 2 else np.ones(len(sc), bool)
        ka2 = max(float(np.mean(sk[hi])) - kdis2, 1e-9)
    else:
        ka2 = np.nan
    return sl, se, rr ** 2, ka2


def analyze(path, folder):
    lig = LIG[folder.split("+")[0]]
    partner = folder.split("+", 1)[1]
    geno = "R79P" if partner.endswith("R79P") else ("WT" if partner.endswith("-WT") else None)
    wells = parse_octet_csv(path)
    wells = [w for w in wells if not w["is_control"] and np.isfinite(w["conc_M"])]
    if geno:
        wells = [w for w in wells if w["label"] == geno]
    wells = [w for w in wells if w["y"].max() > 0.02]
    if len(wells) < 3:
        return None, "fewer than 3 usable sensors"
    tb = boundary(wells)
    if tb is None:
        return None, "no association/dissociation boundary found"
    keep = []
    for w in wells:
        m = w["t"] >= tb
        yd = w["y"][m]
        if len(yd) < 100:
            continue
        if yd[-100:].mean() > yd[:100].mean():
            continue            # upward drift: not physical for dissociation
        keep.append(w)
    if len(keep) < 3:
        return None, "fewer than 3 sensors after drift filter"
    curves = []
    for w in keep:
        m = w["t"] >= tb
        td, yd = sub(w["t"][m] - tb, w["y"][m], 200, 200)
        curves.append((td, yd))
    t_dis = keep[0]["t"][-1] - tb
    kdis1, kdis2, ekdis1, ekdis2, R2d, tet, pinned = stage1(curves)
    ka1, eka1, r2k, ka2 = stage2(keep, tb, kdis2)
    KD1 = kdis1 / ka1 if np.isfinite(ka1) and ka1 > 0 else np.nan
    KD2 = kdis2 / ka2 if np.isfinite(ka2) and ka2 > 0 else np.nan
    eKD1 = KD1 * np.hypot(ekdis1 / kdis1, eka1 / ka1) if np.isfinite(KD1) and ka1 > 0 else np.nan
    eKD2 = KD2 * (ekdis2 / kdis2) if np.isfinite(KD2) else np.nan
    decay_frac = 1 - np.exp(-kdis2 * t_dis)
    return dict(
        interaction=f"{lig} + {partner}",
        experiment=split_name(path)[0], fit=split_name(path)[1],
        KD1=KD1, KD1_err=eKD1, KD2=KD2, KD2_err=eKD2,
        ka1=ka1, ka1_err=eka1, ka2=ka2,
        kdis1=kdis1, kdis1_err=ekdis1, kdis2=kdis2, kdis2_err=ekdis2,
        R2_dissoc=R2d, r2_kobs=r2k, n_sensors=len(keep),
        t_assoc_s=tb, t_dissoc_s=t_dis,
        tetramer_frac=tet, slow_decay_observed=decay_frac,
        kdis2_reliable=bool((not pinned) and decay_frac > 0.05),
        t12_dimer_min=np.log(2) / kdis1 / 60,
        t12_tetramer_d=np.log(2) / kdis2 / 86400), None


def main():
    files = sorted(glob.glob(DATA_DIR + "/EB*Interactions - Preprocessed Data FIles/*/*.csv"))
    rows, skipped = [], []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        try:
            r, why = analyze(f, folder)
        except Exception as e:
            r, why = None, f"error: {type(e).__name__}: {e}"[:90]
        if r is None:
            skipped.append((folder, os.path.basename(f), why))
            print(f"SKIP {folder:18s} {os.path.basename(f)[:44]:44s} {why}")
        else:
            rows.append(r)
            print(f"OK   {r['interaction']:30s} {r['experiment'][:26]:26s} fit {r['fit'][:8]:8s} "
                  f"KD1={r['KD1']:.2e} kdis1={r['kdis1']:.2e} kdis2={r['kdis2']:.2e} "
                  f"R2={r['R2_dissoc']:.3f} rel={r['kdis2_reliable']}")
    out = os.path.join(WORK_DIR, "batch_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out.replace("batch_results", "batch_skipped"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["folder", "file", "reason"]); w.writerows(skipped)
    print(f"\n{len(rows)} fitted, {len(skipped)} skipped -> {out}")


if __name__ == "__main__":
    main()
