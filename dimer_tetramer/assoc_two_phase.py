"""
Two-phase association model for the EphB-ephrin-Fc sensorgrams:  FAST phase = DIMER,  SLOW phase = TETRAMER.

    R_i(t) = Af_i (1 - exp(-kf_i t)) + As_i (1 - exp(-ks_i t))          sensor i, concentration C_i
    kf_i   = Ka2 * C_i + kf0          (dimer,    fast: pseudo-first-order k_obs, slope = Ka2)
    ks_i   = Ka1 * C_i + ks0          (tetramer, slow: slope = Ka1)

The two rate lines are fitted GLOBALLY over every kept sensor of a run (amplitudes Af_i, As_i free per sensor, because
sensors are loaded differently). This replaces the earlier per-sensor biexponential + "take the component with the
larger amplitude" rule, which called the gradual phase "dimer" whenever it carried most of the signal (81 nM, 2026.96).
Free per-sensor biexponentials are also degenerate (fast/slow roles flip between sensors, 'slow' phases pinned at
A = 3.0, k = 3e-6 as a linear drift); tying the rates to concentration removes that.

Each slope is profiled (fixed on a grid, everything else refitted): it is reported as a number only if the sum of
squares rises by more than PROFILE_TOL on BOTH sides; flat on the low side = upper bound only; flat on both = n.d.

The colouring of the association fit line uses the SLOPE-DOMINANCE crossover: the time at which the tetramer component
starts rising faster than the dimer component,
    t_x = ln(Af kf / (As ks)) / (kf - ks)           (0 if As ks >= Af kf: tetramer-dominated from the start)
before t_x the fitted curve is dimer-dominated, after it tetramer-dominated.
"""
import numpy as np
from scipy.optimize import least_squares

SC = np.array([1e5, 1e-3, 1e4, 1e-4])      # parameter scales: Ka2 (1/Ms), kf0 (1/s), Ka1 (1/Ms), ks0 (1/s)
ORDER_MIN = 3.0                            # at the top concentration the fast rate must be >= 3x the slow rate
_LB = np.array([0.0, 0.0, 0.0, 1e-3])      # kf0 >= 0, ks0 >= 1e-7 /s
_UB = np.array([50.0, 100.0, 50.0, 50.0])  # Ka2 <= 5e6, kf0 <= 0.1, Ka1 <= 5e5, ks0 <= 5e-3


def _sub_idx(n, n_log=350, n_lin=350):
    return np.unique(np.r_[np.geomspace(1, n - 1, n_log).astype(int), np.linspace(0, n - 1, n_lin).astype(int)])


def two_phase(t, Af, kf, As, ks):
    return Af * (-np.expm1(-kf * t)) + As * (-np.expm1(-ks * t))


def crossover(Af, kf, As, ks):
    """time (s) until which the dimer (fast) component dominates the slope of the fitted curve; 0 if it never does"""
    if Af <= 0 or kf <= ks or Af * kf <= As * ks:
        return 0.0
    return float(np.log(Af * kf / max(As * ks, 1e-300)) / (kf - ks))


def _pack(ws, tb):
    T, Y, I = [], [], []
    for j, w in enumerate(ws):
        m = w["t"] <= tb
        ta, ya = w["t"][m], w["y"][m]
        i = _sub_idx(len(ta))
        T.append(ta[i]); Y.append(ya[i]); I.append(np.full(len(i), j))
    C = np.array([w["conc_nM"] for w in ws]) * 1e-9
    return np.concatenate(T), np.concatenate(Y), np.concatenate(I), C


def _resid(p, T, Y, I, C, n):
    kf = (p[0] * SC[0] * C + p[1] * SC[1])[I]
    ks = (p[2] * SC[2] * C + p[3] * SC[3])[I]
    return p[4:4 + n][I] * (-np.expm1(-kf * T)) + p[4 + n:][I] * (-np.expm1(-ks * T)) - Y


def _starts(ymax, n):
    out = []
    for a2 in (3e4, 2e5, 1e6):
        for f0 in (3e-4, 1e-2):
            for a1 in (1e3, 2e4):
                for s0 in (1e-4, 4e-4):
                    for fr in (0.3, 0.8):
                        out.append(np.r_[np.r_[a2, f0, a1, s0] / SC, np.full(n, fr * ymax), np.full(n, (1 - fr) * ymax)])
    return out


def _one(p0, T, Y, I, C, n, fixed=None):
    """fixed = {param index (0 = Ka2, 2 = Ka1): value in natural units} pins that slope (profile scan)"""
    lb = np.r_[_LB, np.zeros(2 * n)]; ub = np.r_[_UB, np.full(2 * n, 3.0)]
    p0 = np.array(p0, float)
    for k, v in (fixed or {}).items():
        lb[k] = v / SC[k] - 1e-9; ub[k] = v / SC[k] + 1e-9; p0[k] = v / SC[k]
    p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)
    s = least_squares(_resid, p0, bounds=(lb, ub), args=(T, Y, I, C, n), x_scale=np.r_[np.ones(4), 0.1 * np.ones(2 * n)],
                      max_nfev=400)
    return s


def _order_ok(p, C):
    top = C.max()
    return (p[0] * SC[0] * top + p[1] * SC[1]) >= ORDER_MIN * (p[2] * SC[2] * top + p[3] * SC[3])


def fit_global(ws, tb, warm=None):
    """global fit over sensors `ws` (dicts with t, y, conc_nM). Returns the best solution that has the fast phase
    >= ORDER_MIN x the slow phase at the top concentration (or the best overall, order_ok=False)."""
    T, Y, I, C = _pack(ws, tb); n = len(ws)
    ymax = float(max(w["y"][w["t"] <= tb].max() for w in ws))
    starts = list(warm or []) + _starts(ymax, n)        # optional extra starts, then a 48-point start grid
    sols = []
    for p0 in starts:
        s = _one(p0, T, Y, I, C, n)
        sols.append((s.cost, s.x))
    sols.sort(key=lambda z: z[0])
    valid = [z for z in sols if _order_ok(z[1], C)]
    cost, x = (valid or sols)[0]
    # alternative minima: other valid solutions within 3 % of the best cost whose slopes differ by > 2x
    alt = [z for z in valid if z[0] <= cost * 1.03 and (max(z[1][0], x[0]) / max(min(z[1][0], x[0]), 1e-9) > 2
                                                        or max(z[1][2], x[2]) / max(min(z[1][2], x[2]), 1e-9) > 2)]
    return dict(x=x, cost=float(cost), order_ok=bool(valid), n=n, C=C, ambiguous=len(alt) > 0,
                Ka2=x[0] * SC[0], kf0=x[1] * SC[1], Ka1=x[2] * SC[2], ks0=x[3] * SC[3],
                Af=x[4:4 + n], As=x[4 + n:], concs_nM=C * 1e9)


KA2_GRID = np.array([0, 1e3, 3e3, 1e4, 3e4, 6e4, 1e5, 1.5e5, 2.5e5, 4e5, 7e5, 1.5e6, 4e6])
KA1_GRID = np.array([0, 1e2, 3e2, 1e3, 2e3, 4e3, 7e3, 1.2e4, 2e4, 4e4, 1e5, 3e5])
PROFILE_TOL = 0.05        # practical chi-square bracket: sum of squares within 5 % of the minimum (BLI points are
                          # strongly correlated, so a formal F-test would be meaningless)


def profile_slope(ws, tb, full, which):
    """Fix one slope on a grid, refit everything else (centre-outward warm starts), and bracket the values whose
    sum of squares stays within PROFILE_TOL of the minimum. The optimum itself is part of the profile (ratio 1).
    Returns dict(lo, hi, resolved, values, ratio, better). lo = 0 / hi = inf: the profile stays flat to that edge of
    the scan, so the slope is NOT determined on that side. `better` = the scan found a lower minimum than `full`."""
    T, Y, I, C = _pack(ws, tb); n = full["n"]
    k = 0 if which == "Ka2" else 2
    grid = KA2_GRID if which == "Ka2" else KA1_GRID
    best, x0 = full[which], full["x"]
    floor = grid[1]                                   # smallest non-zero scan value; anything below counts as 0
    if best >= floor:                                 # finer points around the optimum so the bracket is not grid-limited
        grid = np.unique(np.r_[grid, best * np.array([0.5, 0.7, 0.85, 0.93, 1.07, 1.15, 1.4, 2.0])])
    lg = lambda v: np.log10(max(v, floor / 10))
    order = sorted(range(len(grid)), key=lambda g: abs(lg(grid[g]) - lg(best)))
    sols = {}
    for gi in order:
        start = sols[min(sols, key=lambda j: abs(lg(grid[j]) - lg(grid[gi])))][1] if sols else x0
        r = min((_one(c, T, Y, I, C, n, fixed={k: float(grid[gi])}) for c in (start, x0)), key=lambda z: z.cost)
        sols[gi] = (r.cost, r.x)
    vals = np.r_[grid, best]
    cost = np.r_[[sols[g][0] for g in range(len(grid))], full["cost"]]
    o = np.argsort(vals, kind="stable"); vals, cost = vals[o], cost[o]
    ratio = cost / full["cost"]
    c0 = int(np.where(o == len(grid))[0][0])          # position of the optimum
    ok = ratio <= 1 + PROFILE_TOL
    lo_i = c0
    while lo_i > 0 and ok[lo_i - 1]:
        lo_i -= 1
    hi_i = c0
    while hi_i < len(vals) - 1 and ok[hi_i + 1]:
        hi_i += 1

    def cross(i_in, i_out):          # log-interpolated crossing of ratio = 1 + tol between an accepted and a rejected point
        a, b = lg(vals[i_in]), lg(vals[i_out]); ra, rb = ratio[i_in], ratio[i_out]
        f = (1 + PROFILE_TOL - ra) / (rb - ra) if rb > ra else 0.0
        return float(10 ** (a + f * (b - a)))
    lo = 0.0 if lo_i == 0 else cross(lo_i, lo_i - 1)
    if lo < floor:
        lo = 0.0
    hi = np.inf if hi_i == len(vals) - 1 else cross(hi_i, hi_i + 1)
    gbest = min(sols, key=lambda g: sols[g][0])        # lowest-cost profile point (used to refit if it beats `full`)
    return dict(lo=lo, hi=hi, resolved=bool(lo > 0 and np.isfinite(hi) and best >= floor), values=vals, ratio=ratio,
                better=bool(ratio.min() < 1 - 1e-3), best_x=sols[gbest][1], best_cost=float(sols[gbest][0]))


def per_sensor_rates(ws, tb, full):
    """Each sensor refitted ALONE (both rates free, started from and held within 10x of the global-fit rates so the
    fast/slow roles cannot swap). Used only to check the linear rate-vs-concentration assumption (rate vs [EphB2] plot);
    the reported constants come from the global fit. share_f / share_s = fraction of that sensor's end-of-association
    signal carried by the fast / slow phase (a rate from a phase carrying < 10 % of the signal is poorly defined)."""
    out = []
    for j, w in enumerate(ws):
        m = w["t"] <= tb
        ta, ya = w["t"][m], w["y"][m]
        i = _sub_idx(len(ta)); ta, ya = ta[i], ya[i]
        C = w["conc_nM"] * 1e-9
        kf = max(full["Ka2"] * C + full["kf0"], 1e-7); ks = max(full["Ka1"] * C + full["ks0"], 1e-8)   # never 0 (box)
        lb = np.array([0, kf / 10, 0, ks / 10]); ub = np.array([3, kf * 10, 3, ks * 10])
        p0 = np.clip([full["Af"][j], kf, full["As"][j], ks], lb + 1e-12, ub - 1e-12)
        s = least_squares(lambda p: two_phase(ta, *p) - ya, p0, bounds=(lb, ub), x_scale=[0.1, kf, 0.1, ks], max_nfev=2000)
        Af, kf_i, As, ks_i = s.x
        if kf_i < ks_i:
            Af, kf_i, As, ks_i = As, ks_i, Af, kf_i
        sig = max(two_phase(tb, Af, kf_i, As, ks_i), 1e-12)
        out.append(dict(conc=w["conc_nM"], kf=float(kf_i), ks=float(ks_i),
                        share_f=float(Af * -np.expm1(-kf_i * tb) / sig), share_s=float(As * -np.expm1(-ks_i * tb) / sig)))
    return out
