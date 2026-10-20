"""
Rate constants for ALL 16 extended runs (14-16 h dissociation), v2: association re-analysed with
FAST phase = DIMER and SLOW phase = TETRAMER in every run (assoc_two_phase.py). Numbering as in the master table:
1 = tetramer, 2 = dimer.

What changed from extended_constants_all16.py (v1, kept unchanged):
  v1 took Ka2 from the per-sensor k_obs of whichever association component had the LARGER amplitude. When the gradual
  phase carried most of the signal (e.g. 2026.96 at 81 nM: sharp phase 10 %, gradual 90 %) the gradual phase was called
  the dimer and Ka1 became "n.d.". v2 assigns by RATE, not amplitude:
  Ka2 (1/Ms)    slope of the fast-phase rate vs [EphB2]:  kf_i = Ka2*C_i + kf0      (global fit over all sensors)
  Ka1 (1/Ms)    slope of the slow-phase rate vs [EphB2]:  ks_i = Ka1*C_i + ks0      (same fit; amplitudes per sensor)
                Each slope is profiled; reported if bounded on both sides (5 % sum-of-squares bracket), "< x" if the
                profile is flat down to zero (upper bound), n.d. if flat on both sides.
  Kdis2, Kdis1  unchanged: fast / slow rate of the global biexponential dissociation fit (Kdis1 an upper bound when no
                tetramer decay was detected in 15 h: 2026.91, 2026.97).
  KD1, KD2      Kdis/Ka in nM, with bound symbols propagated.
Association fit lines are coloured by phase: vermilion where the dimer (fast) component dominates the slope of the
fitted curve, blue where the tetramer (slow) component does (crossover t_x, assoc_two_phase.crossover).

Outputs (analysis folder): Extended_constants_all16_v2_9-30-26.xlsx, Extended_sensorgrams_all16_v2_9-30-26.pdf,
Extended_sensorgrams_all16_v2_overview.png.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import sys
import math
import pickle

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_fit import sub  # noqa: E402
from eb3_extended_constants import evaluate_workbook  # noqa: E402
from extended_constants_all16 import RUNS  # noqa: E402
import assoc_two_phase as A  # noqa: E402

W = WORK_DIR
OUT_XLSX = os.path.join(W, "Extended_constants_all16_v2_9-30-26.xlsx")
OUT_PDF = os.path.join(W, "Extended_sensorgrams_all16_v2_9-30-26.pdf")
OUT_PNG = os.path.join(W, "Extended_sensorgrams_all16_v2_overview.png")
INK, INK2, GRID, GREY = "#1f1f1f", "#5a5a58", "#e6e6e3", "#9A9A96"
DIMER, TETRAMER = "#D55E00", "#0072B2"          # Okabe-Ito vermilion / blue (colour-blind safe pair)
LN2 = math.log(2)
EARLY_S = 600                                   # "first 10 min" panel and R² window for the dimer phase


# ------------------------------------------------------------------ analysis
def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def load(run_list=RUNS):
    """run_list = (experiment, Global Fit #, ligand, analyte); entries come from fit_cache.pkl plus, if present,
    fit_cache_extra.pkl (runs outside the EB* folders, built by build_extra_cache.py)"""
    by = load_cache()
    return [analyze_run(by[(f"{lig} + {analyte}", exp)], exp, gf, lig, analyte) for exp, gf, lig, analyte in run_list]


def load_cache():
    """{(interaction, experiment): cache entry} over fit_cache.pkl + fit_cache_extra.pkl (successful fits only)"""
    cache = pickle.load(open(os.path.join(W, "fit_cache.pkl"), "rb"))
    extra = os.path.join(W, "fit_cache_extra.pkl")
    if os.path.exists(extra):
        cache.update(pickle.load(open(extra, "rb")))
    return {(e["r"]["interaction"], e["r"]["experiment"]): e for e in cache.values() if "error" not in e}


def analyze_run(e, exp, gf, lig, analyte, strict=True):
    """v2 association analysis of one cached run. strict (the 17 extended runs): assert >= 14 h dissociation, fast >= 3x
    slow at the top concentration, and no lower minimum found by the profile scans. Non-strict (runs of any length):
    record those conditions in x["fit_issues"] instead, refitting from the profile's better minimum when one is found."""
    r, s = e["r"], e["s"]; tb = r["tb"]
    if strict:
        assert s["dissoc_h"] >= 14, exp
    issues = []
    order = sorted(range(len(r["keep"])), key=lambda i: r["keep"][i]["conc_nM"])
    ws = [r["keep"][i] for i in order]
    f = A.fit_global(ws, tb)
    if strict:
        assert f["order_ok"], exp
    elif not f["order_ok"]:
        issues.append("fast phase < 3x the slow phase at the top concentration")
    p2 = A.profile_slope(ws, tb, f, "Ka2"); p1 = A.profile_slope(ws, tb, f, "Ka1")
    if strict:
        assert not (p2["better"] or p1["better"]), exp              # the profile found no lower minimum
    else:
        for _ in range(3):                                           # refit from the profile's lower minimum
            if not (p2["better"] or p1["better"]):
                break
            bx = min((p2, p1), key=lambda p: p["best_cost"])["best_x"]
            f = A.fit_global(ws, tb, warm=[bx])
            p2 = A.profile_slope(ws, tb, f, "Ka2"); p1 = A.profile_slope(ws, tb, f, "Ka1")
        if p2["better"] or p1["better"]:
            issues.append("profile scan still finds a slightly lower minimum")
    sensors, full_y, full_yh, early_y, early_yh = [], [], [], [], []
    for j, (i, w) in enumerate(zip(order, ws)):
        C = w["conc_nM"] * 1e-9
        Af, As = float(f["Af"][j]), float(f["As"][j])
        kf, ks = f["Ka2"] * C + f["kf0"], f["Ka1"] * C + f["ks0"]
        m = w["t"] <= tb
        ta, ya = w["t"][m], w["y"][m]; yh = A.two_phase(ta, Af, kf, As, ks)
        me = ta <= EARLY_S
        full_y.append(ya); full_yh.append(yh); early_y.append(ya[me]); early_yh.append(yh[me])
        A1d, A2d = float(r["A1s"][i]), float(r["A2s"][i])       # dissociation amplitudes (fast-off, slow-off)
        sensors.append(dict(w=w, conc=w["conc_nM"], Af=Af, kf=kf, As=As, ks=ks, tx=A.crossover(Af, kf, As, ks),
                            r2=r2(ya, yh), A1d=A1d, A2d=A2d, ymax=float(ya.max())))
    top = max(sn["conc"] for sn in sensors)
    st = max([sn for sn in sensors if sn["conc"] == top], key=lambda sn: sn["ymax"])
    measured = s["tet_category"] == "measured"
    kdis1 = r["ks"] if measured else (s["tet_kdis2_upper"] if s["tet_kdis2_upper"] is not None else np.nan)
    fy, fyh = np.concatenate(full_y), np.concatenate(full_yh)
    ey, eyh = np.concatenate(early_y), np.concatenate(early_yh)
    x = dict(exp=exp, gf=gf, lig=lig, analyte=analyte, e=e, buffer=s["buffer"], tb=tb, top=top, sensors=sensors,
             fit=f, prof2=p2, prof1=p1, ka2=f["Ka2"], ka1=f["Ka1"], kf0=f["kf0"], ks0=f["ks0"],
             kdis2=r["kf"], kdis1=kdis1, kdis1_bound=not measured,
             kf_top=st["kf"], ks_top=st["ks"], tx_top=st["tx"],
             fast_share_top=st["Af"] * -math.expm1(-st["kf"] * tb) / A.two_phase(tb, st["Af"], st["kf"], st["As"], st["ks"]),
             fastoff_share_top=st["A1d"] / max(st["A1d"] + st["A2d"], 1e-12),
             r2_top=st["r2"], r2_all=r2(fy, fyh), r2_early=r2(ey, eyh), r2_min=min(sn["r2"] for sn in sensors),
             r2_dis=r["R2d"], assoc_min=s["assoc_min"], dis_h=s["dissoc_h"], n_fit=len(ws), n_all=len(e["mask"]),
             fit_issues=issues)
    x["alone"] = A.per_sensor_rates(ws, tb, f)
    x["ka2_kind"] = kind(p2); x["ka1_kind"] = kind(p1)
    x["KD2_nM"] = x["kdis2"] / x["ka2"] * 1e9 if x["ka2_kind"] == "value" else np.nan
    x["KD1_nM"] = x["kdis1"] / x["ka1"] * 1e9 if x["ka1_kind"] == "value" else np.nan
    x["ka1_top_apparent"] = (x["ks_top"] - kdis1) / (top * 1e-9)          # v1 definition, for comparison only
    return x


def kind(p):
    if p["resolved"]:
        return "value"
    if p["lo"] == 0 and np.isfinite(p["hi"]):
        return "upper"
    return "nd"


# ------------------------------------------------------------------ text helpers (bound symbols)
def ka_text(x, which):
    k, p, v = x[f"{which}_kind"], x["prof2" if which == "ka2" else "prof1"], x[which]
    if k == "value":
        return f"{v:.3e}"
    if k == "upper":
        return f"< {p['hi']:.2e}"
    return "n.d."


def ka_range(x, which):
    k, p = x[f"{which}_kind"], x["prof2" if which == "ka2" else "prof1"]
    if k == "value":
        return f"{p['lo']:.2e} – {p['hi']:.2e}"
    if k == "upper":
        return f"0 – {p['hi']:.2e} (flat to 0)"
    return "flat on both sides"


def kd2_text(x):
    k = x["ka2_kind"]
    if k == "value":
        return f"{x['KD2_nM']:.3g}"
    if k == "upper":
        return f"> {x['kdis2'] / x['prof2']['hi'] * 1e9:.3g}"
    return "n.d."


def kd1_text(x):
    k, b = x["ka1_kind"], x["kdis1_bound"]
    if k == "value":
        return ("< " if b else "") + f"{x['KD1_nM']:.3g}"
    if k == "upper" and not b:
        return f"> {x['kdis1'] / x['prof1']['hi'] * 1e9:.3g}"
    return "n.d."                              # Ka1 n.d., or Kdis1 < and Ka1 < (bounds in opposite directions)


def kdis1_text(x):
    return ("< " if x["kdis1_bound"] else "") + f"{x['kdis1']:.3e}"


def t1_text(x):
    return ("> " if x["kdis1_bound"] else "") + f"{LN2 / x['kdis1'] / 86400:.2f}"


def notes(x):
    n = []
    if x["ka2_kind"] != "value":
        n.append("Ka2: the fast-phase rate does not change with [EphB2] within the 5 % bracket"
                 + (" (upper bound only)" if x["ka2_kind"] == "upper" else ""))
    if x["ka1_kind"] != "value":
        n.append(f"Ka1: the slow-phase rate is ~independent of [EphB2] (k_slow {x['ks0']:.1e} + Ka1*C); "
                 + ("upper bound only" if x["ka1_kind"] == "upper" else "not determinable"))
    if x["kdis1_bound"]:
        n.append("no tetramer decay detected in 15 h: Kdis1 is an upper bound (KD1 <, tetramer t1/2 >)")
    if x["ka2_kind"] == "value" and x["kf0"] < 1e-6:
        n.append(f"dimer k_obs intercept ~0 (Kdis2 = {x['kdis2']:.1e}/s)")
    elif x["ka2_kind"] == "value" and not 1 / 3 <= x["kf0"] / x["kdis2"] <= 3:
        n.append(f"dimer k_obs intercept ({x['kf0']:.1e}/s) differs from Kdis2 ({x['kdis2']:.1e}/s) by "
                 f"{max(x['kf0'] / x['kdis2'], x['kdis2'] / x['kf0']):.0f}x")
    if x["assoc_min"] < 60:
        n.append(f"{x['assoc_min']:.0f} min association")
    return "; ".join(n)


# ------------------------------------------------------------------ figures
def conc_colours(x):
    concs = sorted({sn["conc"] for sn in x["sensors"]})
    return dict(zip(concs, plt.cm.viridis(np.linspace(0.05, 0.9, len(concs)))))


def assoc_fit_line(a, sn, tb, t_end, scale, lw, z=4):
    """fitted association curve, vermilion where the dimer component dominates the slope, blue after the crossover"""
    tt = np.unique(np.r_[0, np.geomspace(0.25, min(tb, t_end), 700), min(sn["tx"], t_end)])
    tt = tt[tt <= min(tb, t_end)]
    yy = A.two_phase(tt, sn["Af"], sn["kf"], sn["As"], sn["ks"])
    d = tt <= sn["tx"]
    if d.sum() >= 2:
        a.plot(tt[d] / scale, yy[d], "-", lw=lw, color=DIMER, zorder=z, solid_capstyle="butt")
    q = tt >= sn["tx"]
    a.plot(tt[q] / scale, yy[q], "-", lw=lw, color=TETRAMER, zorder=z, solid_capstyle="butt")


def sensorgram(ax, x, small=False, legend=True):
    """full sensorgram in hours: dots = data (viridis by concentration, grey = excluded sensors), association fit by
    phase, dissociation fit by concentration"""
    e = x["e"]; r = e["r"]; tb = x["tb"]; cols = conc_colours(x)
    end = x["sensors"][0]["w"]["t"][-1]
    lw, ms = (1.1, 1.3) if small else (1.4, 1.6)
    for w, m in zip(e["parsed"]["wells"], e["mask"]):
        if not m:
            ta, ya = sub(w["t"], w["y"], 200, 200)
            ax.plot(ta / 3600, ya, ".", ms=ms * 0.7, color=GREY, alpha=0.4)
    for sn in x["sensors"]:
        w = sn["w"]; c = cols[sn["conc"]]
        ma, md = w["t"] <= tb, w["t"] >= tb
        ta, ya = sub(w["t"][ma], w["y"][ma], 150, 120)
        td, yd = sub(w["t"][md] - tb, w["y"][md], 200, 150)
        ax.plot(np.r_[ta, td + tb] / 3600, np.r_[ya, yd], ".", ms=ms, alpha=0.35, color=c, zorder=2)
        assoc_fit_line(ax, sn, tb, tb, 3600, lw)
    for w, A1, A2 in zip(r["keep"], r["A1s"], r["A2s"]):                  # dissociation fit (unchanged)
        tt = np.r_[0, np.geomspace(1, end - tb, 400)]
        ax.plot((tt + tb) / 3600, A1 * np.exp(-r["kf"] * tt) + A2 * np.exp(-r["ks"] * tt), "-", lw=lw,
                color=cols[w["conc_nM"]], label=f"{w['conc_nM']:g} nM", zorder=3)
    ax.axvline(tb / 3600, color=INK2, ls=":", lw=0.9)
    ys = np.concatenate([sn["w"]["y"] for sn in x["sensors"]]); hi = float(ys.max())
    ax.set_ylim(min(0, float(np.percentile(ys, 0.1))) - 0.05 * hi, hi * 1.12)
    ax.set_xlim(-0.2, end / 3600 + 0.2)
    ax.text(tb / 3600, hi * 1.08, "  association | dissociation", fontsize=6.5 if small else 7.5, color=INK2, va="center")
    style(ax, "time (h)", small)
    if legend:
        h, l = ax.get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=6.5, loc="upper right", frameon=False,
                  bbox_to_anchor=(1.0, 0.93), title="dissociation fit", title_fontsize=6.5)


def early_panel(ax, x, small=False):
    """association only, first 10 min, in minutes: shows the dimer (fast) phase and the start of the tetramer phase"""
    tb = x["tb"]; cols = conc_colours(x); t_end = min(EARLY_S, tb)
    lw, ms = (1.1, 1.8) if small else (1.5, 2.4)
    for sn in x["sensors"]:
        w = sn["w"]; m = w["t"] <= t_end
        ta, ya = sub(w["t"][m], w["y"][m], 120, 90)
        ax.plot(ta / 60, ya, ".", ms=ms, alpha=0.45, color=cols[sn["conc"]], zorder=2)
        assoc_fit_line(ax, sn, tb, t_end, 60, lw)
    ys = np.concatenate([sn["w"]["y"][sn["w"]["t"] <= t_end] for sn in x["sensors"]]); hi = float(ys.max())
    ax.set_ylim(min(0, float(np.percentile(ys, 0.5))) - 0.04 * hi, hi * 1.1)
    ax.set_xlim(-0.15, t_end / 60 + 0.1)
    style(ax, "time (min)", small)


def style(ax, xlabel, small):
    ax.set_xlabel(xlabel, fontsize=8 if small else 9); ax.set_ylabel("response (nm)", fontsize=8 if small else 9)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7 if small else 8, colors=INK2)


def rate_panel(ax, x, small=True):
    """fast (dimer) and slow (tetramer) rates vs [EphB2]: lines = global fit (k = Ka*C + k0), markers = each sensor
    refitted alone (filled if that phase carries >= 10 % of the sensor's signal, hollow otherwise); dotted = Kdis2"""
    cc = np.array([sn["conc"] for sn in x["sensors"]])
    cg = np.geomspace(cc.min() / 1.6, cc.max() * 1.6, 200)
    lf, ls_ = x["fit"]["Ka2"] * cg * 1e-9 + x["kf0"], x["fit"]["Ka1"] * cg * 1e-9 + x["ks0"]
    ax.plot(cg, lf, "-", color=DIMER, lw=1.4); ax.plot(cg, ls_, "-", color=TETRAMER, lw=1.4)
    if np.isfinite(x["kdis2"]):                                       # '—' when there is no dimer phase
        ax.axhline(x["kdis2"], color=DIMER, ls=":", lw=0.9)
    for p in x["alone"]:
        for k, sh, c, mk in ((p["kf"], p["share_f"], DIMER, "o"), (p["ks"], p["share_s"], TETRAMER, "s")):
            ax.plot(p["conc"], k, mk, ms=4.2 if small else 5, mec=c, mew=1.0, mfc=c if sh >= 0.1 else "white", zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    lo = np.nanmin([lf.min(), ls_.min(), x["kdis2"]]) / 4; hi = max(lf.max(), ls_.max()) * 4
    ax.set_ylim(lo, hi); ax.set_xlim(cg[0], cg[-1])
    ax.set_xlabel("[EphB2] (nM)", fontsize=7.5 if small else 9); ax.set_ylabel("rate (1/s)", fontsize=7.5 if small else 9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.5 if small else 8, colors=INK2)
    ax.text(0.02, 0.98, "fast (dimer)", transform=ax.transAxes, fontsize=6.3, color=DIMER, va="top")
    ax.text(0.02, 0.89, "slow (tetramer)", transform=ax.transAxes, fontsize=6.3, color=TETRAMER, va="top")
    ax.text(0.98, 0.02, "dotted = Kdis2\nhollow = phase <10% of signal", transform=ax.transAxes, fontsize=5.6,
            color=INK2, ha="right", va="bottom", linespacing=1.2)


PHASE_HANDLES = [Line2D([], [], color=DIMER, lw=2), Line2D([], [], color=TETRAMER, lw=2)]
PHASE_LABELS = ["dimer (fast phase) dominates the fitted slope", "tetramer (slow phase) dominates the fitted slope"]


def run_page(pdf, x, k, n):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.955, f"{x['lig']} + {x['analyte']}   —   {x['exp']}   (Global Fit # {x['gf']})",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.05, 0.928, f"{x['buffer']} · association {x['assoc_min']:.0f} min · dissociation {x['dis_h']:.1f} h · "
             f"{x['n_fit']} of {x['n_all']} sensors fitted · dots = data by concentration (grey = excluded sensors)",
             fontsize=9, color=INK2)
    fig.text(0.95, 0.955, f"page {k} / {n}", fontsize=8, color="#777", ha="right")
    fig.legend(PHASE_HANDLES, PHASE_LABELS, loc="upper right", bbox_to_anchor=(0.975, 0.925), fontsize=7.5,
               frameon=False, title="association fit line", title_fontsize=7.5, alignment="left")
    ax = fig.add_axes([0.07, 0.4, 0.56, 0.49]); sensorgram(ax, x)
    ae = fig.add_axes([0.71, 0.63, 0.26, 0.2]); early_panel(ae, x, small=True)
    ae.set_title(f"association, first {EARLY_S // 60} min (dimer phase)", fontsize=8.5, color=INK2, loc="left")
    az = fig.add_axes([0.71, 0.4, 0.26, 0.17]); sensorgram(az, x, small=True, legend=False)
    az.set_xlim(-0.05, x["tb"] / 3600 + 1.0)
    az.set_title("association + first hour of dissociation", fontsize=8.5, color=INK2, loc="left")
    st = max([sn for sn in x["sensors"] if sn["conc"] == x["top"]], key=lambda sn: sn["ymax"])
    rows = [("Ka1  tetramer (1/Ms)", ka_text(x, "ka1"), f"slope of slow rate vs [EphB2]; 5 % χ² range {ka_range(x, 'ka1')}"),
            ("Kdis1 tetramer (1/s)", kdis1_text(x), f"half-life {t1_text(x)} days" +
             ("  (no decay in 15 h: bound)" if x["kdis1_bound"] else "")),
            ("KD1  tetramer (nM)", kd1_text(x), "Kdis1 / Ka1"),
            ("Ka2  dimer (1/Ms)", ka_text(x, "ka2"), f"slope of fast rate vs [EphB2]; 5 % χ² range {ka_range(x, 'ka2')}"),
            ("Kdis2 dimer (1/s)", f"{x['kdis2']:.3e}", f"half-life {LN2 / x['kdis2'] / 60:.1f} min"),
            ("KD2  dimer (nM)", kd2_text(x), "Kdis2 / Ka2"),
            (f"signal at {x['top']:g} nM", f"{100 * x['fast_share_top']:.0f}% / {100 * (1 - x['fast_share_top']):.0f}%",
             "dimer / tetramer share at the end of association"),
            (f"rates at {x['top']:g} nM", f"{x['kf_top']:.2e} / {x['ks_top']:.2e}",
             f"fast / slow (1/s); dimer dominates the first {st['tx']:.0f} s"),
            ("R² association, top conc", f"{x['r2_top']:.4f}", "global association fit"),
            ("R² association, all curves", f"{x['r2_all']:.4f}",
             f"first {EARLY_S // 60} min {x['r2_early']:.4f}; worst sensor {x['r2_min']:.4f}"),
            ("R² dissociation", f"{x['r2_dis']:.4f}", "global fit, all curves (Kdis1, Kdis2)")]
    y = 0.315
    for name, val, note in rows:
        fig.text(0.05, y, name, fontsize=8.6, color=INK, family="DejaVu Sans Mono")
        fig.text(0.265, y, val, fontsize=8.6, color=INK, family="DejaVu Sans Mono", weight="bold")
        fig.text(0.415, y, note, fontsize=7.5, color=INK2)
        y -= 0.026
    ak = fig.add_axes([0.815, 0.075, 0.16, 0.225]); rate_panel(ak, x)
    ak.set_title("rate vs [EphB2]", fontsize=8.5, color=INK2, loc="left")
    pdf.savefig(fig); plt.close(fig)


def table_pages(pdf, runs):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.03, 0.955, "Extended runs (14-16 h dissociation): rate constants and fit quality — v2 (fast = dimer, "
             "slow = tetramer)", fontsize=13, weight="bold", color=INK)
    fig.text(0.03, 0.93, "1 = tetramer, 2 = dimer. One global association fit per run: fast rate = Ka2·[EphB2] + kf0 "
             "(dimer), slow rate = Ka1·[EphB2] + ks0 (tetramer), amplitudes per sensor.\nKdis from the global "
             "dissociation fit (unchanged). < / > = bound; n.d. = not determinable (rate does not change with [EphB2] "
             "within the 5 % χ² bracket).",
             fontsize=7.8, color=INK2, va="top", linespacing=1.5)
    ax = fig.add_axes([0.01, 0.1, 0.98, 0.8]); ax.axis("off")
    head = ["GF", "Run", "Ligand", "Analyte", "Buffer", "Ka1\n(1/Ms)", "Kdis1\n(1/s)", "KD1\n(nM)", "Ka2\n(1/Ms)",
            "Kdis2\n(1/s)", "KD2\n(nM)", "dimer share\ntop conc", "R² assoc\nall", "R² assoc\n10 min", "R²\ndissoc"]
    rows = [[str(x["gf"]), x["exp"].split()[0], x["lig"].replace("Ephrin", "E").replace("-Fc", ""), x["analyte"],
             x["buffer"], ka_text(x, "ka1"), kdis1_text(x), kd1_text(x), ka_text(x, "ka2"), f"{x['kdis2']:.3e}",
             kd2_text(x), f"{100 * x['fast_share_top']:.0f}%", f"{x['r2_all']:.4f}", f"{x['r2_early']:.4f}",
             f"{x['r2_dis']:.4f}"] for x in runs]
    wd = [0.03, 0.06, 0.04, 0.08, 0.05, 0.08, 0.085, 0.06, 0.08, 0.075, 0.055, 0.06, 0.055, 0.055, 0.05]
    tb = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center", colWidths=[w / sum(wd) for w in wd])
    tb.auto_set_font_size(False); tb.set_fontsize(7.3); tb.scale(1, 1.62)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
    nd1 = [x["exp"].split()[0] for x in runs if x["ka1_kind"] != "value"]
    nd2 = [x["exp"].split()[0] for x in runs if x["ka2_kind"] != "value"]
    fig.text(0.03, 0.06, f"Ka1 bound / n.d. ({', '.join(nd1)}): the slow-phase rate barely changes with [EphB2], so a "
             "bimolecular Ka1 is not determined (upper bound shown).\n"
             f"Ka2 bound / n.d. ({', '.join(nd2)}): the fast-phase rate does not change with [EphB2].  "
             "< / > on Kdis1 (2026.91, .97): no tetramer decay detected in 15 h.  2026.90: 20-min association.",
             fontsize=7.4, color=INK2, linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)
    stats_page(pdf, runs)


KEYS = (("ka1", "Ka1 (1/Ms)", "{:.3e}"), ("kdis1", "Kdis1 (1/s)", "{:.3e}"), ("KD1_nM", "KD1 (nM)", "{:.3g}"),
        ("ka2", "Ka2 (1/Ms)", "{:.3e}"), ("kdis2", "Kdis2 (1/s)", "{:.3e}"), ("KD2_nM", "KD2 (nM)", "{:.3g}"))


def numeric(x, key):
    """the value if it is a measurement (no bound, not n.d.), else None"""
    if key == "ka1":
        return x["ka1"] if x["ka1_kind"] == "value" else None
    if key == "ka2":
        return x["ka2"] if x["ka2_kind"] == "value" else None
    if key == "kdis1":
        return None if x["kdis1_bound"] else x["kdis1"]
    if key == "KD1_nM":
        return x["KD1_nM"] if x["ka1_kind"] == "value" and not x["kdis1_bound"] else None
    if key == "KD2_nM":
        return x["KD2_nM"] if x["ka2_kind"] == "value" else None
    return x[key]


def grp(runs, g, lig="EphrinB3-Fc", buf="PBST"):
    return [x for x in runs if x["analyte"] == g and x["lig"] == lig and x["buffer"] == buf]


def ttest_rows(runs):
    wt, mu = grp(runs, "EphB2-WT"), grp(runs, "EphB2-R79P")
    out = []
    for key, name, f in KEYS:
        a = [v for v in (numeric(x, key) for x in wt) if v is not None]
        b = [v for v in (numeric(x, key) for x in mu) if v is not None]
        p = stats.ttest_ind(a, b, equal_var=True).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan
        out.append(dict(key=key, name=name, f=f, a=a, b=b, p=p))
    return out


def stats_page(pdf, runs):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.95, "EphB2-WT vs EphB2-R79P", fontsize=13, weight="bold", color=INK)
    fig.text(0.04, 0.935, "Student's t-test, two-tailed, equal variance, measured values only (bounds and n.d. excluded). "
             "Only EphrinB3-Fc in PBST has replicates (n = 3 each);\nthe PBSTB sets have one run per variant, shown below "
             "for reference (not tested). Buffers are not pooled.", fontsize=8.3, color=INK2, va="top", linespacing=1.5)
    ax2 = fig.add_axes([0.03, 0.5, 0.94, 0.38]); ax2.axis("off")
    ms = lambda v, f: (f"{f.format(np.mean(v))} ± {f.format(np.std(v, ddof=1))} (n={len(v)})" if len(v) >= 2 else
                       (f"{f.format(v[0])} (n=1)" if v else "none measured"))
    rows2 = []
    for t in ttest_rows(runs):
        ratio = f"{np.mean(t['b']) / np.mean(t['a']):.2f}" if t["a"] and t["b"] else "–"
        pv = "n/a" if not np.isfinite(t["p"]) else ("<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")
        rows2.append([t["name"], ms(t["a"], t["f"]), ms(t["b"], t["f"]), ratio, pv])
    tb2 = ax2.table(cellText=rows2, colLabels=["EphrinB3-Fc, PBST", "WT mean ± SD", "R79P mean ± SD", "R79P / WT",
                                               "Student's p"], loc="upper center", cellLoc="center")
    tb2.auto_set_font_size(False); tb2.set_fontsize(8.4); tb2.scale(1, 1.8)
    wt, mu = grp(runs, "EphB2-WT"), grp(runs, "EphB2-R79P")
    fig.text(0.04, 0.475, "Per run (PBST): Ka1 WT " + ", ".join(ka_text(x, "ka1") for x in wt) + "; R79P " +
             ", ".join(ka_text(x, "ka1") for x in mu) + ".  p = n/a where a group has fewer than 2 measured values.",
             fontsize=7.6, color=INK2)
    ax3 = fig.add_axes([0.03, 0.08, 0.94, 0.36]); ax3.axis("off")
    rows3 = []
    for lig in ("EphrinB3-Fc", "EphrinB2-Fc"):
        a, b = grp(runs, "EphB2-WT", lig, "PBSTB")[0], grp(runs, "EphB2-R79P", lig, "PBSTB")[0]
        for name, fn in (("Ka1 (1/Ms)", lambda x: ka_text(x, "ka1")), ("Kdis1 (1/s)", kdis1_text), ("KD1 (nM)", kd1_text),
                         ("Ka2 (1/Ms)", lambda x: ka_text(x, "ka2")), ("Kdis2 (1/s)", lambda x: f"{x['kdis2']:.3e}"),
                         ("KD2 (nM)", kd2_text)):
            rows3.append([f"{lig}, PBSTB", name, f"{fn(a)} ({a['exp'].split()[0]})", f"{fn(b)} ({b['exp'].split()[0]})"])
    tb3 = ax3.table(cellText=rows3, colLabels=["Set", "Constant", "WT (single run)", "R79P (single run)"],
                    loc="upper center", cellLoc="center")
    tb3.auto_set_font_size(False); tb3.set_fontsize(7.6); tb3.scale(1, 1.3)
    for t in (tb2, tb3):
        for (i, j), c in t.get_celld().items():
            c.set_edgecolor(GRID)
            if i == 0:
                c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
    pdf.savefig(fig); plt.close(fig)


def overview_png(runs, path=OUT_PNG, tag="v2"):
    """4 x 4 runs; each cell = first 10 min of association (left, shows the dimer phase) + full sensorgram (right)"""
    nrow = int(np.ceil(len(runs) / 4))
    fig = plt.figure(figsize=(24, 4 * nrow))
    gs = fig.add_gridspec(nrow, 8, width_ratios=[1, 2.3] * 4, wspace=0.32, hspace=0.42, left=0.035, right=0.99,
                          top=1 - 1.12 / (4 * nrow), bottom=0.64 / (4 * nrow))
    for k, x in enumerate(runs):
        r, c = divmod(k, 4)
        ae = fig.add_subplot(gs[r, 2 * c]); early_panel(ae, x, small=True)
        ae.set_title(f"first {EARLY_S // 60} min", fontsize=8, color=INK2, loc="left")
        ax = fig.add_subplot(gs[r, 2 * c + 1]); sensorgram(ax, x, small=True, legend=False)
        ax.set_ylabel("")
        ax.set_title(f"{x['lig'].replace('-Fc', '')} + {x['analyte']}  {x['exp'].split()[0]}  (GF {x['gf']}, {x['buffer']})",
                     fontsize=9, weight="bold", color=INK, loc="left")
    fig.legend(PHASE_HANDLES, PHASE_LABELS, loc="upper right", bbox_to_anchor=(0.99, 0.995), fontsize=10, frameon=False,
               ncol=2, title="association fit line", title_fontsize=10)
    fig.suptitle(f"All {len(runs)} extended runs ({tag}: fast phase = dimer, slow phase = tetramer): dots = data by concentration, "
                 "lines = fits", fontsize=13, weight="bold", color=INK, x=0.01, y=0.985, ha="left")
    fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ workbook
COLS = ["Global Fit #", "Run", "Ligand", "Analyte", "Buffer", "Association (min)", "Dissociation (h)",
        "Top concentration (nM)",
        "Ka2 dimer (1/Ms) = slope of fast-phase rate vs [EphB2]", "Ka2 5% χ² range, low", "Ka2 5% χ² range, high",
        "Dimer k_obs intercept kf0 (1/s)", "Kdis2 dimer (1/s)", "KD2 dimer (nM)",
        "Ka1 tetramer (1/Ms) = slope of slow-phase rate vs [EphB2]", "Ka1 5% χ² range, low", "Ka1 5% χ² range, high",
        "Tetramer k_obs intercept ks0 (1/s)", "Kdis1 tetramer (1/s)", "Kdis1 type", "KD1 tetramer (nM)",
        "Fast (dimer) rate at top conc (1/s)", "Slow (tetramer) rate at top conc (1/s)",
        "v1-style Ka1 from top conc only = (slow rate − Kdis1)/[EphB2]top (1/Ms), for comparison",
        "Dimer share of association signal at top conc", "Fast-off share of dissociation amplitude at top conc",
        "Dimer-dominated time at top conc (s)",
        "R² association, top concentration", "R² association, all curves", "R² association, first 10 min, all curves",
        "R² association, worst sensor", "R² dissociation, global fit (Kdis1, Kdis2)",
        "Tetramer half-life (days)", "Dimer half-life (min)", "Notes"]


def write_xlsx(runs):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter as gl
    wb = Workbook()
    F, FB = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    FILL = PatternFill("solid", fgColor="F2F2EF"); BOX = Border(bottom=Side(style="thin", color="D9D9D6"))
    col = {name: j for j, name in enumerate(COLS, 1)}
    L = {name: gl(j) for name, j in col.items()}

    def header(ws, cols, widths, freeze="E2"):
        for j, (h, w) in enumerate(zip(cols, widths), 1):
            c = ws.cell(row=1, column=j, value=h); c.font = FB; c.fill = FILL; c.border = BOX
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            ws.column_dimensions[c.column_letter].width = w
        ws.row_dimensions[1].height = 72; ws.freeze_panes = freeze

    ws = wb.active; ws.title = "Constants"
    widths = [8, 15, 12, 12, 8, 10, 10, 10, 14, 11, 11, 12, 12, 10, 14, 11, 11, 12, 13, 11, 10, 12, 12, 16, 11, 11, 11,
              11, 11, 11, 11, 13, 11, 11, 80]
    header(ws, COLS, widths)
    num = lambda v: float(f"{v:.10g}")
    E3, E4 = "0.000E+00", "0.00E+00"
    for i, x in enumerate(runs):
        rr = i + 2; x["row"] = rr
        p2, p1 = x["prof2"], x["prof1"]
        v = {"Global Fit #": x["gf"], "Run": x["exp"], "Ligand": x["lig"], "Analyte": x["analyte"], "Buffer": x["buffer"],
             "Association (min)": round(x["assoc_min"], 1), "Dissociation (h)": round(x["dis_h"], 2),
             "Top concentration (nM)": x["top"],
             COLS[8]: num(x["ka2"]) if x["ka2_kind"] == "value" else ka_text(x, "ka2"),
             "Ka2 5% χ² range, low": num(p2["lo"]) if x["ka2_kind"] == "value" else 0,
             "Ka2 5% χ² range, high": num(p2["hi"]) if np.isfinite(p2["hi"]) else "flat",
             "Dimer k_obs intercept kf0 (1/s)": num(x["kf0"]), "Kdis2 dimer (1/s)": num(x["kdis2"]),
             "KD2 dimer (nM)": f"={L['Kdis2 dimer (1/s)']}{rr}/{L[COLS[8]]}{rr}*1E9" if x["ka2_kind"] == "value"
             else kd2_text(x),
             COLS[14]: num(x["ka1"]) if x["ka1_kind"] == "value" else ka_text(x, "ka1"),
             "Ka1 5% χ² range, low": num(p1["lo"]) if x["ka1_kind"] == "value" else 0,
             "Ka1 5% χ² range, high": num(p1["hi"]) if np.isfinite(p1["hi"]) else "flat",
             "Tetramer k_obs intercept ks0 (1/s)": num(x["ks0"]), "Kdis1 tetramer (1/s)": num(x["kdis1"]),
             "Kdis1 type": "upper bound" if x["kdis1_bound"] else "measured",
             "KD1 tetramer (nM)": f"={L['Kdis1 tetramer (1/s)']}{rr}/{L[COLS[14]]}{rr}*1E9"
             if x["ka1_kind"] == "value" and not x["kdis1_bound"] else kd1_text(x),
             "Fast (dimer) rate at top conc (1/s)": num(x["kf_top"]), "Slow (tetramer) rate at top conc (1/s)": num(x["ks_top"]),
             COLS[23]: f"=({L['Slow (tetramer) rate at top conc (1/s)']}{rr}-{L['Kdis1 tetramer (1/s)']}{rr})/"
                       f"({L['Top concentration (nM)']}{rr}*1E-9)",
             "Dimer share of association signal at top conc": num(x["fast_share_top"]),
             "Fast-off share of dissociation amplitude at top conc": num(x["fastoff_share_top"]),
             "Dimer-dominated time at top conc (s)": round(x["tx_top"], 1),
             "R² association, top concentration": num(x["r2_top"]), "R² association, all curves": num(x["r2_all"]),
             "R² association, first 10 min, all curves": num(x["r2_early"]),
             "R² association, worst sensor": num(x["r2_min"]),
             "R² dissociation, global fit (Kdis1, Kdis2)": num(x["r2_dis"]),
             "Tetramer half-life (days)": f"=LN(2)/{L['Kdis1 tetramer (1/s)']}{rr}/86400",
             "Dimer half-life (min)": f"=LN(2)/{L['Kdis2 dimer (1/s)']}{rr}/60", "Notes": notes(x)}
        fmt = {COLS[8]: E3, "Ka2 5% χ² range, low": E4, "Ka2 5% χ² range, high": E4, "Dimer k_obs intercept kf0 (1/s)": E3,
               "Kdis2 dimer (1/s)": E3, "KD2 dimer (nM)": "0.000", COLS[14]: E3, "Ka1 5% χ² range, low": E4,
               "Ka1 5% χ² range, high": E4, "Tetramer k_obs intercept ks0 (1/s)": E3, "Kdis1 tetramer (1/s)": E3,
               "KD1 tetramer (nM)": "0.0000", "Fast (dimer) rate at top conc (1/s)": E3,
               "Slow (tetramer) rate at top conc (1/s)": E3, COLS[23]: E3,
               "Dimer share of association signal at top conc": "0%",
               "Fast-off share of dissociation amplitude at top conc": "0%", "R² association, top concentration": "0.0000",
               "R² association, all curves": "0.0000", "R² association, first 10 min, all curves": "0.0000",
               "R² association, worst sensor": "0.0000", "R² dissociation, global fit (Kdis1, Kdis2)": "0.0000",
               "Tetramer half-life (days)": "0.00", "Dimer half-life (min)": "0.0"}
        for name, val in v.items():
            c = ws.cell(row=rr, column=col[name], value=val); c.font = F
            if name in fmt:
                c.number_format = fmt[name]
            if isinstance(val, str) and not val.startswith("=") and name not in ("Run", "Ligand", "Analyte", "Buffer",
                                                                               "Kdis1 type", "Notes"):
                c.alignment = Alignment(horizontal="right")
        ws.cell(row=rr, column=col["Notes"]).alignment = Alignment(wrap_text=True, vertical="top")

    # ---- WT vs R79P (EphrinB3-Fc, PBST). Formulas when all six runs are measured; otherwise typed (measured only).
    st = wb.create_sheet("WT vs R79P")
    header(st, ["Constant (EphrinB3-Fc, PBST)", "WT mean", "WT SD", "WT n", "R79P mean", "R79P SD", "R79P n", "R79P / WT",
                "Student's t-test p (two-tailed, equal variance)", "How computed"], [28, 13, 13, 7, 13, 13, 8, 10, 18, 60],
           freeze="A2")
    wt = grp(runs, "EphB2-WT"); mu = grp(runs, "EphB2-R79P")
    wr, mr = [x["row"] for x in wt], [x["row"] for x in mu]
    assert wr == list(range(wr[0], wr[-1] + 1)) and mr == list(range(mr[0], mr[-1] + 1)) and len(wr) == len(mr) == 3
    colmap = {"ka1": L[COLS[14]], "kdis1": L["Kdis1 tetramer (1/s)"], "KD1_nM": L["KD1 tetramer (nM)"],
              "ka2": L[COLS[8]], "kdis2": L["Kdis2 dimer (1/s)"], "KD2_nM": L["KD2 dimer (nM)"]}
    fmts = {"ka1": E3, "kdis1": E3, "KD1_nM": "0.000", "ka2": E3, "kdis2": E3, "KD2_nM": "0.00"}
    for i, t in enumerate(ttest_rows(runs)):
        rr = i + 2; c = colmap[t["key"]]
        allnum = len(t["a"]) == 3 and len(t["b"]) == 3
        if allnum:
            a, b = f"Constants!{c}{wr[0]}:{c}{wr[-1]}", f"Constants!{c}{mr[0]}:{c}{mr[-1]}"
            vals = [t["name"], f"=AVERAGE({a})", f"=STDEV({a})", f"=COUNT({a})", f"=AVERAGE({b})", f"=STDEV({b})",
                    f"=COUNT({b})", f"=E{rr}/B{rr}", f"=TTEST({a},{b},2,2)", "formulas on the Constants sheet"]
        else:
            sd = lambda v: float(np.std(v, ddof=1)) if len(v) >= 2 else "–"
            vals = [t["name"], float(np.mean(t["a"])) if t["a"] else "–", sd(t["a"]), len(t["a"]),
                    float(np.mean(t["b"])) if t["b"] else "–", sd(t["b"]), len(t["b"]),
                    float(np.mean(t["b"]) / np.mean(t["a"])) if t["a"] and t["b"] else "–",
                    float(t["p"]) if np.isfinite(t["p"]) else "n/a",
                    "typed: measured values only (bounds / n.d. excluded); p needs >= 2 measured values per group"]
        for j, val in enumerate(vals, 1):
            st.cell(row=rr, column=j, value=val).font = F
        for cc in (2, 3, 5, 6):
            st.cell(row=rr, column=cc).number_format = fmts[t["key"]]
        st.cell(row=rr, column=8).number_format = "0.00"; st.cell(row=rr, column=9).number_format = "0.000"
    st.cell(row=9, column=1, value=("WT = GF 69, 70, 73; R79P = GF 71, 72, 74. The PBSTB sets (EphrinB3: GF 67 vs 68; "
                                    "EphrinB2: GF 60 vs 61) have one run per variant and are not tested. Buffers are not pooled.")
            ).font = Font(name="Arial", size=9, italic=True)

    nt = wb.create_sheet("Notes"); nt.column_dimensions["A"].width = 130
    for i, t in enumerate([
        "v2 of the 16 extended runs (14-16 h dissociation) = the blue-highlighted rows of 'MASTER TABLE (BEST Colored)'. "
        "Generated 2026-09-30. 1 = tetramer, 2 = dimer. v1 (Extended_constants_all16_9-30-26.xlsx) is unchanged.",
        "Change from v1: the association is split by RATE, not amplitude. Fast phase = dimer, slow phase = tetramer, in every "
        "run and at every concentration. v1 used the larger-amplitude component as the dimer, which labelled the gradual phase "
        "'dimer' whenever it carried most of the signal (2026.90, .96, .97).",
        "Association model (one global fit per run, all fitted sensors): R_i(t) = Af_i(1-exp(-kf_i t)) + As_i(1-exp(-ks_i t)); "
        "kf_i = Ka2*C_i + kf0 (dimer), ks_i = Ka1*C_i + ks0 (tetramer); amplitudes Af_i, As_i free per sensor; the fast rate "
        "must be >= 3x the slow rate at the top concentration. 48-point start grid; best sum of squares kept.",
        "Ka2 and Ka1 = the two slopes. Each slope was profiled (fixed on a grid, everything else refitted). The 5% chi-square "
        "range = slopes whose sum of squares is within 5% of the minimum (a practical bracket: BLI points are strongly "
        "correlated, so a formal F-test is meaningless). Reported as a number when the range is closed on both sides; "
        "'< x' when it stays open down to zero (upper bound); n.d. when open on both sides.",
        "Kdis2, Kdis1 (unchanged from v1): fast and slow rates of the global biexponential dissociation fit. Kdis1 type "
        "'upper bound' = no tetramer decay detected in 15 h; KD1 and the tetramer half-life are then bounds.",
        "KD = Kdis/Ka x 1e9 (nM). If Ka is an upper bound, KD is a lower bound (>). KD1 is n.d. when Ka1 is an upper bound and "
        "Kdis1 is also an upper bound (bounds in opposite directions).",
        "Consistency checks (columns L, R, Y, Z): for a simple 1:1 step the k_obs intercept equals the off-rate. The dimer "
        "intercept kf0 matches Kdis2 within ~3x in the PBST runs, but is ~10-70x larger in 2026.90, .96, .97 (PBSTB). "
        "The tetramer intercept ks0 (1e-4 - 7e-4 /s) is 100-1000x larger than Kdis1, i.e. the slow phase is mostly a "
        "concentration-independent step, which is why Ka1 is often only bounded.",
        "Column X keeps the v1 definition of Ka1 (top concentration only, assumes the intercept is Kdis1) for comparison; "
        "it is not used anywhere else.",
        "R2 association: the global fit on the full 0.5 s data (top sensor; all sensors pooled; first 10 min pooled = the "
        "dimer-phase window; worst single sensor). R2 dissociation = pooled global dissociation fit (unchanged).",
        "PDF: association fit lines are vermilion where the dimer (fast) component dominates the slope of the fitted curve and "
        "blue where the tetramer (slow) component does; crossover t_x = ln(Af kf / As ks)/(kf - ks) (column AA at the top "
        "concentration). Dissociation fit lines keep the concentration colours.",
        "Typed values are fit outputs; KD1, KD2 (when measured), the v1-style Ka1, the half-lives and the WT vs R79P statistics "
        "(when all six runs are measured) are formulas.",
        "Code: dimer_tetramer/extended_constants_all16_v2.py (+ assoc_two_phase.py)"], 1):
        c = nt.cell(row=i, column=1, value=t); c.font = F; c.alignment = Alignment(wrap_text=True)
    wb.save(OUT_XLSX)
    return L


def main():
    runs = load()
    with PdfPages(OUT_PDF) as pdf:
        table_pages(pdf, runs)
        for k, x in enumerate(runs, 3):
            run_page(pdf, x, k, len(runs) + 2)
        pdf.infodict()["Title"] = "All 16 extended runs v2: fast phase = dimer, slow phase = tetramer"
    overview_png(runs)
    L = write_xlsx(runs)
    cell, n = evaluate_workbook(OUT_XLSX)
    for x in runs:
        rr = x["row"]
        got = cell("Constants", f"{L[COLS[23]]}{rr}"); want = x["ka1_top_apparent"]
        assert abs(got - want) <= 1e-7 * abs(want), (rr, got, want)
        if x["ka2_kind"] == "value":
            got = cell("Constants", f"{L['KD2 dimer (nM)']}{rr}"); assert abs(got - x["KD2_nM"]) <= 1e-7 * x["KD2_nM"]
        if x["ka1_kind"] == "value" and not x["kdis1_bound"]:
            got = cell("Constants", f"{L['KD1 tetramer (nM)']}{rr}"); assert abs(got - x["KD1_nM"]) <= 1e-7 * x["KD1_nM"]
    for i, t in enumerate(ttest_rows(runs)):
        if np.isfinite(t["p"]):
            got = cell("WT vs R79P", f"I{i + 2}"); assert abs(got - t["p"]) <= 1e-6 * t["p"], (t["key"], got, t["p"])
    print(f"workbook: {n} formulas evaluated, all finite and matching Python")
    print(f"{'GF':3s} {'run':9s} {'lig':4s} {'analyte':11s} {'buf':6s} {'Ka1':>11s} {'Kdis1':>11s} {'KD1 nM':>8s} "
          f"{'Ka2':>11s} {'Kdis2':>10s} {'KD2 nM':>7s} {'dim%':>5s} {'R2all':>6s} {'R2 10m':>6s} {'R2dis':>6s}  kf0/Kdis2")
    for x in runs:
        print(f"{x['gf']:3d} {x['exp'].split()[0]:9s} {x['lig'][6:8]:4s} {x['analyte']:11s} {x['buffer']:6s} "
              f"{ka_text(x, 'ka1'):>11s} {kdis1_text(x):>11s} {kd1_text(x):>8s} {ka_text(x, 'ka2'):>11s} "
              f"{x['kdis2']:10.3e} {kd2_text(x):>7s} {100 * x['fast_share_top']:4.0f}% {x['r2_all']:6.4f} "
              f"{x['r2_early']:6.4f} {x['r2_dis']:6.4f}  {x['kf0'] / x['kdis2']:.2g}")
    for t in ttest_rows(runs):
        print(f"  EB3 PBST {t['key']:7s} WT {t['a']}  R79P {t['b']}  p = {t['p']:.3g}")
    print(OUT_PDF); print(OUT_PNG); print(OUT_XLSX)


if __name__ == "__main__":
    main()
