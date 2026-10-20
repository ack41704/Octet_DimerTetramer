> **Superseded.** This is the v1 description (2026-09-29). The current model is **v3**, described in [METHODS.md](METHODS.md). v3 changes how association phases are assigned (by rate: fast = dimer, slow = tetramer, from one global fit per run) and reports point values with flags instead of bounds. The sensor selection, dissociation fit and window tests below are unchanged and still apply. File names and dates refer to the working analysis folder; in this repository the scripts are in `dimer_tetramer/` and read data from `$OCTET_DATA_DIR`.

# The dimer → tetramer kinetic model for ephrinB-Fc / Eph BLI data

This document covers what the model assumes, what the software actually computes, every rule that decides whether a number is reported as a measurement, a bound or "not reliable", and how each rule was checked. It describes the analysis behind:

- the **"Dimer-Tetramer Model" tab** in `Global Fit - Master Table on 9-29-26.numbers`;
- **`Dimer-Tetramer_All_Curve_Fits_9-29-26.pdf`**, which has one page per experiment in master-table order;
- the GUI (`gui_app.py`) and the CSV exports in this folder.

**Numbering.** Everything user-facing (the Numbers tab, the PDF and the GUI cards) uses **1 = tetramer, 2 = dimer**. The code and CSV files number steps in assembly order, the opposite way, so every CSV column names the species ("dimer_ka", "tetramer_kdis") and never relies on a number.

| Quantity | Master table / PDF / GUI | Code and CSV column |
|---|---|---|
| Dimer association (1/Ms) | **Ka2** | `ka1`, `dimer_ka_M1s1` |
| Dimer dissociation (1/s) | **Kdis2** | `kdis1` (= `kf`), `dimer_kdis_s1` |
| Dimer KD (nM) | **KD2** | `KD1_nM`, `dimer_KD_nM` |
| Tetramer closure (1/s) | **Ka1** | `ka2`, `tetramer_ka_s1` |
| Tetramer dissociation (1/s) | **Kdis1** | `kdis2` (= `ks`), `tet_kdis2` |
| Tetramer KD (dimensionless) | **KD1** | `KD2`, `tetramer_KD` |

---

## 1. What is being modelled, biophysically

**On the sensor.** The ligand is an **ephrinB-Fc fusion** (ephrin-B1, -B2 or -B3). The Fc is a disulfide-linked homodimer, so every immobilized ligand molecule carries **two ephrin ectodomains ("arms")** held close together.

**In solution.** The analyte is a **monomeric Eph receptor** ligand-binding domain: EphB1-His, EphB2-His, EphB4-His or EphA4-His, or EphB2-WT and the EphB2-R79P mutant.

**Two interfaces and two steps.** Crystal structures of Eph–ephrin complexes show two kinds of contact:

1. **The high-affinity heterodimerization interface.** One Eph binds one ephrin, forming a 1:1 heterodimer. On the sensor this happens independently on each Fc arm. It is a **bimolecular** reaction, so its rate depends on analyte concentration.
2. **The lower-affinity tetramerization interfaces.** Two 1:1 heterodimers pack against each other into a circular 2:2 heterotetramer. On an Fc both heterodimers are tethered to the same molecule, so closure is **intramolecular (first-order)**. Its rate and equilibrium constant are therefore concentration-independent, and the equilibrium constant is **dimensionless**. It is the ratio of closed to open doubly-bound Fc, with the local "effective concentration" of the partner heterodimer absorbed into the constant.

**Why dissociation is biphasic.** A heterodimer that is not part of a tetramer loses its Eph at the heterodimer off-rate, which takes minutes. An Eph locked into a closed tetramer can only leave after the tetramer reopens, which takes days. Buffer-only dissociation therefore shows:

- a **fast phase**: loss of Eph from open heterodimers (the **dimer** off-rate);
- a **slow phase**: Eph held in tetramers (the **tetramer** lifetime).

**R79P** sits at the tetramer interface. The model's central prediction is that it should mainly shorten the slow phase, and it does: tetramer t½ is 8.46 ± 3.29 d for WT versus 2.85 ± 0.82 d for R79P (EphrinB3-Fc, PBST, n = 3 each). R79P also changes the fast phase: dimer Kdis2 is 5.2e-4 versus 1.4e-3 s⁻¹, and KD2 is 4.1 versus 9.7 nM. See §10.1 for why the fast-phase rate is not a completely "pure" dimer constant.

---

## 2. The mechanistic model (four states)

Each immobilized Fc ligand is in one of four states, written as fractions of all ligands:

| State | Meaning | Eph bound | Signal weight |
|---|---|---|---|
| f0 (L₂) | free Fc | 0 | 0 |
| f1 (L₂·A) | one arm occupied (1:1 dimer on one arm) | 1 | ½ |
| f2 (L₂·A₂) | both arms occupied, **open** | 2 | 1 |
| fT (T) | both arms occupied, **closed 2:2 tetramer** | 2 | 1 |

Scheme, in the user's numbering (2 = dimer step, 1 = tetramer step):

```
            2·ka2·[A]              ka2·[A]               ka1
   L2   ⇌─────────────⇌   L2·A  ⇌─────────────⇌  L2·A2  ⇌──────⇌   T
             kdis2                 2·kdis2               kdis1
```

The factors of 2 are statistical. A free Fc has two empty arms that analyte can hit, and a doubly-bound open Fc has two Eph that can leave. With these factors, `ka2` and `kdis2` are **per-arm** constants, so KD2 = kdis2/ka2 is the affinity of one ephrin for one Eph.

Rate equations, where A is the analyte concentration (constant during association, 0 during dissociation):

```
df0/dt = −2·ka2·A·f0 + kdis2·f1
df1/dt = +2·ka2·A·f0 − kdis2·f1 − ka2·A·f1 + 2·kdis2·f2
df2/dt = +ka2·A·f1 − 2·kdis2·f2 − ka1·f2 + kdis1·fT
dfT/dt = +ka1·f2 − kdis1·fT
f0 + f1 + f2 + fT = 1

BLI response:  R(t) = Rmax · (½·f1 + f2 + fT)
```

The response counts **bound Eph mass**. Closing an open f2 into T adds no mass, so it does not change the signal by itself. It changes the signal only by shifting the binding equilibria (see §2.2).

Derived constants:

- **KD2 = kdis2 / ka2** (M, reported in nM): the per-arm heterodimer affinity.
- **KD1 = kdis1 / ka1** (dimensionless) = 1/K_close. For example, KD1 = 4.5e-3 means a doubly-bound Fc spends about 220 times longer closed than open.
- **t½ = ln 2 / kdis** for each step. Dimer t½ is reported in minutes and tetramer t½ in days.

**Model assumptions (what it leaves out):**
- Constant analyte concentration: no depletion, no mass-transport limitation.
- The two arms are identical and independent before closure: no cooperativity in the dimer step.
- A tetramer must reopen before either Eph can leave (no direct T → f1 path).
- No rebinding during dissociation.
- Homogeneous ligand on the sensor.
- No baseline drift after reference subtraction.

### 2.1 What the model predicts for dissociation (A = 0)

- **Without closure (ka1 = 0), the decay is exactly single-exponential.** With f2 = 1 at the start, f2 = e^(−2k·t), f1 = 2(e^(−k·t) − e^(−2k·t)), and R/Rmax = ½·f1 + f2 = e^(−k·t) with k = kdis2. Each Eph leaves independently, so the fast phase measures the per-arm dimer off-rate directly.
- **With closure, a slow phase appears.** When the tetramer off-rate is much smaller than the others, the slow eigenvalue of the dissociation is:

```
λ_slow ≈ kdis1 · 2·kdis2 / (2·kdis2 + ka1)
```

  A reopened tetramer can either lose an Eph (rate 2·kdis2) or re-close (rate ka1). Only the first path takes signal away.
- **The fast phase** is dominated by kdis2, with a contribution at 2·kdis2 + ka1 from the open-bivalent state.

### 2.2 What the model predicts for association

- **Fast phase.** Each arm fills with observed rate **k_obs = ka2·[A] + kdis2**, the standard pseudo-first-order result. It is linear in [A] with slope ka2.
- **Slow phase.** Closure then pulls doubly-bound Fc into T. Let θ = [A]/([A] + KD2) be the arm occupancy. If the arm steps equilibrate quickly compared with closure, the open-doubly-bound fraction is θ² and the closure relaxes at

```
k_slow ≈ ka1·θ² + kdis1
```

  The signal it adds is proportional to (1 − θ). This matters in two ways:
  - At fully saturating [A], closure is invisible in the association trace.
  - Reading the slow rate as ka1 + kdis1 is only correct when the top concentrations nearly saturate the dimer step (θ² → 1).

---

## 3. Why the ODE is not fitted directly

Fitting the four-state ODE freely to each experiment was tried first and abandoned for three reasons:

1. **The parameters are not identifiable from one experiment.** ka1 and kdis1 trade off against the amplitude split. With 13-min associations and 0.7–4 h dissociations, many parameter sets fit equally well.
2. **Shared-parameter fits flipped sign.** In a joint fit, WT and R79P swapped which tetramer was more stable. That contradicted the lab's direct manual half-life reads (R79P ≈ 3 d, WT ≈ 6 d).
3. **A floating baseline absorbs the slow phase.** An offset term let the fit call the tetramer "baseline", so a fitted offset hides exactly the parameter of interest.

The pipeline therefore extracts each constant from the part of the data that determines it, using **empirical exponentials whose rates map onto the model** (§2.1–2.2). The two-stage estimates track the lab's manual half-life reads (r = 0.81). The full ODE (`bivalent_model.py`) is kept as the interpretive framework and for simulation.

---

## 4. What the pipeline computes, step by step

All code is in `dimer_tetramer/`. `plot_all_fits.py` holds the single copy of the fitting logic, and the PDF, GUI, CSV exports and Numbers tab all call it.

### 4.1 Input
- **Files.** Octet "Preprocessed Data" CSV exports (reference-subtracted and aligned by the Octet software), one per experiment/global-fit, under `EB1/EB2/EB3 Interactions - Preprocessed Data FIles/<interaction folder>/`.
- **Parsing.** `parse_octet.py` reads each sensor's time and response traces, its analyte concentration and its Max-Control flag.
- **Genotype.** For interaction folders ending in `-WT` or `-R79P`, only sensors with that genotype label are kept. Files such as "2026.84a (WT) & (R79P)" contain both genotypes and are split between the two rows.
- **Buffer.** Taken from the experiment name: "PBSTB" → PBSTB, "casein" → PBST+casein, otherwise PBST.

### 4.2 Association → dissociation boundary
The boundary is the time of the maximum of the **mean** trace across sensors with signal (non-control, peak > 0.02 nm). The maximum must fall between 3% and 90% of the record, otherwise there is no usable dissociation step. Three files fail this or the parsing step (§8).

### 4.3 Which sensors are fitted (the GUI checklist defaults)
Rules are applied in order. Counts are across all 75 fitted experiments.

| Rule | Criterion | Excluded |
|---|---|---|
| control well | labelled Max Control | 236 |
| near-zero signal | peak ≤ 0.02 nm | 167 |
| dissociation too short | fewer than 100 dissociation points | 0 |
| drifts upward | mean of last 100 dissociation points > mean of first 100 (not physical for dissociation) | 41 |
| **dominant sensor** | peak > 3 × the largest peak of every other remaining sensor | 2 |
| **failed replicate** | at a concentration with ≥ 2 sensors (within 2%), peak < ⅓ of its replicate | 9 |

**The two new rules.** The last two rules were added in this revision and are physical sanity checks:
- **Dominant sensor.** Binding saturates, so no concentration step can raise the response more than 3× above every other sensor. In 2026.26 and 2026.27 a "182 nM" sensor read 12–18× its 182.3 nM replicate, probably a mislabelled control. Because the fit is unweighted, that one sensor carried about 99% of the squared signal and dominated the global fit: dimer t½ came out as 404 min instead of 2 min.
- **Failed replicate.** A sensor can fail to bind (unloaded or reference tip) but cannot bind 3× more than an identical sensor, so the low one is dropped. Removing these made the dimer Ka2 regression pass in several runs; for example, 2026.26 went from R² 0.03 to 1.00 and 2026.84a (R79P) from 0.47 to 0.99.

In total, 364 of 819 sensors are fitted. Every exclusion appears, with its reason, in the GUI checklist and on the PDF page (grey traces), and any of them can be overridden in the GUI.

### 4.4 Stage 1: global biexponential fit of the dissociation (gives Kdis2 and Kdis1)
- **Data.** For every fitted sensor, dissociation time t′ = t − t_boundary. To weight the early fast decay and the long tail equally, each sensor is subsampled to the union of 200 log-spaced and 200 linearly spaced points.
- **Model.** All sensors are fitted at once with **shared rates** and sensor-specific amplitudes:

```
R_i(t′) = a_i·exp(−kf·t′) + b_i·exp(−ks·t′)
kf = fast rate → dimer Kdis2      ks = slow rate → tetramer Kdis1
```

  The decay goes to **zero**, with no baseline offset (see §3, point 3).
- **Parameter bounds.** Rates are fitted as log₁₀ values:
  - kf ∈ [1e-5, 0.316] s⁻¹;
  - ks ∈ [1e-8, 3.16e-4] s⁻¹;
  - amplitudes ∈ [0, 3] nm.
- **Optimizer.**
  - Method: trust-region-reflective least squares (`scipy.optimize.least_squares`, xtol = ftol = 1e-14, up to 20 000 evaluations), unweighted residuals.
  - Start amplitudes: 30% and 70% of each sensor's first dissociation point.
  - Multistart: four (kf, ks) starts, (8e-4, 1.5e-6), (5e-3, 5e-6), (1e-4, 5e-7) and (2e-3, 5e-5). The lowest-cost solution is kept, so it can only match or beat a single start.
- **Reproducibility.** Re-running is bit-for-bit reproducible.
- **"Slow phase (% of R)".** Σb_i / Σ(a_i + b_i): the share of the signal at the start of dissociation that is held in the slow (tetramer) phase.

### 4.5 Stage 2: association fits and dimer Ka2
- **Per-sensor fit.** Each fitted sensor's association trace (≥ 40 points; 300 log-spaced points used) is fitted separately:

```
R(t) = c1·(1 − exp(−k1·t)) + c2·(1 − exp(−k2·t))       (k1 ≥ k2 after sorting)
```

  Bounds are c ∈ [0, 3] nm, k1 ∈ [1e-5, 1] and k2 ∈ [1e-6, 0.1] s⁻¹. Starting values are 0.5 × the final response for each amplitude, with k1 = 5e-3 and k2 = 4e-4 s⁻¹.
- **k_obs for the dimer step.** Consider the components that hold ≥ 15% of the amplitude and have k < 0.05 s⁻¹. The one with the larger amplitude is taken as that sensor's k_obs.
- **Ka2 (dimer).** The slope of a linear regression of k_obs on [A] in molar, which needs ≥ 3 sensors: **k_obs = Ka2·[A] + intercept**. In theory the intercept equals Kdis2. It is shown on the PDF panel but not used, because Kdis2 comes from the dissociation fit. The regression R² is the QC statistic for Ka2.

### 4.6 Tetramer Ka1
- **Which components count.** From the per-sensor association fits, take the slow component (k2) of every sensor where the two components are distinct, the slow one holds ≥ 10% of the amplitude, and k2 < 0.05 s⁻¹.
- **Formula.** Average k2 over the **two highest concentrations** and subtract Kdis1: **Ka1 = ⟨k_slow⟩_top2 − Kdis1**.
- **Not determinable** if the result is ≤ 0, i.e. no slow association faster than the tetramer off-rate. This happens in 2026.89.

### 4.7 Derived values
- **KD2** = Kdis2 / Ka2 × 10⁹ (nM).
- **KD1** = Kdis1 / Ka1 (dimensionless).
- **t½** = ln 2 / kdis.
- **Failed-phase cases.** If Stage 1 put the fast rate on its lower bound (kf ≤ 1.05e-5 s⁻¹), no dimer phase was found, and Kdis2, KD2 and the dimer t½ are "not determinable". This happens in one run: in 2026.40, after a 2 h association, the dissociation is essentially flat for 2.2 h. If the slow rate is on its floor (≤ 2e-8 s⁻¹), no tetramer decay was detected (§5).

### 4.8 Fit-quality numbers, and why there are three
- **R² (dissociation, global fit)** — column K and the GUI. 1 − SS_res/SS_tot over all fitted points pooled, which is the Octet-style global R². Median 0.997 (range 0.952–1.000). Pooling credits the model for the differences in amplitude *between* sensors, so it runs high.
- **R² dissociation (median per sensor)**. Each sensor's own R² about its own mean, which is stricter. Median 0.975 (IQR 0.949–0.989). This is the number that exposes a poor shape fit. The worst cases are 2026.66c and 2026.97; see their PDF pages.
- **R² association (median per sensor)**. Median 0.998. With four free parameters per trace, this mainly confirms that the traces are smooth.
- **R² Ka2 regression**. How well k_obs follows a straight line in [A]. This is the QC statistic for the dimer Ka2.

---

## 5. Tetramer: measured, bounded, or not resolvable

### 5.1 The window problem, and the evidence for it
The tetramer lives for days. A dissociation step of 0.7–4 h sees only about 0.3–4% of its decay, so it cannot pin the rate down. The window was tested directly (`tetramer_window_test.py`): the **13 runs with a measured 15 h tetramer** were truncated to shorter windows and refitted with the same method.

| Window | 1 h | 2 h | 3 h | 4 h | 8 h |
|---|---|---|---|---|---|
| Tetramer Kdis1, short window / 15 h (median) | 17.1× | 7.1× | 4.5× | 3.1× | 1.7× |
| Model-free terminal slope / 15 h (median) | 9.4× | 5.2× | 3.3× | 2.3× | 1.04× |

- **Short windows always read too fast.** The short-window rate was an **overestimate in 13 of 13 runs at ≤ 3 h and 12 of 13 at 4 h**, with a smallest ratio of 0.92.
- **The terminal-slope check rules out a fitting artefact.** The model-free terminal slope, the late-phase −slope/level that a manual read uses, shows the same bias. The data themselves decay faster early than late: the slow phase is not a single exponential.

The dimer off-rate was checked the same way on the six EB3 + EphB2 15 h runs (`truncation_test.py`):

| Window | 0.7 h | 2 h | 4 h | 8 h |
|---|---|---|---|---|
| Dimer Kdis2 overestimate | 3–5× | 1.4–2× | 1.1–1.35× | within ~10% |

Association length was checked on the same runs:
- **Ka2** is robust to it: truncating the 90-min association to 13 min changes Ka2 by only 2–13%.
- **Ka1** is not: at 13 min it came out anywhere from about 1× to 60× off, or not determinable.

### 5.2 The rule every run goes through (`tetramer_resolution` in `plot_all_fits.py`)

| Category | When | What is reported | Runs |
|---|---|---|---|
| **measured** | ≥ 8 h dissociation, slow phase detected (ks > 2e-8 s⁻¹), dimer phase resolved | Kdis1 = ks, t½ = ln2/ks | 13 |
| **lower bound (short window)** | < 8 h window | Kdis1 < ks / 0.9, i.e. t½ > 0.9·ln2/ks. Short windows overestimate the rate (13/13 runs); the 0.9 factor covers the one 0.92 case, so the bound held in every test at every window. | 57 |
| **lower bound (15 h, no decay)** | ≥ 8 h, but ks sits on its 1e-8 floor | 95% upper bound on the rate from the **sensor-to-sensor spread** of the late-phase decay (see below) | 2 |
| **no bound** | no slow phase and too little late signal to bound it | "—" | 3 |

**How the 15 h, no-decay bound is computed.** For each sensor with ≥ 50 points after max(5 dimer half-lives, 40% of the window) and a mean level ≥ 0.05 nm, a line is fitted and its rate is −slope/level. The bound is mean + t₀.₉₇₅,ₙ₋₁·SD/√n across sensors.

**Runs with no dimer phase (2026.40).** The biexponential split is then not a dimer/tetramer split, so the same sensor-spread statistic is applied to the **whole** window. Any faster component only raises the apparent rate, so the bound stays conservative.

**Why not a profile-likelihood bound.** Sensors are independent measurements, but the ~2000 points within one curve are strongly autocorrelated. A within-curve profile likelihood treats them as independent and gives absurdly tight bounds: > 402 d and > 319 d, against sensor-spread bounds of > 63.9 d and > 5.8 d on the same runs. It was rejected for that reason.

### 5.3 Reference values
Each non-measured run whose interaction has **measured 15 h runs in the same buffer** carries that group's measured t½ (mean ± SD, n) in a separate "reference" column. Twelve runs have one. It is the interaction's value, **not a measurement of that run**, and is never counted as a replicate.

---

## 6. QC rules used in the master table and the PDF

These decide whether a cell shows a plain number, a bound (`<`, `>`), a value in parentheses (not reliable from this run, shown for reference, with the status column saying why) or "—" (not determinable).

| Parameter | Plain number requires | Otherwise | Runs passing |
|---|---|---|---|
| Tetramer Kdis1, t½ | category "measured" | `<` / `>` bound, or "—" | 13 of 75 |
| Tetramer Ka1 | ≥ 60 min association **and** a resolved dimer phase **and** top-two-concentration saturation θ² ≥ 0.25 | (value) | 11 of 75 |
| Tetramer KD1 | measured Kdis1 and reliable Ka1 | (value), or a bound | — |
| Dimer Ka2 | Ka2 > 0 and regression R² ≥ 0.9 | (value), or "—" if there is no positive concentration dependence | 25 of 75 (14 of 16 long-association runs, 11 of 59 13-min runs) |
| Dimer Kdis2, t½ | ≥ 8 h dissociation and a resolved dimer phase | (value): overestimated by the window as tabulated in §5.1 | 15 of 75 |
| Dimer KD2 | reliable Ka2 and reliable Kdis2 | (value) | — |

**Why tetramer Ka1 has a saturation rule.** By §2.2, the slow association rate is ka1·θ² + kdis1, not ka1 + kdis1.
- In the **PBST 15 h runs** the top two concentrations reach θ² = 0.47–0.84. Ka1 there is somewhat low (by 1/θ² ≈ 1.2–2.1×) but usable.
- In four **PBSTB 15 h runs** with weak dimer binding (2026.94, 2026.96, 2026.98 and 2026.99), KD2 is 57–590 nM against top concentrations of 27–81 nM, giving θ² = 0.01–0.13. The slow association rate there is not the closure rate, so those Ka1 values are shown in parentheses. The other PBSTB 15 h runs reach θ² = 0.34–0.88 and pass.

**Why dimer Ka2 so often fails at 13 min.** In the 13-min runs, the per-sensor two-exponential fit cannot reliably separate the arm-binding phase from the closure phase. The chosen component flips between them from one concentration to the next; for example, in 2026.79a the 27 nM sensor yields 5.8e-6 s⁻¹, between neighbours of 2.0e-3 (9 nM) and 2.5e-2 (81 nM). Changing which component is picked does not fix this (tested). A global association fit with shared Ka2 is the natural next step. A quick test fitted most runs well (R² > 0.99), but its parameters hit their bounds in about 20 runs and its 13-min Ka2 values ran about 1.5–4× above the 90-min replicates of the same interaction, so it needs proper development before use.

---

## 7. Group-mean rows

- **PBST only.** Buffer changes the tetramer off-rate 2–7×, in inconsistent directions: WT is 7× faster in PBSTB, R79P 1.7× slower, EphB2-His 4.6× slower. Runs in different buffers are therefore never pooled, and the status column points to any measured non-PBST runs.
- **QC-passing runs only**, each cell with its own n. Ka2 can come from runs whose Kdis2 does not pass, because Ka2 does not depend on the dissociation window.
- **Tetramer groups with no measured PBST run** report a **bound**. The bound is the ⌈n/2⌉-th smallest per-run bound (the median; the lower of the two when n = 2). It remains a valid lower bound as long as most per-run bounds are valid, so one noisy short run cannot set the group value. The status gives the range of the per-run bounds.
- **Dimer Kdis2 when no PBST run has ≥ 8 h dissociation** is shown as an approximate mean in parentheses, "(m ± sd; n = k)", and is overestimated as described in §5.1.

**Headline values** (EphrinB3-Fc, PBST, mean ± SD over the three 15 h runs; Ka2 over four runs):

| | EphB2-WT | EphB2-R79P |
|---|---|---|
| Tetramer t½ (d) | **8.46 ± 3.29** | **2.85 ± 0.82** |
| Tetramer Kdis1 (s⁻¹) | 1.04e-6 ± 0.35e-6 | 3.01e-6 ± 1.03e-6 |
| Tetramer Ka1 (s⁻¹) | 2.31e-4 ± 0.61e-4 | 4.02e-4 ± 1.50e-4 |
| Tetramer KD1 (–) | 4.5e-3 ± 1.4e-3 | 8.1e-3 ± 3.1e-3 |
| Dimer Ka2 (M⁻¹s⁻¹) | 1.20e5 ± 0.29e5 | 1.16e5 ± 0.65e5 |
| Dimer Kdis2 (s⁻¹) | 5.18e-4 ± 0.67e-4 | 1.36e-3 ± 0.30e-3 |
| Dimer t½ (min) | 22.6 ± 3.1 | 8.7 ± 1.7 |
| Dimer KD2 (nM) | 4.06 ± 1.11 | 9.66 ± 1.75 |

---

## 8. Files that could not be fit (not in the master table)

| File | Reason |
|---|---|
| EB2+EphA4-His / 2026.59b (casein) - Fit 34 | No sensor has a readable analyte concentration: a different export layout. Re-export in the standard layout. |
| EB2+EphB1-His / 2026.42 - Fit 19 | The mean signal peaks at 99% of the 253-min record, so no dissociation step was recorded. |
| EB2+EphB2-His / 2026.90 (PBSTB) - Fit 59 | The mean signal peaks at 2% of the 980-min record, so there is no usable association→dissociation boundary. |

---

## 9. Validation summary
- **Manual reads.** The two-stage estimates track the lab's manual half-life reads (r = 0.81). WT and R79P are correctly ordered, as the manual reads were.
- **Window effects.** Truncation tests quantify every window effect (§5.1), and the lower-bound rule held in 13 of 13 runs at every window.
- **Determinism.** Re-running all 78 files reproduces every constant bit-for-bit. The 65 runs whose sensor set did not change in this revision came out identical.
- **Consistency across outputs.** The GUI, CSV files, PDF and Numbers tab all read the same cache and the same code, and the Numbers export checks that the other 8 tabs are unchanged.

## 10. Known limitations, and which way they bias the numbers

1. **Reported rates are observable (macroscopic) rates.** By §2.1, the slow dissociation rate is kdis1·2kdis2/(2kdis2 + ka1). In the 13 measured runs, ka1/(2·kdis2) = 0.07–0.59, so the microscopic reopening rate is 1.07–1.59× faster than the reported Kdis1.
   - The reported tetramer t½ is the lifetime of the **signal**: how long the tetramer-held Eph stays bound. That is the physically meaningful quantity, but it is not the pure reopening constant.
   - Likewise, the fast phase mixes kdis2 with a 2·kdis2 + ka1 component. A less stable tetramer can therefore shift the apparent dimer Kdis2, which should be kept in mind when reading the R79P dimer difference.
2. **Tetramer Ka1 is underestimated by 1/θ²**, about 1.2–2.1× in the PBST 15 h runs (§6). It can be corrected as (k_slow − Kdis1)/θ² using the run's own KD2. The correction is not applied yet.
   - The association slow phase is also larger at the top concentrations than the model alone predicts (§2.2), so part of it probably reflects processes outside the model: ligand heterogeneity, rebinding, or slow conformational change.
   - **Ka1 is the least model-anchored parameter.**
3. **The slow phase is not strictly single-exponential.** The biexponential estimate still falls between 8 h and 15 h: the 8 h values are about 1.7× the 15 h values. The model-free terminal slope, however, has converged by 8 h (1.04×). The 15 h "measured" rates are therefore close to, but possibly still slightly above, the asymptotic rate, and the tetramer may be somewhat more stable than reported.
4. **The fits are unweighted.** High-signal sensors dominate. The dominant-sensor rule removes the extreme cases.
5. **There are no terms for mass transport, rebinding, baseline drift or nonspecific binding.** Upward-drifting sensors are excluded rather than modelled.
6. **Buffer dependence** is real and large (§7). Values are comparable only within a buffer.
7. **Alternative explanation.** Biphasic kinetics can also come from two ligand populations (the "heterogeneous ligand" model). The tetramer interpretation is supported by the selective shortening of the slow phase by the tetramer-interface mutant R79P.

## 11. Reproducing everything

```bash
# from the repository root, with OCTET_DATA_DIR and OCTET_WORK_DIR set (see README)
python3 dimer_tetramer/export_full_dataset.py   # fit all 78 files → cache, CSVs, All_78_Fits.pdf, tetramer categories (~5 min)
python3 dimer_tetramer/fill_master_tab.py       # → "Global Fit - Master Table on 9-29-26.numbers" (new file; source untouched)
python3 dimer_tetramer/export_master_pdf.py     # → Dimer-Tetramer_All_Curve_Fits_9-29-26.pdf
python3 dimer_tetramer/gui_app.py               # interactive checklist + refit
```

---

## 12. Manuscript paragraph (Methods)

> **Kinetic model.** BLI sensorgrams were analysed with a two-step model in which each arm of the immobilized, bivalent ephrinB-Fc binds one monomeric Eph receptor to form a 1:1 heterodimer, and the two heterodimers on the same Fc then close into a 2:2 heterotetramer by a first-order, intramolecular step. Because the full model is not identifiable from single experiments, constants were extracted in two stages. Dissociation phases from all sensors of an experiment were fitted globally to a biexponential decay with shared rates; the fast rate was assigned to heterodimer dissociation (k_d,dimer) and the slow rate to tetramer dissociation (k_d,tet). k_a,dimer was the slope of the fast observed association rate against analyte concentration, and k_a,tet the slow association rate at the two highest concentrations minus k_d,tet. K_D,dimer = k_d,dimer/k_a,dimer; K_D,tet = k_d,tet/k_a,tet is dimensionless; half-lives are ln2/k_d. Truncating 15-h dissociations showed that shorter windows overestimate k_d,tet in every case, so runs with dissociations under 8 h report only a lower bound on the tetramer half-life. Sensors without signal, with upward drift, or inconsistent with a same-concentration replicate were excluded, and values were compared only within the same buffer.
