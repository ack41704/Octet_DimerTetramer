# Methods: the v3 dimer–tetramer model

Methods text for the kinetic analysis implemented in this repository (v3). It is written to match the code:
`plot_all_fits.py` (sensor selection, dissociation fit), `assoc_two_phase.py` (global association fit),
`extended_constants_all16_v3.py` / `all_runs_v3.py` (constants, window-aware flags, t-tests).

Numbering: in the tables and figures, **species 1 = tetramer and species 2 = dimer** (Ka1, Kdis1, KD1 = tetramer;
Ka2, Kdis2, KD2 = dimer). Below, the constants are written by species name (ka,dimer, kd,tet, …).

**Placeholders.** Items in [brackets] (instrument model, biosensor type, temperature, shake speed, loading level,
buffer compositions, software version) are not in the data files and must be filled in from the experimental records.

The earlier v1 description, kept for its sensor-selection and window-test detail, is in
[MODEL_DESCRIPTION_v1.md](MODEL_DESCRIPTION_v1.md).

---

## One-paragraph version

Biolayer interferometry sensorgrams were analyzed with a two-step model in which each Fc-dimerized ephrinB-Fc molecule, sparsely immobilized on the biosensor, first binds one EphB2 analyte to form a 1:1 heterodimer, after which a second EphB2 binds the other ephrinB protomer and the two Eph–ephrin pairs close into a 2:2 circular heterotetramer; closure conserves bound mass and produces no change in response, and because heterotetramers dissociate far more slowly than heterodimers, the response contributed by tetramer-bound EphB2 persists through dissociation. Accordingly, both phases were treated as biphasic, with the faster process always assigned to the heterodimer and the slower to the tetramer (assignment by rate, not amplitude). Reference-subtracted sensorgrams were exported from Octet Data Analysis; sensors were excluded if they were max-control (A20-I) wells, peaked at ≤ 0.02 nm, drifted upward during dissociation, exceeded three times the peak of every other sensor, or fell below one-third of a same-concentration replicate. Dissociation phases from all retained sensors were fitted globally to a biexponential decay with shared rates and per-sensor amplitudes, giving kd,dimer (fast) and kd,tet (slow); where no slow decay was detected, kd,tet was set to the upper 95% confidence limit of the sensor-to-sensor late-phase decay rate. Association phases were fitted globally to a two-phase model, R_i(t) = A_f,i(1 − e^(−k_f,i·t)) + A_s,i(1 − e^(−k_s,i·t)), with observed rates linear in analyte concentration and amplitudes free per sensor; ka,dimer was taken as the slope of the fast observed rate against [EphB2]. This slope was reported only if it was bounded on both sides by a 5% sum-of-squares profile and at least three sensors carried a fast phase making up ≥ 10% of their signal. Otherwise ka,dimer was estimated from the top concentration as (k_fast − kd,dimer)/[EphB2]top and reported as approximate. Because the slow observed rate depended only weakly on concentration, ka,tet was obtained from a two-exponential fit to the highest-concentration trace as (k_slow − kd,tet)/[EphB2]top, and was reported only when that fit had R² ≥ 0.95. Equilibrium constants were calculated as KD,dimer = kd,dimer/ka,dimer and KD,tet = kd,tet/ka,tet (both in nM), and half-lives as ln 2/kd. Truncation of 15-h runs showed that dissociation windows shorter than 8 h overestimate both off-rates, and that associations shorter than 60 min under-sample the slow phase, so constants from such runs were flagged and not compared with those from 90-min/15-h runs. Each run was treated as an independent measurement, and EphB2-WT and R79P were compared by two-tailed Student's t-test only within sets of the same ligand, buffer and protocol. All fitting used bounded trust-region least squares in custom Python code (Python 3.13, NumPy 2.3, SciPy 1.16).

---

## Full methods

**Kinetic model.** The Fc-dimerized ephrinB-Fc ligand presents two Eph-binding sites, and it was sparsely immobilized on the biosensor surface. Binding of soluble EphB2 analyte was modeled as two steps:

1. **Dimer formation.** One EphB2 molecule binds one ephrinB protomer to form a 1:1 heterodimer, which increases the BLI response.
2. **Tetramer formation.** A second EphB2 molecule binds the other ephrinB protomer of the same Fc dimer, producing an equal response increment. The two Eph–ephrin pairs then close into a circular 2:2 heterotetramer. Closure conserves bound mass and therefore produces no further change in response.

Heterodimers dissociate quickly. Heterotetramers dissociate extremely slowly, so the response contributed by EphB2 in tetramers persists through the dissociation phase.

Both phases of the sensorgram are therefore biphasic:

- **Association:** a fast phase, assigned to heterodimer formation (association rate constant ka,dimer), and a slow phase, assigned to the EphB2 binding that accompanies tetramer assembly (ka,tet).
- **Dissociation:** a fast phase, assigned to heterodimer dissociation (kd,dimer), and a slow phase, assigned to tetramer dissociation (kd,tet).

Phases were assigned by rate, not by amplitude. Even where the slow phase carried most of the signal, the faster phase was always assigned to the dimer. (In the data tables, species 1 = tetramer and species 2 = dimer.)

Standard 1:1 global fitting in the instrument software assigns a single off-rate per experiment. That value blends the fast dimer and slow tetramer processes, and in several runs it reached the software's lower limit of 1 × 10⁻⁷ s⁻¹. The model described here resolves the two dissociation processes separately.

**Data acquisition and preprocessing.** Experiments were run on an Octet [model] (Sartorius) at [temperature] °C with [shake speed] rpm orbital shaking. EphrinB-Fc was loaded onto [biosensor type] biosensors to [~0.3 nm], and EphB2-[His/WT/R79P] was used as the analyte at [concentrations]. Running buffers were PBST, PBST + casein, or PBSTB [compositions].

Sensorgrams were reference-subtracted and aligned in Octet Data Analysis [version] and exported as preprocessed data. Runs that were not reference-subtracted were excluded.

The association-to-dissociation boundary was taken as the time of maximum mean response across sensors, requiring at least 5 min of association. A sensor was excluded if any of the following held:

- it was a control well, i.e. it contained the A20-I max-control inhibitor;
- its peak response was ≤ 0.02 nm;
- its dissociation phase had fewer than 100 points;
- it drifted upward during dissociation (mean of the last 100 points above the mean of the first 100);
- its peak exceeded 3× the peak of every other sensor;
- its peak was below one-third of that of a replicate at the same concentration.

**Dissociation phase.** The dissociation phases of all retained sensors were fitted globally to a biexponential decay to a zero baseline, with rates shared across sensors and amplitudes fitted per sensor:

R_i(t) = A_f,i·e^(−kd,dimer·t) + A_s,i·e^(−kd,tet·t)

Fitting details:

- **Sampling:** each sensor contributed up to 400 points, a combination of logarithmically and linearly spaced points.
- **Bounds:** 10⁻⁵–0.32 s⁻¹ for the fast rate, 10⁻⁸–3.2 × 10⁻⁴ s⁻¹ for the slow rate, and 0–3 nm for amplitudes.
- **Algorithm:** bounded trust-region least squares from four starting points; the lowest-cost solution was kept.

Interpretation of the slow rate depended on the length of the dissociation window:

- **≥ 8 h with a detectable slow decay:** kd,tet is reported as a measurement.
- **< 8 h:** the fitted slow rate is an apparent rate and overestimates kd,tet. When the 15-h runs were truncated to ≤ 3 h, the apparent rate exceeded the full-window rate in 13 of 13 runs.
- **No slow decay detected:** kd,tet was set to a detection limit, the upper 95% confidence limit of the late-phase decay rate. Each sensor's late-phase rate was its linear-fit slope divided by its mean level. Across sensors, the limit is mean + t(0.975, n−1)·SD/√n.

kd,dimer from windows shorter than 8 h is also overestimated. Truncation of the 15-h runs gave overestimates of about 3–5× at 0.7 h, 1.4–2× at 2 h and 1.1–1.35× at 4 h.

**Association phase: dimer.** The association phases of all retained sensors were fitted globally to a two-phase model:

R_i(t) = A_f,i·(1 − e^(−k_f,i·t)) + A_s,i·(1 − e^(−k_s,i·t)),  with  k_f,i = ka,dimer·[EphB2]_i + k_f,0  and  k_s,i = k_s,1·[EphB2]_i + k_s,0

- **Rates:** the fast and slow observed rates are linear in analyte concentration and shared across sensors.
- **Amplitudes:** free per sensor, to allow for differences in sensor loading.
- **Constraint:** the fast rate was required to be at least 3× the slow rate at the top concentration.
- **Fitting:** up to 700 log- and linearly spaced points per sensor; best of a 48-point grid of starting values.

ka,dimer is the slope of the fast observed rate against [EphB2].

The slope was reported only when the concentration series determined it. Determinacy was tested with a 1-D profile: ka,dimer was fixed on a grid, all other parameters were refitted at each grid point, and the slope had to keep the sum of squared residuals within 5% of its minimum on both sides. A practical criterion was used instead of a formal F-test because successive BLI points are strongly autocorrelated. In addition, at least three sensors had to carry a fast phase making up ≥ 10% of their association signal.

Where either test failed, ka,dimer was estimated from the top concentration as (k_fast,top − kd,dimer)/[EphB2]_top and reported as approximate.

**Association phase: tetramer.** The association trace at the highest EphB2 concentration was fitted on its own to a sum of two exponentials, using 1,500 evenly spaced points. Then:

ka,tet = (k_slow,top − kd,tet)/[EphB2]_top

The top concentration was used for three reasons:

- it gives the largest signal;
- it best resolves the slow phase;
- across the series the slow rate depends only weakly on concentration, so a concentration slope is poorly constrained.

ka,tet was not reported if the top-trace fit had R² < 0.95. Where kd,tet could not be determined, it was taken as 0 when computing ka,tet. For display, each fitted association curve was divided at the time t_x = ln(A_f·k_f / (A_s·k_s))/(k_f − k_s), after which the slow phase dominates the slope of the fit.

**Derived constants.** KD,dimer = kd,dimer/ka,dimer and KD,tet = kd,tet/ka,tet, both reported in nM. Half-lives were calculated as t½ = ln 2/kd. Fit quality is reported as R² for the top-concentration association fit, the global association fit and the global dissociation fit.

**Statistics.** Each run was treated as one independent measurement (n = runs). EphB2-WT and EphB2-R79P were compared by two-tailed Student's t-test with equal variance, only within sets with the same ligand, buffer and association/dissociation protocol and at least two runs per variant. Buffers and protocols were never pooled.

**Software.** Analyses used custom Python code (Python 3.13; NumPy 2.3; SciPy 1.16; Matplotlib 3.10) [available at: repository URL].

**Assumptions and limitations.**

- The model assumes tetramer closure is mass-neutral, and that the fast and slow phases correspond to dimer and tetramer processes respectively.
- ka,tet is an operational constant defined at the top concentration. In long PBST runs the slow association rate was largely independent of concentration, which may indicate a first-order closure step rather than bimolecular binding.
- Constants from runs with associations shorter than 60 min, or dissociations shorter than 8 h, are reported but are not comparable with those from 90-min/15-h runs.
