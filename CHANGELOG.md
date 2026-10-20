# Changelog

## 0.1.0 (2026-10-20)

First repository snapshot of the working analysis.

- **v3 dimer–tetramer model.**
  - Association phases are assigned by rate (fast = dimer, slow = tetramer) from one global two-phase fit per run.
  - Ka2 comes from the profiled fast-rate slope, with a flagged top-concentration fallback.
  - Ka1 comes from the top-concentration trace and requires a fit with R² ≥ 0.95.
  - Point values for every constant, with window-aware flags A / D / S / L / T / N.
- **All-runs analysis.** All-runs analysis with a per-buffer PDF and master-table copies. `add_new_runs_v3.py` adds new
  runs without refitting the existing ones.
- **Figures.** Publication figures and the half-life t-test table for EphrinB3-Fc + EphB2-WT vs R79P.
- **Validation.** Window-rule validation scripts (dissociation and association truncation tests).
- **Repository.** Portable paths (`OCTET_DATA_DIR`, `OCTET_WORK_DIR`), a sync tool from the working copy, data-free
  tests and CI.
