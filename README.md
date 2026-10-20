# Octet_DimerTetramer

Kinetic analysis of biolayer-interferometry (Octet BLI) sensorgrams for **Eph receptor binding to Fc-dimerized
ephrin ligands**, using a two-step **heterodimer → heterotetramer** model. Standard 1:1 global fitting assigns a
single off-rate per experiment, which blends the fast heterodimer process with the very slow tetramer process. This
code resolves the two processes, and it reports the association, dissociation and affinity constants for each,
together with flags that say how far each value can be trusted.

> The method is described in our paper (see [Citation](#citation)). The repository contains **no data**: point
> `OCTET_DATA_DIR` at your own Octet exports (see [Quick start](#quick-start)).

## The model

Each immobilized ephrin-Fc molecule has two ephrin arms. Soluble Eph analyte (EphB2-His, -WT or -R79P, among others)
binds in two steps:

1. **Heterodimer** (species **2**). One Eph binds one ephrin arm (a 1:1 complex). Association and dissociation are
   fast (minutes).
2. **Heterotetramer** (species **1**). A second Eph binds the other arm, and the two Eph–ephrin pairs close into a
   circular 2:2 complex. Closure conserves mass, so it adds no response. Tetramers dissociate extremely slowly (days).

| Quantity | Tables / figures | Meaning |
|---|---|---|
| Ka2, Kdis2, KD2 | dimer | fast association slope, fast dissociation rate, KD2 = Kdis2/Ka2 (nM) |
| Ka1, Kdis1, KD1 | tetramer | slow association constant, slow dissociation rate, KD1 = Kdis1/Ka1 (nM) |
| t½ | both | ln 2 / Kdis |

**Numbering:** 1 = tetramer and 2 = dimer in every table, PDF and figure. Some older code and CSV columns number the
steps in assembly order, the other way round, so their columns are named by species (`dimer_kdis_s1`,
`tetramer_ka_s1`, …).

### How each constant is computed (v3)

- **Sensor selection.** A sensor is excluded if it is:
  - a max-control (A20-I) well;
  - flat (peak ≤ 0.02 nm);
  - drifting upward during dissociation;
  - more than 3× every other sensor;
  - below ⅓ of a same-concentration replicate.

  The association→dissociation boundary is the peak of the mean trace.
- **Dissociation → Kdis2, Kdis1.** One global biexponential fit per run (`plot_all_fits.stage1_full`), with rates
  shared across sensors and amplitudes fitted per sensor:
  - fast rate = Kdis2 (dimer);
  - slow rate = Kdis1 (tetramer).
- **Association → Ka2.** One global two-phase fit per run (`assoc_two_phase.fit_global`):
  - model: `R_i(t) = Af_i(1−e^(−kf_i t)) + As_i(1−e^(−ks_i t))`, with `kf_i = Ka2·C_i + kf0` and
    `ks_i = Ka1·C_i + ks0`;
  - phases are assigned **by rate, not amplitude**: fast = dimer, slow = tetramer;
  - Ka2 is the fast-rate slope. It is used only if a 5 % sum-of-squares profile bounds it on both sides
    (`profile_slope`) and ≥ 3 sensors carry a fast phase of ≥ 10 % of their signal;
  - otherwise Ka2 falls back to `(k_fast,top − Kdis2)/C_top` and is flagged **T**.
- **Association → Ka1.** `(k_slow,top − Kdis1)/C_top`, from a two-exponential fit to the top-concentration trace.
  Not reported if that fit has R² < 0.95.
- **Window-aware flags.** Every value is a point value: no "<", ">" or n.d. in the value cells. The flags say how to
  read it:

| Flag | Meaning |
|---|---|
| **A** | association < 60 min: the slow phase is only partly sampled, so Ka1/KD1 are not comparable with 90-min runs |
| **D** | dissociation < 8 h: Kdis2 (and KD2) overestimated |
| **S** | Kdis1 is the apparent rate of a < 8 h window: it overestimates the true rate, and KD1 with it |
| **L** | Kdis1 is a detection / decay limit (no slow decay detected) |
| **T** | Ka2 approximate (top-concentration formula) |
| **N** | a value is not determinable (—); the reason is in the run's notes |

- **Statistics.** WT vs R79P comparisons use a two-tailed, equal-variance Student's t-test, with n = runs. Tests are
  run only within one ligand × buffer × protocol set (association / dissociation length). Buffers and protocols are
  never pooled.

The full methods text, including a one-paragraph manuscript version, is in [docs/METHODS.md](docs/METHODS.md). The
superseded v1 description is in [docs/MODEL_DESCRIPTION_v1.md](docs/MODEL_DESCRIPTION_v1.md); it keeps the sensor-rule
and window-test detail, which still applies.

## Quick start

```bash
python -m pip install -r requirements.txt          # add requirements-gui.txt for the GUI
export OCTET_DATA_DIR=/path/to/octet_exports        # input CSVs + master-table .numbers files
export OCTET_WORK_DIR=/path/to/octet_work           # fit caches + every generated PDF / CSV / figure
python -m pytest -q                                 # data-free tests (simulated sensorgrams)
```

Run the scripts from the repository root, e.g. `python dimer_tetramer/all_runs_v3.py`. Each script names its dated
input and output files in constants at the top (`SRC`, `OUT`, `NUM_IN`, …). For a new batch, change those names; the
scripts never overwrite their input workbook.

### Data layout expected in `OCTET_DATA_DIR`

```
EB1 Interactions - Preprocessed Data FIles/<interaction>/<run> - Fit <n>.csv     # e.g. EB1+EphB2-His/
EB2 Interactions - Preprocessed Data FIles/<interaction>/...
EB3 Interactions - Preprocessed Data FIles/<interaction>/...
<run> - Preprocessed Data[ - Fit <n>].csv          # single runs outside the folders (listed in build_extra_cache.EXTRA)
Global Fit - Master Table on <date>.numbers        # run list / layout read by the table writers
```

The inputs are Octet Data Analysis **"Preprocessed Data"** CSV exports: reference-subtracted, with five header lines
and then X/Y column pairs per well (see `parse_octet.py`).

## Pipeline

| Step | Script | Output (in `OCTET_WORK_DIR` unless noted) |
|---|---|---|
| 1 | `export_full_dataset.py` | fits every file in the EB* folders → `fit_cache.pkl`, per-point CSVs, all-fits PDF |
| 2 | `build_extra_cache.py` | runs outside the folders → `fit_cache_extra.pkl` (fits only files not yet cached) |
| 3 | `extended_constants_all16_v3.py` | v3 constants for the ≥ 14 h-dissociation runs (xlsx + PDF) |
| 4 | `all_runs_v3.py` | v3 applied to every master-table run: all-runs PDF + a new tab in a **copy** of the master table (`OCTET_DATA_DIR`) |
| 5 | `split_by_buffer_v3.py` | the same PDF split by buffer (no refitting) + results CSV |
| 6 | `update_master_by_buffer.py` | per-buffer tables in a new master-table copy |
| 7 | `add_new_runs_v3.py` | adds new runs **without refitting the existing ones**; new dated CSV, PDF, workbook |
| — | `figure_EB3_WT_vs_R79P.py`, `figure_EB3_WT_vs_R79P_halflife.py`, `table_EB3_WT_vs_R79P_halflife.py` | publication figures (PDF / SVG with editable text, 600-dpi PNG) and the half-life t-test table |
| — | `gui_app.py` | PyQt5 viewer: per-sensor checklist, refit, CSV export |

### Script map

| Module | Role |
|---|---|
| `parse_octet.py` | Octet CSV parser |
| `batch_fit.py`, `plot_all_fits.py` | boundary detection, sensor rules, global dissociation fit, per-run plots (single copy of the fitting logic) |
| `assoc_two_phase.py` | global two-phase association fit, slope profiles, slope-dominance crossover |
| `extended_constants_all16{,_v2,_v3}.py` | the extended-run analysis, v1 → v3 (v3 imports the v2 per-run step) |
| `resolve_tetramer.py`, `fill_master_tab.py`, `build_tab.py`, `export_master_pdf.py`, `regen_pdf_from_cache.py`, `eb3_extended_constants.py` | v1 pipeline and earlier analyses (kept for reproducibility) |
| `truncation_test.py`, `tetramer_window_test.py`, `tetramer_bounds_check.py`, `assoc_truncation.py` | validation of the window rules: truncate the 15-h runs and compare with the full-window fit |
| `paths.py` | `DATA_DIR` / `WORK_DIR` from the environment (the only module written for the repository) |

## Validation

- **Window rules.** These come from controlled truncation of the 15-h dissociation runs (the scripts in the last row
  above):
  - a short dissociation window always overestimates the slow (tetramer) rate;
  - the fast-rate bias shrinks with the window length (values in [docs/METHODS.md](docs/METHODS.md));
  - associations shorter than 60 min under-sample the slow phase.
- **Tests** (`tests/`, no lab data needed):
  - parameter recovery on simulated sensorgrams: the global biexponential recovers Kdis2/Kdis1, and the global
    two-phase fit recovers Ka2, kf0, Ka1 and ks0, with Ka2 bracketed by its profile;
  - the Octet parser on a synthetic export;
  - repository hygiene: no machine paths, every script compiles, every import is local or declared, the sync rewrite
    behaves as documented.

## Limitations

- Assigning the fast phase to the dimer and the slow phase to the tetramer is a model assumption, as is mass-neutral
  closure.
- Ka1 is an operational constant defined at the top concentration. In long PBST runs the slow association rate barely
  depends on concentration, which is consistent with a first-order closure step.
- Constants from runs with < 60-min associations or < 8-h dissociations are reported (flagged) but are not comparable
  with those from 90-min / 15-h runs.

## Maintaining this copy

`dimer_tetramer/` is a **snapshot** of the working analysis folder. Edit the scripts there, then refresh:

```bash
python tools/sync_from_working_copy.py /path/to/Octet_DimerTetramer_Analysis --dry-run
python tools/sync_from_working_copy.py /path/to/Octet_DimerTetramer_Analysis
python -m pytest -q
```

The sync rewrites absolute paths to `DATA_DIR` / `WORK_DIR` and refuses to write if any machine path remains or if a
rewritten file fails to compile.

## Citation

If you use the dimer–tetramer model, please cite our paper:

> Wang, H., Khambete, A., Sprouse, F., Karsi, M., Cortez, K., Bellinghausen, E., Khuu, T., Block, N., Talebian, A., Ahmed, M. S., Gedara, C. M. U., Ortiz, M., & Henkemeyer, M. (2025). Eph-Ephrin Tetramerization Inhibitors Target Bidirectional Signaling to Combat Pain and Addiction. *bioRxiv*. https://doi.org/10.64898/2025.12.08.692997

```bibtex
@article{wang2025eph,
  title   = {Eph-Ephrin Tetramerization Inhibitors Target Bidirectional Signaling to Combat Pain and Addiction},
  author  = {Wang, Hanghang and Khambete, Amav and Sprouse, Frances and Karsi, Melody and Cortez, Karina and Bellinghausen, Eric and Khuu, Tamdan and Block, Natalie and Talebian, Asghar and Ahmed, Mahmoud Salama and Gedara, Chinthaka Mahesh Udamulle and Ortiz, Michael and Henkemeyer, Mark},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.64898/2025.12.08.692997},
  url     = {https://www.biorxiv.org/content/10.64898/2025.12.08.692997v2}
}
```

GitHub's **Cite this repository** button gives the same reference (from [`CITATION.cff`](CITATION.cff)), which also records the software version.

## License

MIT; see [LICENSE](LICENSE).
