"""
Octet Dimer-Tetramer Fit Viewer -- PyQt5 GUI

Browse, (re)fit, and inspect Octet BLI datasets with the bivalent ephrinB3-Fc
dimer->tetramer model. Wraps the exact same fitting code that produced
batch_results.csv, the "Dimer-Tetramer Model" tab in the master table, and the
diagnostic PDF -- draw_fit_panels(), summary_from_result(), parse_for_gui(),
default_mask() and fit_wells() are all imported, not re-derived, so numbers and
plots here always match those deliverables.

Per-sensor checklist: every well in the file is listed (including controls and
near-zero ones, which the batch pipeline silently drops) with a checkbox and,
for anything auto-excluded, a reason. Toggling a box re-fits ONLY that
experiment (parsing is cached, so this is fast) and updates the plot and the
two parameter cards live.

Usage:  python3 gui_app.py
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import sys
import os
import glob
import csv
import pickle
import traceback

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets, QtGui

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_all_fits import (parse_for_gui, default_mask, fit_wells,  # noqa: E402
                           draw_fit_panels, summary_from_result, group_references)
from batch_fit import split_name  # noqa: E402
from build_tab import build as build_master_tab, SRC as DEFAULT_MASTER_SRC  # noqa: E402

DEFAULT_GLOB = DATA_DIR + "/EB*Interactions - Preprocessed Data FIles/*/*.csv"
HERE = WORK_DIR
CACHE_PATH = os.path.join(HERE, "fit_cache.pkl")

COLS = ["Interaction", "Experiment", "Buffer", "Assoc\n(min)", "Dissoc\n(h)",
        "KD1 dimer\n(nM)", "kdis1\n(s-1)", "kdis2\n(s-1)", "R2 kobs", "Dimer\nOK?",
        "Tetramer\nt\u00bd", "Status"]


class ParamCard(QtWidgets.QGroupBox):
    """Compact labeled-value panel for one half of the model (dimer or tetramer).
    `number` is the user's chosen digit for this card (1 or 2) -- baked into every
    row label (Ka1/Kdis1/KD1 or Ka2/Kdis2/KD2), not just the card title."""

    ROWS = [("ka", "Ka{n} ({unit})"), ("kdis", "Kdis{n} (s⁻¹)"),
            ("kd", "KD{n} ({kdunit})"), ("thalf", "t½"), ("r2", "R{n}² ({r2label})")]

    def __init__(self, title, number, accent, unit, kdunit, r2label, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"QGroupBox {{ font-weight: 600; border: 1px solid palette(mid); "
                            f"border-left: 4px solid {accent}; border-radius: 4px; "
                            f"margin-top: 8px; padding-top: 4px; }}"
                            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}")
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.labels = {}
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        for key, tmpl in self.ROWS:
            caption = tmpl.format(n=number, unit=unit, kdunit=kdunit, r2label=r2label)
            val = QtWidgets.QLabel("—")
            val.setFont(mono)
            form.addRow(caption + ":", val)
            self.labels[key] = val

    def set_values(self, ka, kdis, kd, thalf_text, r2, resolvable=True):
        def fmt(v, sig="{:.3e}"):
            return sig.format(v) if (v is not None and np.isfinite(v)) else "—"
        self.labels["ka"].setText(fmt(ka))
        if resolvable:
            self.labels["kdis"].setText(fmt(kdis))
            self.labels["kd"].setText(fmt(kd, "{:.3g}"))
            self.labels["thalf"].setText(thalf_text)
        else:
            self.labels["kdis"].setText("not resolved")
            self.labels["kd"].setText("not resolved")
            self.labels["thalf"].setText("not resolved")
        self.labels["r2"].setText(fmt(r2, "{:.4f}"))

    def set_texts(self, **texts):
        """Set rows from pre-formatted strings (used for bounds like '< 1.2e-06')."""
        for key, text in texts.items():
            self.labels[key].setText(text)

    def clear(self):
        for lbl in self.labels.values():
            lbl.setText("—")


TET_CATEGORY_LABEL = {
    "measured": "measured",
    "lower_bound_15h": "no decay detected in 15 h -- lower bound",
    "lower_bound_short": "short window -- lower bound only",
    "no_bound": "cannot be bounded",
}


def tetramer_texts(s):
    """Card strings for the tetramer ('1') card, by resolution category. Bounds are shown
    as bounds ('<', '>') so a short-window number can never be read as a measurement."""
    def sci(v):
        return f"{v:.3e}" if v is not None and np.isfinite(v) else "—"
    cat = s.get("tet_category", "measured" if s.get("tetramerOK") else "no_bound")
    base = dict(ka=sci(s["ka2"]), r2=f"{s['R2_dissoc']:.4f}")
    if cat == "measured":
        return dict(base, kdis=sci(s["kdis2"]), kd=f"{s['KD2']:.3g}" if np.isfinite(s["KD2"]) else "—",
                    thalf=f"{s['t12_tetramer_d']:.2f} d")
    if cat in ("lower_bound_15h", "lower_bound_short"):
        kd_up = s.get("tet_KD2_upper")
        return dict(base, kdis=f"< {s['tet_kdis2_upper']:.2e}",
                    kd=f"< {kd_up:.3g}" if kd_up is not None else "—",
                    thalf=f"> {s['tet_thalf_lower_d']:.2f} d")
    return dict(base, kdis="no bound", kd="no bound", thalf="no bound")


def tetramer_table_cells(s):
    """(kdis2 cell, tetramer cell) for the results table."""
    cat = s.get("tet_category", "measured" if s.get("tetramerOK") else "no_bound")
    if cat == "measured":
        return f"{s['kdis2']:.2e}", f"{s['t12_tetramer_d']:.1f} d"
    if cat in ("lower_bound_15h", "lower_bound_short"):
        return f"<{s['tet_kdis2_upper']:.1e}", f">{s['tet_thalf_lower_d']:.2g} d"
    return "—", "—"


class FitWorker(QtCore.QThread):
    """Runs parse_for_gui()+fit_wells() off the GUI thread. Each job may carry an
    already-parsed dict (so re-fits after a checkbox toggle skip re-reading the
    CSV entirely) and an explicit mask (so a toggle only refits that one file)."""
    progress = QtCore.pyqtSignal(int, int, str)
    one_done = QtCore.pyqtSignal(str, object, object, object, str)  # path, parsed, result, summary, error
    finished_all = QtCore.pyqtSignal()

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs  # list of (path, folder, parsed_or_None, mask_or_None)

    def run(self):
        total = len(self.jobs)
        for i, (path, folder, parsed, mask) in enumerate(self.jobs):
            self.progress.emit(i, total, os.path.basename(path))
            try:
                if parsed is None:
                    parsed = parse_for_gui(path, folder)
                if parsed is None:
                    self.one_done.emit(path, None, None, None, "no usable fit (boundary/sensors)")
                    continue
                if mask is None:
                    mask, _ = default_mask(parsed)
                keep_idx = [j for j, m in enumerate(mask) if m]
                if len(keep_idx) < 3:
                    self.one_done.emit(path, parsed, None, None,
                                       f"only {len(keep_idx)} sensor(s) checked (need >=3)")
                    continue
                experiment, fit = split_name(path)
                r = fit_wells(parsed, keep_idx, experiment, fit)
                s = summary_from_result(r)
                self.one_done.emit(path, parsed, r, s, "")
            except Exception as e:
                self.one_done.emit(path, None, None, None,
                                   f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")
        self.progress.emit(total, total, "")
        self.finished_all.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Octet Dimer-Tetramer Fit Viewer")
        self.resize(1500, 900)

        self.folders = {}       # path -> folder (for re-parsing)
        self.parsed = {}        # path -> parse_for_gui() result (cached, never refetched)
        self.masks = {}         # path -> list[bool], current checklist state
        self.reasons = {}       # path -> list[str-or-None], auto-exclusion reasons
        self.results = {}       # path -> fit_wells() result
        self.summaries = {}     # path -> summary_from_result() result
        self.errors = {}        # path -> error string
        self.worker = None
        self._selected_path = None
        self._checklist_guard = False  # prevents itemChanged feedback loops while repopulating

        self.debounce = QtCore.QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(400)
        self.debounce.timeout.connect(self._commit_checklist_refit)

        self._build_ui()
        QtCore.QTimer.singleShot(50, self.on_load_default)

    # ---------- UI construction ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()
        self.btn_default = QtWidgets.QPushButton("Load default dataset")
        self.btn_add_files = QtWidgets.QPushButton("Add files...")
        self.btn_add_folder = QtWidgets.QPushButton("Add folder...")
        self.btn_run = QtWidgets.QPushButton("Run fit on pending")
        self.btn_export_csv = QtWidgets.QPushButton("Export results CSV...")
        self.btn_export_tab = QtWidgets.QPushButton("Add tab to master table...")
        for b in (self.btn_default, self.btn_add_files, self.btn_add_folder, self.btn_run,
                  self.btn_export_csv, self.btn_export_tab):
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- left: results table ----
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        splitter.addWidget(self.table)

        # ---- right: checklist + cards + plot ----
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)

        rlay.addWidget(QtWidgets.QLabel("Sensors in this experiment (uncheck to exclude from the fit):"))
        self.checklist = QtWidgets.QListWidget()
        self.checklist.setMaximumHeight(130)
        self.checklist.itemChanged.connect(self.on_checklist_changed)
        rlay.addWidget(self.checklist)

        # Numbering here is deliberately the REVERSE of the master-table CSV schema:
        # per the user's explicit choice, "1" = tetramer (the tight, slow site) and
        # "2" = dimer (the weak, fast site) -- matching Sartorius' own site-1/site-2
        # convention, not this project's assembly-order convention used everywhere
        # else (batch_results.csv, the Numbers tab, the manuscript draft). Each card
        # still names its physical species explicitly so the two conventions can
        # never be silently confused for one another.
        cards = QtWidgets.QHBoxLayout()
        self.card_tetramer = ParamCard("Tetramer", 1, "#0F6E56",
                                       "s⁻¹", "dimensionless", "dissociation fit")
        self.card_dimer = ParamCard("Dimer", 2, "#D85A30",
                                    "M⁻¹s⁻¹", "nM", "ka2 regression")
        cards.addWidget(self.card_tetramer)
        cards.addWidget(self.card_dimer)
        rlay.addLayout(cards)

        self.caption = QtWidgets.QLabel(" ")
        self.caption.setWordWrap(True)
        rlay.addWidget(self.caption)

        self.figure = Figure(figsize=(9, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(1, 2)
        # Standard button-based toolbar for pan/zoom -- deliberately NOT wiring
        # scroll-wheel to zoom (this project's convention: scroll must never zoom a canvas).
        self.toolbar = NavigationToolbar2QT(self.canvas, right)
        rlay.addWidget(self.toolbar)
        rlay.addWidget(self.canvas, 1)

        splitter.addWidget(right)
        splitter.setSizes([620, 780])

        self.status = QtWidgets.QLabel("Ready.")
        self.statusBar().addWidget(self.status, 1)

        self.btn_default.clicked.connect(self.on_load_default)
        self.btn_add_files.clicked.connect(self.on_add_files)
        self.btn_add_folder.clicked.connect(self.on_add_folder)
        self.btn_run.clicked.connect(self.on_run_pending)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_tab.clicked.connect(self.on_export_tab)

    # ---------- adding files ----------

    def _find_row(self, path):
        """Locate a path's CURRENT row by its UserRole tag. Required because
        QTableWidget with setSortingEnabled(True) silently reorders rows as soon
        as cell contents differ (confirmed: with sorting on, insertRow+setItem
        can even leave a freshly-set cell reading back as None) -- any row index
        captured earlier, including one cached in a dict, can be wrong by the
        time it's used. Always look up fresh, never cache a row number."""
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.data(QtCore.Qt.UserRole) == path:
                return r
        return None

    def _add_path(self, path, status="pending"):
        """Inserts path if new, with exactly one write+resort (see _write_row's
        docstring). No-op if already present -- callers that need to update an
        existing row's status use _set_row(path, status), not this."""
        if path in self.folders:
            return
        self.folders[path] = os.path.basename(os.path.dirname(path))
        was_sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._write_row(row, path, status)
        self.table.setSortingEnabled(was_sorting)

    def on_load_default(self):
        files = sorted(glob.glob(DEFAULT_GLOB))
        if not files:
            self.status.setText("Default dataset folders not found under ~/Downloads.")
            return

        cache = self._load_cache_file()
        for f in files:
            entry = cache.get(f)
            if entry is None:
                self._add_path(f)
                continue
            if "error" in entry:
                self.errors[f] = entry["error"]
            else:
                self.parsed[f] = entry["parsed"]
                self.masks[f] = entry["mask"]
                self.reasons[f] = entry["reasons"]
                self.results[f] = entry["r"]
                self.summaries[f] = (entry["s"] if "tet_category" in entry["s"]
                                     else summary_from_result(entry["r"]))  # pre-resolution cache
            self._add_path(f)  # inserts (no-op if already present); status corrected next line
            self._set_row(f, status=("error" if f in self.errors else "fitted"))

        self._apply_group_refs()
        n_cached = len(self.results) + len(self.errors)
        if n_cached:
            dim_ok = sum(1 for s in self.summaries.values() if s["dimerOK"])
            tet_ok = sum(1 for s in self.summaries.values() if s["tetramerOK"])
            self.status.setText(
                f"Loaded {len(files)} files, {n_cached} from cache ({os.path.basename(CACHE_PATH)}) -- "
                f"{len(self.results)} fitted, {dim_ok} dimer-OK, {tet_ok} tetramer-OK.")
        pending = [f for f in files if f not in self.results and f not in self.errors]
        if pending:
            self._start_worker([(p, self.folders[p], None, None) for p in pending])

    def _load_cache_file(self):
        if not os.path.exists(CACHE_PATH):
            return {}
        try:
            with open(CACHE_PATH, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            self.status.setText(f"Could not read cache ({type(e).__name__}: {e}) -- fitting fresh.")
            return {}

    def on_add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add Octet CSV files", DATA_DIR, "CSV files (*.csv)")
        for p in paths:
            self._add_path(p)
        if paths:
            self.status.setText(f"Added {len(paths)} file(s).")

    def on_add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add folder (recursive)", DATA_DIR)
        if not d:
            return
        files = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        for f in files:
            self._add_path(f)
        self.status.setText(f"Added {len(files)} file(s) from {os.path.basename(d)}.")

    # ---------- fitting ----------

    def on_run_pending(self):
        pending = [p for p in self.folders if p not in self.results and p not in self.errors]
        if not pending:
            self.status.setText("Nothing pending -- everything already fitted.")
            return
        jobs = [(p, self.folders[p], self.parsed.get(p), self.masks.get(p)) for p in pending]
        self._start_worker(jobs)

    def _start_worker(self, jobs):
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(jobs))
        self.worker = FitWorker(jobs)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.one_done.connect(self.on_worker_one_done)
        self.worker.finished_all.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_progress(self, i, total, name):
        self.progress.setValue(i)
        if name:
            self.status.setText(f"Fitting {i+1}/{total}: {name}")

    def on_worker_one_done(self, path, parsed, r, s, err):
        if parsed is not None and path not in self.parsed:
            self.parsed[path] = parsed
            if path not in self.masks:
                mask, reasons = default_mask(parsed)
                self.masks[path] = mask
                self.reasons[path] = reasons
        if err:
            self.errors[path] = err
            self.results.pop(path, None)
            self.summaries.pop(path, None)
        else:
            self.errors.pop(path, None)
            self.results[path] = r
            self.summaries[path] = s
        changed = self._apply_group_refs()
        self._set_row(path, status=("error" if err else "fitted"))
        for other in changed:
            if other != path:
                self._set_row(other, status="fitted")
        if path == self._selected_path:
            self._refresh_detail_for_selection()

    def _apply_group_refs(self):
        """Recompute same-interaction+buffer measured references and attach them to every
        non-measured run. A refit (e.g. after a checklist toggle) can change a group's
        measured mean, so this runs after every result lands. Returns paths whose
        reference changed."""
        items = [(p, self.results[p], self.summaries[p]) for p in self.summaries if p in self.results]
        refs = group_references([(r, s) for _, r, s in items])
        changed = []
        for p, r, s in items:
            ref = refs.get((r["interaction"], s["buffer"])) if s.get("tet_category") != "measured" else None
            new = ((ref["n"], ref["thalf_mean_d"], ref["thalf_sd_d"], ref["kdis2_mean"])
                   if ref else (None, None, None, None))
            old = (s.get("tet_group_n"), s.get("tet_group_thalf_mean_d"),
                   s.get("tet_group_thalf_sd_d"), s.get("tet_group_kdis2_mean"))
            if new != old:
                (s["tet_group_n"], s["tet_group_thalf_mean_d"],
                 s["tet_group_thalf_sd_d"], s["tet_group_kdis2_mean"]) = new
                changed.append(p)
        return changed

    def on_worker_finished(self):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        n_ok, n_err = len(self.results), len(self.errors)
        dim_ok = sum(1 for s in self.summaries.values() if s["dimerOK"])
        tet_ok = sum(1 for s in self.summaries.values() if s["tetramerOK"])
        self.status.setText(f"Done. {n_ok} fitted ({n_err} failed) -- "
                            f"{dim_ok} dimer-OK, {tet_ok} tetramer-OK.")

    # ---------- table ----------

    def _set_row(self, path, status):
        """Re-resolves path's row and writes it, as ONE atomic sort-disabled
        operation. This is the only entry point callers should use post-insert.
        Never pass a row index between calls -- see _write_row's docstring for
        why even that is unsafe."""
        was_sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        row = self._find_row(path)
        if row is not None:
            self._write_row(row, path, status)
        self.table.setSortingEnabled(was_sorting)

    def _write_row(self, row, path, status):
        """Pure write, no sorting side effects of its own -- callers (_set_row,
        _add_path) are responsible for bracketing this with exactly one
        setSortingEnabled(False)/(restore) pair. Calling this twice per file
        with sorting toggled back on in between (the original bug here) lets
        the SECOND write's row index go stale before it's used, since
        re-enabling sorting can immediately relocate the row the FIRST write
        just created."""
        folder = self.folders[path]
        interaction = self.parsed[path]["interaction"] if path in self.parsed else folder
        experiment = os.path.splitext(os.path.basename(path))[0]
        s = self.summaries.get(path)
        vals = [interaction, experiment, "", "", "", "", "", "", "", "", "", status]
        if s is not None:
            vals[2] = s["buffer"]
            vals[3] = f"{s['assoc_min']:.0f}"
            vals[4] = f"{s['dissoc_h']:.1f}"
            vals[5] = f"{s['KD1_nM']:.2f}" if np.isfinite(s["KD1_nM"]) else ""
            vals[6] = f"{s['kdis1']:.2e}"
            vals[7], vals[10] = tetramer_table_cells(s)
            vals[8] = f"{s['r2_kobs']:.3f}" if np.isfinite(s["r2_kobs"]) else ""
            vals[9] = "Y" if s["dimerOK"] else "N"
        for c, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(str(v))
            if status == "error":
                item.setForeground(QtGui.QBrush(QtGui.QColor("#a32d2d")))
            elif status == "pending":
                item.setForeground(QtGui.QBrush(QtGui.QColor("#888780")))
            elif s is not None and c == 9 and vals[c] == "N":
                item.setForeground(QtGui.QBrush(QtGui.QColor("#a32d2d")))
            elif s is not None and c in (7, 10) and s.get("tet_category") != "measured":
                # bounds and unboundable runs are greyed so they never read as measurements
                item.setForeground(QtGui.QBrush(QtGui.QColor("#888780")))
            self.table.setItem(row, c, item)
        self.table.item(row, 0).setData(QtCore.Qt.UserRole, path)

    def _table_selected_path(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.data(QtCore.Qt.UserRole) if item else None

    # ---------- selection / checklist / cards / plot ----------

    def on_row_selected(self):
        path = self._table_selected_path()
        if path is None:
            return
        self._selected_path = path
        self._populate_checklist(path)
        self._refresh_detail_for_selection()

    def _populate_checklist(self, path):
        self._checklist_guard = True
        self.checklist.clear()
        parsed = self.parsed.get(path)
        if parsed is None:
            self.checklist.addItem("(not parsed yet -- run the fit first)")
            self._checklist_guard = False
            return
        mask = self.masks.get(path)
        reasons = self.reasons.get(path)
        if mask is None:
            mask, reasons = default_mask(parsed)
            self.masks[path] = mask
            self.reasons[path] = reasons
        for w, checked, reason in zip(parsed["wells"], mask, reasons):
            label = f"{w['conc_nM']:g} nM"
            if w["is_control"]:
                label += "  (+ Max Control)"
            if reason:
                label += f"   — {reason}"
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            if reason:
                item.setForeground(QtGui.QBrush(QtGui.QColor("#888780")))
            self.checklist.addItem(item)
        self._checklist_guard = False

    def on_checklist_changed(self, item):
        if self._checklist_guard or self._selected_path is None:
            return
        mask = [self.checklist.item(i).checkState() == QtCore.Qt.Checked
                for i in range(self.checklist.count())]
        self.masks[self._selected_path] = mask
        n_checked = sum(mask)
        self.status.setText(f"{n_checked} sensor(s) selected -- refitting in a moment...")
        self.debounce.start()

    def _commit_checklist_refit(self):
        path = self._selected_path
        if path is None or path not in self.parsed:
            return
        mask = self.masks.get(path)
        if mask is None or sum(mask) < 3:
            self.status.setText("Need at least 3 checked sensors to fit.")
            return
        jobs = [(path, self.folders[path], self.parsed[path], mask)]
        self._start_worker(jobs)

    def _refresh_detail_for_selection(self):
        path = self._selected_path
        r = self.results.get(path)
        if r is None:
            for ax in self.axes:
                ax.cla()
            self.canvas.draw_idle()
            self.card_dimer.clear()
            self.card_tetramer.clear()
            err = self.errors.get(path)
            self.caption.setText("Not fitted yet." if not err else f"Error: {err}")
            return
        s = self.summaries[path]
        draw_fit_panels(self.axes, r)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.card_dimer.set_values(
            s["ka1"], s["kdis1"], s["KD1_nM"],
            f"{s['t12_dimer_min']:.1f} min", s["r2_kobs"],
            resolvable=s.get("dimer_phase_resolved", True))
        self.card_tetramer.set_texts(**tetramer_texts(s))

        cat = s.get("tet_category", "measured" if s["tetramerOK"] else "no_bound")
        note = f"  Tetramer (\"1\"): {TET_CATEGORY_LABEL.get(cat, cat)}."
        if cat != "measured":
            if s.get("tet_bound_method"):
                note += f" Basis: {s['tet_bound_method']}."
            if s.get("tet_group_n"):
                sd = s.get("tet_group_thalf_sd_d")
                sd_txt = "" if sd is None else " ± %.2f" % sd
                note += (f" Reference -- measured {s['buffer']} 15 h replicates of this interaction: "
                         f"t½ = {s['tet_group_thalf_mean_d']:.2f}{sd_txt} d "
                         f"(n={s['tet_group_n']}). That is the group's value, not this run's measurement.")
            else:
                note += (f" No measured 15 h replicate exists for this interaction in {s['buffer']}"
                         f" -- a 15 h dissociation run is needed for a measurement.")
        if not s["dimerOK"]:
            note += " Ka2 regression is weak (R2²≤0.9) -- KD2 (dimer, \"2\") here is unreliable."
        # Built locally (not via the shared page_title()) so the numbering here always
        # matches the cards above: page_title()/the PDF/the master-table CSV use the
        # OPPOSITE convention (1=dimer) on purpose, and mixing the two on one screen
        # would be exactly the confusion this whole renumbering was meant to resolve.
        caption = (f"{r['interaction']}   |   {r['experiment']}   |   {s['buffer']}   |   "
                  f"assoc={s['assoc_min']:.0f} min  dissoc={s['dissoc_h']:.1f} h   "
                  f"Kdis2(dimer)={s['kdis1']:.2e} s-1   Ka2(dimer)={s['ka1']:.2e} M-1s-1")
        self.caption.setText(caption + note)

    # ---------- export ----------

    def _current_rows_for_export(self):
        rows = []
        for path, s in self.summaries.items():
            r = self.results[path]
            rows.append(dict(
                interaction=r["interaction"], experiment=r["experiment"], fit=r["fit"] or "",
                KD1=s["KD1_nM"] / 1e9 if np.isfinite(s["KD1_nM"]) else None,
                KD1_err=None, KD2=s["KD2"] if s["tetramerOK"] else None, KD2_err=None,
                ka1=s["ka1"], ka1_err=None, ka2=s["ka2"],
                kdis1=s["kdis1"], kdis1_err=None,
                kdis2=s["kdis2"] if s["tetramerOK"] else None, kdis2_err=None,
                R2_dissoc=s["R2_dissoc"], r2_kobs=s["r2_kobs"], n_sensors=s["n_sensors"],
                t_assoc_s=s["assoc_min"] * 60, t_dissoc_s=s["dissoc_h"] * 3600,
                tetramer_frac=None, slow_decay_observed=s["decay_frac"],
                kdis2_reliable=s["tetramerOK"],
                t12_dimer_min=s["t12_dimer_min"],
                t12_tetramer_d=s["t12_tetramer_d"] if s["tetramerOK"] else None,
                apparent_overall_t_half_days=None,
                tet_category=s.get("tet_category"), tet_kdis2_upper=s.get("tet_kdis2_upper"),
                tet_thalf_lower_d=s.get("tet_thalf_lower_d"), tet_KD2_upper=s.get("tet_KD2_upper"),
                tet_bound_method=s.get("tet_bound_method"), tet_group_n=s.get("tet_group_n"),
                tet_group_thalf_mean_d=s.get("tet_group_thalf_mean_d"),
                tet_group_thalf_sd_d=s.get("tet_group_thalf_sd_d")))
        return rows

    def on_export_csv(self):
        if not self.summaries:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No fitted results yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export results CSV", os.path.join(HERE, "gui_results.csv"), "CSV files (*.csv)")
        if not path:
            return
        rows = self._current_rows_for_export()
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        self.status.setText(f"Exported {len(rows)} rows -> {path}")

    def on_export_tab(self):
        if not self.summaries:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No fitted results yet.")
            return
        src, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Master table to add a tab to", os.path.dirname(DEFAULT_MASTER_SRC),
            "Numbers files (*.numbers)")
        if not src:
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save updated master table as", src.replace(".numbers", " (updated).numbers"),
            "Numbers files (*.numbers)")
        if not out:
            return
        tmp_csv = os.path.join(HERE, "_gui_export_tmp.csv")
        rows = self._current_rows_for_export()
        with open(tmp_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        try:
            out_path, n, _ = build_master_tab(csv_path=tmp_csv, src_path=src, out_path=out)
            QtWidgets.QMessageBox.information(self, "Done", f"Wrote {n} rows to a new tab in:\n{out_path}")
            self.status.setText(f"Master table updated -> {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}\n\n"
                                           f"{traceback.format_exc()}")
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
