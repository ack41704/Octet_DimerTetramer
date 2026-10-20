#!/usr/bin/env python3
"""
Refresh dimer_tetramer/ from the working analysis folder.

The repository is a SNAPSHOT of the working copy: edit the scripts there and re-sync, never edit dimer_tetramer/*.py
here directly (paths.py is the one repo-owned module), or the two copies diverge.

    python tools/sync_from_working_copy.py /path/to/Octet_DimerTetramer_Analysis --dry-run
    python tools/sync_from_working_copy.py /path/to/Octet_DimerTetramer_Analysis

Each file in FILES is copied with one deterministic change: the working copy's absolute paths become portable ones
(dimer_tetramer/paths.py: DATA_DIR for input data, WORK_DIR for caches and outputs). Nothing is written if any
absolute path (/Users/..., /Volumes/...) would remain or if a rewritten file does not compile.
"""
import argparse
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO, "dimer_tetramer")

FILES = [
    # input parsing, dissociation fit, sensor selection, fit cache
    "parse_octet.py", "batch_fit.py", "plot_all_fits.py", "resolve_tetramer.py", "export_full_dataset.py",
    "build_extra_cache.py", "fill_master_tab.py", "export_master_pdf.py", "regen_pdf_from_cache.py",
    # v3 dimer-tetramer model
    "assoc_two_phase.py", "eb3_extended_constants.py", "extended_constants_all16.py",
    "extended_constants_all16_v2.py", "extended_constants_all16_v3.py", "all_runs_v3.py",
    "split_by_buffer_v3.py", "update_master_by_buffer.py", "add_new_runs_v3.py",
    # publication figures and tables
    "figure_EB3_WT_vs_R79P.py", "figure_EB3_WT_vs_R79P_halflife.py", "table_EB3_WT_vs_R79P_halflife.py",
    # validation of the window rules
    "truncation_test.py", "tetramer_window_test.py", "tetramer_bounds_check.py", "assoc_truncation.py",
    # interactive sensor checklist / per-run fits
    "gui_app.py", "build_tab.py",
]

ANALYSIS_DIR = r"/Users/amav/Downloads/Octet_DimerTetramer_Analysis/"
RULES = [
    # code references inside notes / docstrings
    (re.escape(ANALYSIS_DIR), "dimer_tetramer/"),
    # outputs and caches lived next to the scripts
    (r"W = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)", "W = WORK_DIR"),
    (r"HERE = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)", "HERE = WORK_DIR"),
    (r'os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\), "batch_results\.csv"\)',
     'os.path.join(WORK_DIR, "batch_results.csv")'),
    (r'open\("fit_cache\.pkl", "rb"\)', 'open(WORK_DIR + "/fit_cache.pkl", "rb")'),
    # input files: f-strings, then plain string literals
    (r'f"/Users/amav/Downloads/', 'f"{DATA_DIR}/'),
    (r'"/Users/amav/Downloads/', 'DATA_DIR + "/'),
    (r'os\.path\.expanduser\("~/Downloads"\)', "DATA_DIR"),
]
ABSOLUTE = re.compile(r"/Users/|/Volumes/")
IMPORT_LINE = "from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)\n"


def rewrite(text):
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    if ("DATA_DIR" in text or "WORK_DIR" in text) and "from paths import" not in text:
        first = next(n for n in ast.parse(text).body if isinstance(n, (ast.Import, ast.ImportFrom)))
        lines = text.splitlines(keepends=True)
        lines.insert(first.lineno - 1, IMPORT_LINE)
        text = "".join(lines)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("working_copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out, problems = {}, []
    for name in FILES:
        src = os.path.join(a.working_copy, name)
        if not os.path.exists(src):
            problems.append(f"missing in the working copy: {name}"); continue
        text = rewrite(open(src, encoding="utf-8").read())
        for i, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE.search(line):
                problems.append(f"{name}:{i}: absolute path remains: {line.strip()[:100]}")
        try:
            compile(text, name, "exec")
        except SyntaxError as e:
            problems.append(f"{name}: does not compile after rewriting: {e}")
        out[name] = text
    if problems:
        print("NOT synced:\n  " + "\n  ".join(problems)); sys.exit(1)
    changed = [n for n, t in out.items() if not os.path.exists(os.path.join(DEST, n))
               or open(os.path.join(DEST, n), encoding="utf-8").read() != t]
    print(f"{len(out)} files checked, {len(changed)} new or changed: {', '.join(changed) or '-'}")
    if a.dry_run:
        return
    os.makedirs(DEST, exist_ok=True)
    for n in changed:
        with open(os.path.join(DEST, n), "w", encoding="utf-8") as fh:
            fh.write(out[n])
    print(f"written to {DEST}")


if __name__ == "__main__":
    main()
