"""
Add new runs to the v3 results WITHOUT touching the existing ones (they stay frozen as approved):
  2026.123  EphrinA5-Fc + EphB2-WT    Global Fit 83   PBST   (ligand and buffer from the user, 2026-10-02)
  2026.124  EphrinA5-Fc + EphB2-R79P  Global Fit 84   PBST
Each new run gets exactly the v3 analysis (all_runs_v3.analyse: v2.analyze_run + window-aware Kdis + v3.finish_run).
Its standard-pipeline fit is in fit_cache_extra.pkl (build_extra_cache.py, which only fits files it has not fitted).

Outputs
  All_runs_v3_results_10-2-26.csv               the 9-30-26 results + the new runs (all 84 rows; plain-text record)
  All_runs_sensorgrams_v3_by_buffer_10-2-26.pdf the by-buffer PDF + the new runs: tables regenerated (new rows after
                                                2026.122, the EphrinA5 group), t-test pages regenerated from the saved
                                                values (unchanged sets; the new EphrinA5 WT / R79P pair is n = 1 each,
                                                listed as not tested), the 74 existing run pages copied verbatim (page
                                                numbers rewritten), the 2 new run pages rendered by v3.run_page
  Global Fit - Master Table on 10-2-26.numbers  copy of the user's 10-1-26 file (with all of the user's edits) + the
                                                new runs appended to the PBST table in the user's layout (17 columns;
                                                single-run group = run row + 2 blank rows; alternating group shading).
                                                The PBST + casein and PBSTB tables are moved down by exactly the added
                                                height, so the spacing between tables is unchanged. Nothing else
                                                changes (formulas included; verified).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import csv
import io
import math
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from numbers_parser import Document, RGB
import pypdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_runs_v3 as AR  # noqa: E402
import extended_constants_all16_v3 as V3  # noqa: E402
import split_by_buffer_v3 as S  # noqa: E402

W = WORK_DIR
NEW_RUNS = [dict(interaction="EphrinA5-Fc + EphB2-WT", label="2026.123", gf="83",
                 path=DATA_DIR + "/2026.123 - Preprocessed Data - Fit 83.csv"),
            dict(interaction="EphrinA5-Fc + EphB2-R79P", label="2026.124", gf="84",
                 path=DATA_DIR + "/2026.124 - Preprocessed Data - Fit 84.csv")]
AFTER = "2026.122"                                      # new rows follow the EphrinA5-Fc + EphB2-His run
NUM_IN = DATA_DIR + "/Global Fit - Master Table on 10-1-26.numbers"     # the user's edited copy
NUM_OUT = DATA_DIR + "/Global Fit - Master Table on 10-2-26.numbers"
CSV_OUT = os.path.join(W, "All_runs_v3_results_10-2-26.csv")
PDF_OUT = os.path.join(W, "All_runs_sensorgrams_v3_by_buffer_10-2-26.pdf")
ROW_H = 20.0                                            # Numbers default row height used throughout these tables
SHADE = (239, 239, 233)


# ------------------------------------------------------------------ analysis of the new runs only
def analyse_new():
    extra = pickle.load(open(os.path.join(W, "fit_cache_extra.pkl"), "rb"))
    out = []
    for n in NEW_RUNS:
        e = extra[n["path"]]
        assert e["r"]["interaction"] == n["interaction"] and e["r"]["experiment"] == n["label"], (n, e["r"]["experiment"])
        b = dict(row=None, interaction=n["interaction"], label=n["label"], gf=n["gf"], e=e)
        b["x"] = AR.analyse(b)
        out.append(b)
    return out


def row_dict(b, head):
    """the new run as a row of the 9-30-26 results table (same columns as split_by_buffer_v3.load_rows)"""
    vals = [b["interaction"]] + AR.numbers_values(b)
    assert len(vals) == len(AR.NCOLS) == len(head)
    d = {}
    for h, n, v in zip(head, AR.NCOLS, vals):
        key = h if h else n                              # the two cleared header cells keep their 9-30 names here
        d[key] = float(v) if isinstance(v, (float, np.floating)) else v
    for h in head:
        d.setdefault(h, "")
    return d


# ------------------------------------------------------------------ PDF
def light(d):
    """a saved results row as the minimal run dict that all_runs_v3's t-test pages need"""
    f = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
    lig, analyte = [p.strip() for p in d["Interaction"].split("+", 1)]
    return dict(exp=d["Octet Experiment"], lig=lig, analyte=analyte, buffer=d["Buffer"], assoc_min=f(d["Association (min)"]),
                dis_h=f(d["Dissociation (h)"]), ka1=f(d["Ka1 tetramer (1/Ms)"]), kdis1=f(d["Kdis1 tetramer (1/s)"]),
                KD1_nM=f(d["KD1 tetramer (nM)"]), ka2=f(d["Ka2 dimer (1/Ms)"]), kdis2=f(d["Kdis2 dimer (1/s)"]),
                KD2_nM=f(d["KD2 dimer (nM)"]))


def build_pdf(rows_all, new):
    keep = [d for d in rows_all if S.run_no(d["Octet Experiment"]) not in S.EXCLUDE]
    rows_by_buf = {b: [d for d in keep if S.buffer_of_row(d) == b] for b, _ in S.BUFFERS}
    runs_light = [light(d) for d in keep if isinstance(d["Association (min)"], float)]
    src = pypdf.PdfReader(S.SRC_PDF)                     # 9-30-26 all-runs PDF: the approved run pages
    stats_src = [i for i, p in enumerate(src.pages) if "EphB2-WT vs EphB2-R79P" in p.extract_text()]
    run_pages = []
    for i, p in enumerate(src.pages):
        txt = p.extract_text()
        if i in stats_src or "rate constants and fit quality" in txt:
            continue
        m = re.search(r"\n([^\n]+?)\s+(\d{4}\.\d+[^\n]*?)\s+\(Global Fit # ([^)]+)\)", "\n" + txt)
        if m:
            run_pages.append((i, m.group(2).strip()))
    kept = [rp for rp in run_pages if S.run_no(rp[1]) not in S.EXCLUDE]
    assert len(kept) == 74 and kept[-1][1] == AFTER, (len(kept), kept[-1])
    n_tab = S.n_table_pages(rows_by_buf)
    n_stat = max(1, int(np.ceil(sum(blk["tested"] for blk in AR.ttest_blocks(runs_light)) / 3)))
    n_total = n_tab + n_stat + len(kept) + len(new)
    head_buf = io.BytesIO()
    with PdfPages(head_buf) as pdf:
        S.table_pages(pdf, rows_by_buf, 1, n_total)
        assert AR.stats_pages(pdf, runs_light, n_tab + 1, n_total) == n_stat
    new_buf = io.BytesIO()
    with PdfPages(new_buf) as pdf:
        for j, b in enumerate(new):
            V3.run_page(pdf, b["x"], n_tab + n_stat + len(kept) + 1 + j, n_total)
    out = pypdf.PdfWriter()
    for p in pypdf.PdfReader(head_buf).pages:
        out.add_page(p)
    k = n_tab + n_stat + 1
    for i, _ in kept:
        out.add_page(S.renumber(src.pages[i], k, n_total)); k += 1
    for p in pypdf.PdfReader(new_buf).pages:
        out.add_page(p)
    out.add_metadata({"/Title": "All Octet runs, v3 dimer-tetramer model, tables by buffer (non-subtracted runs excluded)"})
    with open(PDF_OUT, "wb") as fh:
        out.write(fh)
    return n_total, n_tab, n_stat, len(kept), stats_src


# ------------------------------------------------------------------ Numbers
def shift_table(doc, table, dy):
    """move a table down by dy points (the drawable position Numbers stores; numbers_parser only reads it)"""
    info = doc._model.objects[doc._model.table_info_id(table._table_id)]
    info.super.geometry.position.y = info.super.geometry.position.y + dy


def update_numbers(new):
    doc = Document(NUM_IN)
    sh = [s for s in doc.sheets if s.name.endswith(S.TAB)][0]
    tabs = {t.name.split(" —")[0]: t for t in sh.tables}
    t = tabs["PBST"]
    header = ["" if h is None else h for h in t.rows(values_only=True)[0]]
    ncol = len(header)
    last = t.rows(values_only=True)[-1]
    assert str(last[1]) == AFTER, last[:3]                                  # the table ends with 2026.122
    last_shaded = t.cell(t.num_rows - 1, 1).style is not None and t.cell(t.num_rows - 1, 1).style.bg_color is not None \
        and tuple(t.cell(t.num_rows - 1, 1).style.bg_color) == SHADE
    # layout (the user's single-run group): run row, then 2 blank rows; groups alternate shading
    plan = [("blank", None, last_shaded), ("blank", None, last_shaded)]      # complete 2026.122's group
    shaded = not last_shaded
    for j, b in enumerate(new):
        plan.append(("run", b, shaded))
        if j < len(new) - 1:
            plan += [("blank", None, shaded), ("blank", None, shaded)]
        shaded = not shaded
    r0 = t.num_rows
    below = [x for x in sh.tables if x.coordinates[1] > t.coordinates[1]]
    before = {x.name: x.coordinates[1] for x in below}
    t.add_row(num_rows=len(plan))                                          # appended after 2026.122
    shade = doc.add_style(name="v3 interaction shade (added rows)", bg_color=RGB(*SHADE))
    fmt = {c: f for c, f in AR.FMT.items() if c < ncol}
    for i, (kind, b, sh_) in enumerate(plan):
        r = r0 + i
        if kind == "run":
            vals = [b["interaction"]] + AR.numbers_values(b)[:ncol - 1]
            for c, v in enumerate(vals):
                if isinstance(v, (float, np.floating)):
                    if np.isfinite(v):
                        t.write(r, c, float(f"{float(v):.12g}"))
                        if c in fmt:
                            t.set_cell_formatting(r, c, fmt[c][0], decimal_places=fmt[c][1])
                    else:
                        t.write(r, c, "—")
                else:
                    t.write(r, c, v)
        if sh_:
            for c in range(ncol):
                t.set_cell_style(r, c, shade)
    dy = len(plan) * ROW_H
    for x in below:
        shift_table(doc, x, dy)
    doc.save(NUM_OUT)
    return plan, dy, before


def verify_numbers(new, plan, dy, before):
    a, b = Document(NUM_IN), Document(NUM_OUT)
    assert [s.name for s in a.sheets] == [s.name for s in b.sheets]
    same = lambda u, v: u == v or (isinstance(u, float) and isinstance(v, float) and
                                   (math.isnan(u) and math.isnan(v) or abs(u - v) <= 1e-12 * max(1, abs(u))))
    n_cells = n_formula = 0
    for sa, sb in zip(a.sheets, b.sheets):
        assert len(sa.tables) == len(sb.tables), sa.name
        for ta, tb in zip(sa.tables, sb.tables):
            ra, rb = ta.rows(), tb.rows()
            assert ta.name == tb.name, sa.name
            extra = len(rb) - len(ra)
            assert extra in (0, len(plan)), (sa.name, ta.name, extra)
            for x, y in zip(ra, rb):                     # every pre-existing cell (incl. formula results) unchanged
                for u, v in zip(x, y):
                    assert same(u.value, v.value), (sa.name, ta.name, u.value, v.value)
                    n_cells += 1
                    n_formula += getattr(u, "_formula_id", None) is not None
            if extra:
                added = tb.rows(values_only=True)[len(ra):]
                runs = [r for r in added if r[1] not in (None, "")]
                assert [str(r[1]) for r in runs] == [n["label"] for n in new], runs
                for r, bb in zip(runs, new):
                    want = [bb["interaction"]] + AR.numbers_values(bb)[:len(r) - 1]
                    for c, (g, w) in enumerate(zip(r, want)):
                        if isinstance(w, (float, np.floating)):
                            assert (g == "—") if not np.isfinite(w) else abs(g - w) <= 1e-9 * abs(w), (c, g, w)
                        else:
                            assert (g or "") == w, (c, g, w)
            if ta.name in before:
                assert abs(tb.coordinates[1] - (before[ta.name] + dy)) < 1e-6, (ta.name, tb.coordinates)
    sh = [s for s in b.sheets if s.name.endswith(S.TAB)][0]
    boxes = sorted((t.coordinates[1], t.coordinates[1] + t.height, t.name[:14]) for t in sh.tables)
    assert all(boxes[i][1] < boxes[i + 1][0] for i in range(len(boxes) - 1)), boxes
    return n_cells, n_formula, boxes


# ------------------------------------------------------------------ main
def main():
    head, rows_all = S.load_rows()                       # the saved 9-30-26 results (82 rows)
    assert len(rows_all) == 82
    new = analyse_new()
    for b in new:
        x = b["x"]
        print(f"{b['interaction']:26s} {b['label']} (GF {b['gf']}): Ka1 {AR.fv(x['ka1'])} Kdis1 {AR.fv(x['kdis1'])} "
              f"KD1 {AR.fv(x['KD1_nM'], '{:.3g}')} | Ka2 {AR.fv(x['ka2'])} Kdis2 {AR.fv(x['kdis2'])} "
              f"KD2 {AR.fv(x['KD2_nM'], '{:.3g}')} | R2 {x['r2_top']:.4f} {x['r2_all']:.4f} {x['r2_dis']:.4f} [{x['flags']}]"
              f" sensors {x['n_fit']}/{x['n_all']}")
    new_rows = [row_dict(b, head) for b in new]
    i = [d["Octet Experiment"] for d in rows_all].index(AFTER) + 1
    rows_new = rows_all[:i] + new_rows + rows_all[i:]
    S.OUT_CSV = CSV_OUT
    S.save_csv(head, rows_new)
    n_total, n_tab, n_stat, n_kept, stats_src = build_pdf(rows_new, new)
    # the regenerated t-test pages: same tested sets and numbers as the 10-1 PDF; only the not-tested list grows
    newpdf = pypdf.PdfReader(PDF_OUT); old = pypdf.PdfReader(S.SRC_PDF)
    norm = lambda t: " ".join(re.sub(r"page \d+ / \d+", "", t).split())
    o1, n1 = norm(old.pages[stats_src[0]].extract_text()), norm(newpdf.pages[n_tab].extract_text())
    assert o1 == n1, "t-test page 1 changed"
    o2 = norm(old.pages[stats_src[1]].extract_text()).split("Not tested")[0]
    n2 = norm(newpdf.pages[n_tab + 1].extract_text()).split("Not tested")[0]
    assert o2 == n2, "t-test page 2 (tested set) changed"
    assert "2026.123" in newpdf.pages[n_tab + 1].extract_text()
    plan, dy, before = update_numbers(new)
    n_cells, n_formula, boxes = verify_numbers(new, plan, dy, before)
    print(f"{CSV_OUT}: {len(rows_new)} rows")
    print(f"{PDF_OUT}: {n_total} pages ({n_tab} table, {n_stat} t-test, {n_kept} copied + {len(new)} new run pages); "
          "t-test results unchanged")
    print(f"{NUM_OUT}: +{len(plan)} rows in the PBST table, lower tables moved down {dy:.0f} pt; {n_cells} existing cells "
          f"({n_formula} formulas) unchanged; tables {[(round(a), round(b_)) for a, b_, _ in boxes]}")


if __name__ == "__main__":
    main()
