"""
All-runs v3 PDF reorganised, NO new analysis: the main data table is split into one table per buffer (PBST,
PBST + casein, PBSTB), each still in MASTER TABLE (BEST Colored) order and shaded by interaction, and the
non-subtracted runs are excluded.

Inputs (the saved 9-30-26 results, read only):
  All_runs_sensorgrams_v3_9-30-26.pdf                       run pages and WT-vs-R79P pages are copied verbatim
  Global Fit - Master Table on 9-30-26.numbers, tab "... Dimer-Tetramer v3 (all runs)"   the table values
Only the page numbers of the copied pages change (their 'page k / n' text is rewritten in place).

Excluded: every run whose data file is tagged "(not subtracted)": 2026.23, .24, .25, .26, .27 (tagged in the master
table too) and 2026.28 (tagged in its file name, "2026.28 (not subtracted) - Fit 12.csv", but not in the master table).
None of them is an EphB2-WT / R79P run, so the t-test pages are unchanged.

Also writes All_runs_v3_results_9-30-26.csv: the full 9-30-26 results table (all 82 rows) as a plain-text backup.
Output: All_runs_sensorgrams_v3_by_buffer_10-1-26.pdf (analysis folder).
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import csv
import io
import os
import re
import sys
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numbers_parser import Document
import pypdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_constants_all16_v2 import INK, INK2, GRID  # noqa: E402
from all_runs_v3 import FLAG_TEXT, ROWS_PER_TABLE_PAGE, TAB  # noqa: E402

W = WORK_DIR
SRC_PDF = os.path.join(W, "All_runs_sensorgrams_v3_9-30-26.pdf")
SRC_NUM = DATA_DIR + "/Global Fit - Master Table on 9-30-26.numbers"
OUT_PDF = os.path.join(W, "All_runs_sensorgrams_v3_by_buffer_10-1-26.pdf")
OUT_CSV = os.path.join(W, "All_runs_v3_results_9-30-26.csv")
EXCLUDE = {"2026.23", "2026.24", "2026.25", "2026.26", "2026.27", "2026.28"}       # non-subtracted runs
BUFFERS = [("PBST", "PBST"), ("PBST+casein", "PBST + casein"), ("PBSTB", "PBSTB")]
PAGE_W, PAGE_H = 11 * 72, 8.5 * 72


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def run_no(label):
    return re.match(r"\s*(\d{4}\.\d+)", label).group(1)


def buffer_of_row(r):
    """buffer column, or (not-analysable rows) the master table's Global Fit # annotation"""
    if r["Buffer"]:
        return r["Buffer"]
    g = r["Global Fit #"].lower()
    return "PBST+casein" if "(casein)" in g else "PBSTB" if "pbstb" in g else "PBST"


def load_rows():
    # the tab was renamed in Numbers after 9-30-26 ("Amav Model's Dimer-Tetramer v3 (all runs)"): match its ending
    t = [s for s in Document(SRC_NUM).sheets if s.name.endswith(TAB)][0].tables[0]
    rows = t.rows(values_only=True)
    head = ["" if h is None else str(h) for h in rows[0]]       # the user cleared two header cells: keep them empty
    out, inter = [], None
    for r in rows[1:]:
        if all(v in (None, "") for v in r):
            continue
        d = {h: ("" if v is None else v) for h, v in zip(head, r)}
        inter = d["Interaction"] or inter
        d["Interaction"] = inter
        d["Global Fit #"] = str(d["Global Fit #"])
        d["Octet Experiment"] = str(d["Octet Experiment"])
        out.append(d)
    return head, out


def save_csv(head, rows):
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for d in rows:
            w.writerow([f"{d[h]:.10g}" if isinstance(d[h], float) else d[h] for h in head])


def fv(v, f="{:.3e}"):
    return f.format(v) if isinstance(v, float) and np.isfinite(v) else "—"


def table_rows(rows):
    body, shade, g, prev = [], [], -1, None
    for d in rows:
        if d["Interaction"] != prev:
            g += 1; prev = d["Interaction"]
        inter = d["Interaction"].replace("Ephrin", "E").replace("-Fc", "")
        gf = d["Global Fit #"].split(" (")[0]
        if not isinstance(d["Association (min)"], float):                     # not analysable
            body.append([gf, d["Octet Experiment"][:22], inter, "", "not analysable", "", "", "", "", "", "", "", "",
                         d["Flags"]])
        else:
            body.append([gf, d["Octet Experiment"][:22], inter,
                         f"{d['Association (min)']:.0f} m / {d['Dissociation (h)']:.1f} h",
                         fv(d["Ka1 tetramer (1/Ms)"]), fv(d["Kdis1 tetramer (1/s)"]), fv(d["KD1 tetramer (nM)"], "{:.3g}"),
                         fv(d["Ka2 dimer (1/Ms)"]), fv(d["Kdis2 dimer (1/s)"]), fv(d["KD2 dimer (nM)"], "{:.3g}"),
                         fv(d["R² association, top conc"], "{:.4f}"), fv(d["R² association, all curves"], "{:.4f}"),
                         fv(d["R² dissociation"], "{:.4f}"), d["Flags"]])
        shade.append(g % 2)
    return body, shade


HEAD = ["GF", "Run", "Interaction", "Assoc /\nDissoc", "Ka1\n(1/Ms)", "Kdis1\n(1/s)", "KD1\n(nM)", "Ka2\n(1/Ms)",
        "Kdis2\n(1/s)", "KD2\n(nM)", "R² assoc\ntop", "R² assoc\nall", "R²\ndissoc", "Flags"]
WD = [0.035, 0.105, 0.135, 0.065, 0.075, 0.075, 0.06, 0.075, 0.075, 0.055, 0.055, 0.055, 0.05, 0.045]


def n_table_pages(rows_by_buf):
    return sum(max(1, int(np.ceil(len(r) / ROWS_PER_TABLE_PAGE))) for r in rows_by_buf.values())


def table_pages(pdf, rows_by_buf, k0, n_total):
    k = k0
    excl = ", ".join(sorted(EXCLUDE))
    for buf, title in BUFFERS:
        body, shade = table_rows(rows_by_buf[buf])
        pages = [list(range(i, min(i + ROWS_PER_TABLE_PAGE, len(body)))) for i in range(0, len(body), ROWS_PER_TABLE_PAGE)]
        for pi, idx in enumerate(pages):
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.03, 0.955, f"{title}: rate constants and fit quality — v3 (fast = dimer, slow = tetramer)" +
                     (f"   [{pi + 1}/{len(pages)}]" if len(pages) > 1 else ""), fontsize=12.5, weight="bold", color=INK)
            fig.text(0.03, 0.93, "1 = tetramer, 2 = dimer. Ka1 = (k_slow − Kdis1)/[EphB2]top (top-concentration slow phase); "
                     "Ka2 = slope of the fast-phase rate vs [EphB2] (global fit); Kdis from the global dissociation fit.\n"
                     f"{title} runs only, in MASTER TABLE (BEST Colored) order; shading alternates by interaction. "
                     f"Non-subtracted runs excluded ({excl}).", fontsize=7.6, color=INK2, va="top", linespacing=1.5)
            fig.text(0.97, 0.955, f"page {k} / {n_total}", fontsize=8, color="#777", ha="right")
            ax = fig.add_axes([0.01, 0.1, 0.98, 0.8]); ax.axis("off")
            tb = ax.table(cellText=[body[i] for i in idx], colLabels=HEAD, loc="upper center", cellLoc="center",
                          colWidths=[w / sum(WD) for w in WD])
            tb.auto_set_font_size(False); tb.set_fontsize(6.6); tb.scale(1, 1.42)
            for (i, j), c in tb.get_celld().items():
                c.set_edgecolor(GRID)
                if i == 0:
                    c.set_facecolor("#f2f2ef"); c.set_text_props(weight="bold", color=INK)
                elif shade[idx[i - 1]]:
                    c.set_facecolor("#f7f7f4")
            fig.text(0.03, 0.035, FLAG_TEXT, fontsize=6.8, color=INK2, linespacing=1.5)
            pdf.savefig(fig); plt.close(fig)
            k += 1
    return k


DIGIT_W = 0.6362 * 8          # width (pt) of one DejaVu Sans digit at the 8 pt page-number size, for right alignment
PAGE_NO = re.compile(rb"(1 0 -0 1 )([\d.]+)( [\d.]+ cm\nBT\n/F\d+ 8 Tf\n0 0 Td\n\[ \(page )(\d+ / \d+)(\) \] TJ)")


def renumber(page, k, n):
    """rewrite the copied page's own 'page k / 86' text (top right) to the new numbering; nothing else on the page
    changes. The right-aligned text keeps its right edge when the number of digits changes."""
    c = page.get_contents()
    data = c.get_data()
    hits = PAGE_NO.findall(data)
    assert len(hits) == 1, len(hits)
    new_txt = f"{k} / {n}".encode()

    def sub(m):
        x = float(m.group(2)) + DIGIT_W * (len(m.group(4)) - len(new_txt))
        return m.group(1) + f"{x:.6f}".encode() + m.group(3) + new_txt + m.group(5)
    c.set_data(PAGE_NO.sub(sub, data))
    page.replace_contents(c)
    return page


def main():
    src_md5 = (md5(SRC_PDF), md5(SRC_NUM))
    head, rows = load_rows()
    assert len(rows) == 82, len(rows)
    save_csv(head, rows)
    keep = [d for d in rows if run_no(d["Octet Experiment"]) not in EXCLUDE]
    dropped = [d["Octet Experiment"] for d in rows if run_no(d["Octet Experiment"]) in EXCLUDE]
    assert len(dropped) == len(EXCLUDE), dropped
    rows_by_buf = {b: [d for d in keep if buffer_of_row(d) == b] for b, _ in BUFFERS}
    assert sum(len(v) for v in rows_by_buf.values()) == len(keep)

    src = pypdf.PdfReader(SRC_PDF)
    stats_pages = [i for i, p in enumerate(src.pages) if "EphB2-WT vs EphB2-R79P" in p.extract_text()]
    run_pages = []                                  # (source page index, row) in the original (BEST) order
    for i, p in enumerate(src.pages):
        if i in stats_pages:
            continue
        txt = p.extract_text()
        m = re.search(r"\n([^\n]+?)\s+(\d{4}\.\d+[^\n]*?)\s+\(Global Fit # ([^)]+)\)", "\n" + txt)
        if m and "rate constants and fit quality" not in txt:
            run_pages.append((i, m.group(2).strip(), m.group(3).strip()))
    assert len(stats_pages) == 2 and len(run_pages) == 80, (len(stats_pages), len(run_pages))
    kept_runs = [rp for rp in run_pages if run_no(rp[1]) not in EXCLUDE]
    assert len(kept_runs) == 80 - len(EXCLUDE), len(kept_runs)

    n_tab = n_table_pages(rows_by_buf)
    n_total = n_tab + len(stats_pages) + len(kept_runs)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        table_pages(pdf, rows_by_buf, 1, n_total)
    out = pypdf.PdfWriter()
    for p in pypdf.PdfReader(buf).pages:
        out.add_page(p)
    k = n_tab + 1
    for i in stats_pages:
        out.add_page(renumber(src.pages[i], k, n_total)); k += 1
    for i, _, _ in kept_runs:
        out.add_page(renumber(src.pages[i], k, n_total)); k += 1
    out.add_metadata({"/Title": "All Octet runs, v3 dimer-tetramer model, tables by buffer (non-subtracted runs excluded)"})
    with open(OUT_PDF, "wb") as fh:
        out.write(fh)
    assert (md5(SRC_PDF), md5(SRC_NUM)) == src_md5, "inputs changed"
    print(f"excluded: {', '.join(dropped)}")
    for b, t in BUFFERS:
        print(f"  {t}: {len(rows_by_buf[b])} runs")
    print(f"{OUT_PDF}: {n_total} pages ({n_tab} table, {len(stats_pages)} t-test, {len(kept_runs)} run pages)")
    print(f"{OUT_CSV}: {len(rows)} rows (all 9-30-26 results); 9-30-26 PDF and Numbers file unchanged")


if __name__ == "__main__":
    main()
