"""
Master-table update matching All_runs_sensorgrams_v3_by_buffer_10-1-26.pdf: NO new analysis, the values are the saved
9-30-26 results (tab "Amav Model's Dimer-Tetramer v3 (all runs)" of Global Fit - Master Table on 9-30-26.numbers).

In a COPY, Global Fit - Master Table on 10-1-26.numbers (the 9-30-26 file keeps the initial all-runs results):
  - the user's tab keeps its name; its results table becomes the PBST table, and PBST + casein and PBSTB tables are
    added below it (same 23 columns). Rows in MASTER TABLE (BEST Colored) order, the interaction named on the first row
    of each group, groups shaded alternately as in the PDF (blank separator rows would not fit above the t-test table)
  - non-subtracted runs removed (split_by_buffer_v3.EXCLUDE: 2026.23-.28)
  - the WT vs R79P table is untouched (none of the removed runs is a WT / R79P run)
Every other tab is verified unchanged.
"""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os
import sys

import numpy as np
from numbers_parser import Document, RGB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_by_buffer_v3 as S  # noqa: E402
from all_runs_v3 import FMT  # noqa: E402

OUT = DATA_DIR + "/Global Fit - Master Table on 10-1-26.numbers"
NAME = "{} — Dimer-Tetramer v3: point values, fast phase = dimer, slow phase = tetramer (non-subtracted runs excluded)"
GAP = 80                                                     # points between tables, as between the existing two


def sheet_of(doc):
    return [s for s in doc.sheets if s.name.endswith(S.TAB)][0]


def load_raw():
    """the tab's header row and its run rows as value lists (by column position; interaction forward-filled into
    column 0). Position, not header text: the user cleared two header cells (the Kdis1 / Kdis2 source columns)."""
    t = sheet_of(Document(S.SRC_NUM)).tables[0]
    rows = t.rows(values_only=True)
    out, inter = [], None
    for r in rows[1:]:
        if all(v in (None, "") for v in r):
            continue
        r = ["" if v is None else v for v in r]
        inter = r[0] or inter
        r[0] = inter
        out.append(r)
    return ["" if h is None else h for h in rows[0]], out


def as_dict(r):
    return {"Octet Experiment": str(r[1]), "Global Fit #": str(r[2]), "Buffer": r[3]}


def fill(t, header, rows, shade):
    """header + one row per run; interaction on the first row of each group; alternate groups shaded"""
    for c, h in enumerate(header):
        t.write(0, c, h)
    prev, g = None, -1
    for i, r in enumerate(rows, 1):
        if r[0] != prev:
            g += 1; prev = r[0]
            t.write(i, 0, r[0])
        else:
            t.write(i, 0, "")
        for c in range(1, len(header)):
            v = r[c]
            if isinstance(v, float):
                t.write(i, c, v)
                if c in FMT:
                    t.set_cell_formatting(i, c, FMT[c][0], decimal_places=FMT[c][1])
            else:
                t.write(i, c, v)
        if g % 2:
            for c in range(len(header)):
                t.set_cell_style(i, c, shade)
    for c, w in ((0, 170), (1, 150), (2, 120), (18, 150), (19, 230), (20, 200), (22, 600)):
        t.col_width(c, w)


def main():
    header, rows = load_raw()
    assert len(rows) == 82 and len(header) == 23, (len(rows), len(header))
    keep = [r for r in rows if S.run_no(str(r[1])) not in S.EXCLUDE]
    by_buf = {b: [r for r in keep if S.buffer_of_row(as_dict(r)) == b] for b, _ in S.BUFFERS}

    doc = Document(S.SRC_NUM)
    sh = sheet_of(doc)
    assert len(sh.tables) == 2, [t.name for t in sh.tables]
    t1, stats_t = sh.tables
    stats_before = stats_t.rows(values_only=True)
    shade = doc.add_style(name="v3 interaction shade", bg_color=RGB(239, 239, 233))

    # PBST: the existing results table, shrunk to its rows
    n1 = 1 + len(by_buf["PBST"])
    t1.delete_row(num_rows=t1.num_rows - n1, start_row=n1)
    t1.name = NAME.format("PBST")
    fill(t1, header, by_buf["PBST"], shade)
    y = t1.coordinates[1] + t1.height + GAP
    made = [t1]
    for b, title in S.BUFFERS[1:]:
        t = sh.add_table(NAME.format(title), x=t1.coordinates[0], y=y, num_rows=1 + len(by_buf[b]), num_cols=len(header),
                         num_header_rows=t1.num_header_rows, num_header_cols=t1.num_header_cols)
        fill(t, header, by_buf[b], shade)
        y = t.coordinates[1] + t.height + GAP
        made.append(t)
    assert y - GAP + 40 <= stats_t.coordinates[1], (y, stats_t.coordinates)   # no overlap with the t-test table
    doc.save(OUT)

    # ---------------- verify the saved file
    a, b = Document(S.SRC_NUM), Document(OUT)
    assert [s.name for s in a.sheets] == [s.name for s in b.sheets]
    for sa, sb in zip(a.sheets, b.sheets):
        if sa.name.endswith(S.TAB):
            continue
        for ta, tb in zip(sa.tables, sb.tables):
            assert ta.name == tb.name and ta.rows(values_only=True) == tb.rows(values_only=True), sa.name
    sb = sheet_of(b)
    tables = sb.tables
    assert len(tables) == 4, [t.name for t in tables]
    assert tables[1].rows(values_only=True) == stats_before                      # t-test table unchanged
    n_vals = 0
    for t, (buf, title) in zip([tables[0], tables[2], tables[3]], S.BUFFERS):
        got = t.rows(values_only=True)
        assert t.name.startswith(title + " —"), t.name
        assert len(got) == 1 + len(by_buf[buf]), (title, len(got))
        assert [("" if h is None else h) for h in got[0]] == header, (title, got[0])
        for r, d in zip(got[1:], by_buf[buf]):
            for c in range(1, len(header)):
                v = d[c]
                if isinstance(v, float):
                    assert abs(r[c] - v) <= 1e-12 * max(1, abs(v)), (title, d[1], c, r[c], v); n_vals += 1
                else:
                    assert (r[c] or "") == v, (title, d[1], c, r[c], v)
        assert not any(S.run_no(str(r[1])) in S.EXCLUDE for r in got[1:])
    boxes = sorted((t.coordinates[1], t.coordinates[1] + t.height, t.name[:22]) for t in tables)
    assert all(boxes[i][1] < boxes[i + 1][0] for i in range(len(boxes) - 1)), boxes
    print(f"{OUT}\n  tab '{sb.name}': " + "; ".join(f"{t.name.split(' —')[0]} {t.num_rows - 1} runs" for t in
                                                     (tables[0], tables[2], tables[3])) + "; WT vs R79P table unchanged")
    print(f"  {n_vals} numeric cells verified against the saved 9-30-26 results; {len(a.sheets) - 1} other tabs unchanged; "
          f"tables do not overlap: {[(round(y0), round(y1)) for y0, y1, _ in boxes]}")
    print(f"  removed: {', '.join(str(r[1]) for r in rows if S.run_no(str(r[1])) in S.EXCLUDE)}")


if __name__ == "__main__":
    main()
