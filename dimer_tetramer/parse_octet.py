"""Parser for Octet 'Preprocessed Data' CSV exports (Sartorius/ForteBio format).

 row0: XML correction-settings blob (ignored)
 row1: sensor location codes, repeated for X,Y of each well
 row2: "Sample Loc: ..., Sample ID: ..., " per well
 row3: "Sample Conc: <val>, Conc Unit: <unit>, " per well
 row4: "X Values, Y Values, " per well
 row5+: pairs of (time_s, response_nm) per well
"""
import numpy as np

UNIT_TO_M = {"nM": 1e-9, "uM": 1e-6, "µM": 1e-6, "mM": 1e-3, "M": 1.0, "pM": 1e-12}


def parse_octet_csv(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header_lines = [f.readline() for _ in range(5)]
        data_lines = f.readlines()

    sample_row = header_lines[2].strip()
    conc_row = header_lines[3].strip()

    sample_ids = []
    for p in sample_row.split("Sample Loc:")[1:]:
        if "Sample ID:" in p:
            sid = p.split("Sample ID:", 1)[1]
            sample_ids.append(sid.strip().rstrip(","))
        else:
            sample_ids.append("")

    concs, units = [], []
    for chunk in conc_row.split(","):
        chunk = chunk.strip()
        if chunk.startswith("Sample Conc:"):
            try:
                concs.append(float(chunk.split(":", 1)[1].strip()))
            except ValueError:
                concs.append(float("nan"))
        elif chunk.startswith("Conc Unit:"):
            units.append(chunk.split(":", 1)[1].strip())

    n_wells = min(len(concs), len(sample_ids)) if sample_ids else len(concs)
    while len(units) < n_wells:
        units.append("nM")

    arr = np.genfromtxt(data_lines, delimiter=",", usecols=range(n_wells * 2),
                        filling_values=np.nan)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    wells = []
    for i in range(n_wells):
        t, y = arr[:, 2 * i], arr[:, 2 * i + 1]
        m = ~np.isnan(t) & ~np.isnan(y)
        t, y = t[m], y[m]
        if len(t) < 50:
            continue
        sid = sample_ids[i] if i < len(sample_ids) else ""
        low = sid.lower()
        wells.append(dict(
            sample_id=sid,
            conc_nM=concs[i],
            conc_M=concs[i] * UNIT_TO_M.get(units[i], 1e-9),
            is_control=("control" in low or "a20" in low or "inhibitor" in low),
            label=("R79P" if "R79P" in sid else ("WT" if "WT" in sid else "")),
            t=t - t[0],
            y=y,
        ))
    return wells
