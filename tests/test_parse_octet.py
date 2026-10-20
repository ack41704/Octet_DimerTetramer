"""parse_octet_csv on a synthetic 'Preprocessed Data' export (5 header lines, then X/Y column pairs per well)."""
import numpy as np

from parse_octet import parse_octet_csv


def _write_export(path, wells):
    n_rows = max(len(t) for _, _, _, t, _ in wells)
    lines = ["<CorrectionSettings>...</CorrectionSettings>",
             ", ".join(f"{loc}, {loc}" for loc, *_ in wells) + ", ",
             "".join(f"Sample Loc: {loc}, Sample ID: {sid}, " for loc, sid, *_ in wells),
             "".join(f"Sample Conc: {c}, Conc Unit: nM, " for _, _, c, *_ in wells),
             "X Values, Y Values, " * len(wells)]
    for i in range(n_rows):
        cells = []
        for *_, t, y in wells:
            cells += [f"{t[i]:.1f}", f"{y[i]:.5f}"] if i < len(t) else ["", ""]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_synthetic_export(tmp_path):
    t = np.arange(100.0, 400.0, 1.0)
    wells = [("A1", "EphB2-WT 81 nM", 81.0, t, 0.3 * (1 - np.exp(-0.01 * (t - 100)))),
             ("B1", "EphB2-R79P", 243.0, t, 0.1 + 0 * t),
             ("C1", "A20-I control", 243.0, t, 0 * t),
             ("D1", "EphB2-WT short", 27.0, t[:30], 0 * t[:30])]          # < 50 points: dropped
    f = tmp_path / "2026.999 - Preprocessed Data.csv"
    _write_export(f, wells)
    out = parse_octet_csv(f)
    assert [w["sample_id"] for w in out] == ["EphB2-WT 81 nM", "EphB2-R79P", "A20-I control"]
    assert [w["label"] for w in out] == ["WT", "R79P", ""]
    assert [w["is_control"] for w in out] == [False, False, True]
    assert np.allclose([w["conc_M"] for w in out], [81e-9, 243e-9, 243e-9])
    assert out[0]["t"][0] == 0 and len(out[0]["t"]) == len(t)               # time re-zeroed at the first point
    assert np.allclose(out[0]["y"], wells[0][4], atol=1e-5)
