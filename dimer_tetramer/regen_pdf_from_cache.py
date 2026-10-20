"""Rebuild All_78_Fits.pdf from fit_cache.pkl -- same plots, current headers, no refitting."""
from paths import DATA_DIR, WORK_DIR  # noqa: E402,F401  (repo copy: portable data / work locations)
import os, sys, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_all_fits import draw_fit_panels, page_title

W = WORK_DIR
cache = pickle.load(open(os.path.join(W, "fit_cache.pkl"), "rb"))
ok = sorted(((p, e) for p, e in cache.items() if "error" not in e), key=lambda z: z[0])
with PdfPages(os.path.join(W, "All_78_Fits.pdf")) as pdf:
    for p, e in ok:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        draw_fit_panels(axes, e["r"])
        fig.suptitle(page_title(e["r"], e["s"]), fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        pdf.savefig(fig)
        plt.close(fig)
print(f"{len(ok)} pages -> All_78_Fits.pdf")
