"""
Data and work locations for the repository copy of the analysis.

The working analysis folder used absolute paths; this copy resolves them from two environment variables
(relative defaults are taken from the current directory):

  OCTET_DATA_DIR   Octet 'Preprocessed Data' CSV exports (the "EB1/EB2/EB3 Interactions - Preprocessed Data FIles"
                   folders and any single-run CSVs) and the master-table .numbers files.     default: ./data
  OCTET_WORK_DIR   fit caches (fit_cache.pkl, fit_cache_extra.pkl) and every generated PDF / workbook / CSV.
                                                                                               default: ./work
"""
import os

DATA_DIR = os.path.abspath(os.environ.get("OCTET_DATA_DIR", "data"))
WORK_DIR = os.path.abspath(os.environ.get("OCTET_WORK_DIR", "work"))
os.makedirs(WORK_DIR, exist_ok=True)
