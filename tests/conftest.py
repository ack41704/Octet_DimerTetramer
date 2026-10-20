"""Make the dimer_tetramer scripts importable (they import each other by bare module name, as in the working copy)
and keep anything they write out of the repository."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OCTET_WORK_DIR", tempfile.mkdtemp(prefix="octet_work_"))
os.environ.setdefault("OCTET_DATA_DIR", tempfile.mkdtemp(prefix="octet_data_"))
sys.path.insert(0, str(ROOT / "dimer_tetramer"))
