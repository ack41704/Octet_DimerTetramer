"""Guards for the repository copy: no machine-specific paths, every script compiles, every local import resolves,
and the sync tool's path rewrite behaves as documented."""
import ast
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "dimer_tetramer"
ABS_PATH = re.compile("/" + "Users/|/" + "Volumes/|/home/|[A-Za-z]:\\\\Users")


def _sync_tool():
    spec = importlib.util.spec_from_file_location("sync_tool", ROOT / "tools" / "sync_from_working_copy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_hardcoded_machine_paths():
    files = list(PKG.rglob("*.py")) + list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md"))
    offenders = [f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}"
                 for p in files for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if ABS_PATH.search(line)]
    assert not offenders, "absolute paths found:\n" + "\n".join(offenders)


def test_every_script_compiles():
    for py in sorted(PKG.glob("*.py")):
        compile(py.read_text(encoding="utf-8"), str(py), "exec")


def test_synced_file_list_matches_package():
    synced = set(_sync_tool().FILES)
    present = {p.name for p in PKG.glob("*.py")} - {"paths.py"}
    assert synced == present, f"in FILES but missing: {synced - present}; present but not synced: {present - synced}"


# top-level packages the scripts may import besides the standard library and each other (requirements*.txt)
DEPENDENCIES = {"numpy", "scipy", "matplotlib", "mpl_toolkits", "openpyxl", "numbers_parser", "pypdf", "PyQt5"}


def test_every_import_is_stdlib_a_dependency_or_local():
    local = {p.stem for p in PKG.glob("*.py")}
    unknown = []
    for py in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for top in {n.split(".")[0] for n in names}:
                if top not in local and top not in DEPENDENCIES and top not in sys.stdlib_module_names:
                    unknown.append(f"{py.name}: {top}")
    assert not unknown, "imports of modules that are not in the repository or requirements: " + ", ".join(unknown)


def test_rewrite_makes_paths_portable():
    sync = _sync_tool()
    src = ('"""doc"""\nimport os\nW = os.path.dirname(os.path.abspath(__file__))\n'
           'A = "/' + 'Users/amav/Downloads/x.numbers"\nB = f"/' + 'Users/amav/Downloads/{run}.csv"\n')
    out = sync.rewrite(src)
    assert "W = WORK_DIR" in out
    assert 'A = DATA_DIR + "/x.numbers"' in out and 'B = f"{DATA_DIR}/{run}.csv"' in out
    assert out.splitlines()[1].startswith("from paths import DATA_DIR, WORK_DIR")   # after the docstring
    assert not ABS_PATH.search(out)
    compile(out, "x", "exec")
