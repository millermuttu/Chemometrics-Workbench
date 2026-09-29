# PyInstaller spec for the packaged application: onedir, one console
# executable, the built frontend beside it. `PROPOSAL.md` §4.2.
#
#   cd frontend && pnpm build && cd ..
#   uv run pyinstaller packaging/workbench.spec --noconfirm
#
# The result is dist/ChemometricsWorkbench/. `tests/smoke_package.py` launches
# it, drives a PCA over HTTP, and fails if the bundle carries scikit-learn or
# chemotools - both are development dependencies (§7), and a package that
# shipped either would be a wrapper around what we claim parity with.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
BUNDLE = ROOT / "frontend" / "dist"
if not (BUNDLE / "index.html").is_file():
    raise SystemExit(f"{BUNDLE} has no index.html: run `pnpm build` in frontend/ first")

a = Analysis(
    [str(ROOT / "src" / "chemometrics_workbench" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    # server.py looks for the bundle at sys._MEIPASS/frontend/dist.
    datas=[(str(BUNDLE), "frontend/dist"), *collect_data_files("chemometrics_workbench")],
    # uvicorn picks its loop and protocol implementations by name at runtime.
    hiddenimports=collect_submodules("uvicorn"),
    excludes=["sklearn", "chemotools", "tkinter", "matplotlib", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="ChemometricsWorkbench",
    # A console window is how 1.0 says "close this to stop the workbench".
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ChemometricsWorkbench")
