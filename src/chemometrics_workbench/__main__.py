"""The launcher: `python -m chemometrics_workbench`, and a packaged application's
entry point. It serves, then opens the workbench in the default browser."""

from chemometrics_workbench.server import main

main(open_browser=True)
