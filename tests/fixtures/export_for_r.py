"""Write the exact matrices the R reference reads, and nothing else.

The point of handing R a file rather than letting it load the datasets itself
is that a reference value must differ from ours because the *algorithm*
differs, never because two readers disagreed about a decimal comma. The
matrices written here are the ones `generate_reference_values.py` computes
against: centred X for PCA, centred X and centred y for PLS, at full double
precision.

    uv run python tests/fixtures/export_for_r.py <directory>

It is a step in the #24 workflow, not part of a normal fixture regeneration -
see `r_mdatools_reference.R` beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from chemometrics_workbench.datasets import load_corn, load_gasoline, load_tecator

TARGETS = {"corn": "moisture", "gasoline": "octane", "tecator": "fat"}
LOADERS = {"corn": load_corn, "gasoline": load_gasoline, "tecator": load_tecator}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/r-reference")
    out.mkdir(parents=True, exist_ok=True)

    for name, target in TARGETS.items():
        dataset = LOADERS[name]()
        spectra = np.asarray(dataset.spectra, dtype=float)
        y = np.asarray(dataset.targets[target], dtype=float)

        # Centred here, and R is told center = FALSE, because pca.md section 2
        # makes centring a pipeline step. mdatools centres by default and would
        # otherwise centre an already-centred matrix.
        centred = spectra - spectra.mean(axis=0)
        y_centred = y - y.mean()

        np.savetxt(out / f"{name}.x.tsv", centred, delimiter="\t", fmt="%.17g")
        np.savetxt(out / f"{name}.y.tsv", y_centred, delimiter="\t", fmt="%.17g")
        print(f"{name}: {centred.shape[0]} x {centred.shape[1]}, target {target}")

    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
