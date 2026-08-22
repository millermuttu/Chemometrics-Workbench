"""Regenerate the measurements in `0001-chemotools.md`.

    CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run --with chemotools==0.4.3 python \
        docs/decisions/0001-chemotools-evidence.py

`chemotools` is deliberately **not** a dependency of this project — that is what
the decision record decides — so this script is run with `uv run --with`, which
installs it into a throwaway environment and leaves `pyproject.toml` alone. It
is committed because a decision whose numbers cannot be re-derived is a
preference with a table attached.

It asserts nothing. Every number it prints is quoted in the record beside the
reading it was given, and a rerun that disagrees is a finding about the record.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable

import numpy as np

# chemotools 0.4.3 emits FutureWarnings for three modules that have moved and
# SyntaxWarnings from docstrings in its own source. Both are quoted in the
# record as maintenance evidence; silenced here so the numbers are readable.
warnings.filterwarnings("ignore")

from chemotools.baseline import (  # noqa: E402
    AsLs,
    PolynomialCorrection,
    RubberbandCorrection,
)
from chemotools.derivative import SavitzkyGolay  # noqa: E402
from chemotools.outliers import HotellingT2, QResiduals  # noqa: E402
from chemotools.scale import NormScaler  # noqa: E402
from chemotools.scatter import (  # noqa: E402
    MultiplicativeScatterCorrection,
    StandardNormalVariate,
)
from sklearn.decomposition import PCA as SklearnPCA  # noqa: E402

from chemometrics_workbench.datasets import (  # noqa: E402
    load_corn,
    load_gasoline,
    load_tecator,
)
from chemometrics_workbench.decomposition import PCA  # noqa: E402
from chemometrics_workbench.preprocessing import (  # noqa: E402
    BaselineCorrectTransformer,
    MSCTransformer,
    NormaliseTransformer,
    SavitzkyGolayTransformer,
    SNVTransformer,
)

LOADERS = {"corn": load_corn, "gasoline": load_gasoline, "tecator": load_tecator}

# The same block the #9 and #10 preprocessing references were generated on. A
# window of 5 leaves four of its eight columns as edge columns, which is what
# makes the Savitzky-Golay comparison a test of the edge convention.
BLOCK = (slice(0, 5), slice(0, 8))
SAVGOL_WINDOW, SAVGOL_POLYORDER = 5, 2
N_COMPONENTS = 5


def difference(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    """Largest absolute difference, and the same relative to the reference's scale."""
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    largest = float(np.abs(ours - theirs).max())
    scale = max(float(np.abs(theirs).max()), np.finfo(np.float64).tiny)
    return largest, largest / scale


def report(label: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    absolute, relative = difference(ours, theirs)
    print(f"  {label:<46} abs {absolute:.3e}   relative {relative:.3e}")


def heading(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def refused(label: str, call: Callable[[], object]) -> None:
    """Print what a kernel does with input it should refuse."""
    try:
        result = call()
        values = np.unique(np.asarray(result, dtype=np.float64))[:3]
        print(f"  {label:<12} accepted it and returned {values}")
    except Exception as error:
        message = str(error).replace("\n", " ")[:100]
        print(f"  {label:<12} {type(error).__name__}: {message}")


blocks = {name: loader().spectra[BLOCK] for name, loader in LOADERS.items()}
spectra = {name: loader().spectra for name, loader in LOADERS.items()}


heading("1. SNV")
for name, block in blocks.items():
    theirs = StandardNormalVariate().fit_transform(block)
    report(f"{name}: ours ddof=1 (our default)", SNVTransformer(1).fit_transform(block), theirs)
    report(
        f"{name}: ours ddof=0 (their convention)", SNVTransformer(0).fit_transform(block), theirs
    )


heading("2. MSC")
for name, block in blocks.items():
    for reference in ("mean", "median"):
        ours = MSCTransformer(reference).fit_transform(block)  # type: ignore[arg-type]
        theirs = MultiplicativeScatterCorrection(method=reference).fit_transform(block)
        report(f"{name}: reference={reference}", ours, theirs)

print()
print("  Does the fitted reference travel to a held-out block, or is it re-estimated?")
for name, full in spectra.items():
    calibration, held_out = full[:20], full[20:30]
    ours_fitted = MSCTransformer("mean").fit(calibration)
    theirs_fitted = MultiplicativeScatterCorrection().fit(calibration)
    ours_held = ours_fitted.transform(held_out)
    theirs_held = theirs_fitted.transform(held_out)
    ours_refit = MSCTransformer("mean").fit(held_out).transform(held_out)
    theirs_refit = MultiplicativeScatterCorrection().fit(held_out).transform(held_out)
    print(
        f"  {name}: refitting on the new block changes the result -- "
        f"ours {not np.allclose(ours_held, ours_refit)}, "
        f"chemotools {not np.allclose(theirs_held, theirs_refit)}"
    )
    report(f"{name}: held-out block", ours_held, theirs_held)


heading("3. Savitzky-Golay derivatives")
print(f"  chemotools signature: {inspect.signature(SavitzkyGolay.__init__)}")
print()
for name, block in blocks.items():
    for deriv in (0, 1, 2):
        ours = SavitzkyGolayTransformer(
            window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, deriv=deriv
        ).fit_transform(block)
        theirs = SavitzkyGolay(
            window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, deriv=deriv, mode="interp"
        ).fit_transform(block)
        report(f"{name}: deriv={deriv}, their mode='interp'", ours, theirs)

print()
print("  Their default mode is 'nearest'; ours fixes 'interp'. Full spectra, window 11:")
for name, full in spectra.items():
    for deriv in (1, 2):
        ours = SavitzkyGolayTransformer(window_length=11, polyorder=2, deriv=deriv).fit_transform(
            full
        )
        theirs = SavitzkyGolay(window_length=11, polyorder=2, deriv=deriv).fit_transform(full)
        report(f"{name}: deriv={deriv}, whole spectrum", ours, theirs)
        report(f"{name}: deriv={deriv}, interior only", ours[:, 5:-5], theirs[:, 5:-5])


heading("4. Baselines")
for name, full in spectra.items():
    block = full[:5]
    report(
        f"{name}: AsLS lam=1e5 p=0.01",
        BaselineCorrectTransformer("asls", lam=1e5, p=0.01, max_iter=20).fit_transform(block),
        AsLs(lam=1e5, penalty=0.01, nr_iterations=20).fit_transform(block),
    )
    report(
        f"{name}: rubberband",
        BaselineCorrectTransformer("rubberband").fit_transform(block),
        RubberbandCorrection().fit_transform(block),
    )
    report(
        f"{name}: polynomial order=2",
        BaselineCorrectTransformer("polynomial", order=2).fit_transform(block),
        PolynomialCorrection(order=2).fit_transform(block),
    )
print()
for cls in (AsLs, PolynomialCorrection, RubberbandCorrection):
    print(f"  {cls.__name__}{inspect.signature(cls.__init__)}")


heading("5. Normalisation")
for name, block in blocks.items():
    for norm, order in (("l1", 1), ("l2", 2)):
        ours = NormaliseTransformer(norm).fit_transform(block)  # type: ignore[arg-type]
        report(f"{name}: norm={norm}", ours, NormScaler(l_norm=order).fit_transform(block))
print()
refused("chemotools", lambda: NormScaler(l_norm=np.inf).fit_transform(blocks["corn"]))
print("  ^ our 'max' and 'area' norms have no equivalent: l_norm must be an integer >= 1.")


heading("6. Contract behaviour")
block = blocks["corn"]
as_float32 = block.astype(np.float32)
print(
    f"  float32 in -> ours {SNVTransformer().fit_transform(as_float32).dtype}, "
    f"chemotools {StandardNormalVariate().fit_transform(as_float32).dtype}"
)

untouched = block.copy()
StandardNormalVariate().fit_transform(block)
SNVTransformer().fit_transform(block)
print(f"  the caller's array is still intact: {np.array_equal(block, untouched)}")

with_nan = block.copy()
with_nan[1, 2] = np.nan
print("  a missing value:")
refused("ours", lambda: SNVTransformer().fit_transform(with_nan))
refused("chemotools", lambda: StandardNormalVariate().fit_transform(with_nan))

print("  a narrower matrix at transform:")
ours_fitted, theirs_fitted = SNVTransformer().fit(block), StandardNormalVariate().fit(block)
refused("ours", lambda: ours_fitted.transform(block[:, :4]))
refused("chemotools", lambda: theirs_fitted.transform(block[:, :4]))

constant = np.full((3, 8), 0.7)
print("  a constant spectrum, whose standard deviation is 1.16e-16 rather than 0.0:")
refused("ours", lambda: SNVTransformer().fit_transform(constant))
refused("chemotools", lambda: StandardNormalVariate().fit_transform(constant))


heading("7. Outside this decision: the limits #11 is blocked on (see #28)")
for name, full in spectra.items():
    centred = full - full.mean(axis=0)
    ours = PCA(N_COMPONENTS).fit(centred)
    theirs_model = SklearnPCA(n_components=N_COMPONENTS, svd_solver="full").fit(centred)
    n = centred.shape[0]

    t2 = HotellingT2(theirs_model, confidence=0.95).fit(centred)
    q = QResiduals(theirs_model, confidence=0.95, method="jackson-mudholkar").fit(centred)
    our_spe_limit = ours.spe_limit(0.05)
    their_spe_limit = float(q.critical_value_)
    our_t2_limit = ours.hotelling_t2_limit(0.05, "new")
    their_t2_limit = float(t2.critical_value_)

    print(f"  {name}:")
    print(
        f"    SPE limit  ours {our_spe_limit:.17g}  theirs {their_spe_limit:.17g}  "
        f"relative {abs(our_spe_limit - their_spe_limit) / their_spe_limit:.3e}"
    )
    print(
        f"    T2 limit   ours {our_t2_limit:.10g} (new-sample F form)  "
        f"theirs {their_t2_limit:.10g}  ratio {our_t2_limit / their_t2_limit:.6f}  "
        f"(n+1)/n = {(n + 1) / n:.6f}"
    )
    print(f"    ours calibration (beta) form: {ours.hotelling_t2_limit(0.05, 'calibration'):.10g}")
    for method in ("chi-square", "percentile"):
        variant = QResiduals(theirs_model, confidence=0.95, method=method).fit(centred)
        print(f"    their Q limit, method={method:<11} {float(variant.critical_value_):.6e}")
