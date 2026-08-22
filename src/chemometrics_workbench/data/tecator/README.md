# Gasoline / octane

**Status: NOT committed. Downloaded on demand and checksum-verified.**

Two reasons, either of which is sufficient.

The data originates with Kalivas (1997), published by Elsevier, and no
licence or public-domain declaration accompanies it. The R packages that
carry it thank Prof. Kalivas for making it available — an acknowledgement,
not a grant.

The copy this loader uses travels inside the R package `pls`, which is
licensed **GPL-2**. This repository is MIT. Committing a GPL-2 file into it
would put part of an MIT distribution under a copyleft licence, which is a
licensing decision this project has not made and does not need to make.
Downloading avoids the question entirely: the user fetches GPL-2 material
from its own distributor, under its own terms.

| | |
| --- | --- |
| Download URL | <https://cran.r-project.org/src/contrib/Archive/pls/pls_2.8-5.tar.gz> |
| Archive SHA-256 | `8029018d4c8921fa4c7ec5081551afdcc55d53271d9920db828483b442a033cf` |
| Member `pls/data/gasoline.RData` SHA-256 | `fdfd17ff9a407d9ee04d0c966aab3e4f2e0390c39724cab99334c74b842acbdd` |
| Checked | 2026-08-22 |
| Loader | `chemometrics_workbench.datasets.load_gasoline()` |

**The URL points at CRAN's `Archive/`, deliberately.** `src/contrib/` holds
only the current release, so a pinned hash there breaks the day CRAN
publishes a new version. Archive URLs are permanent. Version 2.8-5 is pinned
because the gasoline data has not changed across `pls` releases; a newer
`pls` would give the same numbers and a different archive hash.

## Reading `.RData` from Python

Requires the `rdata` package, which is in the `dev` dependency group rather
than the runtime dependencies — the application does not read R files, only
the parity work does. `load_gasoline()` imports it lazily and says so if it
is missing.

`rdata`'s default conversion fails on this file: the `gasoline` object is an
R data frame whose `NIR` column is a 60 × 401 `AsIs` matrix, and pandas
refuses a two-dimensional column. The loader converts with an empty
`constructor_dict` instead, which yields the raw arrays and skips the data
frame constructor.

## Cache location

`$CHEMOMETRICS_DATA_HOME`, or `$XDG_CACHE_HOME/chemometrics-workbench/datasets`,
or `~/.cache/chemometrics-workbench/datasets`. Delete the file to force a
re-download.

## What the loader returns

60 gasoline samples × 401 wavelengths, diffuse reflectance as log(1/R), from
900 nm to 1700 nm in 2 nm steps. The axis is parsed from the matrix's own
dimnames (`"900 nm"`, `"902 nm"`, …), not reconstructed.

The single target is `octane`, ranging 83.4 to 89.6.

## Reference

Kalivas, John H. (1997), "Two data sets of near infrared spectra",
*Chemometrics and Intelligent Laboratory Systems* 37, 255–259.
