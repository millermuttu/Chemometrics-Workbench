# NIR corn

**Status: NOT committed. Downloaded on demand and checksum-verified.**

Eigenvector Research publish this dataset for download, and the entry credits
the permission that made that possible — "The data was originally taken at
Cargill. Many thanks for Mike Blackburn for letting us distribute it." That
is a permission granted to Eigenvector to distribute it, not a licence
granting anyone else the same right, and the page states no terms of use for
the datasets it offers. Redistribution rights could not be established, so
this repository does not carry the file.

The loader downloads it on first use and refuses to proceed unless the bytes
hash to the value pinned below.

| | |
| --- | --- |
| Download URL | <https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip> |
| Archive SHA-256 | `8a2d1a03648b6ad334caaafa5d8377bf945ba1477a5082c4963d705d07cca795` |
| Member `corn.mat` SHA-256 | `e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5` |
| Checked | 2026-08-22 |
| Source page | <https://eigenvector.com/resources/data-sets/> |
| Loader | `chemometrics_workbench.datasets.load_corn(instrument=...)` |

The archive also contains a `__MACOSX/._corn.mat` resource fork. It is
ignored; only `corn.mat` is read, and it is hashed separately so that a
repackaged archive with identical contents is still recognised.

## Cache location

`$CHEMOMETRICS_DATA_HOME`, or `$XDG_CACHE_HOME/chemometrics-workbench/datasets`,
or `~/.cache/chemometrics-workbench/datasets`. Delete the file to force a
re-download.

## What the loader returns

80 corn samples measured on three NIR spectrometers — `m5`, `mp5` and `mp6`
— over 1100–2498 nm at 2 nm intervals, giving 700 channels. Pick one with the
`instrument` argument; `m5` is the default because it is the instrument most
parity results in the literature are quoted against.

The wavelength axis is read from the file's own `axisscale` field, not
reconstructed.

Targets are moisture, oil, protein and starch, in percent, and are the same
for all three instruments — the same 80 samples were measured on each. That
is the point of the dataset: it is the standard benchmark for calibration
transfer between instruments.

`corn.mat` also holds NBS glass standards (`m5nbs`, `mp5nbs`, `mp6nbs`) used
for instrument standardisation. The loader does not expose them yet; add a
loader when a calibration-transfer feature needs them.

## Reference

Eigenvector Research, Inc., "Corn", data originally from Cargill.
