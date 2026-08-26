# Reader fixtures

Real data, small files. Every one is a slice of the committed Tecator dataset —
the first 8 samples and the first 12 of its 100 NIT absorbance channels, with
`fat` and `moisture` taken from the same rows — written out in the layouts
`PROPOSAL.md` §6 names. Publishing a result from Tecator obliges you to name the
instrument and company; see `src/chemometrics_workbench/data/tecator/README.md`.

| File | What it exercises |
| --- | --- |
| `tecator_subset.csv` | Comma delimiter, decimal point, identifier column, wavelength headers, one target |
| `tecator_subset_eu.csv` | Semicolon delimiter, **decimal comma**, a text metadata column, two targets |
| `tecator_transposed.csv` | Samples in columns, the axis down the first column |
| `spectra_whitespace.txt` | No header, whitespace delimited — what an instrument dumps |

The numbers in all four are identical to `load_tecator().spectra[:8, :12]` to
four decimal places, which is what lets one test assert that the same data
read through two different layouts comes back the same.
