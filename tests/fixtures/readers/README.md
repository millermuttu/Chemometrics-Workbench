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
| `tecator_subset.xlsx` | A workbook: a merged title row over a blank row, a second sheet, typed numbers |
| `tecator_subset.jdx` | A JCAMP-DX LINK block of eight spectra, `(X++(Y..Y))` with a `##YFACTOR` |
| `compressed_forms.jdx` | Hand-written, five ordinates: SQZ, DIF and DUP in one file |

The numbers in all six are identical to `load_tecator().spectra[:8, :12]` to
four decimal places, which is what lets one test assert that the same data
read through two different layouts — and through two different readers — comes
back the same. The workbook and the JCAMP file keep their wavelengths as full
floats where the CSV prints them to six significant digits, which is the only
difference between them.

`compressed_forms.jdx` is the exception: it is not Tecator data. It is five
ordinates chosen so that SQZ, DIF and DUP can each be asserted against a number
written out in the test — `A00 J K` then `A03 J T` is 100, 101, 103, 104, 104.
