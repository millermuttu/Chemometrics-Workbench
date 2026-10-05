# 0006 — Phase 5 data comes from open, licensed datasets on the internet

**Date:** 2026-10-02
**Status:** accepted
**Decides:** where Phase 5's reader fixtures, classification data and exit-run data come from (`feature_list.json`, Phase 5).

---

## Decision

**Every file Phase 5 develops, tests or demonstrates against is downloaded from a public source under a licence that allows redistribution.** No maintainer-supplied or synthetic instrument files stand in for a reader fixture, and nothing is committed without its licence and citation beside it. A synthetic set is still allowed where the method needs known ground truth (CARS's recovery test), because no real file can provide that.

Each source was checked on 2026-10-02 through the Zenodo records API or the GitHub repository's licence:

| Use | Source | Licence | What it holds |
| --- | --- | --- | --- |
| Multi-class classification, SIMCA, outliers, exit run | Quadram Institute, [`QIBChemometrics/fresh-meat-ftir`](https://github.com/QIBChemometrics/fresh-meat-ftir) | CC0-1.0 | FTIR spectra of 60 minced meats (20 each of chicken, turkey, pork) in duplicate, one CSV (648 kB). Three classes. |
| Second multi-class set | Quadram Institute, [`QIBChemometrics/FTIR-Spectra-Olive-Oils`](https://github.com/QIBChemometrics/FTIR-Spectra-Olive-Oils) | CC0-1.0 | 120 FTIR spectra of 60 extra-virgin olive oils from four regions, one CSV (820 kB). Four classes. |
| `reader-mat` | Duval, *MLNIRdata*, [10.5281/zenodo.16781223](https://doi.org/10.5281/zenodo.16781223) | CC-BY-4.0 | 208 NIR spectra of hydrocarbon mixtures plus densities, a MATLAB 5.0 MAT-file (5.3 MB) with the same data as CSV for cross-checking. |
| `reader-spc` | *Identifying Archaeological Bone and Antler…*, [10.5281/zenodo.16108826](https://doi.org/10.5281/zenodo.16108826) | CC-BY-4.0 | 32 Galactic SPC files from an FTIR instrument, about 60 kB each. |
| `reader-spa` | *FTIR test for MXene-Based Ecoflex Dry Electrodes*, [10.5281/zenodo.17691644](https://doi.org/10.5281/zenodo.17691644) | CC-BY-4.0 | OMNIC SPA files, some with the instrument's own CSV export of the same spectrum, which is the reader's cross-check. |
| `reader-spa`, second instrument | *Heavy-ion irradiation of … dust analogs*, [10.5281/zenodo.21396092](https://doi.org/10.5281/zenodo.21396092) | CC-BY-4.0 | About 80 SPA files of about 10 kB. |
| `reader-asd` | *Eaton Fire Ash Mapping and Analysis: Field Spectroscopy*, [10.5281/zenodo.16538735](https://doi.org/10.5281/zenodo.16538735) | CC-BY-4.0 | Raw ASD files of 35 kB each, plus the measurements as CSV for cross-checking. |

**Variety beyond these** — old-format SPC, multi-file SPC and `xyxy` layouts, which the bone dataset may not include — may come from [`rohanisaac/spc`](https://github.com/rohanisaac/spc)'s `test_data/`, which is GPL-3.0. That is acceptable for test fixtures in this MIT repository because they are data shipped beside the code, never linked into it or packaged, provided its licence file is committed with them. Prefer a CC-BY or CC0 file when one covers the same case.

## Where the files live

**`dataset/` at the repository root holds every file downloaded for development, and it is gitignored.** It was populated on 2026-10-02 from the sources above, about 56 MB in all:

| Folder | Contents |
| --- | --- |
| `dataset/classification/` | `FreshMeatFTIR.csv` and `FTIR_Spectra_olive_oils.csv` |
| `dataset/mat/` | The MLNIRdata MAT-file and its CSV twins |
| `dataset/spc/` | The 31 bone and antler SPC files |
| `dataset/spa/ecoflex/`, `dataset/spa/dust/` | The two SPA records, with the ecoflex CSV exports |
| `dataset/asd/` | The ASD archive, unpacked into three batches, plus the measurements CSV |
| `dataset/tecator.csv` | The stray root copy; the shipped one is `docs/examples/tecator.csv` |

The folder is a local cache, not a source of truth. CI never sees it, so nothing under `tests/` may depend on it unless it skips when the file is absent. Whatever a test or a page needs in CI is committed somewhere else:

- **Reader fixtures** are a handful of files copied from `dataset/` into `tests/fixtures/readers/<format>/`, with a `LICENSE.md` naming the source, the DOI or URL, the licence and the authors, as `tests/fixtures/readers/opus/` already does.
- **A classification set** a test or worked example needs is copied whole beside it with its licence and citation, as `docs/examples/tecator.csv` is. Both Quadram CSVs are under 1 MB.
- **The MLNIRdata MAT-file (5.3 MB)** is not committed. The committed fixture is a slice re-saved with `scipy.io.savemat` and marked as derived in its `LICENSE.md`. A test reads the original from `dataset/` when it is present.
- **Every derived value carries attribution.** A worked example or exit-run record that uses one of these files cites it.

## Consequence for the exit criterion

The two multi-class sets are CSV, not one of the new formats. The exit criterion is therefore split into two claims, each still demonstrated over HTTP:

1. The classification, outlier and selection workflow runs on the Quadram meat set.
2. A file from each new format, taken from the sources above, imports through the application and matches its source's own export.
