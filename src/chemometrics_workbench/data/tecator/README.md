# Tecator meat

**Status: committed.** The only one of the three reference datasets this
repository carries, because it is the only one whose terms permit it.

## The permission, quoted from the file itself

`tecator.txt` opens with the statement below, and the committed file is the
copy of it. Redistribution is conditional on the note travelling with the data,
so **the note must not be stripped from `tecator.txt`** — reformatting or
trimming that header would remove the thing the licence depends on.

> These data are recorded on a Tecator Infratec Food and Feed Analyzer working
> in the wavelength range 850 - 1050 nm by the Near Infrared Transmission (NIT)
> principle. Each sample contains finely chopped pure meat with different
> moisture, fat and protein contents.
>
> If results from these data are used in a publication we want you to mention
> the instrument and company name (Tecator) in the publication. In addition,
> please send a preprint of your article to
>
>     Karin Thente, Tecator AB,
>     Box 70, S-263 21 Hoganas, Sweden
>
> The data are available in the public domain with no responsability from the
> original data source. The data can be redistributed as long as this
> permission note is attached.
>
> For more information about the instrument - call Perstorp Analytical's
> representative in your area.

## The obligation this puts on a published result

**If you publish a result from this dataset you must name the instrument and
company (Tecator).** That is a condition of use rather than a courtesy, and it
applies to `docs/parity-report.md` as much as to a paper.

It is stated here because this file is where the terms live. `datasets.py`
repeats it on `load_tecator` for whoever is reading the loader.

| | |
| --- | --- |
| Source | StatLib, <http://lib.stat.cmu.edu/datasets/tecator> |
| File | `tecator.txt`, committed |
| SHA-256 | `e435c0538cd706473d87525da595d75dbff43a58bf2694104bd948081c3790d7` |
| Checked | 2026-09-05 — host and path resolve; the page itself refuses a direct fetch |
| Loader | `chemometrics_workbench.datasets.load_tecator()` |

The checksum is verified on every read, the same as for the two downloaded
datasets. It is pinned in `datasets.py` as `TECATOR_SHA256` and repeated in
`SHA256SUMS` beside this file.

**The file is stored with LF endings and the hash is of those bytes.** A
checkout that rewrites them to CRLF changes every line and therefore the
digest, and the loader refuses the file — on Windows only, which is what made
it expensive to find. `.gitattributes` marks `*.txt` as `-text` to stop git
rewriting them, and `test_the_committed_data_is_checked_out_byte_for_byte`
asserts it. The check was not relaxed: a checksum that tolerates a
transformation is not checking anything.

## What the loader returns

240 meat samples, 100 near-infrared transmission absorbance channels over
850–1050 nm, with reference moisture, fat and protein in percent.

**The wavelength axis is reconstructed, not read.** The file gives a range and
a channel count and carries no axis vector, so `load_tecator` builds
`linspace(850, 1050, 100)`. If a reference value ever disagrees at the fourth
digit on Tecator alone, suspect the axis before suspecting the kernel — corn
and gasoline both read theirs from the file.

**The 22 principal components in the file are discarded.** They are the
original authors' preprocessing rather than raw data, and a kernel compared
against them would be compared against someone else's decomposition.

The 240 samples appear in five contiguous groups — C (129), M (43), T (43),
E1 (8) and E2 (17) — which the loader encodes into sample ids so the published
split survives any later reordering of rows.

## Reference

Tecator Infratec Food and Feed Analyzer, Tecator AB (later Perstorp
Analytical). Distributed through StatLib.
