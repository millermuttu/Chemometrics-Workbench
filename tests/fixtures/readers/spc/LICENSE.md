# SPC sample files

Two sources, under two licences.

## `bone_sample_1.spc` — CC-BY-4.0

Copied unchanged from **Identifying Archaeological Bone and Antler and Methods
to Soften Them Using Fourier Transform Infrared (FTIR) Spectroscopy: Test
Studies** by Grzegorz Osipowicz, Aleksandra Lisowska-Gaczorek, Mariusz Bosiak,
Ryszard Grygiel and Krzysztof Szostek, Zenodo (2025),
<https://doi.org/10.5281/zenodo.16108826>, licensed **CC-BY-4.0**
(<https://creativecommons.org/licenses/by/4.0/>). It is the file
`sample 1 (9).SPC`, renamed: a new-format, single-spectrum SPC file written by
OMNIC, 14935 points from 4000 to 400 cm-1, with absorbance stored as scaled
integers.

## `rohanisaac/` — GPL-3.0

`nir.spc`, `s_xy.spc`, `DOERNER.spc`, `m_ordz.spc`, `m_evenz.spc` and `ms.spc`
are copied unchanged from the `test_data/` folder of `rohanisaac/spc`
(<https://github.com/rohanisaac/spc>, commit
`f35b90fedb995778ad11b276582d89e233774e5e`). Each `*.txt` beside them is that
project's own text conversion of the file, from `test_data/txt/`, and is what
the tests compare against. The project is licensed **GPL-3.0**, and its licence
is committed beside the files as `COPYING`.

Decision 0006 (`docs/decisions/0006-phase-5-data-sources.md`) allows this.
These files are data shipped beside the code as test fixtures. They are never
linked into the application or packaged with it, and they carry their licence.
They cover the layouts the bone files do not: an old-format file (`DOERNER`),
an old-format multifile (`m_ordz`), a new-format multifile (`nir`, `m_evenz`),
an explicit x array (`s_xy`), and the `xyxy` layout (`ms`), which the reader
refuses.

`m_evenz.spc` has subfiles with a negative exponent. `rohanisaac/spc` reads
that exponent as zero, so its text conversion is not compared on those
subfiles (see `readers/spc.py`).
