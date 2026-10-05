# MATLAB MAT-file sample

Derived from **MLNIRdata: 208 near-infrared spectroscopic spectra and densities
of hydrocarbon mixtures** by Laurent Duval (IFP Energies nouvelles), version
1.0.0, Zenodo, <https://doi.org/10.5281/zenodo.16781223>, licensed
**CC-BY-4.0** (<https://creativecommons.org/licenses/by/4.0/>). The data were
used in Alsouki et al., *Chemometrics and Intelligent Laboratory Systems*
(2023), <https://doi.org/10.1016/j.chemolab.2023.104813>.

**These files are changed from the original**, as CC-BY-4.0 requires us to say.
They are slices, made by `scipy.io.savemat` and `numpy.savetxt` in the
`reader-mat` work (#284), so that the fixture is 25 KB rather than 15 MB:

- `mlnir_slice.mat` holds the first 12 spectra and every 50th variable of each
  array in `MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat`. The arrays
  keep their original names and their original layout: variables down the rows
  and samples across the columns, with each axis a column vector and the
  density a row vector.
- `mlnir_slice_data.csv`, `mlnir_slice_axis.csv` and `mlnir_slice_density.csv`
  hold the same slice of the publisher's own CSV export of the same data
  (`MLNIR_matrixX_NirSpectrumData.csv`, `MLNIR_matrixX_NirSpectrumDataAxis.csv`,
  `MLNIR_matrixY_NirPropertyDensityNormalized.csv`). The test compares the
  reader's output against them.

The full files are not committed. Download them into `dataset/mat/` from the
DOI above (`docs/decisions/0006-phase-5-data-sources.md`), and the test that
reads the whole MAT-file runs as well.
