# meat.csv: where it comes from

`meat.csv` is derived from `FreshMeatFTIR.csv` in the Quadram Institute's
[`QIBChemometrics/fresh-meat-ftir`](https://github.com/QIBChemometrics/fresh-meat-ftir)
repository, which is dedicated to the public domain under **CC0-1.0**. The
measurements are described in Al-Jowder O, Kemsley E K and Wilson R H (1997),
"Mid-infrared spectroscopy and authenticity problems in selected meats: a
feasibility study", *Food Chemistry* 59, 195–201,
<https://doi.org/10.1016/S0308-8146(96)00289-0>.

The source holds 120 mid-infrared ATR spectra, 448 points from 1005 to
1868 cm⁻¹. They are two acquisitions, RunA and RunB, of each of 60 samples of
minced chicken, pork and turkey, 20 of each. `meat.csv` changes it in four
ways:

1. **One row per sample.** Each sample's two runs are averaged. Two runs of one
   sample are not two samples, and a cross-validation that put one run in
   training and the other in the held-out fold would test the model on what it
   had already seen.
2. **Samples in rows**, with the axis descending, as FT-IR software writes it.
3. **Two label columns** taken from the source's column names: `meat`
   (chicken, pork, turkey) and `supplier` (A to E). `sample` is the meat's
   initial, the supplier and the sample number: `PD01` is pork from supplier D,
   sample 1.
4. **Pairs by position.** The source labels both runs of pork, supplier D,
   sample 1 `RunB`. Its columns otherwise come in RunA, RunB pairs, so the pair
   is taken by position.

All 20 turkey samples come from supplier E, and no chicken or pork does.
