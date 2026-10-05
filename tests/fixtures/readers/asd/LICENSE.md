# ASD sample files — CC-BY-4.0

From **Eaton Fire Ash Mapping and Analysis: Field Spectroscopy** by Mark
Wronkiewicz, Ceth W. Parker, Francisco Ochoa, Red Willow Coleman, Isaac N.
Aguilar, Gregory S. Okin, K. Dana Chadwick and Philip G. Brodrick, Zenodo
(2025), <https://doi.org/10.5281/zenodo.16538735>, licensed **CC-BY-4.0**
(<https://creativecommons.org/licenses/by/4.0/>). Decision 0006 names it.

`SP_00019.asd`, `SP_00006.asd`, `SP_00064.asd` and `SP_00052.asd` are copied
unchanged from `raw_asd_data_20250728.zip` (`batch01_Spectral9998/`): ASD
FieldSpec 4, file version 7, reflectance with the white reference stored,
2151 channels from 350 to 2500 nm.

`measurements.csv` holds four rows of the record's
`Eaton Fire Ash Sampling - Spectroscopy Measurements 20250728.csv` (samples
JPL02 to JPL05), unchanged, with the `asd_file_name` column added from its
`Spectroscopy Metadata` file. The record processed these with Terraspec: the
stored ratio times the white panel's calibration curve, with the VNIR detector
spliced to the SWIR. That is what the tests factor out.
