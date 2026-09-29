# Exploring a dataset with PCA

This example takes the Tecator meat spectra from import to an outlier you can name. It shows how to
read the spectra, how preprocessing changes them, what a PCA's scores, loadings and explained
variance say, and how Hotelling's T² and SPE pick out samples that do not fit.

Every number on this page is checked by the test suite against what the workbench computes
(`tests/test_examples.py`). If you follow the steps, you will see the same numbers.

## The data

Download **[tecator.csv](tecator.csv)** and its **[permission note](tecator-permission.txt)**.
The data were recorded on a **Tecator Infratec Food and Feed Analyzer**. There are 240 samples of
chopped meat, each a near-infrared transmission spectrum of 100 channels from 850 to 1050 nm, with
the moisture, fat and protein content measured by chemistry. The redistribution terms require the
note to travel with the data. If you publish a result from it, name the instrument and the company,
Tecator.

Sample names carry the set the original authors assigned: `C` calibration, `M` monitoring, `T` test.

## 1. Import

In a new project, click **Import data**, then **Choose file**, and pick `tecator.csv`. Nothing is
imported yet. The preview states what it detected, and you can correct each part:

- a comma delimiter;
- samples in rows;
- 100 spectral columns, read as a wavelength axis from the header;
- `moisture`, `fat` and `protein` as targets;
- `fat_class` as metadata.

![The import preview](../images/screens/ex-import.png)

Click **Import 240 × 100**. The dataset opens as a table.

## 2. Raw spectra, and what preprocessing does to them

Click **Pipeline**. In **Build a pipeline**, add these steps in order: choose each in **Step**, then
click **Add**.

1. **SNV**, which centres and scales each spectrum by its own mean and standard deviation. It
   removes the offset and scale differences that come from path length and particle size.
2. **SG d1 w11**, a Savitzky–Golay first derivative over 11 points. It removes what is left of the
   baseline and sharpens overlapping bands.
3. **Mean centre**, which subtracts the column means. PCA describes variation about the mean.
4. **PCA**, with five components.

Click **Validate**, then **Save**, then **Run pipeline**.

![The pipeline](../images/screens/ex-pca-pipeline.png)

In the project outline, double-click **SG d1 w11**. The spectra view draws the raw spectra and the
processed ones on the same wavelength axis. **Layers** switches between them. The raw spectra
differ mostly by offset and slope. After SNV and the derivative, what is left is where they differ
in shape, around the fat and water bands between 900 and 950 nm.

![Raw and processed spectra](../images/screens/ex-spectra.png)

## 3. Scores, loadings and explained variance

Double-click **PCA 5 PC**. The header reads `240 × 100`, and says how much of the variance each
component carries:

| Component | Explained variance |
| --- | --- |
| PC1 | 79.5% |
| PC2 | 13.1% |
| PC3 | 5.8% |
| PC1 to PC5 | 99.5% |

Five components describe almost all of the preprocessed variation.

- **Scores** place each sample on the components. Samples close together have similar spectra. Use
  **Scores y axis** to look at PC3, which carries less variance but may separate something PC1
  and PC2 do not.
- **Loadings** show which wavelengths make up each component. A loading that peaks where the
  spectra differ in shape is what the scores are measuring.
- **Explained variance** is the table above, drawn with its cumulative curve.

![PCA results](../images/screens/ex-pca.png)

## 4. An outlier, found and explained

The ellipse on the scores plot is the Hotelling's T² limit at α = 0.05. A sample outside it is far
from the others *within* the model. **Diagnostics** gives the limits and counts the samples beyond
them. The T² limit of 10.9 has 18 samples above the T² limit, and 11 above the SPE limit. SPE, the
squared residual, measures how much of a spectrum the model does not describe at all. Together that
is 23 of 240 samples beyond a limit. At α = 0.05 you expect about one sample in twenty beyond each
limit by chance, so that count alone is not alarming. What matters is how far beyond a sample is.

The most extreme is **M002**, with a T² of 26.8, well over twice the limit. Click its row in the
diagnostics table. **Contributions** shows which wavelengths put it there, and they sum to its T².
Switch the contribution plot to SPE to see what the model misses in it.

![The contributions of M002](../images/screens/ex-outlier.png)

A sample beyond a limit is a question, not a verdict. It may be a measurement error, a sample
unlike the rest, or simply the edge of the population. Look at its spectrum and its reference
values before removing anything.

Next: **[a calibration with PLS](pls.md)**, on the same pipeline.
