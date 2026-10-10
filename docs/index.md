# Chemometrics Workbench

An open-source, local-first workbench for spectroscopic data. You import spectra, build a
preprocessing pipeline on a canvas, and fit PCA, PLS regression or two-class PLS-DA. You can compare
runs step by step and export a model that predicts without this application. It is aimed at research
and teaching work that would otherwise need Unscrambler, SIMCA or OPUS.

Everything runs on your machine. The application starts a small server on `127.0.0.1` and opens its
screens in your browser. Nothing leaves the computer.

- **[Install](install.md)**: download a package, open it, and see the workbench.
- **[Your first PCA](first-pca.md)**: from a CSV file to a scores plot in a few minutes.
- **[The screens](screens.md)**: what each part of the window does.
- **[Model export](export.md)** and **[the HTML report](report.md)**: taking results elsewhere.
- **Worked examples** on the Tecator meat spectra: **[exploring a dataset with PCA](examples/pca.md)**
  and **[a calibration with PLS](examples/pls.md)**, with the data to download. A third,
  **[classifying meat](examples/classification.md)**, compares PLS-DA, LDA, kNN and SIMCA on
  mid-infrared spectra, and **[validating a classifier honestly](examples/validation.md)** checks
  one with a grouped cross-validation, a permutation test and a nested selection.
- **How-to**: **[find and exclude outliers](how-to/outliers.md)** and
  **[select variables](how-to/variable-selection.md)** for a PLS model.

Every number the workbench reports is compared against an independent implementation or a
published value, and the **[parity report](parity-report.md)** lists each comparison and how
closely it agrees. The **Reference** section holds the specifications the algorithms are built
from.
