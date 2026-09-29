# Your first PCA

This takes a file of spectra to a scores plot. It is the same path the application's end-to-end
walkthrough test drives on every change.

You need a file with **one spectrum per row** (or per column) and a header row naming the
variables, usually the wavelengths. The workbench reads:

- delimited text: `.csv`, `.txt`, `.tsv`, `.dat`;
- Excel: `.xlsx`, `.xlsm`;
- JCAMP-DX: `.jdx`, `.dx`, `.jcm`;
- Bruker OPUS files.

To follow along with real data, use the Tecator meat NIR set: 240 samples by 100 channels, with
`fat`, `moisture` and `protein` columns.

## 1. Import

A new project says **This project is empty** and offers one action, **Import data**. Click it, then
**Choose file**.

Nothing is committed yet. The import screen shows what it detected: the delimiter, whether samples
run along rows or columns, which columns are spectra and which are targets or metadata. Correct
anything it got wrong, then click **Import *n* × *p***. The dataset opens as a table.

## 2. Build the pipeline

Click **Pipeline** in the tab bar. The canvas shows your dataset as the source node. On the right
side of the canvas, **Build a pipeline** lists the steps:

1. choose **SNV** in **Step** and click **Add**;
2. choose **SG d1 w11** (a Savitzky–Golay first derivative, window 11) and click **Add**;
3. choose **PCA** and click **Add**.

Each step appears on the canvas as you add it. Click **Validate**, which should report
`valid · 3 steps`, then **Save**. The pipeline is saved in the project, so a reload or a restart
keeps it.

## 3. Run

Click **Run pipeline**. The status bar at the bottom shows progress and a **Cancel** button, and
the window stays usable while it runs. When it says **Done**, every node on the canvas is marked
complete.

## 4. Read the result

In the project outline on the left, double-click **PCA 5 PC**. The results tab shows:

- the scores plot with its Hotelling T² ellipse;
- the loadings, against the wavelength axis;
- the explained variance per component;
- the diagnostics, T² against Q residuals.

Hover a point to name its sample. The header gives each component's share of the variance.

![PCA results on Tecator](images/screens/pca.png)

## Next

- Change a step's parameters in the inspector on the right, then click **Apply and re-run**.
- Branch the pipeline to compare two preprocessing routes. See [the screens](screens.md).
- Add a split and a **PLS 5 LV** node to fit a regression, then click **Save model** on its results.
