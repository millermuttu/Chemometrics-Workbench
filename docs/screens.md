# The screens

The window has three regions around a tabbed document area:

- **Project outline**, on the left, lists your datasets, the pipeline as an outline, experiments
  (every run) and saved models. Click an entry to preview it in a tab. Double-click it to keep the
  tab open.
- **Document area**, in the middle, holds the tabs: the pipeline canvas, spectra, results, a
  dataset, an experiment or a model. **Split view** puts two tabs side by side. When the tabs
  overflow the bar, a menu lists the rest.
- **Inspector**, on the right, shows whatever is selected: a node's parameters, a model's metrics,
  and the provenance record.

The **status bar** at the bottom shows the running job, its progress and **Cancel**. The theme
switch offers **Light** and **Dark**.

## Pipeline canvas

The pipeline is a graph. A source dataset branches into competing preprocessing routes, each
ending in a model, so you can compare them. Node states read at a glance: not yet run, queued,
running, complete (with its headline metric), stale because something upstream changed, or failed
(with its message).

![The pipeline canvas](images/screens/pipeline.png)

Build with the step list (**Step**, **Add**, **Validate**, **Save**), drag nodes to arrange them,
and click **Run pipeline**. Selecting a node on the canvas or in the outline opens its tab.

## Spectra

A preprocessing node opens as its spectra: raw and processed overlaid on the wavelength axis. Large
sets draw a shaded band for the full set plus a drawn subset, and the counts above the plot say how
many of each are shown. **Highlight sample** draws one spectrum over the rest and names it.
**Layers** switches between raw, processed and both.

![The spectra view](images/screens/spectra.png)

## Results

A PCA, PLS or PLS-DA node opens as a grid of plots in one tab. PCA shows scores, loadings,
explained variance and diagnostics. PLS adds predicted against measured with the 1:1 line,
residuals, the RMSECV curve, VIP and the regression coefficients, and PLS-DA shows its
classification results in place of the regression panels. The diagnostics table lists the samples
outside the limits. Click one to open its contribution plot, which shows the variables that put it
there.

![PLS results](images/screens/pls.png)

**Save model** keeps the fitted node in the project's model registry.

## Experiments

Every run is kept. An experiment tab shows **What it ran** (the pipeline as it was, parameter by
parameter), **Against what** (the dataset version) and **What it scored**. **Compare with…** picks a
second run and opens a step-by-step comparison that names exactly the parameters that differ.

## Models

A saved model opens to its record: its metrics, where it came from, and the `.cwmodel` file it is
stored in, with that file's checksum. See [Model export](export.md) for taking one elsewhere.

## Dataset

The dataset tab is the sample and variable table, with metadata and target columns and the import
summary.
