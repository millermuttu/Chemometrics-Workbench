# Classifying meat with PLS-DA, LDA, kNN and SIMCA

This example tells three meats apart by their mid-infrared spectra. It covers:

- a cross-validation stratified by class, so that every fold holds every class;
- PLS-DA and its confusion matrices;
- three other classifiers on the same preprocessing, compared by cross-validated accuracy;
- why SIMCA's numbers do not compare with the other three.

It starts from an empty project. Every number on this page is checked by the test suite against
what the workbench computes (`tests/test_examples.py`).

## The data

Download **[meat.csv](meat.csv)**. It holds 60 samples of minced chicken, pork and turkey, 20 of
each, measured by mid-infrared ATR spectroscopy at the Quadram Institute. The original data is in the
public domain (CC0). [Where it comes from](meat-source.md) says what was changed to make
this file, and cites the study.

## 1. Import

Open the workbench and click **Import data**. Choose `meat.csv`. The preview reads it as
**60 × 448**, on a wavenumber axis from 1868 to 1005 cm⁻¹. It picks out `meat` and `supplier` as
label columns, because they hold text rather than numbers. Click **Import 60 × 448**.

![The import preview](../images/screens/ex-meat-import.png)

## 2. A stratified cross-validation and a PLS-DA

Click **Pipeline**. The canvas holds one node, **Source**, the dataset. Drag from its right-hand port onto an
empty part of the canvas, and pick a step from the menu that opens. Add three steps this way, each
dragged from the one before, then click **Save**:

1. **K-fold 10**: ten-fold cross-validation, shuffled with seed 42.
2. **Mean centre**. It goes below the split, so it is refitted in every fold.
3. **PLS-DA 5 LV**. Its class column is `meat`, the first label column. The inspector also
   offers `supplier`.

Double-click the **K-fold 10** node in the outline. In the inspector, set **Stratify By** to
`meat` and click **Apply and re-run**. Each fold then holds two samples of each meat. An unstratified
fold of six samples could hold no pork at all, and the model would never be tested on pork in that
fold.

![The classification pipeline](../images/screens/ex-meat-pipeline.png)

## 3. Reading the PLS-DA

Open the PLS-DA node's results. PLS-DA fits one dummy response per class and assigns each sample
to the class whose response is largest. The **RMSECV** panel is the cross-validated error of
those dummy responses, and its note reads **lowest at A = 5**. Five components is the default,
so leave it.

**Confusion** has one table per set. Rows are the observed class, columns the assigned one, and a
perfect table has numbers only on its diagonal. The **Cross-validated** table counts every sample
once, from the fold where it was held out. Here it is **20, 20 and 20** on the diagonal, with
nothing off it. **Calibration metrics** gives **Accuracy (CV) 1.000**.

![The PLS-DA results](../images/screens/ex-plsda.png)

## 4. Three other classifiers

Back on **Pipeline**, drag from the right-hand port of **Mean centre** to an empty part of the
canvas and pick **LDA 5 PC**. Do the same twice more for **kNN k5** and **SIMCA 3 PC**. Each new
node has the same split and centring above it as the PLS-DA. Click **Save**, then **Run pipeline**.

LDA and kNN both work on the first five principal components of the centred spectra, and both
assign every sample to exactly one class, like PLS-DA. So their cross-validated accuracies compare
directly with its:

| | Accuracy (CV) | Misassigned |
| --- | --- | --- |
| PLS-DA 5 LV | **1.000** | none |
| LDA 5 PC | **0.967** | 2 of 60 |
| kNN k5 | **0.983** | 1 of 60 |

On 60 samples, one misassignment is a difference of 0.017. That is not evidence that one of these
methods is better than another in general. It means all three separate these meats almost
perfectly.

## 5. SIMCA answers a different question

SIMCA does not pick one class for each sample. It fits a separate PCA to each class and asks each
class model, in turn, whether the sample fits. A sample can be accepted by one model, by several,
or by none. Its results show **Acceptance** in place of **Confusion**: rows are the observed class,
a ✓ column counts the samples that class model accepted, and **none** counts the samples no model
accepted.

Cross-validated, SIMCA's sensitivity is **0.750** and its specificity **0.983**. Most of the
missed sensitivity is in the **none** column: **15 of 60** samples are accepted by no model. That
is SIMCA doing its job, not failing at it. A class model built from 20 samples at a 95% acceptance
limit draws a tight boundary, and a sample outside every boundary is reported as unlike anything it
knows. A sample that is none of the three meats would be reported the same way, while PLS-DA, LDA
and kNN would each force it into one of the three. Use SIMCA when that question matters.

![SIMCA's acceptance tables](../images/screens/ex-simca.png)

## A caution about this dataset

All 20 turkey samples come from supplier E, and no chicken or pork does. A classifier that
recognised supplier E's processing rather than turkey meat would score the same on this data.
The cross-validation cannot tell those apart, because every fold has the same confounding. Only
turkey from another supplier would show which one the model learned.

The raw measurements hold two runs of every sample. [Validating a classifier
honestly](validation.md) uses them, with a cross-validation grouped by sample, a permutation test
and a nested check of a variable selection.
