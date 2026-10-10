# Validating a classifier honestly

[Classifying meat](classification.md) used one spectrum per sample. This example uses the raw
measurements, with two spectra of every sample, and asks how far the numbers a classifier reports
can be trusted. It covers:

- a cross-validation **grouped** by sample, so that both runs of a sample are always in the same fold;
- the model that is reported and exported, which is fitted on every sample;
- a **permutation test**: is the model better than chance?
- a second classifier, an SVM, on the same folds;
- a variable selection checked by **nested** cross-validation.

It starts from an empty project. Every number on this page is checked by the test suite against
what the workbench computes (`tests/test_examples.py`).

## The data

Download **[meat-raw.csv](meat-raw.csv)**. It holds the same 60 samples of chicken, pork and
turkey as `meat.csv`, but keeps both of the instrument's runs of each sample, so it has 120 rows.
The `sample` column says which sample a row belongs to: `CA01A` and `CA01B` are runs A and B of
sample `CA01`. [Where it comes from](meat-source.md) says how the file was made.

## 1. Import

Click **Import data** and choose `meat-raw.csv`. The preview reads it as **120 × 448** on a
wavenumber axis, with four label columns: `meat`, `supplier`, `sample` and `run`. Click
**Import 120 × 448**.

## 2. A grouped cross-validation

On **Pipeline**, drag from the Source node's port and add, each from the one before,
**K-fold 10**, **Mean centre** and **PLS-DA 5 LV**, then click **Save**. The PLS-DA's class
column is `meat`.

Double-click **K-fold 10** in the outline. In the inspector, set **Group By** to `sample` and click
**Apply and re-run**.

Two runs of one sample are nearly the same spectrum. An ungrouped K-fold deals the 120 rows into
folds one by one, so most samples have one run in training and the other held out, and the model
is tested on spectra it has, in effect, already seen. A grouped K-fold deals the 60 samples
instead, so a held-out run's twin is held out with it. Grouping and stratifying are exclusive: a
split does one or the other.

![The grouped pipeline](../images/screens/ex-grouped-pipeline.png)

## 3. Reading the PLS-DA

Open the PLS-DA's results. **Calibration metrics** reads **Accuracy (CV) 0.992**: **1 of 120**
spectra is assigned to the wrong meat. The cross-validated RMSECV is **0.205**.

Is that better than the ungrouped number? On this data, hardly. Set **Group By** back to
nothing and the accuracy is 0.992 either way, with an RMSECV of 0.202 without grouping. The three
meats are far apart, so seeing a sample's twin barely helps. That is a finding about this dataset,
not a reason to skip grouping: on a harder problem, a leak like this can be the difference between
a model that works and one that only appears to. Put **Group By** back to `sample`.

**Confusion**'s **Calibration** table counts **120** spectra: the model on screen, and the one
**Save model** and the exports carry, is fitted on all of them. Cross-validation estimated the
error of the recipe; it did not pick one fold's model. The **Held out (fold 0)** table is fold zero's
12 spectra, 6 samples with both runs, assigned by the model fitted without them, and it gives
**Accuracy (held out) 1.000**.

![The grouped PLS-DA](../images/screens/ex-grouped-plsda.png)

## 4. Is it better than chance?

Scroll to **Permutation test**, leave **100** permutations and click **Run permutation test**. The
workbench shuffles the meat labels 100 times, reruns the whole cross-validation on each shuffle,
and draws the accuracies it gets, with the real model's 0.992 marked among them.
The note reads **p = 0.00990**, which is 1/101: the smallest p-value 100 permutations can give,
because none of them did as well as the real labels. To show a smaller p-value, run more
permutations.

The labels are shuffled spectrum by spectrum, not sample by sample, even under a grouped split, so
a shuffle can give a sample's two runs different meats
([the permutation test](../algorithms/metrics-and-validation.md#14-the-permutation-test) says so).

## 5. A second classifier

Back on **Pipeline**, drag from **Mean centre**'s port and add **SVM rbf 5 PC**, then **Save**
and **Run pipeline**. It is scored on the same grouped folds. SVM's **Accuracy (CV)** is
**0.942**, with 7 of 120 misassigned: worse than PLS-DA here, on the
same five principal components the SVM's kernel is taken on.

## 6. A selection, checked

Open the PLS-DA's results again. In **Variable importance**, set the view to **Select variables**.
With **VIP ≥ 1** it keeps **129 of 448** wavenumbers. Click **Validate (nested)**.

A selection made on every sample and then cross-validated on the same samples is optimistic: the
samples that score it also chose it. Nested validation reruns the selection inside each outer
fold, on that fold's training samples only, and scores it on the held-out ones. The note reads
**Nested RMSECV 0.2352 · selected on every sample 0.2285 · 10 outer × 5 inner folds**. The honest
error is the larger one. Neither beats the full spectrum's 0.205, so here VIP selection is not
worth applying.

## What to report

The grouped cross-validated accuracy, 0.992, with the permutation p-value beside it, and the
model fitted on all 120 spectra. Not the ungrouped number, and not a selection's error without
its nested check. The caution at the end of [classifying meat](classification.md) still applies:
every turkey sample comes from one supplier.
