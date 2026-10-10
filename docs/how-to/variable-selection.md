# Select variables for a PLS model

A PLS model uses every variable it is given. Selection keeps only some of them. That can make a
model simpler, cheaper to measure, or easier to explain, and sometimes more accurate. This page
runs the three selection methods the workbench has on one model, applies each, and shows how to
read the result without fooling yourself.

It continues from **[a calibration with PLS](../examples/pls.md)**, on the **PLS 4 LV** node that
example ends with. Its RMSECV is **2.46** on all 100 wavelengths. Every number on this page is
checked by the test suite (`tests/test_examples.py`).
[Variable selection](../algorithms/variable-selection.md) defines each method.

## How a selection is applied

Every method below ends the same way. **Apply selection** adds two nodes to the pipeline: a
**Select** step on the PLS's input that keeps the chosen variables, and a copy of the PLS below it.
The original PLS stays where it was, so the two can be compared. The selection is a step in the
recipe, so it is saved, versioned and exported like any other step.

Open the PLS node's results. In the **Variable importance** panel, set the view to
**Select variables**. **Select by** picks the method.

## VIP threshold

**VIP ≥** with a threshold of `1` keeps every wavelength whose VIP is at least 1, which means it
contributes more than an average share. It reads **22 of 100**. Click **Apply selection**. The copy
on those 22 wavelengths has an RMSECV of **2.56**.

VIP needs no extra computation, which makes it a fast first look. It is not a search for the
best subset, and here it does a little worse than the full spectrum.

## iPLS

**iPLS** cuts the spectrum into equal intervals, 20 by default, and fits a PLS on each. It then
adds intervals one at a time while the cross-validated error keeps falling. Click **Run iPLS**.
The plot draws one bar per interval, at its RMSECV.

The best interval is **901 to 909 nm**, five wavelengths, with an RMSECV of **2.51**. No second
interval improves on it, so iPLS keeps those five: **5 of 100**. That is slightly worse than the
2.46 of all 100 wavelengths. Five wavelengths that come within 0.05 of the full spectrum is a
useful finding if you want a cheaper instrument. If you only want the best model, keep the full
spectrum.

## CARS

**CARS** repeatedly shrinks the set of variables, keeping the ones with the largest
coefficients, and keeps the subset with the lowest RMSECV. It is random, so it runs with a fixed
seed and gives the same answer each time. Click **Run CARS** with the default 50 sampling runs.
It keeps **7 of 100**. Apply it, and the copy reads an RMSECV of **2.28**, lower than the full
spectrum's.

## Why the lower number is not to be trusted yet

CARS tried 50 subsets and kept the one with the lowest cross-validated error, using the same folds
that now score the copy. The lowest of 50 noisy estimates is lower than the true error, so 2.28
is optimistic. VIP and iPLS have the same problem in smaller measure: they also chose their
variables from every sample.

The workbench says so. After you apply any of these selections, **Validate** reports a warning on
the **Select** step that the variables were chosen on the samples that validate the model.

## Validate the selection (nested)

To get an honest error for a selection, it has to be scored on samples it never saw. **Validate
(nested)**, beside **Apply selection**, does that without a new batch: it reruns the selection
inside each of the ten outer folds, on that fold's training samples only, with an inner five-fold
cross-validation of its own, and scores each fold's selected model on the fold's held-out samples.

With **Select by** on CARS and 50 runs, click **Validate (nested)**. The note reads
**Nested RMSECV 2.385 · selected on every sample 2.285 · 10 outer × 5 inner folds**. The ten
outer folds kept between 6 and 11 wavelengths each: CARS does not pick the same ones twice. So
2.28 was optimistic by about 0.1. The honest 2.38 is still below the full spectrum's 2.46, so the
selection is worth something, just less than it first appeared.

The same check on VIP ≥ 1 gives the opposite: VIP's nested RMSECV is 2.473, below the 2.56 its
copy reported. A threshold on VIP is a much smaller search than CARS, so it has less room to
flatter itself.

## How sure is a VIP?

A VIP is computed from one model on one set of samples. **Bootstrap**, in the **Variable
importance** panel's header, refits the whole chain on 200 resamples of the samples, drawn with
replacement, and draws a 95% band around each wavelength's VIP. **21 of the 22** wavelengths with a
VIP of at least 1 have the bottom of their band at or above 1 as well, so the selection is stable.
**7** wavelengths have bands that straddle 1. Whether those are in or out depends on which samples
were measured.
