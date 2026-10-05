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
the **Select** step that the variables were chosen on the samples that validate the model. To get
an honest error for a selected model, test it on samples the selection never saw, such as a new
batch measured after the selection was made.
