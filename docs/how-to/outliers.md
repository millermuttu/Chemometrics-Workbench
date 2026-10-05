# Find and exclude outliers

The workbench flags samples that a model fits badly or that sit far from the rest. It never
removes them for you. This page shows how to read the flags and how to exclude a sample while
keeping the original data.

It continues from **[a calibration with PLS](../examples/pls.md)**, on the **PLS 4 LV** node that
example ends with. Every number on this page is checked by the test suite (`tests/test_examples.py`).
[Outlier diagnostics](../algorithms/outliers.md) defines each rule.

## 1. Read the flags

Open the PLS node's results and scroll to the **Outliers** row. It has three panels:

- **Influence**: Hotelling's T² against Q, the distance within the model against the distance
  from it, with a limit line on each axis.
- **Leverage vs residual**: how far each sample is from the centre of the scores, against how far
  its reference value is from the model's prediction.
- **Flagged samples**: every sample that breaks at least one rule, with the rules it breaks.

Hover over a point to see which sample it is.

**Flagged samples** reads **84 of 240**. The model on screen was fitted on the 240 samples, every
one: below a split, cross-validation measures the error and the model you see is refitted on every
sample. 67 of the 84 are flagged only by
**robust distance**. Robust distance measures each sample against the most tightly clustered part
of the data, and Tecator's fat content runs from 0.9% to 58.5%, so the samples at either end are far
from that core without being wrong. One rule on its own is a reason to look at a sample, not to
remove it.

Look for the samples that break several rules. Three break four each:

| Sample | Rules |
| --- | --- |
| **C018** | T² · Q · leverage · robust distance |
| **C022** | T² · Q · leverage · robust distance |
| **E1007** | T² · leverage · residual · robust distance |

C018 and C022 are extreme in the spectra: far out within the model (T², leverage) and badly
described by it (Q). E1007 is extreme in the spectra and also has a large **residual**: its
measured fat does not match what its spectrum predicts.

## 2. Decide

A flag says a sample is unusual. Whether it is wrong is a question about the measurement, which
only you can answer. Check the lab record for a transcription error or a sample that was handled
differently, and look at the spectrum. Exclude a sample when you have found a reason it does not
belong. Do not exclude it because the RMSECV goes down: removing the samples a model fits worst
always lowers its error, whether or not they were mistakes.

## 3. Exclude

Tick the three samples in **Flagged samples** and click **Exclude 3 and rerun**. The workbench:

1. writes a new version of the dataset without those three rows. The original version stays as it
   was, and the new one records which rows were left out and which version it came from;
2. moves the pipeline onto the new version and runs it.

The dataset now has 237 samples, and the PLS's RMSECV goes from 2.46 to **2.38**. To see what changed between
the two runs, open the newest run in the outline and pick the earlier one under **Compare with**.
The comparison marks the source node, because its dataset version is now different.

To undo it, double-click the new version in the outline and click **Restore v1**. Nothing was
deleted.
