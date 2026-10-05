# A calibration with PLS

This example builds a model that predicts fat content from a spectrum. It covers:

- validating the model honestly with cross-validation;
- choosing the number of components from the RMSECV curve;
- reading VIP and the coefficients;
- exporting the model and running it without the workbench;
- comparing two preprocessing choices step by step.

It continues from **[exploring a dataset with PCA](pca.md)**, in the same project. Every number on
this page is checked by the test suite against what the workbench computes
(`tests/test_examples.py`).

## 1. A branch for the calibration

The PCA and the calibration share their preprocessing, SNV then the Savitzky–Golay derivative, so
the calibration is a branch off the same node rather than a second pipeline.

Click **Pipeline**. Drag from the right-hand port of **SG d1 w11** onto an empty part of the canvas.
A menu of steps opens, and the step you pick is attached to the node you dragged from. Add three
steps this way, each dragged from the one before:

1. **K-fold 10**: ten-fold cross-validation, shuffled with seed 42. The seed is on the node, so the
   folds are reproducible and recorded.
2. **Mean centre**. It goes **below** the split, so the centring is refitted inside every fold. A
   mean computed from all 240 samples would leak the held-out samples into the model and make the
   validation look better than it is.
3. **PLS 5 LV**.

![Adding a step from a node's port](../images/screens/ex-branch-menu.png)

Click **Save**. Then double-click the new PLS node in the outline. In the inspector, set **Target**
to `fat` and **N Components** to `10`, and click **Apply and re-run**.

![The pipeline with its calibration branch](../images/screens/ex-pls-pipeline.png)

## 2. How many components

Open the PLS node's results. The **RMSECV** panel is the cross-validation error for each number of
components. RMSECV falls from 4.86 with one component to 2.19 with ten, and it falls less with each
component added.

![The RMSECV curve over ten components](../images/screens/ex-rmsecv.png)

The lowest value is not the right choice by itself. Later components fit smaller and smaller
effects, some of them noise particular to these 240 samples. A common rule is to take the fewest
components whose RMSECV is within one standard deviation of the lowest. **Calibration metrics**
gives that deviation as the **RMSECV spread** across the ten folds, a spread of 0.32. So look for
the first component count below 2.51. That is **4 components**.

Set **N Components** to `4` and click **Apply and re-run**. The header and **Calibration metrics**
now read:

| | |
| --- | --- |
| Cross-validated | RMSECV **2.46**, Q² **0.970** |
| Calibration | RMSEC **2.37**, R² **0.973** |

RMSEC is measured on the samples the model was fitted to, and RMSECV on samples it had not seen.
The model itself is fitted on every sample; the folds are only how its error is estimated.
The two are close, which is what a model that is not overfitted looks like. Both are in the units
of fat, percent by weight. **Predicted vs measured** plots each sample against the 1:1 line, with
the held-out fold marked.

## 3. Which wavelengths matter

**Variable importance** shows VIP for each wavelength. By construction the mean squared VIP is 1,
so a VIP above 1 marks a wavelength that contributes more than its share. Here the strongest is
near 941 nm, in the C–H absorption region where fat absorbs.

![The PLS results](../images/screens/ex-pls.png)

The same panel offers **Coefficients, raw axis**, the model written as one coefficient per raw
wavelength. On this chain it declines and says why. SNV divides each spectrum by its own standard
deviation, so it cannot be folded into fixed coefficients and has to be recomputed for each new
spectrum. That is also why the export below carries SNV as a step rather than folding it away.

## 4. Export, and predict without the workbench

On the PLS results, click **Save model** to keep it in the project's registry. Then click
**Export Python**. The browser saves `pls_predict.py`, one file that needs nothing but NumPy.

```python
import numpy as np
from pls_predict import predict

spectra = np.genfromtxt("tecator.csv", delimiter=",", skip_header=1, usecols=range(1, 101))
print(predict(spectra)[:5].round(2))
```

Run it next to the downloaded `tecator.csv`. The predictions match the workbench's own within the
tolerance [the export format](../model-export.md) states and CI tests. **Export JSON** saves the
same model as data, for a language other than Python.

## 5. Comparing two preprocessing choices

Every run is kept under **Experiments** in the outline. To see what the derivative's window does,
double-click **SG d1 w11**, set **Window Length** to `15`, and click **Apply and re-run**. The node
now reads **SG d1 w15**, and the calibration re-runs to RMSECV 2.44.

Open the newest run in **Experiments** and choose the previous one in **Compare with…**. The
comparison walks both pipelines step by step and names exactly what differs: the Savitzky–Golay
node's `window_length`, 11 in one and 15 in the other. Nothing else changed, so the difference in
RMSECV belongs to that one parameter.

![Two runs compared step by step](../images/screens/ex-compare.png)

That is the question a notebook makes hard to answer, *what exactly is different between these two
models*, answered from the record rather than from memory.
