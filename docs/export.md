# Model export

A fitted PLS model leaves the workbench in three forms:

| Form | What it is | Needs |
| --- | --- | --- |
| **Model artifact** (`.cwmodel`) | The complete model: fitted parameters, the preprocessing chain, the provenance record. | This application, or Python with NumPy (it is a zip of JSON and `.npy` files). |
| **JSON model** | The preprocessing to re-run, plus coefficients, intercept and scaling, in a documented schema. | Anything that reads JSON. |
| **Prediction snippet** (`.py`) | One Python file with a `predict` function. | NumPy only. |

The JSON model and the snippet reproduce the application's predictions within a relative
tolerance of `1e-4`, and CI tests that on every change. [The export format](model-export.md)
explains where that number comes from, and which chains can be exported. A chain containing a
baseline correction is refused by name rather than approximated.

## The model artifact

On a PLS results tab, click **Save model**. The model appears under **Models** in the project
outline, and its tab shows the file it was written to: `models/<id>.cwmodel` inside the project
folder (see [Install](install.md) for where that is). Copy that file to take the model elsewhere.
[The artifact format](model-artifact.md) specifies it.

## The JSON model and the snippet

On a PLS or PLS-DA results tab, click **Export JSON** or **Export Python**. Your browser saves
`<node>_model.json` or `<node>_predict.py`. If the chain cannot be exported, a baseline correction
for example, the sentence saying why appears beside the button instead.

The snippet runs anywhere NumPy does:

```python
import numpy as np
from pls_predict import predict

y = predict(np.loadtxt("new_spectra.csv", delimiter=","))
```

The rows must have the same variables, in the same order, as the calibration data.
