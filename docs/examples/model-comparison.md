# Comparing models

This example continues the [PLS example](pls.md) on the same Tecator download and asks the question Phase 7 was built for: **on these spectra, which model?** It fits five more regressions beside PLS under one cross-validation, and then, on the [classification example](classification.md)'s meat set, two more classifiers beside PLS-DA. Every number quoted here is recomputed by `tests/test_examples.py` from the application itself.

## 1. One split, one centring

Import [`tecator.csv`](tecator.csv) and build the PLS example's chain: **SNV**, **SG d1 w11**, then **K-fold 10** (seed 42) and **Mean centre**. Every model below hangs off that one centring node, so they are all cross-validated on the same ten folds and their RMSECVs can be read against each other.

![The comparison's pipeline: one split, one centring, six regressions](../images/screens/ex-compare-pipeline.png)

## 2. The regressions

Drag from the centring node's port and pick each model from the **Regression** heading of the menu: **PLS 5 LV**, **Ridge α1**, **Elastic Net α1**, **SVR rbf 5 PC** and **OPLS 1+1**. A model the dataset has no target for would be listed there disabled, with the reason; Tecator has three.

Each new model models the dataset's first target, moisture: set **Target** to **fat** in its inspector, as the PLS example does. The defaults are a starting point, not an answer. Open each model's tab and read its **RMSECV** curve, which is drawn against the model's own parameter - components for PLS, the penalty α for Ridge and Elastic Net, C for SVR, orthogonal components for OPLS - and set the parameter near the curve's minimum in the inspector:

| Model | RMSECV | Q² |
| --- | --- | --- |
| PLS 5 LV | **2.39** | 0.972 |
| Ridge α 3e-5 | **2.18** | 0.977 |
| Elastic Net α 0.001 | **2.91** | 0.959 |
| SVR rbf, C 50 | **0.84** | 0.997 |
| OPLS 1+4 | **2.39** | 0.972 |
| MLR on 22 VIP variables | **2.29** | 0.975 |

![SVR's curve over C](../images/screens/ex-compare-svr.png)

What the table says:

- **OPLS 1+4 is PLS 5.** One predictive and four orthogonal components predict exactly as five PLS components ([`opls.md`](../algorithms/opls.md) §4), so the two rows agree to the last digit. OPLS is not a better predictor; it is a better *picture*, separating what predicts fat from the largest variation that does not.
- **Ridge** shrinks every direction rather than keeping five whole, and here that does a little better than PLS.
- **Elastic Net** keeps **91 of 100** variables at this α - a light selection.
- **SVR** with an RBF kernel on five principal component scores is the clear winner here, which suggests the relation of fat to these spectra is not quite linear. Its curve is still falling at the top of its grid: it is lowest at C = 57.6, and C 50 is the round number beside it.

Choosing a parameter at a curve's minimum and then quoting that same curve's minimum as the expected error is optimistic ([`metrics-and-validation.md`](../algorithms/metrics-and-validation.md) §9). For a number to publish, validate the choice too: the [validation example](validation.md) shows how.

## 3. MLR on a selection

MLR cannot be fitted on the whole spectrum - 100 collinear variables - and the canvas says so by name. Open the PLS tab, choose **Select variables** in the variable importance panel, and **Apply selection** at VIP ≥ 1: the selection becomes a `select_variables` step in the pipeline. Drag from it and add **MLR**, with its target set to fat: on those 22 variables it is the last row of the table above. The selection was chosen by looking at every sample, so the warning beside it applies: its RMSECV is optimistic in the same way.

## 4. Two more classifiers

On the meat set, under the classification example's stratified ten folds, add **Logistic 5 PC** and **QDA 5 PC** from the **Classification** heading beside PLS-DA:

| Classifier | Accuracy (CV) |
| --- | --- |
| PLS-DA 5 LV | **1.000** |
| Logistic 5 PC | **0.967** |
| QDA 5 PC | **0.967** |

Logistic regression is the first classifier that reports **class probabilities**: its tab lists every sample's assignment beside the probability of each class, so a sample assigned with 0.51 reads differently from one assigned with 0.99. QDA gives every meat its own covariance on the scores - with twenty samples a class and five components it has enough rows for each.

![Logistic regression's class probabilities](../images/screens/ex-compare-logistic.png)
