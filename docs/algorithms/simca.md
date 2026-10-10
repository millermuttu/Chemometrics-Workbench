# SIMCA — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md), whose decomposition, $T^2$ and SPE each class model is built from and which this does not restate; [`classification.md`](classification.md) for class order; [`metrics-and-validation.md`](metrics-and-validation.md) for folds.

---

## 1. What SIMCA is, and what it is not

SIMCA (soft independent modelling of class analogy) fits **one PCA model per class** and asks each model whether a sample looks like its class. It is a *class-modelling* method, not a discriminant one. A sample may be accepted by one class, by several, or by none, and "none" is a legitimate answer: it says the sample resembles nothing the models were built from.

The classifier therefore assigns no single class, and it has no $N \times N$ confusion matrix in `classification.md`'s sense. It reports an **acceptance table** (§5).

---

## 2. Class models

The class column is read as in `classification.md` §1: classes in Unicode order. For each class $C_k$, with calibration rows $X_k$:

1. Centre by the class's own mean: $\bar{x}_k$ is the column mean of $X_k$. **Class centring is part of SIMCA, not a pipeline step.** Each class needs its own centre, and a pipeline centring node can only centre everything by one mean. A centring node above SIMCA is harmless: it shifts every class mean by the same vector.
2. Fit a PCA with $A$ components to $X_k - \bar{x}_k$, exactly as `pca.md` specifies.

The same $A$ is used for every class. A class with $A$ or fewer calibration samples cannot support $A$ components and is refused, with the class named.

---

## 3. Distances to a class

For any sample $x$ and class $k$, with $t = (x - \bar{x}_k) P_k$:

- $T^2_k$ is `pca.md` §7's Hotelling $T^2$ of $t$ on class $k$'s eigenvalues.
- $Q_k$ is `pca.md` §8's SPE of $x - \bar{x}_k$ against class $k$'s loadings.

The limits are at $\alpha = 0.05$, the application's fixed significance:

- $T^2_{\lim,k}$ is the **new-sample** (F) form of `pca.md` §7, for $n_k$ and $A$. A sample being classified is a new sample to the model, even when it is one of the model's own calibration rows. One form for every decision keeps the rule a single rule.
- $Q_{\lim,k}$ is `pca.md` §8's Jackson–Mudholkar limit. Where it is outside its domain the run says so, as a PCA's would. A class whose $A$ components leave no residual has no $Q$ limit and is refused, with the class named.

The **reduced distance** to class $k$ is

$$h_k = \max\!\left(\frac{T^2_k}{T^2_{\lim,k}}, \; \frac{Q_k}{Q_{\lim,k}}\right)$$

---

## 4. Acceptance

**Class $k$ accepts $x$ when $h_k \le 1$**: both distances are within their limits. Each sample gets one decision per class, so a sample has a row of $N$ decisions, and every combination of them is possible.

This is the classic two-limit rule. Other packages combine $T^2$ and $Q$ into one statistic with a fitted distribution; see §8.

---

## 5. Reported quantities

For each set (calibration, the cross-validated set below a split, and fold zero's held-out rows), as `pls-da.md` §6 names them:

- **The acceptance table.** An $N \times N$ count, where rows are the observed class and columns are the class model. Cell $(j, k)$ counts samples of $C_j$ accepted by model $k$. A row's counts may sum to more than the class size, or to less. Beside it, per observed class, is the number of samples **no** model accepted.
- **Per class**, for model $k$:
  - sensitivity: the fraction of $C_k$'s samples that model $k$ accepts;
  - specificity: the fraction of every other class's samples that model $k$ rejects.
  
  Either is absent when its denominator is zero (`metrics-and-validation.md` §11).
- **Pooled**, in the flat metrics with the usual suffixes:
  - `sensitivity`: the fraction of samples their own class model accepts;
  - `specificity`: the fraction of (sample, other class) pairs whose model rejects the sample.
- **The reduced distances** $h$, an $n \times N$ matrix per set. A Coomans plot draws two of its columns against each other, with the line $h = 1$ on each axis.
- Each class model's size: $n_k$, $A$, and both limits.

---

## 6. Cross-validation

As everywhere: below a split, each fold's class models are fitted on that fold's training rows of its own preprocessed array. Each held-out row is decided by its fold's models. The cross-validated acceptance table pools every fold's held-out decisions, so it holds one row of $N$ decisions per sample. The reported models are refitted on every sample (`metrics-and-validation.md` §9, #330); the held-out set is fold zero's rows, decided by fold zero's models.

---

## 7. Export

The model artifact carries every class model: its mean, loadings, eigenvalues and both limits. **The JSON model and the Python snippet carry them too** (#306). A SIMCA decision needs the whole preprocessed spectrum, not a dot product with one coefficient vector, so the foldable tail of the chain travels as an explicit affine map rather than folded into coefficients. `model-export.md` §6 specifies that form and its size. The snippet returns, for each spectrum, which class models accept it.

---

## 8. Known divergences

| Area | Here | Elsewhere |
| --- | --- | --- |
| Decision rule | Both $T^2$ and $Q$ within their limits | R `mdatools` defaults to a combined statistic with data-driven limits; older SIMCA uses $Q$ alone, or an F-test on the residual |
| $T^2$ limit | F form for new samples, for every decision | Some packages use the calibration form for a model's own rows |
| Scaling | Per-class centring only | Some packages also autoscale each class |
| Components | One $A$ for every class | Chosen per class, often by cross-validation |

---

## 9. Parity

**Not compared against another implementation.** The R `mdatools` reference this plan named would need R, which this project's development environment does not carry, and scikit-learn has no SIMCA. The claim rests on these instead, each in `tests/test_classification.py`:

- Every class model equals `decomposition.PCA` fitted to that class's centred rows. That PCA carries its own parity claims against scikit-learn.
- Every distance and every acceptance follows §3 and §4, recomputed independently from that PCA.
- Every tally follows §5, recomputed from the decisions.

The parity report lists SIMCA with this reason.
