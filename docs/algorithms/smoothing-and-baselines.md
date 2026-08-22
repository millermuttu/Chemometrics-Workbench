# Smoothing, derivatives and baseline correction — algorithm specification

Status: **normative**. This document fixes the conventions the implementation must follow. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md) for the shared conventions on centring, missing values and numerical policy; [`pls-regression.md`](pls-regression.md) §7 for what a fitted model exports, which is where the convolution matrix of §4 is used.

This is the half of preprocessing where implementations genuinely disagree. The scaling and scatter-correction kernels — SNV, MSC, centring, autoscaling, normalisation — have no convention freedom worth documenting at this length, and their conventions live in the module docstring of `preprocessing.py`. Everything here has at least one free choice, and every one of them is made below.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $x$ | One spectrum, $p$ values, indexed $i = 0 \dots p-1$ |
| $p$ | Number of variables |
| $w$ | Savitzky–Golay window length, odd |
| $h$ | Half-window, $h = (w-1)/2$ |
| $m$ | Polynomial order of the local fit |
| $d$ | Derivative order |
| $\Delta$ | Spacing between neighbouring variables (`delta`) |
| $M$ | The $p \times p$ matrix the whole filter is |
| $z$ | An estimated baseline, $p$ values |

---

## 2. Savitzky–Golay

For each variable $i$ a polynomial of degree $m$

$$q(t) = \sum_{j=0}^{m} c_j t^{j}$$

is fitted by unweighted least squares to the $w$ values in the window centred on $i$, in the window-local coordinate $t = -h \dots h$. The filtered value is $q$ — or its $d$-th derivative — evaluated at $t = 0$:

$$\hat{x}_i = \frac{q^{(d)}(0)}{\Delta^{d}} = \frac{d!\,c_d}{\Delta^{d}}$$

The least-squares fit is linear in the window, so the whole operation collapses into one weight vector per output position and the polynomial is never actually formed at transform time.

**Requirements**: $w$ odd and at least 3, $0 \le m < w$, $0 \le d \le m$, and $p \ge w$. The kernel enforces the same conditions the schema does rather than trusting them, because a transformer built directly in a script never passes through the schema. The one place the two differ is that the schema also caps $d$ at 2, which is a statement about what the application offers rather than about what the filter is defined for; the kernel does not repeat that cap.

A window of $w = m+1$ interpolates the window exactly and smooths nothing; it is legal and pointless, and is not rejected.

---

## 3. Edge handling — `interp`, and nothing else

Within $h$ variables of either end there is no centred window. **The convention is `interp`**: the polynomial fitted to the *first* full window is evaluated at $t = -h \dots -1$ to give the first $h$ outputs, and the polynomial fitted to the last full window is evaluated at $t = 1 \dots h$ to give the last $h$. This is `scipy.signal.savgol_filter(..., mode="interp")`.

**Nothing is padded, reflected, repeated or wrapped**, so no value the filter returns depends on data that was invented.

**The mode is not configurable.** A recipe that records "Savitzky–Golay, window 11, order 2, first derivative" is reproducible only if the edge mode is a property of the software rather than a choice buried in a call. Making it configurable would put a parameter in the schema whose value nobody sets deliberately and whose effect nobody notices until two packages disagree at the bounds.

What other packages do there, and why a comparison fails at the first and last variable and nowhere else:

| Mode | First $h$ values |
| --- | --- |
| `interp` (ours, and SciPy's default) | Fitted from the first full window, evaluated off-centre |
| `mirror` | The spectrum reflected about its first value, excluding it, then filtered normally |
| `nearest` | The first value repeated $h$ times as padding |
| `wrap` | The last $h$ values used as padding — meaningless for a spectrum |
| `constant` | Zeros as padding, which pulls the first values towards zero |
| *(several chemometrics tools)* | Left unfiltered, or the block dropped, shortening the spectrum |

A tool in the last row changes the variable count, which is worse than a numerical disagreement: the axis no longer matches the data. Our filter always returns $p$ values.

**Two properties pin this convention in tests**, and both are exercised in `test_preprocessing.py` and `test_parity.py`:

1. A polynomial of degree $\le m$ passes through the filter unchanged **including at the first and last variable**. Under any padded mode it does not, because the padding is not on the polynomial.
2. The parity block is 8 variables wide with $h = 2$, so four of its eight columns are edge columns, and the first and last are compared separately as well as within the block.

---

## 4. The filter is a matrix, and the matrix is exported

Because every output is a fixed linear combination of inputs, the whole filter is one matrix $M$ of shape $p \times p$:

$$X_{\text{filtered}} = X M^{\top}$$

$M$ is banded with bandwidth $w$ in its interior; its first and last $h$ rows are dense within the end window, which is exactly what §3 describes.

**This matrix is public** (`SavitzkyGolayTransformer.convolution_matrix()`) and it is the reason Savitzky–Golay is treated differently from SNV and MSC. `pls-regression.md` §7 records that SNV and MSC cannot be folded into exported regression coefficients, because both depend on the sample being predicted. Savitzky–Golay does not, so it folds:

$$b_{\text{raw}} = M^{\top} b_{\text{filtered}}$$

An exported model that used only Savitzky–Golay and centring is therefore a bare coefficient vector applied to raw spectra. A kernel that returned only the filtered values would make that impossible.

---

## 5. Derivative scaling — per index by default

$\Delta$ defaults to 1, so **a derivative is per variable index**, not per nanometre or per wavenumber. `SavitzkyGolay` in the schema carries no spacing field, so a pipeline recipe always means per index.

Both conventions are defensible and they are not the same numbers: they differ by the constant factor $\Delta^{d}$. Nothing about a model's fit changes — the regression coefficients absorb the constant exactly — but a coefficient plot, a VIP profile and any number quoted against another package all move. A caller who wants per-axis-unit derivatives passes `delta` to the transformer and is responsible for recording that they did, because the recipe cannot.

Second derivatives invert peaks: a peak becomes a negative trough between two positive lobes. That is expected, and it is why the sign of a second-derivative spectrum is never "corrected".

---

## 6. Baseline correction

Three methods, all row-wise, all stateless. Like SNV, a baseline is a property of the spectrum in front of it, so **baseline correction cannot be folded into exported coefficients** and an exported model that uses it carries a residual preprocessing chain (`pls-regression.md` §7).

The baseline itself is available separately from the corrected spectrum (`BaselineCorrectTransformer.baseline()`), because the UI draws it over the raw data and a baseline that only exists inside a subtraction cannot be inspected.

### 6.1 Asymmetric least squares (AsLS)

Eilers and Boelens (2005). Alternate between a penalised least-squares solve for the baseline $z$

$$(W + \lambda D_2^{\top} D_2)\, z = W x$$

where $W = \operatorname{diag}(w)$ and $D_2$ is the second-difference operator of shape $(p-2) \times p$, and a reweighting that is asymmetric in the residual sign:

$$w_i = \begin{cases} p_{\text{asym}} & x_i > z_i \\ 1 - p_{\text{asym}} & x_i \le z_i \end{cases}$$

With $p_{\text{asym}}$ small, points above the current baseline are almost ignored and the baseline is pulled into the valleys, which is the whole idea. Weights start at 1.

**Convergence criterion: the weight vector stops changing.** The weights are a step function of the residual sign, so once the sign pattern repeats the iteration is at a fixed point *exactly*, not approximately — which is why the criterion is equality rather than a tolerance on $z$.

**Iteration cap: `max_iter`, default 20.** Hitting the cap is not an error; the last iterate is a usable baseline. It is recorded: `n_iterations_` and `converged_` hold one entry per spectrum after a transform, because a baseline that hit the cap is a different claim from one that settled and the caller has no other way to tell.

**Defaults: $\lambda = 10^{5}$, $p_{\text{asym}} = 0.01$**, the values used throughout the original paper. Both depend on the instrument and on the sampling density, neither is universal, and the schema carries both so that a recipe records what was actually used.

### 6.2 Rubberband

The lower convex hull of the points $(i, x_i)$, linearly interpolated between its vertices — a taut band stretched under the spectrum. Computed by Andrew's monotone chain, which needs no sort because the abscissae are the indices in order.

Two consequences, both asserted in tests: the baseline never rises above the spectrum, so **a rubberband-corrected spectrum is non-negative**; and the hull always includes the first and last variable, so **both ends are corrected to exactly zero**. The second is a real limitation — a spectrum whose first variable sits on a peak gets a baseline that is too high across that peak. Range selection before correction is the answer, not a fudged hull.

There are no parameters.

### 6.3 Polynomial

A least-squares polynomial of degree `order` (default 2) fitted to the whole spectrum and subtracted. Unlike the other two it does not attempt to avoid peaks: a strong peak drags the fit upward and is partly subtracted from itself. It is included because it is what is wanted for a slow instrumental drift with no strong peaks, and because it is what several instrument vendors' software does.

### 6.4 Axis units

All three estimate the baseline against the variable *index*, not the axis.

For the polynomial method this is exact rather than approximate: polynomials in the index and polynomials in any affine transform of the index span the same space, so the fitted baseline is identical either way. The index is mapped onto $[-1, 1]$ before the fit — again exact in the same sense, and done because a raw index over 700 variables raised to the fourth power spans $10^{11}$, at which point the fit of a high-order baseline is decided by the least-squares cutoff rather than by the data.

For AsLS and rubberband it assumes **uniform spacing**, which all three reference datasets have. On a non-uniform axis the second-difference penalty of §6.1 is no longer a second derivative and $\lambda$ means something slightly different at each end. That is a real limitation and it is not currently checked for.

---

## 7. Reported quantities

| Quantity | Defined in |
| --- | --- |
| Filtered or differentiated spectra | §2, §3, §5 |
| The convolution matrix $M$ | §4 |
| Estimated baseline per spectrum | §6 |
| AsLS iterations and convergence flag per spectrum | §6.1 |

Nothing else. In particular no smoothness or signal-to-noise figure is reported, because none of the ones in common use has a definition two packages agree on.

---

## 8. Known divergences from other packages

| | |
| --- | --- |
| **Edge mode** | Fixed at `interp`. Packages defaulting to `mirror` or `nearest`, and packages that leave the first and last half-window unfiltered, agree everywhere except the bounds. See §3. |
| **Derivative scaling** | Per variable index unless `delta` is given. Packages that ask for the axis report a derivative per axis unit, differing by $\Delta^{d}$. See §5. |
| **Baseline axis** | Estimated against the index, assuming uniform spacing. See §6.4. |
| **AsLS convergence** | Weights unchanged, capped at 20 iterations. Implementations that instead run a fixed number of iterations, or stop on a tolerance on $\|z\|$, return a slightly different baseline. |
| **Rubberband ends** | Both ends are corrected to exactly zero. Some tools extrapolate the hull beyond the data instead. |

---

## 9. Parity references

Savitzky–Golay is compared against **SciPy** (`scipy.signal.savgol_filter`, `mode="interp"`, window 5, order 2, derivatives 0, 1 and 2), on the 5 × 8 block described in the fixture notes, for each of corn, gasoline and tecator.

SciPy is a runtime dependency of this project, so it is worth stating what that comparison is worth: `scipy.signal` is **not on the kernel's code path**. SciPy solves a least-squares system per output position; the kernel builds one convolution matrix from the pseudo-inverse of the window's Vandermonde matrix. Two routes to the same filter, and the comparison catches a wrong sign, a wrong scaling, an off-by-one in the window and — most usefully — a wrong edge convention.

**The three baseline methods have no external reference.** Neither SciPy nor scikit-learn implements baseline correction and no other implementation is installed here, so the fixture records them as `unsourced` gaps and they are checked against defining properties instead: a rubberband baseline lies on the lower convex hull and never above the spectrum, a degree-$d$ polynomial baseline removes a degree-$d$ polynomial exactly, and AsLS reaches an exact fixed point of its own reweighting. Those are properties rather than a second opinion. `pybaselines` is the obvious reference implementation and adding it is a dependency decision, not a kernel one.

---

## 10. Deliberately not specified here

- **Other smoothers** — moving average, Whittaker, Gaussian, wavelet. None is in the schema. Savitzky–Golay covers the spectroscopic case and adds a derivative for free.
- **Automatic window selection.** The window is the user's choice and its effect on the model is visible through cross-validation, which is the honest way to choose it.
- **Peak detection**, which is what a baseline is usually a step towards elsewhere. This project models spectra whole.
