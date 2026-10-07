# ECB announcement event study, and inference when clusters are few

A small, self-contained toolkit for two things that come up together in empirical
macro-finance: measuring how sovereign yields respond to central bank announcements,
and getting the standard errors right when the panel has a handful of clusters
rather than thousands.

Everything runs on public data, downloaded on demand from the original source.

---

## What this is

**1. The baseline event study.** For each euro area country and maturity, the
intraday change in the sovereign yield around a Governing Council announcement is
regressed on the monetary policy surprise measured over the same window:

```
dy(c, m, t) = alpha + beta * S(t) + e(t)
```

This is the specification of Altavilla, Brugnolini, Gurkaynak, Motto and Ragusa
(2019), and the point of reproducing it here is that it is a known quantity: if the
code is right, the maturity gradient comes out.

**2. The inference layer, which is the real content.** The same panel is estimated
with three variance estimators, so that they can be compared on identical data:

- cluster-robust CRV1 with the usual small-sample correction;
- Driscoll-Kraay, robust to cross-sectional dependence;
- the restricted wild cluster bootstrap (Cameron, Gelbach and Miller 2008;
  Roodman, MacKinnon, Nielsen and Webb 2019), with Rademacher and Webb weights.

The bootstrap is the part worth reading. With few clusters, the asymptotic
cluster-robust t-test rejects too often, and the fix is to impose the null, resample
residuals cluster by cluster, and read the p-value off the bootstrap distribution of
the t statistic. Two details are implemented explicitly because they change answers:

- **Rademacher weights are exhausted, not sampled, when 2^G is small.** With four
  clusters there are only sixteen distinct sign vectors. Drawing 999 times from a set
  of sixteen adds sampling noise to a quantity that can simply be computed exactly.
  The code enumerates all 2^G vectors when 2^G <= B, which makes the p-value exact
  and independent of the seed.
- **Webb's six-point weights** are available for the same reason: at G = 4,
  Rademacher can only ever return a multiple of 1/16, so the finest achievable
  p-value is 0.0625. Webb gives 6^G draws instead.

---

## Running it

```bash
pip install -r requirements.txt
python run.py
python tests/test_inference.py         # ou : python -m pytest tests -q
```

Data is downloaded on first run into `data/` (about 380 KB) and cached.

```bash
python run.py --window "Press Conference Window"   # fenetre de la conference de presse
python run.py --surprise OIS_3M                    # une autre mesure de surprise
python run.py --cluster pays                       # 4 clusters au lieu de 12
python run.py --no-winsor                          # sans ecretage 1/99
```

The `--cluster pays` variant is the interesting one to run: it drops the number of
clusters from twelve to four and prints a warning about what Rademacher can and
cannot deliver at that size.

---

## Result on the current vintage of the data

315 announcements, Monetary Event Window, surprise measured by the one-year OIS,
outcomes winsorised at 1 and 99 per cent, Newey-West with four lags.

| Country | 2Y | 5Y | 10Y | monotone |
|---|---|---|---|---|
| DE | 0.955 | 0.742 | 0.387 | yes |
| FR | 0.930 | 0.779 | 0.408 | yes |
| IT | 0.913 | 0.799 | 0.514 | yes |
| ES | 0.821 | 0.691 | 0.425 | yes |

The response declines with maturity in all four countries, which is the expected
sign pattern for a surprise measured at the one-year point.

Pooled, with entity fixed effects on the twelve country-maturity cells:
beta = 0.696, clustered standard error 0.061, Driscoll-Kraay 0.058, and a bootstrap
p-value below 0.001 under both Rademacher and Webb weights. The three methods agree,
as they should with twelve clusters.

### The same regression, clustered on four countries

`python run.py --cluster pays` changes nothing but the level at which the errors are
clustered. The estimate is identical, and everything else moves:

| | 12 cells | 4 countries |
|---|---|---|
| clustered standard error | 0.0614 | 0.0195 |
| t statistic | 11.3 | 35.7 |
| Driscoll-Kraay standard error | 0.0576 | 0.0576 |
| bootstrap p, Rademacher | < 0.001 | **0.1250** |
| bootstrap p, Webb | < 0.001 | **0.0060** |

Three things are visible at once. The clustered standard error *falls* by a factor of
three when clusters are made coarser, which is the wrong direction and a warning sign
rather than a result. The Rademacher p-value is 0.125, which is 2/16 and the second
attainable value on a grid whose floor is 0.0625: at four clusters that test simply
cannot report significance at 1 per cent, whatever the data say. Webb, drawing from
6^4 = 1296 sign vectors instead of 16, returns 0.006.

The lesson is not that one number is right. It is that the level of clustering, the
choice of weights and the reported p-value are one decision, not three, and that a
result surviving at twelve clusters may have nothing to say at four.

---

## Tests

Six tests, on simulated data only. The one that matters checks the size of the test:
under the null, with twelve clusters and errors carrying a common cluster component,
the bootstrap rejects about 2 per cent of the time at a nominal 5 per cent and the
asymptotic test about 4 per cent. Both hold their level at that number of clusters;
the gap opens up further down. A second test verifies that at G = 4 the p-value is
exact, lands on a multiple of 1/16, and does not move with the seed.

---

## Sources

- Altavilla, C., Brugnolini, L., Gurkaynak, R., Motto, R. and Ragusa, G. (2019).
  "Measuring Euro Area Monetary Policy." *Journal of Monetary Economics*, 108, 162-179.
  Database: https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx
- Cameron, A.C., Gelbach, J. and Miller, D. (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." *Review of Economics and Statistics*, 90(3), 414-427.
- Webb, M. (2014). "Reworking Wild Bootstrap Based Inference for Clustered Errors."
  Queen's Economics Department Working Paper 1315.
- Roodman, D., MacKinnon, J., Nielsen, M. and Webb, M. (2019). "Fast and Wild:
  Bootstrap Inference in Stata Using boottest." *The Stata Journal*, 19(1), 4-60.
- Driscoll, J. and Kraay, A. (1998). "Consistent Covariance Matrix Estimation with
  Spatially Dependent Panel Data." *Review of Economics and Statistics*, 80(4), 549-560.

MIT licence for the code. The data keeps the licence of its source.
