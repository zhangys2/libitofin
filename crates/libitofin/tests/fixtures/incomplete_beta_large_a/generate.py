"""Large-parameter incomplete beta oracle for libitofin #1463.

Every input below made the 100-iteration continued fraction of
`incomplete_beta` fail on main (f4091255), or sits in a dense window around
one that did. Rows are computed with mpmath at 60 significant digits and
rounded to 17; inputs are binary64 values printed exactly.

- `quantile,nu,p,x`: the Student-t quantile, by bisection on the regularized
  incomplete beta. `p` is `0.041 + 0.0015 * i / 2000000` (the #1463 sweep
  grid) for `i` within 100 of each failing index, plus the failing points of
  a 1e-12 step window at `nu = 10000`.
- `cdf,nu,t,P`: the Student-t cdf at the failing `t` (the three of the ticket
  gate and those of a dense grid on `[-1.76, -1.70]`).
- `binomial,n,k,P`: `P(X <= k)` for `X ~ Binomial(n, 1/2)` at the first ten
  failing `k` for `n = 100000`, summed exactly in integers.

Regenerate from the repository root:

    uv run --with mpmath==1.3.0 python -I \
        crates/libitofin/tests/fixtures/incomplete_beta_large_a/generate.py \
        > crates/libitofin/tests/fixtures/incomplete_beta_large_a/cases.csv
"""

import mpmath

mpmath.mp.dps = 60

GRID = 2000000
FAILING_INDICES = {1000.0: 950593, 10000.0: 357772}
EXTRA_QUANTILES = {
    10000.0: [0.041268131335, 0.041268187683, 0.041268212967, 0.04126836897],
}
FAILING_T = {
    1000.0: [-1.733299323535898, -1.75789628],
    10000.0: [
        -1.7323749613717625, -1.7328069158985884, -1.75384409, -1.74110702,
        -1.74098003, -1.73697761, -1.73540933,
    ],
}
BINOMIAL_N = 100000
FAILING_K = [49788, 49803, 49805, 49807, 49808, 49809, 49812, 49814, 49816, 49817]


def lower_cdf(nu, t):
    z = nu / (nu + t * t)
    return mpmath.betainc(nu / 2, mpmath.mpf(1) / 2, 0, z, regularized=True) / 2


def quantile(nu, p):
    nu, p = mpmath.mpf(nu), mpmath.mpf(p)
    lo, hi = mpmath.mpf(-3), mpmath.mpf(0)
    for _ in range(220):
        mid = (lo + hi) / 2
        if lower_cdf(nu, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binomial_cdf(n, k):
    term, total = 1, 1
    for j in range(1, k + 1):
        term = term * (n - j + 1) // j
        total += term
    return mpmath.mpf(total) / mpmath.mpf(2) ** n


def fmt(x):
    return mpmath.nstr(x, 17, min_fixed=-1, max_fixed=1)


def main():
    print(f"# mpmath {mpmath.__version__}, dps {mpmath.mp.dps}; kind,param,input,value")
    for nu, i0 in FAILING_INDICES.items():
        ps = [0.0410 + 0.0015 * i / GRID for i in range(i0 - 100, i0 + 101)]
        for p in ps + EXTRA_QUANTILES.get(nu, []):
            print(f"quantile,{nu!r},{p!r},{fmt(quantile(nu, p))}")
    for nu, ts in FAILING_T.items():
        for t in ts:
            print(f"cdf,{nu!r},{t!r},{fmt(lower_cdf(mpmath.mpf(nu), mpmath.mpf(t)))}")
    for k in FAILING_K:
        print(f"binomial,{BINOMIAL_N},{k},{fmt(binomial_cdf(BINOMIAL_N, k))}")


if __name__ == "__main__":
    main()
