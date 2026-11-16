"""Oracle for libitofin #1468 (incomplete beta power terms at large a, b).

Rows are computed with mpmath at 60 significant digits and rounded to 17;
inputs are binary64 values printed exactly.

- `lanczos,z,value`: Boost's `lanczos13m53::lanczos_sum_expG_scaled(z)`, the
  exact `Gamma(z) e^(z - 1/2) / (z + g - 1/2)^(z - 1/2)` at Boost's binary64 g.
- `binomial,n,p,k,P`: `P(X <= k)` for `X ~ Binomial(n, p)`, that is
  `I_{1-p}(n - k, k + 1)` at the binary64 roundings of `n - k` and `k + 1`
  (the arguments the Rust cdf passes), for `k` at `-2..=2` half standard
  deviations around the mean. `p` is 1/2 or 3/4, so `1 - p` is exact in
  binary64. mpmath's `betainc` does not return at these sizes, so the
  integral of the Beta density is taken directly: tanh-sinh quadrature of
  `exp((a-1) ln t + (b-1) ln(1-t) - ln B(a, b))` over a window of 60
  standard deviations below the mode, split at every standard deviation.
  The route matches exact integer binomial sums at `n = 100000` to `1e-35`
  and the complement integral `1 - I_p(k + 1, n - k)` at `n = 1e16` to
  `1e-32`.
- `student,nu,t,P`: the Student-t cdf, by tanh-sinh quadrature of the density
  (with `loggamma`) from `-inf` to `t`; it matches mpmath's `betainc` at
  `nu = 1e6, 1e12, 1e16` to `1e-33`. `nu` runs to either side of `2^52`
  (`1/eps`), where the cdf switches to the normal. `|t| < sqrt(3)` takes
  the centre form and `|t| > sqrt(3)` the forward one; the forward rows stop
  at `nu = 1e6` below `2^52`, where the continued fraction still cancels
  (a documented deferral in `studentt.rs`).
- `student_quantile,nu,p,x`: the Student-t quantile, by bisection on that cdf.

Regenerate from the repository root:

    uv run --with mpmath==1.3.0 python -I \
        crates/libitofin/tests/fixtures/ibeta_power_terms/generate.py \
        > crates/libitofin/tests/fixtures/ibeta_power_terms/cases.csv
"""

import mpmath

mpmath.mp.dps = 60

LANCZOS_G = mpmath.mpf(6.024680040776729583740234375)
LANCZOS_Z = [1e-3, 0.25, 0.5, 1.0, 1.5, 2.0, 3.5, 10.0, 47.25, 1e3, 1e6, 5e11, 5e15, 1e20]


def lanczos_scaled(z):
    z = mpmath.mpf(z)
    gh = LANCZOS_G - mpmath.mpf(1) / 2
    return mpmath.exp(mpmath.loggamma(z) + z - mpmath.mpf(1) / 2 - (z - mpmath.mpf(1) / 2) * mpmath.log(z + gh))


BINOMIAL = [(10**12, 0.5), (10**14, 0.5), (10**16, 0.5), (10**12, 0.75)]


def ibeta_quad(a, b, x):
    a, b, x = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(x)
    lnb = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    mode = (a - 1) / (a + b - 2)
    sigma = mpmath.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def density(t):
        return mpmath.exp((a - 1) * mpmath.log(t) + (b - 1) * mpmath.log1p(-t) - lnb)

    lo = max(mpmath.mpf(0), mode - 60 * sigma)
    pieces = int(mpmath.ceil((x - lo) / sigma)) + 1
    return mpmath.quad(density, [lo + (x - lo) * i / pieces for i in range(pieces + 1)])


STUDENT_NU = [1e6, 1e12, 4503599627370496.0, 4503599627370498.0, 1e16]
STUDENT_CENTRE_T = [-1.73, -0.5, 0.5, 1.73]
STUDENT_FORWARD_NU = [1e6, 4503599627370498.0, 1e16]
STUDENT_FORWARD_T = [-5.0, -2.5, -1.75, 2.5]
STUDENT_QUANTILE = [(1e9, 0.05), (1e12, 0.05)]


def student_cdf(nu, t):
    nu, t = mpmath.mpf(nu), mpmath.mpf(t)
    ln_c = mpmath.loggamma((nu + 1) / 2) - mpmath.loggamma(nu / 2) - mpmath.log(nu * mpmath.pi) / 2

    def density(s):
        return mpmath.exp(ln_c - (nu + 1) / 2 * mpmath.log1p(s * s / nu))

    lower = min(t, -t)
    tail = mpmath.quad(density, [-mpmath.inf, lower - 40, lower - 10, lower - 3, lower])
    return tail if t <= 0 else 1 - tail


def student_quantile(nu, p):
    lo, hi = mpmath.mpf(-3), mpmath.mpf(0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if student_cdf(nu, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binomial_ks(n, p):
    mean, sd = n * mpmath.mpf(p), mpmath.sqrt(n * mpmath.mpf(p) * (1 - mpmath.mpf(p)))
    return [int(mpmath.nint(mean + j * sd / 2)) for j in range(-2, 3)]


def fmt(x):
    return mpmath.nstr(x, 17, min_fixed=-1, max_fixed=1)


def main():
    print(f"# mpmath {mpmath.__version__}, dps {mpmath.mp.dps}; kind,inputs...,value")
    for z in LANCZOS_Z:
        print(f"lanczos,{z!r},{fmt(lanczos_scaled(z))}")
    for n, p in BINOMIAL:
        for k in binomial_ks(n, p):
            value = ibeta_quad(float(n - k), float(k + 1), 1 - p)
            print(f"binomial,{n},{p!r},{k},{fmt(value)}")
    for nu in STUDENT_NU:
        ts = STUDENT_CENTRE_T + (STUDENT_FORWARD_T if nu in STUDENT_FORWARD_NU else [])
        for t in ts:
            print(f"student,{nu!r},{t!r},{fmt(student_cdf(nu, t))}")
    for nu, p in STUDENT_QUANTILE:
        print(f"student_quantile,{nu!r},{p!r},{fmt(student_quantile(nu, mpmath.mpf(p)))}")


if __name__ == "__main__":
    main()
