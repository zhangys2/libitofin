"""Student-t cdf oracle near t = 0 for libitofin #1461.

`cdf,nu,t,P` rows: the Student-t cdf `P(T <= t)` at `t = +-1e-3, +-1e-5,
+-1e-7` for `nu = 3, 30, 1000`, plus the tail guard `nu = 1000, t = -10`.
Computed with mpmath at 50 significant digits as
`1/2 + sign(t) I_y(1/2, nu/2) / 2` with `y = t^2 / (nu + t^2)` (centre) and
`I_x(nu/2, 1/2) / 2` with `x = nu / (nu + t^2)` (tail), from the exact binary64
`t`, and rounded to 17 significant digits.

Regenerate from the repository root:

    uv run --with mpmath==1.3.0 python -I \
        crates/libitofin/tests/fixtures/studentt_cdf_centre/generate.py \
        > crates/libitofin/tests/fixtures/studentt_cdf_centre/cases.csv
"""

import mpmath

mpmath.mp.dps = 50

NUS = [3.0, 30.0, 1000.0]
TS = [1e-3, 1e-5, 1e-7]
TAIL = [(1000.0, -10.0), (3.0, -3.0), (30.0, -3.0)]


def switch_ts(nu):
    import math
    t_star = math.sqrt(3.0 * nu / (nu + 2.0))
    out = []
    for s in (-1.0, 1.0):
        for t in (math.nextafter(t_star, 0.0), math.nextafter(t_star, math.inf)):
            out.append(s * t)
    return out


def cdf(nu, t):
    nu = mpmath.mpf(nu)
    t = mpmath.mpf(t)
    t2 = t * t
    if abs(t) < 1:
        y = t2 / (nu + t2)
        half = mpmath.betainc(mpmath.mpf(0.5), nu / 2, 0, y, regularized=True) / 2
        return mpmath.mpf(0.5) + mpmath.sign(t) * half
    x = nu / (nu + t2)
    tail = mpmath.betainc(nu / 2, mpmath.mpf(0.5), 0, x, regularized=True) / 2
    return tail if t < 0 else 1 - tail


def main():
    print(f"# mpmath {mpmath.__version__}, dps {mpmath.mp.dps}; kind,nu,t,P")
    rows = [("centre", nu, s * t) for nu in NUS for t in TS for s in (-1.0, 1.0)]
    rows += [("switch", nu, t) for nu in NUS for t in switch_ts(nu)]
    rows += [("tail", nu, t) for nu, t in TAIL]
    for kind, nu, t in rows:
        print(f"{kind},{nu!r},{t!r},{mpmath.nstr(cdf(nu, t), 17, strip_zeros=False)}")


if __name__ == "__main__":
    main()
