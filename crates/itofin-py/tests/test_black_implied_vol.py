"""Black implied standard deviation and annualized volatility bindings.

Prices are generated from the Black-76 formula (the same expression as
`black_formula`: discount * sign * (shifted_forward * N(sign d1) - shifted_strike * N(sign d2))).
The solver must recover the volatility that produced each price.
"""

# standard library
import math

# pypi/conda library
import pytest

# itofin library
from itofin import ItofinError
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_std_dev, black_formula_implied_volatility

ABS_TOL = 1e-8


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_price(option_type, strike, forward, std_dev, discount, displacement=0.0) -> float:
    sign = 1.0 if option_type is OptionType.Call else -1.0
    shifted_forward = forward + displacement
    shifted_strike = strike + displacement
    d1 = math.log(shifted_forward / shifted_strike) / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    undiscounted = sign * (
        shifted_forward * _norm_cdf(sign * d1) - shifted_strike * _norm_cdf(sign * d2)
    )
    return discount * undiscounted


# option type, strike, forward, expiry, vol, discount, displacement
_CASES = [
    (OptionType.Call, 100.0, 100.0, 1.0, 0.20, 1.0, 0.0),
    (OptionType.Call, 120.0, 100.0, 0.5, 0.35, 0.99, 0.0),
    (OptionType.Put, 120.0, 100.0, 0.25, 0.50, 0.97, 0.0),
    (OptionType.Put, 80.0, 100.0, 2.0, 0.15, 0.95, 0.01),
    (OptionType.Call, 100.0, 110.0, 0.75, 0.40, 0.98, 0.02),
]


@pytest.mark.parametrize(
    ("option_type", "strike", "forward", "expiry", "vol", "discount", "displacement"),
    _CASES,
)
def test_std_dev_and_volatility_recover_black_price(
    option_type, strike, forward, expiry, vol, discount, displacement
):
    std_dev = vol * math.sqrt(expiry)
    price = _black_price(option_type, strike, forward, std_dev, discount, displacement)
    recovered_std = black_formula_implied_std_dev(
        option_type,
        strike,
        forward,
        price,
        discount=discount,
        displacement=displacement,
        accuracy=1e-12,
    )
    recovered_iv = black_formula_implied_volatility(
        option_type,
        strike,
        forward,
        expiry,
        price,
        discount=discount,
        displacement=displacement,
        accuracy=1e-12,
    )
    assert recovered_std == pytest.approx(std_dev, abs=ABS_TOL)
    assert recovered_iv == pytest.approx(vol, abs=ABS_TOL)
    assert recovered_iv == pytest.approx(recovered_std / math.sqrt(expiry), abs=1e-12)
    assert math.isfinite(recovered_std)
    assert math.isfinite(recovered_iv)


def test_omitted_guess_matches_explicit_seed_path():
    option_type = OptionType.Call
    strike, forward, expiry, vol = 100.0, 100.0, 1.0, 0.20
    price = _black_price(option_type, strike, forward, vol, 1.0)
    omitted = black_formula_implied_std_dev(
        option_type, strike, forward, price, accuracy=1e-12
    )
    explicit_none = black_formula_implied_std_dev(
        option_type, strike, forward, price, guess=None, accuracy=1e-12
    )
    seeded = black_formula_implied_std_dev(
        option_type, strike, forward, price, guess=0.1, accuracy=1e-12
    )
    assert omitted == pytest.approx(vol, abs=ABS_TOL)
    assert explicit_none == pytest.approx(omitted, abs=1e-12)
    assert seeded == pytest.approx(omitted, abs=ABS_TOL)


@pytest.mark.parametrize(
    "call",
    [
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, -0.1),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, math.nan),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, math.inf),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, 1.0, discount=0.0),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, 1.0, discount=-1.0),
        lambda: black_formula_implied_std_dev(OptionType.Call, math.nan, 100.0, 1.0),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, math.inf, 1.0),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, 1.0, guess=-1.0),
        lambda: black_formula_implied_std_dev(OptionType.Call, 100.0, 100.0, 1.0, max_iterations=-1),
        lambda: black_formula_implied_std_dev(
            OptionType.Call, 100.0, 100.0, 1.0, max_iterations=2**32
        ),
        # Deep in-the-money call priced at zero: put-call parity has no solution.
        lambda: black_formula_implied_std_dev(OptionType.Call, 80.0, 100.0, 0.0),
    ],
)
def test_std_dev_maps_core_failures_to_itofin_error(call):
    with pytest.raises(ItofinError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 0.0, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, -0.25, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, math.nan, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, math.inf, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, math.nan),
        lambda: black_formula_implied_volatility(OptionType.Call, math.nan, 100.0, 1.0, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, math.nan, 1.0, 1.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, discount=math.nan),
        lambda: black_formula_implied_volatility(
            OptionType.Call, 100.0, 100.0, 1.0, 1.0, displacement=math.nan
        ),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, accuracy=0.0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, accuracy=-1e-8),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, accuracy=math.nan),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, max_iterations=0),
        lambda: black_formula_implied_volatility(OptionType.Call, 100.0, 100.0, 1.0, 1.0, max_iterations=-5),
        lambda: black_formula_implied_volatility(OptionType.Call, 80.0, 100.0, 1.0, 0.0),
    ],
)
def test_volatility_rejects_invalid_inputs(call):
    with pytest.raises(ItofinError):
        call()
