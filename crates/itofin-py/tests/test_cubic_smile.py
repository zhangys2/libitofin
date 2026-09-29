"""Standard-deviation-coordinate cubic mid-IV smile bindings."""

# standard library
import math

# pypi/conda library
import pytest

# itofin library
from itofin import ItofinError
from itofin.termstructures import CubicSmileSection

_FORWARD = 100_000.0
_EXPIRY = 30.0 / 365.0
_ATM_VOL = 0.60
_SCALE = _ATM_VOL * math.sqrt(_EXPIRY)
_DEFAULT_POINTS = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0]


def _strike(x_std: float) -> float:
    return _FORWARD * math.exp(x_std * _SCALE)


def _linear_smile(std_points=(-2.0, -1.0, 0.0, 1.0, 2.0)):
    # Deliberately reverse nodes: the binding/core sort paired source inputs.
    points = list(reversed(std_points))
    strikes = [_strike(point) for point in points]
    mid_ivs = [0.30 + 0.02 * point for point in points]
    return CubicSmileSection(strikes, mid_ivs, _FORWARD, _EXPIRY, _ATM_VOL)


def test_default_sample_grid_and_source_nodes_are_exposed_sorted():
    smile = _linear_smile()
    assert smile.std_dev_points == _DEFAULT_POINTS
    assert smile.node_std_dev_points == pytest.approx([-2, -1, 0, 1, 2])
    assert smile.node_mid_ivs == pytest.approx([0.26, 0.28, 0.30, 0.32, 0.34])
    assert len(smile.sampled_mid_ivs) == len(_DEFAULT_POINTS)
    assert smile.sampled_mid_ivs[0] is None
    assert smile.sampled_mid_ivs[4] == pytest.approx(0.30)
    assert smile.sampled_mid_ivs[-1] is None


def test_standard_deviation_and_strike_queries_recover_linear_curve():
    smile = _linear_smile()
    point = 0.4
    expected = 0.30 + 0.02 * point
    assert smile.volatility_at_std_dev(point) == pytest.approx(expected, abs=1e-13)
    assert smile.volatility(_strike(point)) == pytest.approx(expected, abs=1e-12)
    assert smile.strike_at_std_dev(point) == pytest.approx(_strike(point), rel=1e-14)
    assert smile.forward == _FORWARD
    assert smile.atm_vol == _ATM_VOL
    assert smile.exercise_time == _EXPIRY


def test_custom_sample_points_and_explicit_extrapolation():
    points = [-2.0, 0.0, 2.0, 3.0]
    smile = CubicSmileSection(
        [_strike(x) for x in [-1.0, 0.0, 1.0]],
        [0.28, 0.30, 0.32],
        _FORWARD,
        _EXPIRY,
        _ATM_VOL,
        std_dev_points=points,
        extrapolate=True,
    )
    assert smile.std_dev_points == points
    assert smile.sampled_mid_ivs == pytest.approx([0.26, 0.30, 0.34, 0.36])
    assert smile.volatility_at_std_dev(2.0) == pytest.approx(0.34)


def test_bad_market_inputs_and_out_of_range_queries_raise_itofin_error():
    valid_strikes = [_strike(-1.0), _strike(1.0)]
    valid_vols = [0.28, 0.32]
    invalid_calls = [
        lambda: CubicSmileSection(valid_strikes, [0.3], _FORWARD, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection([_FORWARD, _FORWARD], valid_vols, _FORWARD, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection([0.0, _FORWARD], valid_vols, _FORWARD, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection(valid_strikes, [0.3, math.nan], _FORWARD, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection(valid_strikes, [0.3, -0.1], _FORWARD, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection(valid_strikes, valid_vols, 0.0, _EXPIRY, _ATM_VOL),
        lambda: CubicSmileSection(valid_strikes, valid_vols, _FORWARD, 0.0, _ATM_VOL),
        lambda: CubicSmileSection(valid_strikes, valid_vols, _FORWARD, _EXPIRY, 0.0),
        lambda: CubicSmileSection(
            valid_strikes, valid_vols, _FORWARD, _EXPIRY, _ATM_VOL, std_dev_points=[]
        ),
    ]
    for call in invalid_calls:
        with pytest.raises(ItofinError):
            call()

    with pytest.raises(ItofinError):
        _linear_smile().volatility_at_std_dev(3.0)
