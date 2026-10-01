"""Total implied variance cubic smile with Roger Lee wing asymptotics tests."""

# standard library
import math

# pypi/conda library
import pytest

# itofin library
from itofin import ItofinError
from itofin.termstructures import ButterflyArbitrageReport, TotalVarianceCubicSmileSection

_FORWARD = 100_000.0
_EXPIRY = 30.0 / 365.0
_ATM_VOL = 0.60
_SCALE = _ATM_VOL * math.sqrt(_EXPIRY)
_DEFAULT_POINTS = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0]


def _strike(x_std: float) -> float:
    return _FORWARD * math.exp(x_std * _SCALE)


def _sample_smile():
    strikes = [_strike(x) for x in _DEFAULT_POINTS]
    # U-shaped smile (vol skew)
    mid_ivs = [0.75, 0.68, 0.64, 0.61, 0.60, 0.61, 0.63, 0.66, 0.72]
    return TotalVarianceCubicSmileSection(strikes, mid_ivs, _FORWARD, _EXPIRY, _ATM_VOL)


def test_total_variance_cubic_smile_accessors_and_properties():
    smile = _sample_smile()
    assert smile.forward == pytest.approx(_FORWARD)
    assert smile.exercise_time == pytest.approx(_EXPIRY)
    assert smile.atm_vol == pytest.approx(_ATM_VOL)
    assert smile.smoothing == pytest.approx(0.01)
    assert len(smile.knots_k) == 9
    assert len(smile.fitted_total_variances) == 9

    # Roger Lee asymptotic slope bounds
    assert 0.0 <= smile.right_wing_slope <= 2.0
    assert -2.0 <= smile.left_wing_slope <= 0.0


def test_total_variance_and_volatility_queries():
    smile = _sample_smile()

    # ATM query
    vol_atm = smile.volatility(_FORWARD)
    assert vol_atm == pytest.approx(0.60, abs=0.01)

    tv_atm = smile.total_variance(0.0)
    assert tv_atm == pytest.approx(vol_atm * vol_atm * _EXPIRY, rel=1e-6)

    # Variance method
    var_atm = smile.variance(_FORWARD)
    assert var_atm == pytest.approx(tv_atm, rel=1e-6)

    # First and second derivatives
    wp = smile.total_variance_derivative(0.0)
    wpp = smile.total_variance_second_derivative(0.0)
    assert math.isfinite(wp)
    assert math.isfinite(wpp)

    # Durrleman density at ATM
    density = smile.durrleman_density(0.0)
    assert density > 0.0


def test_roger_lee_extreme_wings_evaluation():
    smile = _sample_smile()

    # Extreme strikes far beyond knot range
    k_deep_put = 20_000.0
    k_deep_call = 400_000.0

    vol_put = smile.volatility(k_deep_put)
    vol_call = smile.volatility(k_deep_call)

    assert vol_put > 0.0
    assert vol_call > 0.0

    # Total variance is monotonic in wings
    log_m_put = math.log(k_deep_put / _FORWARD)
    log_m_call = math.log(k_deep_call / _FORWARD)
    assert smile.total_variance(log_m_call) > smile.total_variance(0.0)
    assert smile.total_variance(log_m_put) > smile.total_variance(0.0)


def test_butterfly_arbitrage_report():
    smile = _sample_smile()
    report = smile.butterfly_report
    assert isinstance(report, ButterflyArbitrageReport)
    assert math.isfinite(report.min_density)
    assert math.isfinite(report.argmin_k)
    assert isinstance(report.has_arbitrage, bool)
    assert report.tolerance == pytest.approx(1e-8)
    assert report.points_checked > 0


def test_invalid_arguments_raise_itofin_error():
    with pytest.raises(ItofinError):
        TotalVarianceCubicSmileSection([100.0], [0.2], _FORWARD, _EXPIRY, _ATM_VOL)

    with pytest.raises(ItofinError):
        TotalVarianceCubicSmileSection([90.0, 100.0], [0.2], _FORWARD, _EXPIRY, _ATM_VOL)

    with pytest.raises(ItofinError):
        TotalVarianceCubicSmileSection([90.0, 100.0], [0.2, 0.2], 0.0, _EXPIRY, _ATM_VOL)

    # Zero mid IV
    with pytest.raises(ItofinError):
        TotalVarianceCubicSmileSection([90.0, 100.0], [0.0, 0.2], _FORWARD, _EXPIRY, _ATM_VOL)

    smile = _sample_smile()
    with pytest.raises(ItofinError):
        smile.volatility(0.0)

    with pytest.raises(ItofinError):
        smile.volatility(-10.0)

    with pytest.raises(ItofinError):
        smile.variance(0.0)

    with pytest.raises(ItofinError):
        smile.variance(-5.0)
