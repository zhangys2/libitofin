"""Unit and integration tests for implied forward and discount factor estimation."""

import math
import pytest
import itofin
from itofin.termstructures import OptionQuotePair, implied_forward


def test_european_vanilla_unanchored_parity():
    """Verify that unanchored WLS regression accurately recovers true forward and discount factor."""
    true_forward = 5125.0
    true_df = 0.982
    expiry_years = 0.25

    strikes = [4900.0, 5000.0, 5100.0, 5125.0, 5150.0, 5200.0, 5300.0]
    quotes = []

    for k in strikes:
        parity_diff = true_df * (true_forward - k)
        call_mid = 100.0 + max(parity_diff, 0.0)
        put_mid = call_mid - parity_diff

        # 1.0 wide spread
        quotes.append((k, call_mid - 0.5, call_mid + 0.5, put_mid - 0.5, put_mid + 0.5))

    res = implied_forward(quotes, expiry_years=expiry_years, convention="european")

    assert res.is_valid
    assert res.status == "Valid"
    assert math.isclose(res.forward, true_forward, abs_tol=1e-4)
    assert math.isclose(res.discount_factor, true_df, abs_tol=1e-4)
    assert res.implied_carry_rate is not None
    assert math.isclose(res.implied_carry_rate, -math.log(true_df) / expiry_years, rel_tol=1e-4)
    assert res.pairs_used == len(strikes)
    assert res.forward_bid_strict <= res.forward <= res.forward_ask_strict


def test_anchored_discount_factor():
    """Verify forward point estimate when discount factor is pre-specified."""
    true_forward = 5000.0
    df = 0.99
    expiry_years = 0.1

    quotes = [
        (4950.0, 80.0, 81.0, 30.5, 31.5),
        (5000.0, 50.0, 51.0, 50.0, 51.0),
        (5050.0, 30.5, 31.5, 80.0, 81.0),
    ]

    res = implied_forward(
        quotes,
        expiry_years=expiry_years,
        convention="european",
        discount_factor=df,
    )

    assert res.is_valid
    assert res.discount_factor == df
    assert 4980.0 < res.forward < 5020.0


def test_crypto_inverse_deribit_parity():
    """Verify coin-numeraire linear parity for Deribit-style inverse crypto options."""
    true_forward = 68000.0
    expiry_years = 0.082

    strikes = [60000.0, 64000.0, 68000.0, 72000.0, 76000.0]
    quotes = []

    for k in strikes:
        # Coin parity: C_btc - P_btc = 1 - K / F
        diff_btc = 1.0 - k / true_forward
        c_mid = 0.08 + max(diff_btc, 0.0)
        p_mid = c_mid - diff_btc
        # 0.002 BTC spread
        quotes.append((k, c_mid - 0.001, c_mid + 0.001, p_mid - 0.001, p_mid + 0.001))

    res = implied_forward(
        quotes,
        expiry_years=expiry_years,
        convention="crypto_inverse",
        spot=67500.0,
    )

    assert res.is_valid
    assert res.status == "Valid"
    assert math.isclose(res.forward, true_forward, abs_tol=1.0)
    assert res.forward_bid_strict <= res.forward <= res.forward_ask_strict


def test_option_quote_pair_instances():
    """Verify acceptance of OptionQuotePair class instances alongside tuples."""
    pairs = [
        OptionQuotePair(5000.0, 100.0, 101.0, 50.0, 51.0),
        OptionQuotePair(5100.0, 50.0, 51.0, 100.0, 101.0),
    ]

    res = implied_forward(pairs, expiry_years=0.5, convention="european")
    assert res.is_valid
    assert math.isclose(res.forward, 5050.0, abs_tol=1e-4)


def test_crossed_quotes_flagged():
    """Verify that crossed quotes trigger WarningCrossedSyntheticQuotes."""
    quotes = [
        (5000.0, 100.0, 101.0, 80.0, 81.0),
        (5050.0, 95.0, 95.5, 50.0, 50.5), # crossed synthetic
        (5100.0, 50.0, 51.0, 100.0, 101.0),
    ]

    res = implied_forward(quotes, expiry_years=0.1, discount_factor=1.0)
    assert not res.is_valid
    assert res.status in ("WarningCrossedSyntheticQuotes", "WarningBoxSpreadArbitrage")


def test_american_equity_requires_spot():
    """Verify that American equity convention validates required spot argument."""
    quotes = [(100.0, 5.0, 5.5, 4.0, 4.5), (105.0, 2.0, 2.5, 6.0, 6.5)]
    with pytest.raises(ValueError, match="requires spot price"):
        implied_forward(quotes, expiry_years=0.25, convention="american", spot=None)


def test_invalid_convention_rejected():
    """Verify that unrecognized conventions raise ValueError."""
    quotes = [(100.0, 5.0, 5.5, 4.0, 4.5), (105.0, 2.0, 2.5, 6.0, 6.5)]
    with pytest.raises(ValueError, match="unknown market convention"):
        implied_forward(quotes, expiry_years=0.25, convention="martian")


def test_insufficient_pairs_error():
    """Verify error raised when fewer than min_pairs valid quotes provided."""
    quotes = [(100.0, 5.0, 5.5, 4.0, 4.5)]
    with pytest.raises(itofin.ItofinError):
        implied_forward(quotes, expiry_years=0.25, min_pairs=2)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _model_quotes(spot, rate, carry, expiry, vol, strikes, half=0.01):
    """Black-priced quotes with a small half spread: parity-consistent inputs for the American path."""
    forward = spot * math.exp((rate - carry) * expiry)
    df = math.exp(-rate * expiry)
    sd = vol * math.sqrt(expiry)
    quotes = []
    for k in strikes:
        d1 = (math.log(forward / k) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        call = df * (forward * _norm_cdf(d1) - k * _norm_cdf(d2))
        put = df * (k * _norm_cdf(-d2) - forward * _norm_cdf(-d1))
        quotes.append((k, call - half, call + half, put - half, put + half))
    return quotes


_AMERICAN_STRIKES = [92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5]
_RATE, _CARRY, _EXPIRY = 0.01, 0.02, 0.5


def _american(half=0.01, **kwargs):
    quotes = _model_quotes(100.0, _RATE, _CARRY, _EXPIRY, 0.25, _AMERICAN_STRIKES, half)
    return implied_forward(
        quotes,
        expiry_years=_EXPIRY,
        convention="american",
        spot=100.0,
        discount_factor=math.exp(-_RATE * _EXPIRY),
        **kwargs,
    )


def test_american_model_path_reports_tier_and_diagnostics():
    res = _american()
    assert res.tier == "model_fast"
    assert res.flags == []
    assert res.accurate_prices_used == 0
    assert res.iterations is not None and res.iterations >= 1
    assert res.solved_carry is not None and abs(res.solved_carry - _CARRY) < 0.02
    assert res.call_put_iv_gap is not None and abs(res.call_put_iv_gap) < 1e-6
    assert res.forward > 0


def test_american_without_discount_factor_stays_on_the_corridor_tier():
    quotes = _model_quotes(100.0, _RATE, _CARRY, _EXPIRY, 0.25, _AMERICAN_STRIKES)
    res = implied_forward(quotes, expiry_years=_EXPIRY, convention="american", spot=100.0)
    assert res.tier == "corridor"
    assert res.flags == []
    assert res.solved_carry is None
    assert res.call_put_iv_gap is None
    assert res.iterations is None
    assert res.accurate_prices_used is None


def test_non_american_results_have_no_american_fields():
    quotes = [(k, 100.0 - k + 0.5, 100.0 - k + 1.5, 0.5, 1.5) for k in (95.0, 100.0, 105.0)]
    res = implied_forward(quotes, expiry_years=0.25, discount_factor=1.0)
    assert res.tier is None
    assert res.flags == []
    assert res.solved_carry is None


def test_accurate_level_spends_accurate_prices():
    res = _american(level="accurate")
    assert res.tier == "model_accurate"
    assert res.accurate_prices_used == 2 * len(_AMERICAN_STRIKES)
    assert res.flags == []


def test_dividend_schedule_makes_auto_accurate_and_fast_ignores_it():
    schedule = [(0.2, 0.5)]
    auto = _american(dividends=schedule)
    assert auto.tier == "model_accurate"
    assert auto.flags == []
    fast = _american(dividends=schedule, level="fast")
    assert fast.tier == "model_fast"
    assert fast.flags == ["schedule_ignored"]


def test_work_budget_is_reported_as_a_latency_fallback():
    res = _american(level="accurate", max_accurate_prices=1)
    assert res.tier == "model_fast"
    assert res.flags == ["latency_fallback"]


def test_zero_passes_and_carry_range_arguments_are_accepted():
    res = _american(
        level="accurate",
        correction_passes=0,
        carry_search_range=(-0.05, 0.20),
        carry_flag_range=(-0.02, 0.10),
        corridor_kappa=1.2,
        atm_vol=0.25,
    )
    assert res.tier == "model_fast"


def test_implausible_carry_range_flags_the_status():
    # A wide spread keeps the synthetic bounds from crossing, which would take precedence.
    res = _american(half=0.3, carry_flag_range=(0.0, 0.005))
    assert res.status == "WarningImplausibleCarry"
    assert not res.is_valid


def test_invalid_american_arguments_are_rejected():
    with pytest.raises(ValueError, match="level"):
        _american(level="bogus")
    with pytest.raises(itofin.ItofinError):
        _american(dividends=[(_EXPIRY + 0.1, 1.0)])
    with pytest.raises(itofin.ItofinError):
        _american(carry_search_range=(0.25, -0.10))
