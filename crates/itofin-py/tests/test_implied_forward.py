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
