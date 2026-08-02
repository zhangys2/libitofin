//! Outright FX forward valuation by covered interest parity.
//!
//! An [`FxForward`] is an agreement to exchange, on `delivery_date`, a
//! `base_notional` of the spot rate's **source** (base) currency for its
//! **target** (quote) currency at an agreed `strike` (quote units per base
//! unit). It sits in the money layer beside [`Money`](crate::money::Money) and
//! [`ExchangeRate`](crate::exchangerate::ExchangeRate) and prices analytically
//! off two discount curves - the source (base) curve and the target (quote)
//! curve - rather than through a pricing engine.
//!
//! With `S` the spot rate (quote per base), `P_b` / `P_q` the base / quote
//! discount factors to delivery, notional `N` and strike `K`, the fair outright
//! rate is `F = S * P_b / P_q` (covered interest parity) and the present value
//! in the quote currency of a long position (receive base, pay quote) is
//! `N * S * P_b - N * K * P_q`. QuantLib groups this with the money / FX layer;
//! there is no single cached `test-suite` oracle, so the behaviour is pinned by
//! the parity identities in the tests.

use crate::errors::QlResult;
use crate::exchangerate::ExchangeRate;
use crate::handle::Handle;
use crate::money::Money;
use crate::require;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::Real;

/// An outright (single-settlement) FX forward.
pub struct FxForward {
    spot: ExchangeRate,
    base_curve: Handle<dyn YieldTermStructure>,
    quote_curve: Handle<dyn YieldTermStructure>,
    base_notional: Real,
    strike: Real,
    delivery_date: Date,
    long: bool,
}

impl FxForward {
    /// Builds an outright FX forward.
    ///
    /// `spot` carries the base (source) and quote (target) currencies and the
    /// spot rate; `base_curve` discounts the base currency and `quote_curve`
    /// the quote currency. `base_notional` is the amount of base currency
    /// exchanged and `strike` the agreed rate in quote units per base unit.
    /// `long` is the side that receives the base currency and pays the quote
    /// currency at delivery.
    ///
    /// # Errors
    ///
    /// Fails when `base_notional` or `strike` is not finite, or when `strike`
    /// is not positive.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(
        spot: ExchangeRate,
        base_curve: Handle<dyn YieldTermStructure>,
        quote_curve: Handle<dyn YieldTermStructure>,
        base_notional: Real,
        strike: Real,
        delivery_date: Date,
        long: bool,
    ) -> QlResult<Self> {
        require!(base_notional.is_finite(), "base notional must be finite");
        require!(strike.is_finite(), "strike must be finite");
        require!(strike > 0.0, "strike must be positive");
        Ok(Self {
            spot,
            base_curve,
            quote_curve,
            base_notional,
            strike,
            delivery_date,
            long,
        })
    }

    /// The base (source) and quote (target) discount factors to delivery.
    fn discount_factors(&self) -> QlResult<(Real, Real)> {
        let base = self
            .base_curve
            .current_link()?
            .discount_date(self.delivery_date, true)?;
        let quote = self
            .quote_curve
            .current_link()?
            .discount_date(self.delivery_date, true)?;
        Ok((base, quote))
    }

    /// The fair outright forward rate `S * P_base / P_quote` (quote per base).
    pub fn fair_forward_rate(&self) -> QlResult<Real> {
        let (base, quote) = self.discount_factors()?;
        Ok(self.spot.rate() * base / quote)
    }

    /// The present value in the quote currency.
    ///
    /// A long forward is worth `N * (S * P_base - K * P_quote)`; the short side
    /// is its negative.
    pub fn npv(&self) -> QlResult<Money> {
        let (base, quote) = self.discount_factors()?;
        let long_value = self.base_notional * (self.spot.rate() * base - self.strike * quote);
        let value = if self.long { long_value } else { -long_value };
        Ok(Money::new(self.spot.target().clone(), value))
    }

    /// The spot exchange rate the forward is struck against.
    pub fn spot(&self) -> &ExchangeRate {
        &self.spot
    }

    /// The agreed forward rate (quote per base).
    pub fn strike(&self) -> Real {
        self.strike
    }

    /// The amount of base currency exchanged.
    pub fn base_notional(&self) -> Real {
        self.base_notional
    }

    /// The delivery (settlement) date.
    pub fn delivery_date(&self) -> Date {
        self.delivery_date
    }

    /// Whether this is the long side (receives base, pays quote).
    pub fn is_long(&self) -> bool {
        self.long
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::Currency;
    use crate::interestrate::Compounding;
    use crate::shared::{Shared, shared};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn eurusd(spot: Real) -> ExchangeRate {
        ExchangeRate::new(Currency::eur(), Currency::usd(), spot)
    }

    #[test]
    fn fair_forward_rate_follows_covered_interest_parity() {
        // Continuous flat curves: F = S * exp((r_quote - r_base) * t).
        let spot = 1.10;
        let r_base = 0.02; // EUR
        let r_quote = 0.05; // USD
        let delivery = today() + 365;
        let fwd = FxForward::new(
            eurusd(spot),
            flat(r_base),
            flat(r_quote),
            1_000_000.0,
            1.15,
            delivery,
            true,
        )
        .unwrap();

        let t = 365.0 / 365.0;
        let expected = spot * ((r_quote - r_base) * t).exp();
        assert!(
            (fwd.fair_forward_rate().unwrap() - expected).abs() < 1e-12,
            "fair forward {} vs parity {expected}",
            fwd.fair_forward_rate().unwrap()
        );
    }

    #[test]
    fn striking_at_the_fair_rate_gives_zero_value() {
        let spot = 1.10;
        let delivery = today() + 200;
        let probe = FxForward::new(
            eurusd(spot),
            flat(0.02),
            flat(0.05),
            1_000_000.0,
            1.0,
            delivery,
            true,
        )
        .unwrap();
        let fair = probe.fair_forward_rate().unwrap();

        let at_fair = FxForward::new(
            eurusd(spot),
            flat(0.02),
            flat(0.05),
            1_000_000.0,
            fair,
            delivery,
            true,
        )
        .unwrap();
        assert!(
            at_fair.npv().unwrap().value().abs() < 1e-6,
            "value at the fair strike should vanish, got {}",
            at_fair.npv().unwrap().value()
        );
        assert_eq!(at_fair.npv().unwrap().currency().code(), "USD");
    }

    #[test]
    fn long_and_short_values_are_opposite() {
        let spot = 1.10;
        let delivery = today() + 400;
        // A strike below the fair forward (~1.14 here) is favourable to the
        // long, who buys the base currency cheaply.
        let strike = 1.05;
        let long = FxForward::new(
            eurusd(spot),
            flat(0.02),
            flat(0.05),
            1_000_000.0,
            strike,
            delivery,
            true,
        )
        .unwrap();
        let short = FxForward::new(
            eurusd(spot),
            flat(0.02),
            flat(0.05),
            1_000_000.0,
            strike,
            delivery,
            false,
        )
        .unwrap();

        assert!(long.fair_forward_rate().unwrap() > strike);
        let long_npv = long.npv().unwrap().value();
        let short_npv = short.npv().unwrap().value();
        assert!(
            (long_npv + short_npv).abs() < 1e-9,
            "long {long_npv} and short {short_npv} should net to zero"
        );
        assert!(
            long_npv > 0.0,
            "long value should be positive here: {long_npv}"
        );
    }

    #[test]
    fn a_non_positive_strike_is_rejected() {
        let result = FxForward::new(
            eurusd(1.10),
            flat(0.02),
            flat(0.05),
            1_000_000.0,
            0.0,
            today() + 100,
            true,
        );
        let Err(err) = result else {
            panic!("a non-positive strike must be rejected");
        };
        assert!(err.message().contains("strike must be positive"));
    }
}
