//! Cross-currency floating-floating basis swap.
//!
//! An [`XccyBasisSwap`] pays a floating (Ibor) leg in a base currency and
//! receives a floating (Ibor) leg, plus a basis spread, in a quote currency.
//! Each leg carries an initial and final exchange of its own notional. Each leg
//! is discounted on its own currency's curve and the quote leg is converted to
//! the base currency at the spot FX rate, so it prices analytically off two
//! curves and a spot [`ExchangeRate`](crate::exchangerate::ExchangeRate),
//! mirroring the value-type approach of
//! [`FxForward`](crate::fxforward::FxForward).
//!
//! With `B` the base-leg present value (base ccy), `Q` the quote-leg present
//! value (quote ccy) and `S` the spot rate (quote per base), the value of a
//! pay-base / receive-quote position is `Q/S − B` in the base currency, or
//! `Q − S·B` in the quote currency. This is the first plain-basis slice;
//! QuantLib's `CrossCurrencyBasisSwap*` resettable/mark-to-market variants are
//! follow-ups. There is no single cached `test-suite` oracle, so the behaviour
//! is pinned by the identities in the tests (a degenerate same-rate swap nets
//! to zero, the two currency views agree through the spot, and the fair basis
//! spread zeroes the value).

use crate::cashflow::{CashFlow, Leg};
use crate::cashflows::{CashFlows, IborLeg, SimpleCashFlow};
use crate::errors::QlResult;
use crate::exchangerate::ExchangeRate;
use crate::handle::Handle;
use crate::indexes::IborIndex;
use crate::money::Money;
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::types::{Real, Spread};

const BASIS_POINT: Real = 1.0e-4;

/// A pay-base / receive-quote cross-currency floating-floating basis swap.
pub struct XccyBasisSwap {
    base_leg: Leg,
    base_curve: Handle<dyn YieldTermStructure>,
    quote_leg: Leg,
    quote_curve: Handle<dyn YieldTermStructure>,
    spot: ExchangeRate,
    quote_basis_spread: Spread,
    settings: Shared<Settings<Date>>,
}

impl XccyBasisSwap {
    /// Builds a cross-currency basis swap.
    ///
    /// The base leg (paid) forecasts off `base_index` and discounts on
    /// `base_curve`; the quote leg (received) forecasts off `quote_index`,
    /// carries `quote_basis_spread`, and discounts on `quote_curve`. `spot` is
    /// the base→quote exchange rate (quote units per base unit). Each leg
    /// exchanges its notional at the schedule's start and end.
    ///
    /// # Errors
    ///
    /// Propagates any leg-building error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_schedule: Schedule,
        base_index: Shared<IborIndex>,
        base_curve: Handle<dyn YieldTermStructure>,
        base_notional: Real,
        base_day_count: DayCounter,
        quote_schedule: Schedule,
        quote_index: Shared<IborIndex>,
        quote_curve: Handle<dyn YieldTermStructure>,
        quote_notional: Real,
        quote_day_count: DayCounter,
        quote_basis_spread: Spread,
        spot: ExchangeRate,
        payment_convention: Option<BusinessDayConvention>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<XccyBasisSwap> {
        let base_leg = Self::build_leg(
            base_schedule,
            base_index,
            base_notional,
            0.0,
            base_day_count,
            payment_convention,
        )?;
        let quote_leg = Self::build_leg(
            quote_schedule,
            quote_index,
            quote_notional,
            quote_basis_spread,
            quote_day_count,
            payment_convention,
        )?;
        Ok(XccyBasisSwap {
            base_leg,
            base_curve,
            quote_leg,
            quote_curve,
            spot,
            quote_basis_spread,
            settings,
        })
    }

    fn build_leg(
        schedule: Schedule,
        index: Shared<IborIndex>,
        notional: Real,
        spread: Spread,
        day_count: DayCounter,
        payment_convention: Option<BusinessDayConvention>,
    ) -> QlResult<Leg> {
        let start = schedule.start_date();
        let end = schedule.end_date();
        let mut builder = IborLeg::new(schedule, index)
            .with_notionals(vec![notional])
            .with_payment_day_counter(day_count)
            .with_spreads(vec![spread]);
        if let Some(convention) = payment_convention {
            builder = builder.with_payment_adjustment(convention);
        }
        let coupons = builder.build()?;

        let mut leg: Leg = Vec::with_capacity(coupons.len() + 2);
        leg.push(shared(SimpleCashFlow::new(-notional, start)?) as Shared<dyn CashFlow>);
        leg.extend(coupons);
        leg.push(shared(SimpleCashFlow::new(notional, end)?) as Shared<dyn CashFlow>);
        Ok(leg)
    }

    /// The present value of the (paid) base leg, in the base currency.
    pub fn base_leg_pv(&self) -> QlResult<Real> {
        let curve = self.base_curve.current_link()?;
        CashFlows::npv(&self.base_leg, &*curve, &self.settings, None, None, None)
    }

    /// The present value of the (received) quote leg, in the quote currency.
    pub fn quote_leg_pv(&self) -> QlResult<Real> {
        let curve = self.quote_curve.current_link()?;
        CashFlows::npv(&self.quote_leg, &*curve, &self.settings, None, None, None)
    }

    /// The present value in the base currency: `Q/S − B`.
    pub fn npv(&self) -> QlResult<Money> {
        let value = self.quote_leg_pv()? / self.spot.rate() - self.base_leg_pv()?;
        Ok(Money::new(self.spot.source().clone(), value))
    }

    /// The present value in the quote currency: `Q − S·B`.
    pub fn npv_in_quote(&self) -> QlResult<Money> {
        let value = self.quote_leg_pv()? - self.spot.rate() * self.base_leg_pv()?;
        Ok(Money::new(self.spot.target().clone(), value))
    }

    /// The basis spread on the quote leg that zeroes the swap value.
    ///
    /// # Errors
    ///
    /// The quote leg must have a non-zero spread sensitivity (a priced,
    /// non-expired swap).
    pub fn fair_quote_basis_spread(&self) -> QlResult<Spread> {
        let curve = self.quote_curve.current_link()?;
        let annuity = CashFlows::bps(&self.quote_leg, &*curve, &self.settings, None, None, None)?
            / BASIS_POINT;
        let npv_in_quote = self.npv_in_quote()?.value();
        Ok(self.quote_basis_spread - npv_in_quote / annuity)
    }

    /// The spot exchange rate (quote per base).
    pub fn spot(&self) -> &ExchangeRate {
        &self.spot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::Currency;
    use crate::indexes::ibor::Euribor;
    use crate::interestrate::Compounding;
    use crate::termstructures::yields::FlatForward;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;

    const N: Real = 1_000_000.0;

    fn today() -> Date {
        Date::new(15, Month::January, 2020)
    }

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        s
    }

    fn curve(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn schedule() -> Schedule {
        let calendar = Target::new();
        let settlement = calendar.advance(
            today(),
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let maturity = calendar.advance(
            settlement,
            5,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );
        MakeSchedule::new()
            .from(settlement)
            .to(maturity)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(calendar)
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build()
    }

    #[test]
    fn a_degenerate_same_rate_swap_nets_to_zero() {
        let s = settings();
        let c = curve(0.03);
        let index = shared(Euribor::six_months(c.clone(), Shared::clone(&s)));
        // Same curve/index/notional both sides, spot = 1, zero basis: identical
        // legs with opposite signs must cancel exactly.
        let swap = XccyBasisSwap::new(
            schedule(),
            Shared::clone(&index),
            c.clone(),
            N,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            c.clone(),
            N,
            Actual360::new(),
            0.0,
            ExchangeRate::new(Currency::eur(), Currency::usd(), 1.0),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&s),
        )
        .unwrap();
        assert!(
            swap.npv().unwrap().value().abs() < 1e-6,
            "npv={}",
            swap.npv().unwrap().value()
        );
    }

    #[test]
    fn each_leg_prices_near_par() {
        // A floating leg with notional exchange, discounted on its forecasting
        // curve, is worth roughly par (net ~0) - a sanity check on the notional
        // exchange signs.
        let s = settings();
        let c = curve(0.03);
        let index = shared(Euribor::six_months(c.clone(), Shared::clone(&s)));
        let swap = XccyBasisSwap::new(
            schedule(),
            Shared::clone(&index),
            c.clone(),
            N,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            c.clone(),
            N,
            Actual360::new(),
            0.0,
            ExchangeRate::new(Currency::eur(), Currency::usd(), 1.0),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&s),
        )
        .unwrap();
        assert!(
            swap.base_leg_pv().unwrap().abs() < 0.02 * N,
            "base leg not near par: {}",
            swap.base_leg_pv().unwrap()
        );
    }

    #[test]
    fn the_two_currency_views_agree_through_the_spot() {
        let s = settings();
        let base_c = curve(0.02);
        let quote_c = curve(0.05);
        let base_index = shared(Euribor::six_months(base_c.clone(), Shared::clone(&s)));
        let quote_index = shared(Euribor::six_months(quote_c.clone(), Shared::clone(&s)));
        let spot_rate = 1.25;
        let swap = XccyBasisSwap::new(
            schedule(),
            base_index,
            base_c,
            N,
            Actual360::new(),
            schedule(),
            quote_index,
            quote_c,
            N * spot_rate,
            Actual360::new(),
            0.004,
            ExchangeRate::new(Currency::eur(), Currency::usd(), spot_rate),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&s),
        )
        .unwrap();

        let npv_base = swap.npv().unwrap();
        let npv_quote = swap.npv_in_quote().unwrap();
        assert_eq!(npv_base.currency().code(), "EUR");
        assert_eq!(npv_quote.currency().code(), "USD");
        assert!(
            (npv_quote.value() - spot_rate * npv_base.value()).abs() < 1e-9,
            "quote {} vs spot*base {}",
            npv_quote.value(),
            spot_rate * npv_base.value()
        );
    }

    #[test]
    fn the_fair_basis_spread_zeroes_the_value() {
        let s = settings();
        let base_c = curve(0.02);
        let quote_c = curve(0.05);
        let spot_rate = 1.25;

        let make = |basis: Spread| {
            XccyBasisSwap::new(
                schedule(),
                shared(Euribor::six_months(base_c.clone(), Shared::clone(&s))),
                base_c.clone(),
                N,
                Actual360::new(),
                schedule(),
                shared(Euribor::six_months(quote_c.clone(), Shared::clone(&s))),
                quote_c.clone(),
                N * spot_rate,
                Actual360::new(),
                basis,
                ExchangeRate::new(Currency::eur(), Currency::usd(), spot_rate),
                Some(BusinessDayConvention::ModifiedFollowing),
                Shared::clone(&s),
            )
            .unwrap()
        };

        let off_market = make(0.005);
        let fair = off_market.fair_quote_basis_spread().unwrap();
        let repriced = make(fair);
        assert!(
            repriced.npv().unwrap().value().abs() < 1e-6,
            "npv at the fair basis spread should vanish, got {}",
            repriced.npv().unwrap().value()
        );
    }
}
