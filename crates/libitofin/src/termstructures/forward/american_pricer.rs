//! American vanilla pricing adapter for the model-based forward path.
//!
//! Wraps the library's American engines behind one interface that takes plain numbers
//! (spot, strike, flat rate, flat continuous yield, volatility) and an expiry in years.
//!
//! The engines need dated curves, but an expiry of a few days or hours would be rounded
//! to whole days. The adapter instead prices a time-scaled equivalent problem: a virtual
//! expiry of [`VIRTUAL_YEARS`] years on an Act/360 calendar, with rate, yield and
//! volatility rescaled so that `r*T`, `q*T` and `sigma^2*T` are unchanged. A Black-Scholes
//! American price depends on the problem only through these three quantities and the
//! dividend times relative to the expiry, so the price is the same and dividend dates are
//! placed to within `1/720` of the expiry.

use crate::cashflows::{Dividend, FixedDividend};
use crate::errors::QlResult;
use crate::exercise::{AmericanExercise, Exercise};
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
use crate::interestrate::Compounding;
use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
use crate::option::OptionType;
use crate::pricingengine::PricingEngine;
use crate::pricingengines::vanilla::{
    BaroneAdesiWhaleyApproximationEngine, BinomialVanillaEngine, FdBlackScholesVanillaEngine,
    JuQuadraticApproximationEngine,
};
use crate::processes::BlackScholesMertonProcess;
use crate::quotes::{Quote, SimpleQuote};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::termstructures::yields::FlatForward;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::{Date, Month};
use crate::time::daycounters::actual360::Actual360;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Size, Time, Volatility};

/// Length of the virtual expiry in years (720 days on Act/360).
const VIRTUAL_YEARS: Time = 2.0;
const VIRTUAL_DAYS: i32 = 720;
const DAYS_PER_YEAR: Real = 360.0;

/// Which American engine prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PricerKind {
    /// Ju quadratic approximation (closed form).
    JuQuadratic,
    /// Barone-Adesi-Whaley approximation (closed form).
    BaroneAdesiWhaley,
    /// Cox-Ross-Rubinstein binomial tree with the given number of steps.
    Binomial(Size),
    /// Finite-difference Black-Scholes grid; the only kind that accepts cash dividends.
    FiniteDifference { t_grid: Size, x_grid: Size },
}

impl PricerKind {
    fn supports_dividends(self) -> bool {
        matches!(self, PricerKind::FiniteDifference { .. })
    }
}

/// An American vanilla pricer bound to one expiry and one dividend schedule.
pub(crate) struct AmericanPricer {
    settings: Shared<Settings<Date>>,
    spot: Shared<SimpleQuote>,
    carry: Shared<SimpleQuote>,
    rate: Shared<SimpleQuote>,
    vol: Shared<SimpleQuote>,
    engine: SharedMut<dyn PricingEngine>,
    expiry: Date,
    /// `VIRTUAL_YEARS / expiry_years`.
    scale: Real,
}

fn quote_handle(quote: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
    Handle::new(Shared::clone(quote) as Shared<dyn Quote>)
}

fn reference_date() -> Date {
    Date::new(15, Month::June, 2026)
}

impl AmericanPricer {
    /// Builds a pricer for `expiry_years` (> 0) with an optional discrete cash-dividend
    /// schedule of `(time in years, amount)` pairs, each time in `(0, expiry_years]`.
    ///
    /// # Errors
    ///
    /// Fails for a non-positive expiry, an invalid dividend, or a dividend schedule given
    /// to a kind that cannot price one.
    pub(crate) fn new(
        kind: PricerKind,
        expiry_years: Time,
        dividends: &[(Time, Real)],
    ) -> QlResult<Self> {
        require!(
            expiry_years.is_finite() && expiry_years > 0.0,
            "expiry must be finite and positive, got {expiry_years}"
        );
        require!(
            dividends.is_empty() || kind.supports_dividends(),
            "{kind:?} cannot price cash dividends"
        );
        let scale = VIRTUAL_YEARS / expiry_years;
        require!(scale.is_finite(), "expiry {expiry_years} is out of range");

        let today = reference_date();
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let spot = shared(SimpleQuote::new(1.0));
        let carry = shared(SimpleQuote::new(0.0));
        let rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.2));
        let flat = |quote: &Shared<SimpleQuote>| -> Shared<dyn YieldTermStructure> {
            shared(FlatForward::new(
                today,
                quote_handle(quote),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>
        };
        let vol_curve = shared(BlackConstantVol::with_quote(
            today,
            None,
            quote_handle(&vol),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>;
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            Handle::new(flat(&carry)),
            Handle::new(flat(&rate)),
            Handle::new(vol_curve),
        ));

        let mut cash_dividends: Vec<Shared<dyn Dividend>> = Vec::with_capacity(dividends.len());
        for &(time, amount) in dividends {
            require!(
                time.is_finite() && time > 0.0 && time <= expiry_years,
                "dividend time must lie in (0, {expiry_years}], got {time}"
            );
            require!(
                amount.is_finite() && amount >= 0.0,
                "dividend amount must be finite and non-negative, got {amount}"
            );
            let days = ((time * scale * DAYS_PER_YEAR).round() as i32).clamp(1, VIRTUAL_DAYS);
            cash_dividends
                .push(shared(FixedDividend::new(amount, today + days)) as Shared<dyn Dividend>);
        }

        let engine: SharedMut<dyn PricingEngine> = match kind {
            PricerKind::JuQuadratic => shared_mut(JuQuadraticApproximationEngine::new(process)),
            PricerKind::BaroneAdesiWhaley => {
                shared_mut(BaroneAdesiWhaleyApproximationEngine::new(process))
            }
            PricerKind::Binomial(steps) => shared_mut(BinomialVanillaEngine::new(process, steps)?),
            PricerKind::FiniteDifference { t_grid, x_grid } => {
                shared_mut(FdBlackScholesVanillaEngine::with_params(
                    process,
                    cash_dividends,
                    t_grid,
                    x_grid,
                    0,
                    FdmSchemeDesc::douglas(),
                ))
            }
        };

        Ok(Self {
            settings,
            spot,
            carry,
            rate,
            vol,
            engine,
            expiry: today + VIRTUAL_DAYS,
            scale,
        })
    }

    /// American price per unit of notional.
    ///
    /// `rate` is the flat continuous risk-free rate, `carry` the flat continuous yield (or
    /// the residual carry beside a dividend schedule) and `vol` the Black volatility, all
    /// per year of the real expiry.
    ///
    /// # Errors
    ///
    /// Fails for non-finite or non-positive spot, strike or volatility, or when the
    /// underlying engine cannot price.
    pub(crate) fn price(
        &self,
        option_type: OptionType,
        strike: Real,
        spot: Real,
        rate: Rate,
        carry: Rate,
        vol: Volatility,
    ) -> QlResult<Real> {
        require!(
            spot.is_finite() && spot > 0.0,
            "spot must be finite and positive, got {spot}"
        );
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be finite and positive, got {strike}"
        );
        require!(
            vol.is_finite() && vol > 0.0,
            "volatility must be finite and positive, got {vol}"
        );
        require!(
            rate.is_finite() && carry.is_finite(),
            "rate and carry must be finite"
        );
        self.spot.set_value(spot);
        self.rate.set_value(rate / self.scale);
        self.carry.set_value(carry / self.scale);
        self.vol.set_value(vol / self.scale.sqrt());

        let exercise = AmericanExercise::over(reference_date(), self.expiry)?;
        let mut option = OneAssetOption::new(
            shared(PlainVanillaPayoff::new(option_type, strike)),
            shared(exercise) as Shared<dyn Exercise>,
            Shared::clone(&self.settings),
        );
        option
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&self.engine));
        option.npv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option::OptionType::{Call, Put};

    const SPOT: Real = 100.0;
    const RATE: Rate = 0.05;
    const VOL: Volatility = 0.25;

    fn price_with(
        kind: PricerKind,
        ty: OptionType,
        strike: Real,
        expiry: Time,
        carry: Rate,
    ) -> Real {
        AmericanPricer::new(kind, expiry, &[])
            .unwrap()
            .price(ty, strike, SPOT, RATE, carry, VOL)
            .unwrap()
    }

    // American put, K = S = 100, r = 5%, q = 0, vol 25%, 180 days (0.5 years). Reference values
    // come from a finite-difference grid of 800 x 1600 (6.021901) and a CRR tree of 8000 steps
    // (6.022181); the engine errors are those measured against it in the pricer inventory.
    const PUT_REFERENCE: Real = 6.0219;

    #[test]
    fn engines_reproduce_the_reference_put_within_their_measured_error() {
        let expiry = 0.5;
        let cases = [
            (PricerKind::JuQuadratic, 0.07),
            (PricerKind::BaroneAdesiWhaley, 0.10),
            (PricerKind::Binomial(400), 0.01),
            (
                PricerKind::FiniteDifference {
                    t_grid: 100,
                    x_grid: 200,
                },
                0.02,
            ),
        ];
        for (kind, tolerance) in cases {
            let price = price_with(kind, Put, 100.0, expiry, 0.0);
            assert!(
                (price - PUT_REFERENCE).abs() <= tolerance,
                "{kind:?}: {price} vs {PUT_REFERENCE}"
            );
        }
    }

    fn black_call(forward: Real, strike: Real, expiry: Time, rate: Rate, vol: Volatility) -> Real {
        let sd = vol * expiry.sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
        let d2 = d1 - sd;
        let n = |x: Real| 0.5 * (1.0 + libm_erf(x / std::f64::consts::SQRT_2));
        (-rate * expiry).exp() * (forward * n(d1) - strike * n(d2))
    }

    #[test]
    fn american_call_without_a_yield_equals_the_european_value_at_any_expiry() {
        // With q = 0 early exercise of a call is never optimal, so the American value equals
        // the European Black-Scholes value. Checking short and long expiries against the closed
        // form shows the time scaling preserves the price (0.01 years is 3.6 days, which whole-day
        // rounding would have moved by ~10%).
        for expiry in [0.01, 0.1, 0.5, 3.0] {
            let american = price_with(PricerKind::Binomial(1000), Call, 100.0, expiry, 0.0);
            let european = black_call(SPOT * (RATE * expiry).exp(), 100.0, expiry, RATE, VOL);
            assert!(
                (american - european).abs() < 2e-3 * european.max(1.0),
                "T={expiry}: {american} vs {european}"
            );
        }
    }

    // erf to ~1e-15 by series for the small arguments used above.
    fn libm_erf(x: Real) -> Real {
        let mut sum = x;
        let mut term = x;
        let x2 = x * x;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    #[test]
    fn finite_difference_prices_a_cash_dividend() {
        // One dividend of 2.0 at day 90 of a 1-year option (r = 4%, q = 0, vol 25%). Reference
        // FD 800x1600 on the spot model: put 9.22208, call 10.70437.
        let fd = PricerKind::FiniteDifference {
            t_grid: 100,
            x_grid: 200,
        };
        let pricer = AmericanPricer::new(fd, 1.0, &[(90.0 / 360.0, 2.0)]).unwrap();
        let put = pricer.price(Put, 100.0, SPOT, 0.04, 0.0, VOL).unwrap();
        let call = pricer.price(Call, 100.0, SPOT, 0.04, 0.0, VOL).unwrap();
        assert!((put - 9.22208).abs() < 0.01, "put {put}");
        assert!((call - 10.70437).abs() < 0.01, "call {call}");
        // The dividend must matter: compare with the same grid without it.
        let plain = AmericanPricer::new(fd, 1.0, &[]).unwrap();
        let plain_put = plain.price(Put, 100.0, SPOT, 0.04, 0.0, VOL).unwrap();
        assert!(put > plain_put + 0.3, "put {put} vs {plain_put}");
    }

    #[test]
    fn dividends_are_only_accepted_by_the_finite_difference_kind() {
        for kind in [
            PricerKind::JuQuadratic,
            PricerKind::BaroneAdesiWhaley,
            PricerKind::Binomial(100),
        ] {
            assert!(AmericanPricer::new(kind, 1.0, &[(0.25, 1.0)]).is_err());
            assert!(AmericanPricer::new(kind, 1.0, &[]).is_ok());
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let kind = PricerKind::JuQuadratic;
        assert!(AmericanPricer::new(kind, 0.0, &[]).is_err());
        assert!(AmericanPricer::new(kind, -1.0, &[]).is_err());
        assert!(AmericanPricer::new(kind, Real::NAN, &[]).is_err());
        let fd = PricerKind::FiniteDifference {
            t_grid: 50,
            x_grid: 100,
        };
        assert!(AmericanPricer::new(fd, 1.0, &[(1.5, 1.0)]).is_err());
        assert!(AmericanPricer::new(fd, 1.0, &[(0.0, 1.0)]).is_err());
        assert!(AmericanPricer::new(fd, 1.0, &[(0.5, -1.0)]).is_err());

        let pricer = AmericanPricer::new(kind, 1.0, &[]).unwrap();
        assert!(pricer.price(Put, 100.0, 0.0, 0.05, 0.0, 0.2).is_err());
        assert!(pricer.price(Put, 0.0, 100.0, 0.05, 0.0, 0.2).is_err());
        assert!(pricer.price(Put, 100.0, 100.0, 0.05, 0.0, 0.0).is_err());
        assert!(
            pricer
                .price(Put, 100.0, 100.0, Real::NAN, 0.0, 0.2)
                .is_err()
        );
    }
}
