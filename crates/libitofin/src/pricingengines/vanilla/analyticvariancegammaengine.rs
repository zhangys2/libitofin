//! Analytic European variance-gamma engine.
//!
//! Port of `ql/experimental/variancegamma/analyticvariancegammaengine.{hpp,cpp}`.
//! The price is the gamma-mixture of Black–Scholes values. The FFT engine, and
//! the Monte Carlo and finite-difference variance-gamma engines, are absent.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    OneAssetOptionEngine, OneAssetOptionResults, OptionArguments, StrikedTypePayoff,
};
use crate::math::gammafunction::log_gamma;
use crate::math::integrals::Integrator;
use crate::math::integrals::kronrod::GaussKronrodNonAdaptive;
use crate::math::integrals::lobatto::GaussLobattoIntegral;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::VarianceGammaProcess;
use crate::require;
use crate::shared::Shared;
use crate::types::Real;

type EngineBase = OneAssetOptionEngine;

/// European variance-gamma prices by integrating Black–Scholes against the gamma density.
pub struct VarianceGammaEngine {
    base: EngineBase,
    process: Shared<VarianceGammaProcess>,
    abs_err: Real,
}

impl VarianceGammaEngine {
    /// Default absolute tolerance `1e-5`, QuantLib's constructor default.
    pub fn new(process: Shared<VarianceGammaProcess>) -> Self {
        Self::build(process, 1.0e-5)
    }

    /// `VarianceGammaEngine(process, absoluteError)`.
    ///
    /// # Errors
    ///
    /// Rejects a non-positive or non-finite absolute error.
    pub fn with_absolute_error(
        process: Shared<VarianceGammaProcess>,
        absolute_error: Real,
    ) -> QlResult<Self> {
        require!(
            absolute_error.is_finite() && absolute_error > 0.0,
            "absolute error must be positive"
        );
        Ok(Self::build(process, absolute_error))
    }

    fn build(process: Shared<VarianceGammaProcess>, abs_err: Real) -> Self {
        let base = EngineBase::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            abs_err,
        }
    }
}

impl AsObservable for VarianceGammaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for VarianceGammaEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        let (exercise, payoff) = {
            let args = self.base.arguments();
            let Some(exercise) = args.exercise.clone() else {
                fail!("no exercise given");
            };
            require!(
                exercise.exercise_type() == ExerciseType::European,
                "not an European Option"
            );
            let Some(payoff) = args.payoff.clone() else {
                fail!("non-striked payoff given");
            };
            (exercise, payoff)
        };

        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let maturity = exercise.last_date();
        let risk_free_discount = risk_free.discount_date(maturity, false)?;
        let dividend_discount = dividend.discount_date(maturity, false)?;
        let t = risk_free
            .require_day_counter()?
            .year_fraction(risk_free.reference_date()?, maturity);
        require!(
            t > 0.0,
            "variance-gamma maturity must be after the reference date"
        );
        let spot = self.process.x0()?;
        let integrand = Integrand::new(
            payoff,
            spot,
            t,
            risk_free_discount,
            dividend_discount,
            self.process.sigma(),
            self.process.nu(),
            self.process.theta(),
        )?;

        let mut infinity = 15.0 * (self.process.nu() * t).sqrt();
        let target = self.abs_err * 1.0e-4;
        let mut tail = integrand.at(infinity);
        let mut expansions = 0usize;
        loop {
            if tail.is_nan() {
                if let Some(message) = integrand.take_error() {
                    fail!("{message}");
                }
                fail!("non-finite variance-gamma tail");
            }
            if tail.abs() <= target {
                break;
            }
            require!(expansions < 40, "variance-gamma tail did not decay");
            infinity *= 1.5;
            tail = integrand.at(infinity);
            expansions += 1;
        }
        if let Some(message) = integrand.take_error() {
            fail!("{message}");
        }
        require!(
            infinity.is_finite() && tail.is_finite(),
            "non-finite variance-gamma tail"
        );

        // QuantLib returns the 87-point value even when the error estimate
        // misses `absErr`. The strict integrator still fails for every other caller.
        let near = GaussKronrodNonAdaptive::new(self.abs_err, 0.0)?
            .allowing_excess_error()
            .integrate(|x| integrand.at(x), 0.0, 0.1)?;
        let far = GaussLobattoIntegral::new(2000, self.abs_err)?.integrate(
            |x| integrand.at(x),
            0.1,
            infinity,
        )?;
        if let Some(message) = integrand.take_error() {
            fail!("{message}");
        }
        let value = near + far;
        require!(value.is_finite(), "non-finite variance-gamma price");
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

struct Integrand {
    payoff: Shared<dyn StrikedTypePayoff>,
    spot: Real,
    time: Real,
    risk_free_discount: Real,
    dividend_discount: Real,
    sigma: Real,
    nu: Real,
    theta: Real,
    omega: Real,
    shape: Real,
    log_gamma: Real,
    error: RefCell<Option<String>>,
}

impl Integrand {
    #[allow(clippy::too_many_arguments)]
    fn new(
        payoff: Shared<dyn StrikedTypePayoff>,
        spot: Real,
        time: Real,
        risk_free_discount: Real,
        dividend_discount: Real,
        sigma: Real,
        nu: Real,
        theta: Real,
    ) -> QlResult<Self> {
        let compensator = 1.0 - theta * nu - (sigma * sigma * nu) / 2.0;
        require!(
            compensator > 0.0,
            "variance-gamma compensator must be positive"
        );
        let shape = time / nu;
        let log_gamma = log_gamma(shape)?;
        require!(
            shape.is_finite() && log_gamma.is_finite() && nu.ln().is_finite(),
            "variance-gamma density is not representable"
        );
        Ok(Self {
            payoff,
            spot,
            time,
            risk_free_discount,
            dividend_discount,
            sigma,
            nu,
            theta,
            omega: compensator.ln() / nu,
            shape,
            log_gamma,
            error: RefCell::new(None),
        })
    }

    fn take_error(&self) -> Option<String> {
        self.error.borrow_mut().take()
    }

    fn at(&self, x: Real) -> Real {
        match self.price(x) {
            Ok(value) if value.is_finite() => value,
            Ok(_) => {
                self.record("non-finite variance-gamma integrand");
                Real::NAN
            }
            Err(err) => {
                self.record(err.message());
                Real::NAN
            }
        }
    }

    fn record(&self, message: impl Into<String>) {
        let mut slot = self.error.borrow_mut();
        if slot.is_none() {
            *slot = Some(message.into());
        }
    }

    fn price(&self, x: Real) -> QlResult<Real> {
        let s0_adj = self.spot
            * (self.theta * x + self.omega * self.time + (self.sigma * self.sigma * x) / 2.0).exp();
        // Keep QuantLib's two-step scaling. `sigma * sqrt(x / t) * sqrt(t)` is
        // not bit-identical to `sigma * sqrt(x)`.
        let mut vol_adj = self.sigma * (x / self.time).sqrt();
        vol_adj *= self.time.sqrt();
        let forward = s0_adj * self.dividend_discount / self.risk_free_discount;
        let black = BlackCalculator::with_striked_payoff(
            &*self.payoff,
            forward,
            vol_adj,
            self.risk_free_discount,
        )?;
        // Log-space gamma density. `exp(lgamma) * nu^(t/nu)` overflows for a
        // small `nu` even when the density itself is moderate.
        require!(
            x > 0.0,
            "variance-gamma density evaluated at a non-positive time"
        );
        let log_density =
            (self.shape - 1.0) * x.ln() - x / self.nu - self.log_gamma - self.shape * self.nu.ln();
        Ok(black.value() * log_density.exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{EuropeanOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::thirty360::{Convention as Thirty360Convention, Thirty360};
    use crate::time::frequency::Frequency;

    /// `variancegamma.cpp` `testVarianceGamma`, analytic engine only, tolerance 0.01.
    #[test]
    fn analytic_matches_the_known_european_values() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let processes = [
            (6000.0, 0.00, 0.05, 0.20, 0.05, -0.50),
            (6000.0, 0.02, 0.05, 0.15, 0.01, -0.50),
        ];
        let options = [
            (Call, 5550.0),
            (Call, 5600.0),
            (Call, 5650.0),
            (Call, 5700.0),
            (Call, 5750.0),
            (Call, 5800.0),
            (Call, 5850.0),
            (Call, 5900.0),
            (Call, 5950.0),
            (Call, 6000.0),
            (Call, 6050.0),
            (Call, 6100.0),
            (Call, 6150.0),
            (Call, 6200.0),
            (Call, 6250.0),
            (Call, 6300.0),
            (Call, 6350.0),
            (Call, 6400.0),
            (Call, 6450.0),
            (Call, 6500.0),
            (Call, 6550.0),
            (Put, 5550.0),
        ];
        let expected = [
            [
                955.1637, 922.7529, 890.9872, 859.8739, 829.4197, 799.6303, 770.5104, 742.0640,
                714.2943, 687.2032, 660.7921, 635.0613, 610.0103, 585.6379, 561.9416, 538.9186,
                516.5649, 494.8760, 473.8464, 453.4700, 433.7400, 234.4870,
            ],
            [
                732.8705, 698.5542, 665.1404, 632.6498, 601.1002, 570.5068, 540.8824, 512.2367,
                484.5766, 457.9064, 432.2273, 407.5381, 383.8346, 361.1102, 339.3559, 318.5599,
                298.7087, 279.7864, 261.7751, 244.6552, 228.4057, 130.9974,
            ],
        ];
        let expiry = today + 360;
        for (p, row) in processes.iter().zip(expected) {
            let (spot, q, r, sigma, nu, theta) = *p;
            let quote =
                |rate: Real| Handle::new(shared(SimpleQuote::new(rate)) as Shared<dyn Quote>);
            let flat = |rate: Real| {
                Handle::new(shared(FlatForward::new(
                    today,
                    quote(rate),
                    Actual360::new(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>)
            };
            let process = shared(
                VarianceGammaProcess::new(quote(spot), flat(q), flat(r), sigma, nu, theta).unwrap(),
            );
            let engine = shared_mut(VarianceGammaEngine::new(Shared::clone(&process)));
            for ((kind, strike), expected_npv) in options.iter().zip(row) {
                let mut option = EuropeanOption::new(
                    shared(PlainVanillaPayoff::new(*kind, *strike))
                        as Shared<dyn StrikedTypePayoff>,
                    shared(EuropeanExercise::new(expiry)),
                    Shared::clone(&settings),
                );
                option
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
                let calculated = option.npv().unwrap();
                let err = (calculated - expected_npv).abs();
                assert!(
                    err <= 0.01,
                    "s={spot} k={strike:?}: {calculated} vs {expected_npv} (err={err})"
                );
            }
        }
    }

    /// `testSingularityAtZero`: nine 30/360 days and `nu = 0.2` give `t/nu < 1`,
    /// so the density is singular at zero. Kronrod samples the open interval
    /// `(0, 0.1)` and Lobatto starts at `0.1`. The call must finish finite.
    #[test]
    fn short_dated_call_integrates_through_the_origin() {
        let today = Date::new(1, Month::January, 2017);
        let maturity = Date::new(10, Month::January, 2017);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Thirty360::with_convention(Thirty360Convention::BondBasis);
        let quote = |rate: Real| Handle::new(shared(SimpleQuote::new(rate)) as Shared<dyn Quote>);
        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::new(
                today,
                quote(rate),
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let process = shared(
            VarianceGammaProcess::new(quote(100.0), flat(0.0), flat(0.05), 0.12, 0.2, -0.14)
                .unwrap(),
        );
        let mut option = EuropeanOption::new(
            shared(PlainVanillaPayoff::new(Call, 98.0)) as Shared<dyn StrikedTypePayoff>,
            shared(EuropeanExercise::new(maturity)),
            settings,
        );
        option.base_mut().set_pricing_engine(
            shared_mut(VarianceGammaEngine::new(process)) as SharedMut<dyn PricingEngine>
        );
        let npv = option.npv().unwrap();
        assert!(npv.is_finite() && npv > 0.0, "{npv}");
    }

    /// `nu = 0.001` used to overflow `exp(lgamma) * nu^(t/nu)` before the density
    /// was formed. The log-space density must still produce a finite call.
    #[test]
    fn small_nu_call_stays_representable() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let quote = |rate: Real| Handle::new(shared(SimpleQuote::new(rate)) as Shared<dyn Quote>);
        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::new(
                today,
                quote(rate),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let process = shared(
            VarianceGammaProcess::new(quote(6000.0), flat(0.0), flat(0.05), 0.20, 0.001, -0.50)
                .unwrap(),
        );
        let mut option = EuropeanOption::new(
            shared(PlainVanillaPayoff::new(Call, 6000.0)) as Shared<dyn StrikedTypePayoff>,
            shared(EuropeanExercise::new(today + 360)),
            settings,
        );
        option.base_mut().set_pricing_engine(
            shared_mut(VarianceGammaEngine::new(process)) as SharedMut<dyn PricingEngine>
        );
        let npv = option.npv().unwrap();
        assert!(npv.is_finite() && npv > 0.0, "{npv}");
    }
}
