//! Merton-76 jump-diffusion engine for European vanilla options.
//!
//! Port of `ql/pricingengines/vanilla/jumpdiffusionengine.{hpp,cpp}`: Poisson
//! mixture of Black–Scholes prices with jump-adjusted rate and vol. NPV only
//! this slice (greeks / `testGreeks` deferred). Convergence uses the value
//! addendum; QL also folds δ/γ/θ/ν/ρ/divρ into `lastContribution`.

use super::AnalyticEuropeanEngine;
use crate::errors::QlResult;
use crate::fail;
use crate::handle::RelinkableHandle;
use crate::instruments::{OneAssetOptionEngine, OneAssetOptionResults, OptionArguments};
use crate::interestrate::Compounding;
use crate::math::distributions::poisson::PoissonDistribution;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::{GeneralizedBlackScholesProcess, Merton76Process};
use crate::require;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::termstructures::yields::FlatForward;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size};

/// Jump-diffusion engine (`jumpdiffusionengine.hpp`).
pub struct JumpDiffusionEngine {
    base: OneAssetOptionEngine,
    process: Shared<Merton76Process>,
    relative_accuracy: Real,
    max_iterations: Size,
}

impl JumpDiffusionEngine {
    /// `JumpDiffusionEngine(process)` with QL defaults `1e-4` / `100`.
    pub fn new(process: Shared<Merton76Process>) -> Self {
        Self::with_parameters(process, 1e-4, 100)
    }

    /// `JumpDiffusionEngine(process, relativeAccuracy, maxIterations)`.
    pub fn with_parameters(
        process: Shared<Merton76Process>,
        relative_accuracy: Real,
        max_iterations: Size,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            relative_accuracy,
            max_iterations,
        }
    }
}

impl AsObservable for JumpDiffusionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for JumpDiffusionEngine {
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
        let (payoff, exercise) = {
            let arguments = self.base.arguments();
            let Some(payoff) = arguments.payoff.clone() else {
                fail!("no payoff given");
            };
            let Some(exercise) = arguments.exercise.clone() else {
                fail!("no exercise given");
            };
            (payoff, exercise)
        };
        let last = exercise.last_date();
        let process = &*self.process;

        let jump_vol = process.log_jump_volatility().current_link()?.value()?;
        let jump_square_vol = jump_vol * jump_vol;
        let mu_plus_half_square_vol =
            process.log_mean_jump().current_link()?.value()? + 0.5 * jump_square_vol;
        let k = mu_plus_half_square_vol.exp() - 1.0;
        let intensity = process.jump_intensity().current_link()?.value()?;
        let lambda = (k + 1.0) * intensity;

        let black_vol = process.black_volatility().current_link()?;
        let variance = black_vol.black_variance_date(last, payoff.strike(), false)?;
        let voldc = black_vol.require_day_counter()?;
        let volcal = black_vol.calendar();
        let t = voldc.year_fraction(black_vol.reference_date()?, last);
        let risk_free = process.risk_free_rate().current_link()?;
        let risk_free_rate = -risk_free.discount_date(last, false)?.ln() / t;
        let rate_ref = risk_free.reference_date()?;

        let poisson = PoissonDistribution::new(lambda * t)?;
        let rf_link = RelinkableHandle::new(Shared::clone(&risk_free));
        let vol_link = RelinkableHandle::new(Shared::clone(&black_vol));
        let bs = shared(GeneralizedBlackScholesProcess::new(
            process.state_variable(),
            process.dividend_yield(),
            rf_link.handle(),
            vol_link.handle(),
        ));
        let mut base_engine = AnalyticEuropeanEngine::new(Shared::clone(&bs));

        let mut value = 0.0;
        let mut last_contribution = 1.0;
        let mut i: Size = 0;
        let min_terms = (lambda * t) as Size;
        while (last_contribution > self.relative_accuracy && i < self.max_iterations)
            || i < min_terms
        {
            let v = ((variance + i as Real * jump_square_vol) / t).sqrt();
            let r = risk_free_rate - intensity * k + (i as Real) * mu_plus_half_square_vol / t;
            rf_link.link_to(shared(FlatForward::with_rate(
                rate_ref,
                r,
                voldc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);
            vol_link.link_to(shared(BlackConstantVol::new(
                rate_ref,
                volcal.clone(),
                v,
                voldc.clone(),
            )) as Shared<dyn BlackVolTermStructure>);
            let term = {
                let results = base_engine
                    .calculate_from_arguments(Shared::clone(&payoff), Shared::clone(&exercise))?;
                let Some(term) = results.instrument.value else {
                    fail!("inner European engine returned no NPV");
                };
                term
            };
            let weight = poisson.pmf(i as u64);
            value += weight * term;
            last_contribution = (term
                / if value.abs() > Real::EPSILON {
                    value
                } else {
                    1.0
                })
            .abs()
                * weight;
            i += 1;
        }
        require!(
            i < self.max_iterations,
            "{i} iterations have been not enough to reach the required {} accuracy",
            self.relative_accuracy
        );
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{EuropeanOption, PlainVanillaPayoff};
    use crate::option::OptionType::Call;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared_mut};
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn quote(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    fn yts(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    /// `jumpdiffusion.cpp` `testMerton76` subset (Haug p.9, QL 1e-2 values).
    /// Columns: K, t, λ, γ, NPV. S=100, q=0, r=0.08, σ=0.25, mean-jump 0.
    #[rustfmt::skip]
    const ROWS: &[(Real, Real, Real, Real, Real)] = &[
        ( 80.0, 0.10,  1.0, 0.25, 20.67),
        ( 80.0, 0.50,  1.0, 0.25, 23.63),
        (100.0, 0.10,  1.0, 0.25,  3.42),
        (100.0, 0.25,  1.0, 0.25,  5.88),
        (100.0, 0.50,  1.0, 0.25,  8.95),
        (120.0, 0.10,  1.0, 0.25,  0.10),
        (120.0, 0.50,  1.0, 0.25,  2.23),
        (100.0, 0.25,  5.0, 0.25,  5.96),
        (100.0, 0.25, 10.0, 0.25,  5.97),
        ( 80.0, 0.50, 10.0, 0.25, 23.61), // Haug 23.28
        (100.0, 0.25,  1.0, 0.50,  5.58),
        (100.0, 0.25,  5.0, 0.50,  5.87),
        (100.0, 0.25,  1.0, 0.75,  5.08),
        (100.0, 0.25, 10.0, 0.75,  5.85),
        (110.0, 0.50,  5.0, 0.75,  4.57),
    ];

    #[test]
    fn merton76_haug_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        for &(k, t, intensity, gamma, expected) in ROWS {
            let j_vol = 0.25 * (gamma / intensity).sqrt();
            let diff_vol = 0.25 * (1.0 - gamma).sqrt();
            let mean_log = (1.0_f64).ln() - 0.5 * j_vol * j_vol;
            let process = shared(Merton76Process::new(
                quote(100.0),
                yts(0.0),
                yts(0.08),
                Handle::new(shared(BlackConstantVol::new(
                    today(),
                    None,
                    diff_vol,
                    Actual360::new(),
                )) as Shared<dyn BlackVolTermStructure>),
                quote(intensity),
                quote(mean_log),
                quote(j_vol),
            ));
            let mut option = EuropeanOption::new(
                shared(PlainVanillaPayoff::new(Call, k)),
                shared(EuropeanExercise::new(today() + (t * 360.0).round() as i32)),
                Shared::clone(&settings),
            );
            option.base_mut().set_pricing_engine(
                shared_mut(JumpDiffusionEngine::new(process)) as SharedMut<dyn PricingEngine>
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 1e-2,
                "K={k} t={t} λ={intensity} γ={gamma}: {calculated} vs {expected}"
            );
        }
    }
}
