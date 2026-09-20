//! Merton-76 jump-diffusion engine for European vanilla options.
//!
//! Port of `ql/pricingengines/vanilla/jumpdiffusionengine.{hpp,cpp}`: Poisson
//! mixture of Black–Scholes prices with jump-adjusted rate and vol. NPV only
//! this slice (greeks / `testGreeks` deferred). Convergence uses the value
//! addendum; QL also folds δ/γ/θ/ν/ρ/divρ into `lastContribution`. Shifted
//! `FlatForward` uses the risk-free day counter (QL uses the vol day counter).

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
        let vol_ref = black_vol.reference_date()?;
        let t = voldc.year_fraction(vol_ref, last);
        let risk_free = process.risk_free_rate().current_link()?;
        let rfdc = risk_free.require_day_counter()?;
        let rate_ref = risk_free.reference_date()?;
        let t_rate = rfdc.year_fraction(rate_ref, last);
        let risk_free_rate = -risk_free.discount_date(last, false)?.ln() / t_rate;

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
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);
            vol_link.link_to(shared(BlackConstantVol::new(
                vol_ref,
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
    use crate::time::daycounter::DayCounter;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }
    fn quote(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }
    #[rustfmt::skip]
    fn yts(rate: Real, dc: DayCounter, d: Date) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            d, rate, dc, Compounding::Continuous, Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    fn price(k: Real, days: i32, vol: Real, lam: Real, mu: Real, jv: Real, rdc: DayCounter, vdc: DayCounter, rd: Date, vd: Date) -> (Real, Shared<Merton76Process>, Shared<Settings<Date>>) {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        let p = shared(Merton76Process::new(
            quote(100.0), yts(0.0, rdc.clone(), rd), yts(0.08, rdc, rd),
            Handle::new(shared(BlackConstantVol::new(vd, None, vol, vdc)) as Shared<dyn BlackVolTermStructure>),
            quote(lam), quote(mu), quote(jv),
        ));
        let mut o = EuropeanOption::new(
            shared(PlainVanillaPayoff::new(Call, k)),
            shared(EuropeanExercise::new(today() + days)),
            Shared::clone(&s),
        );
        o.base_mut().set_pricing_engine(
            shared_mut(JumpDiffusionEngine::new(Shared::clone(&p))) as SharedMut<dyn PricingEngine>,
        );
        (o.npv().unwrap(), p, s)
    }

    #[test]
    #[rustfmt::skip]
    fn merton76_haug_and_compensator() {
        let a360 = Actual360::new();
        let d = today();
        let rows: [(Real, Real, Real, Real, Real); 3] = [
            (80.0, 0.10, 1.0, 0.25, 20.67),
            (80.0, 0.50, 10.0, 0.25, 23.61),
            (100.0, 0.25, 10.0, 0.75, 5.85),
        ];
        for (k, t, lam, g, exp) in rows {
            let jv = 0.25 * (g / lam).sqrt();
            let (npv, _, _) = price(
                k, (t * 360.0).round() as i32, 0.25 * (1.0 - g).sqrt(), lam,
                -0.5 * jv * jv, jv, a360.clone(), a360.clone(), d, d,
            );
            assert!((npv - exp).abs() <= 1e-2, "{k} {lam} {g}: {npv} vs {exp}");
        }
        let (npv, _, _) = price(100.0, 360, 0.20, 1.0, 0.20, 0.25, a360.clone(), a360, d, d);
        assert!((npv - 18.93059).abs() <= 1e-4, "{npv}");
    }

    #[test]
    #[rustfmt::skip]
    fn risk_free_time_basis_matches_analytic_european() {
        let (jd, p, s) = price(
            100.0, 365, 0.20, 0.0, 0.0, 0.0,
            Actual365Fixed::new(), Actual360::new(), today() - 10, today(),
        );
        let mut eu = EuropeanOption::new(
            shared(PlainVanillaPayoff::new(Call, 100.0)),
            shared(EuropeanExercise::new(today() + 365)),
            s,
        );
        eu.base_mut().set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(shared(
            GeneralizedBlackScholesProcess::new(
                p.state_variable(), p.dividend_yield(), p.risk_free_rate(), p.black_volatility(),
            ),
        ))) as SharedMut<dyn PricingEngine>);
        let ae = eu.npv().unwrap();
        assert!((jd - ae).abs() <= 1e-10, "{jd} vs {ae}");
    }
}
