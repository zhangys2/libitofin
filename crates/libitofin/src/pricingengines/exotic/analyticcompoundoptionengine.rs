//! Analytic compound-option engine (Wystup 2002).
//!
//! Port of `ql/pricingengines/exotic/analyticcompoundoptionengine.{hpp,cpp}`.
//! NPV only; δ/γ/ν/θ follow-up. Critical spot via Brent + `black_formula`.

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{CompoundArguments, CompoundResults, StrikedTypePayoff, TypePayoff};
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<CompoundArguments, CompoundResults>;

/// Pricing engine for European compound options.
pub struct AnalyticCompoundOptionEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    n: CumulativeNormalDistribution,
}

impl AnalyticCompoundOptionEngine {
    /// `AnalyticCompoundOptionEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(CompoundArguments::default(), CompoundResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            n: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticCompoundOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticCompoundOptionEngine {
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
        let a = self.base.arguments();
        let m_pay = a.mother_payoff.expect("validated");
        let d_pay = a.daughter_payoff.expect("validated");
        let m_ex = a.mother_exercise.as_ref().expect("validated");
        let d_ex = a.daughter_exercise.as_ref().expect("validated");
        let str_m = m_pay.strike();
        let str_d = d_pay.strike();
        require!(str_d > 0.0, "Daughter strike must be positive");
        require!(str_m > 0.0, "Mother strike must be positive");
        let s = self.process.x0()?;
        require!(s > 0.0, "negative or null underlying given");
        let phi = Real::from(d_pay.option_type() as i32);
        let w = Real::from(m_pay.option_type() as i32);

        let rf = self.process.risk_free_rate().current_link()?;
        let qy = self.process.dividend_yield().current_link()?;
        let vol = self.process.black_volatility().current_link()?;
        let mat_m = m_ex.last_date();
        let mat_d = d_ex.last_date();
        let t_m = self.process.time(&mat_m)?;
        let t_d = self.process.time(&mat_d)?;
        let help_mat = rf.reference_date()? + (mat_d - mat_m);
        let t_help = self.process.time(&help_mat)?;
        let std_help = vol.black_vol_date(help_mat, str_d, true)? * t_help.sqrt();
        let q_help = qy.discount_date(help_mat, false)?;
        let r_help = rf.discount_date(help_mat, false)?;
        let d_ty = d_pay.option_type();
        let solved = Brent::new().with_max_evaluations(1000).solve_bracketed(
            |spot| {
                let fwd = spot * q_help / r_help;
                black_formula(d_ty, str_d, fwd, std_help, r_help, 0.0).unwrap_or(Real::NAN) - str_m
            },
            1.0e-6,
            str_d,
            1.0e-6,
            str_d * 1000.0,
        )?;

        let dd_d = qy.discount(t_d, false)?;
        let rd_d = rf.discount(t_d, false)?;
        let dd_m = qy.discount(t_m, false)?;
        let rd_m = rf.discount(t_m, false)?;
        let sd_m = vol.black_vol_date(mat_m, str_m, true)? * t_m.sqrt();
        let sd_d = vol.black_vol_date(mat_d, str_d, true)? * t_d.sqrt();
        let x = ((rd_m * solved / (s * dd_m)) * (0.5 * sd_m * sd_m).exp()).ln() / sd_m;
        let fwd_d = s * dd_d / rd_d;
        let d_plus = (fwd_d / str_d).ln() / sd_d + 0.5 * sd_d;
        let d_minus = d_plus - sd_d;
        let n2 = BivariateCumulativeNormalDistributionDr78::new(w * (t_m / t_d).sqrt())?;
        let value = phi * w * s * dd_d * n2.value(-phi * w * (x - sd_m), phi * d_plus)
            - phi * w * str_d * rd_d * n2.value(-phi * w * x, phi * d_minus)
            - w * str_m * rd_m * self.n.value(-phi * w * x);
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticCompoundOptionEngine`] to `option`.
pub fn set_analytic_compound_option_engine(
    option: &mut crate::instruments::CompoundOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticCompoundOptionEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{CompoundOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            reference,
            None,
            quote_handle(quote),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    fn price(tm: OptionType, td: OptionType, km: Real, kd: Real, s: Real, q: Real, r: Real, t_m: Real, t_d: Real, v: Real) -> Real {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(s))),
            flat_rate(today, &shared(SimpleQuote::new(q))),
            flat_rate(today, &shared(SimpleQuote::new(r))),
            flat_vol(today, &shared(SimpleQuote::new(v))),
        ));
        let mother: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + (t_m * 360.0).round() as i32));
        let daughter: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + (t_d * 360.0).round() as i32));
        let mut option = CompoundOption::new(
            PlainVanillaPayoff::new(tm, km), mother, PlainVanillaPayoff::new(td, kd), daughter, settings,
        );
        set_analytic_compound_option_engine(&mut option, process);
        option.npv().unwrap()
    }

    /// `compoundoption.cpp` `testValues` NPV subset (Haug / sitmo @ 1e-3).
    #[test]
    fn compound_option_haug_npv() {
        use OptionType::{Call, Put};
        type Row = (OptionType, OptionType, Real, Real, Real);
        #[rustfmt::skip]
        let rows: [Row; 4] = [
            (Put,  Call, 50.0, 520.0, 21.1965),
            (Call, Call, 50.0, 520.0, 17.5945),
            (Call, Put,  50.0, 520.0, 18.7128),
            (Put,  Put,  50.0, 520.0, 15.2601),
        ];
        for (tm, td, km, kd, expected) in rows {
            let got = price(tm, td, km, kd, 500.0, 0.03, 0.08, 0.25, 0.5, 0.35);
            assert!(
                (got - expected).abs() <= 1e-3,
                "{tm:?} on {td:?}: expected {expected}, got {got}"
            );
        }
    }
}
