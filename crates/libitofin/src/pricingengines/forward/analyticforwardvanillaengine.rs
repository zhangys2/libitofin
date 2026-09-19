//! Analytic forward vanilla engine.
//!
//! Port of `ql/pricingengines/forward/forwardengine.hpp` specialised to
//! `ForwardVanillaEngine<AnalyticEuropeanEngine>`: rebases the Black–Scholes
//! process to the reset date via implied yield/vol curves, prices a vanilla
//! with strike `moneyness × spot`, then scales NPV and greeks by the dividend
//! discount to the reset date.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    ForwardOptionArguments, Greeks, MoreGreeks, OneAssetOptionResults, PlainVanillaPayoff,
};
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::{AnalyticEuropeanEngine, BinomialVanillaEngine};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackVolTermStructure, ImpliedVolTermStructure};
use crate::termstructures::yields::ImpliedTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size};

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// `ForwardVanillaEngine<AnalyticEuropeanEngine>`.
pub struct AnalyticForwardVanillaEngine {
    base: ForwardEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticForwardVanillaEngine {
    /// `ForwardVanillaEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }

    /// Fills arguments and calculates; used by the quanto forward engine.
    pub(crate) fn calculate_from_arguments(
        &mut self,
        arguments: &ForwardOptionArguments,
    ) -> QlResult<&OneAssetOptionResults> {
        {
            let dest = self.base.arguments_mut();
            dest.payoff = arguments.payoff.clone();
            dest.exercise = arguments.exercise.clone();
            dest.moneyness = arguments.moneyness;
            dest.reset_date = arguments.reset_date;
            dest.settings = arguments.settings.clone();
        }
        PricingEngine::calculate(self)?;
        Ok(self.base.results())
    }
}

impl AsObservable for AnalyticForwardVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticForwardVanillaEngine {
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
        let (option_type, moneyness, reset_date, exercise) = {
            let args = self.base.arguments();
            let Some(payoff) = args.payoff.as_ref() else {
                fail!("no payoff given");
            };
            let Some(exercise) = args.exercise.as_ref() else {
                fail!("no exercise given");
            };
            let Some(moneyness) = args.moneyness else {
                fail!("null moneyness given");
            };
            (
                payoff.option_type(),
                moneyness,
                args.reset_date,
                Shared::clone(exercise),
            )
        };

        let spot = self.process.x0()?;
        if spot.is_nan() || spot <= 0.0 {
            fail!("negative or null underlying given");
        }

        let payoff = shared(PlainVanillaPayoff::new(option_type, moneyness * spot))
            as Shared<dyn crate::instruments::StrikedTypePayoff>;

        let fwd_process = rebase_forward_process(&self.process, reset_date)?;
        let (value, original_greeks, more_greeks) = {
            let mut original = AnalyticEuropeanEngine::new(fwd_process);
            let original_results = original
                .calculate_from_arguments(Shared::clone(&payoff), Shared::clone(&exercise))?;
            (
                original_results.instrument.value,
                original_results.greeks,
                original_results.more_greeks,
            )
        };
        write_forward_results(
            &self.process,
            reset_date,
            moneyness,
            value,
            original_greeks,
            more_greeks,
            self.base.results_mut(),
        )
    }
}

/// Attach an [`AnalyticForwardVanillaEngine`] to a forward vanilla option.
pub fn set_analytic_forward_vanilla_engine(
    option: &mut crate::instruments::ForwardVanillaOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine =
        shared_mut(AnalyticForwardVanillaEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

/// `ForwardVanillaEngine<BinomialVanillaEngine<CoxRossRubinstein>>`.
pub struct BinomialForwardVanillaEngine {
    base: ForwardEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Size,
}

impl BinomialForwardVanillaEngine {
    /// `ForwardVanillaEngine<TestBinomialEngine>` (`time_steps` is 300 in QL).
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Size,
    ) -> QlResult<Self> {
        require!(
            time_steps >= 2,
            "at least 2 time steps required, {time_steps} provided"
        );
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Ok(Self {
            base,
            process,
            time_steps,
        })
    }
}

impl AsObservable for BinomialForwardVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BinomialForwardVanillaEngine {
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
        let (option_type, moneyness, reset_date, exercise) = {
            let args = self.base.arguments();
            let Some(payoff) = args.payoff.as_ref() else {
                fail!("no payoff given");
            };
            let Some(exercise) = args.exercise.as_ref() else {
                fail!("no exercise given");
            };
            let Some(moneyness) = args.moneyness else {
                fail!("null moneyness given");
            };
            (
                payoff.option_type(),
                moneyness,
                args.reset_date,
                Shared::clone(exercise),
            )
        };

        let spot = self.process.x0()?;
        if spot.is_nan() || spot <= 0.0 {
            fail!("negative or null underlying given");
        }

        let payoff = shared(PlainVanillaPayoff::new(option_type, moneyness * spot))
            as Shared<dyn crate::instruments::StrikedTypePayoff>;
        let fwd_process = rebase_forward_process(&self.process, reset_date)?;
        let (value, original_greeks, more_greeks) = {
            let mut original = BinomialVanillaEngine::new(fwd_process, self.time_steps)?;
            let original_results = original
                .calculate_from_arguments(Shared::clone(&payoff), Shared::clone(&exercise))?;
            (
                original_results.instrument.value,
                original_results.greeks,
                original_results.more_greeks,
            )
        };
        write_forward_results(
            &self.process,
            reset_date,
            moneyness,
            value,
            original_greeks,
            more_greeks,
            self.base.results_mut(),
        )
    }
}

fn rebase_forward_process(
    process: &GeneralizedBlackScholesProcess,
    reset_date: Date,
) -> QlResult<Shared<GeneralizedBlackScholesProcess>> {
    let dividend_yield = Handle::new(shared(ImpliedTermStructure::new(
        process.dividend_yield(),
        reset_date,
    )) as Shared<dyn YieldTermStructure>);
    let risk_free_rate = Handle::new(shared(ImpliedTermStructure::new(
        process.risk_free_rate(),
        reset_date,
    )) as Shared<dyn YieldTermStructure>);
    let black_volatility = Handle::new(shared(ImpliedVolTermStructure::new(
        process.black_volatility(),
        reset_date,
    )) as Shared<dyn BlackVolTermStructure>);
    Ok(shared(GeneralizedBlackScholesProcess::new(
        process.state_variable(),
        dividend_yield,
        risk_free_rate,
        black_volatility,
    )))
}

fn write_forward_results(
    process: &GeneralizedBlackScholesProcess,
    reset_date: Date,
    moneyness: Real,
    value: Option<Real>,
    original_greeks: Greeks,
    more_greeks: MoreGreeks,
    results: &mut OneAssetOptionResults,
) -> QlResult<()> {
    let rfdc = process
        .risk_free_rate()
        .current_link()?
        .require_day_counter()?;
    let div_curve = process.dividend_yield().current_link()?;
    let divdc = div_curve.require_day_counter()?;
    let rf_ref = process.risk_free_rate().current_link()?.reference_date()?;
    let reset_time = rfdc.year_fraction(rf_ref, reset_date);
    let disc_q = div_curve.discount_date(reset_date, true)?;

    let mut greeks = Greeks::default();
    results.instrument.value = value.map(|v| disc_q * v);

    if let (Some(delta), Some(strike_sens)) =
        (original_greeks.delta, more_greeks.strike_sensitivity)
    {
        greeks.delta = Some(disc_q * (delta + moneyness * strike_sens));
    }
    greeks.gamma = Some(0.0);
    if let Some(value) = results.instrument.value {
        let q_zero = div_curve
            .zero_rate_date(
                reset_date,
                divdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                true,
            )?
            .rate();
        greeks.theta = Some(q_zero * value);
    }
    if let Some(vega) = original_greeks.vega {
        greeks.vega = Some(disc_q * vega);
    }
    if let Some(rho) = original_greeks.rho {
        greeks.rho = Some(disc_q * rho);
    }
    if let (Some(value), Some(div_rho)) = (results.instrument.value, original_greeks.dividend_rho) {
        greeks.dividend_rho = Some(-reset_time * value + disc_q * div_rho);
    }
    results.greeks = greeks;
    results.more_greeks = MoreGreeks::default();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::{Rate, Time, Volatility};

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    fn flat_rate(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::new(today(), None, vol, Actual360::new()))
                as Shared<dyn BlackVolTermStructure>,
        )
    }

    /// `forwardoption.cpp` `testValues` — Haug p.37 / VBA (tol 1e-4).
    #[test]
    fn haug_forward_vanilla_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let rows = [(OptionType::Call, 4.4064), (OptionType::Put, 8.2971)];
        for (option_type, expected) in rows {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(60.0)) as Shared<dyn Quote>),
                flat_rate(0.04),
                flat_rate(0.08),
                flat_vol(0.30),
            ));
            let payoff = shared(PlainVanillaPayoff::new(option_type, 0.0))
                as Shared<dyn crate::instruments::StrikedTypePayoff>;
            let exercise = shared(EuropeanExercise::new(today() + time_to_days(1.0)));
            let mut option = ForwardVanillaOption::new(
                1.1,
                today() + time_to_days(0.25),
                payoff,
                exercise,
                Shared::clone(&settings),
            );
            set_analytic_forward_vanilla_engine(&mut option, process);
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 1.0e-4,
                "{option_type:?}: {calculated} vs {expected}"
            );
        }
    }

    /// `forwardoption.cpp` `testGreeksInitialization`: inner CRR binomial
    /// omits δ/ρ/divRho/vega, so `ForwardVanillaEngine` must omit them too.
    #[test]
    fn forward_greeks_initialization() {
        use crate::exercise::Exercise;
        use crate::instruments::VanillaOption;
        use crate::pricingengine::PricingEngine;
        use crate::pricingengines::vanilla::BinomialVanillaEngine;
        use crate::shared::{SharedMut, shared_mut};
        use crate::time::period::Period;
        use crate::time::timeunit::TimeUnit;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            flat_rate(0.04),
            flat_rate(0.01),
            flat_vol(0.11),
        ));
        let payoff = shared(PlainVanillaPayoff::new(OptionType::Call, 0.0))
            as Shared<dyn crate::instruments::StrikedTypePayoff>;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(
            today() + Period::new(1, TimeUnit::Years),
        ));
        let reset = today() + Period::new(6, TimeUnit::Months);

        let mut ctrl = VanillaOption::new(
            Shared::clone(&payoff),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        ctrl.base_mut().set_pricing_engine(shared_mut(
            BinomialVanillaEngine::new(Shared::clone(&process), 300).unwrap(),
        ) as SharedMut<dyn PricingEngine>);

        let mut option = ForwardVanillaOption::new(0.9, reset, payoff, exercise, settings);
        option.base_mut().set_pricing_engine(shared_mut(
            BinomialForwardVanillaEngine::new(process, 300).unwrap(),
        ) as SharedMut<dyn PricingEngine>);

        let check = |name: &str,
                     ctrl: crate::errors::QlResult<Real>,
                     fwd: crate::errors::QlResult<Real>| {
            if ctrl.is_err() {
                assert!(fwd.is_err(), "Forward {name} invalid");
            }
        };
        check("delta", ctrl.delta(), option.delta());
        check("rho", ctrl.rho(), option.rho());
        check("dividendRho", ctrl.dividend_rho(), option.dividend_rho());
        check("vega", ctrl.vega(), option.vega());
    }
}
