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
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{shared, Shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackVolTermStructure, ImpliedVolTermStructure};
use crate::termstructures::yields::ImpliedTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;

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

        let dividend_yield = Handle::new(shared(ImpliedTermStructure::new(
            self.process.dividend_yield(),
            reset_date,
        )) as Shared<dyn YieldTermStructure>);
        let risk_free_rate = Handle::new(shared(ImpliedTermStructure::new(
            self.process.risk_free_rate(),
            reset_date,
        )) as Shared<dyn YieldTermStructure>);
        let black_volatility = Handle::new(shared(ImpliedVolTermStructure::new(
            self.process.black_volatility(),
            reset_date,
        )) as Shared<dyn BlackVolTermStructure>);

        let fwd_process = shared(GeneralizedBlackScholesProcess::new(
            self.process.state_variable(),
            dividend_yield,
            risk_free_rate,
            black_volatility,
        ));

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

        let rfdc = self
            .process
            .risk_free_rate()
            .current_link()?
            .require_day_counter()?;
        let div_curve = self.process.dividend_yield().current_link()?;
        let divdc = div_curve.require_day_counter()?;
        let rf_ref = self
            .process
            .risk_free_rate()
            .current_link()?
            .reference_date()?;
        let reset_time = rfdc.year_fraction(rf_ref, reset_date);
        let disc_q = div_curve.discount_date(reset_date, true)?;

        let mut greeks = Greeks::default();
        let results = self.base.results_mut();
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
        if let (Some(value), Some(div_rho)) =
            (results.instrument.value, original_greeks.dividend_rho)
        {
            greeks.dividend_rho = Some(-reset_time * value + disc_q * div_rho);
        }
        results.greeks = greeks;
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

/// Attach an [`AnalyticForwardVanillaEngine`] to a forward vanilla option.
pub fn set_analytic_forward_vanilla_engine(
    option: &mut crate::instruments::ForwardVanillaOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    use crate::shared::{shared_mut, SharedMut};
    let engine =
        shared_mut(AnalyticForwardVanillaEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}
