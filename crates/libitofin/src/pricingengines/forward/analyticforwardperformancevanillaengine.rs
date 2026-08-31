//! Analytic forward performance vanilla engine.
//!
//! Port of `ql/pricingengines/forward/forwardperformanceengine.hpp` specialised to
//! `ForwardPerformanceVanillaEngine<AnalyticEuropeanEngine>`: same reset-date
//! implied process as [`AnalyticForwardVanillaEngine`], but NPV and greeks are
//! scaled by the risk-free discount to reset divided by spot (performance option).

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
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackVolTermStructure, ImpliedVolTermStructure};
use crate::termstructures::yields::ImpliedTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// `ForwardPerformanceVanillaEngine<AnalyticEuropeanEngine>`.
pub struct AnalyticForwardPerformanceVanillaEngine {
    base: ForwardEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticForwardPerformanceVanillaEngine {
    /// `ForwardPerformanceVanillaEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }

    /// Fills arguments and calculates; used by the quanto forward performance engine.
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

impl AsObservable for AnalyticForwardPerformanceVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticForwardPerformanceVanillaEngine {
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

        let (inner_value, original_greeks) = {
            let mut original = AnalyticEuropeanEngine::new(fwd_process);
            let original_results = original
                .calculate_from_arguments(Shared::clone(&payoff), Shared::clone(&exercise))?;
            (original_results.instrument.value, original_results.greeks)
        };

        let risk_free = self.process.risk_free_rate().current_link()?;
        let rfdc = risk_free.require_day_counter()?;
        let rf_ref = risk_free.reference_date()?;
        let reset_time = rfdc.year_fraction(rf_ref, reset_date);
        let disc_r = risk_free.discount_date(reset_date, true)? / spot;

        let value = inner_value.map(|v| disc_r * v);
        let mut greeks = Greeks {
            delta: Some(0.0),
            gamma: Some(0.0),
            ..Greeks::default()
        };
        if let Some(v) = value {
            let r_zero = risk_free
                .zero_rate_date(
                    reset_date,
                    rfdc,
                    Compounding::Continuous,
                    Frequency::NoFrequency,
                    true,
                )?
                .rate();
            greeks.theta = Some(r_zero * v);
        }
        if let Some(vega) = original_greeks.vega {
            greeks.vega = Some(disc_r * vega);
        }
        if let (Some(v), Some(rho)) = (value, original_greeks.rho) {
            greeks.rho = Some(-reset_time * v + disc_r * rho);
        }
        if let Some(div_rho) = original_greeks.dividend_rho {
            greeks.dividend_rho = Some(disc_r * div_rho);
        }

        let results = self.base.results_mut();
        results.instrument.value = value;
        results.greeks = greeks;
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

/// Attach an [`AnalyticForwardPerformanceVanillaEngine`] to a forward vanilla option.
pub fn set_analytic_forward_performance_vanilla_engine(
    option: &mut crate::instruments::ForwardVanillaOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine = shared_mut(AnalyticForwardPerformanceVanillaEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}
