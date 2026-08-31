//! Forward (strike-resetting) vanilla-option engine.
//!
//! Port of `ql/pricingengines/forward/forwardengine.hpp` specialised to
//! `ForwardVanillaEngine<AnalyticEuropeanEngine>`.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{
    ForwardVanillaArguments, ForwardVanillaEngineBase, OneAssetOptionResults, PlainVanillaPayoff,
    StrikedTypePayoff,
};
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::ImpliedVolTermStructure;
use crate::termstructures::yields::ImpliedTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;

/// `ForwardVanillaEngine<AnalyticEuropeanEngine>`.
pub struct ForwardVanillaEngine {
    base: ForwardVanillaEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl ForwardVanillaEngine {
    /// Builds the engine on a Black-Scholes process.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = ForwardVanillaEngineBase::new(
            ForwardVanillaArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for ForwardVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for ForwardVanillaEngine {
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
        let results = forward_vanilla_calculate(&self.process, self.base.arguments())?;
        *self.base.results_mut() = results;
        Ok(())
    }
}

/// Shared forward-vanilla calculation used by [`ForwardVanillaEngine`] and
/// [`QuantoForwardEuropeanEngine`](crate::pricingengines::QuantoForwardEuropeanEngine).
pub fn forward_vanilla_calculate(
    process: &GeneralizedBlackScholesProcess,
    arguments: &ForwardVanillaArguments,
) -> QlResult<OneAssetOptionResults> {
    arguments.validate()?;
    let payoff = arguments.payoff.as_ref().unwrap();
    let exercise = arguments.exercise.as_ref().unwrap();
    let moneyness = arguments.moneyness.unwrap();
    let reset_date = arguments.reset_date.unwrap();

    let spot = process.x0()?;
    if spot.is_nan() || spot <= 0.0 {
        fail!("negative or null underlying given");
    }
    let strike = moneyness * spot;
    let reset_payoff = shared(PlainVanillaPayoff::new(payoff.option_type(), strike))
        as Shared<dyn StrikedTypePayoff>;

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
    ))
        as Shared<dyn crate::termstructures::volatility::BlackVolTermStructure>);

    let fwd_process = shared(GeneralizedBlackScholesProcess::new(
        process.state_variable(),
        dividend_yield,
        risk_free_rate,
        black_volatility,
    ));

    let mut original = AnalyticEuropeanEngine::new(fwd_process);
    let original_results =
        original.calculate_from_arguments(Shared::clone(&reset_payoff), Shared::clone(exercise))?;
    let inner_value = original_results.instrument.value;
    let mut greeks = original_results.greeks;
    let more_greeks = original_results.more_greeks;
    let error_estimate = original_results.instrument.error_estimate;
    let valuation_date = original_results.instrument.valuation_date;
    let additional_results = original_results.instrument.additional_results.clone();

    let dividend = process.dividend_yield().current_link()?;
    let risk_free = process.risk_free_rate().current_link()?;
    let rfdc = risk_free.require_day_counter()?;
    let divdc = dividend.require_day_counter()?;
    let reset_time = rfdc.year_fraction(risk_free.reference_date()?, reset_date);
    let disc_q = dividend.discount_date(reset_date, false)?;

    let value = disc_q * inner_value.unwrap_or(0.0);

    if let (Some(delta), Some(strike_sensitivity)) = (greeks.delta, more_greeks.strike_sensitivity)
    {
        greeks.delta = Some(disc_q * (delta + moneyness * strike_sensitivity));
    }
    greeks.gamma = Some(0.0);
    let q_zero = dividend
        .zero_rate_date(
            reset_date,
            divdc,
            Compounding::Continuous,
            Frequency::NoFrequency,
            false,
        )?
        .rate();
    greeks.theta = Some(q_zero * value);
    if let Some(vega) = greeks.vega {
        greeks.vega = Some(disc_q * vega);
    }
    if let Some(rho) = greeks.rho {
        greeks.rho = Some(disc_q * rho);
    }
    if let Some(div_rho) = greeks.dividend_rho {
        greeks.dividend_rho = Some(-reset_time * value + disc_q * div_rho);
    }

    Ok(OneAssetOptionResults {
        instrument: crate::instrument::InstrumentResults {
            value: Some(value),
            error_estimate,
            valuation_date,
            additional_results,
        },
        greeks,
        more_greeks,
    })
}
