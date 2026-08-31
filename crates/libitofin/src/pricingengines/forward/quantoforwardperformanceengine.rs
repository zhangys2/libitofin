//! Analytic quanto forward performance vanilla engine.
//!
//! Port of `ql/pricingengines/quanto/quantoengine.hpp` specialised to
//! `QuantoEngine<ForwardVanillaOption, ForwardPerformanceVanillaEngine<AnalyticEuropeanEngine>>`.

use std::any::Any;

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{ForwardOptionArguments, OneAssetOptionResults};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::forward::AnalyticForwardPerformanceVanillaEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// `QuantoEngine<ForwardVanillaOption, ForwardPerformanceVanillaEngine<AnalyticEuropeanEngine>>`.
pub struct QuantoForwardPerformanceEuropeanEngine {
    base: ForwardEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoForwardPerformanceEuropeanEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        let observer = base.observer();
        foreign_risk_free_rate.register_observer(&observer);
        exchange_rate_volatility.register_observer(&observer);
        correlation.register_observer(&observer);
        Self {
            base,
            process,
            foreign_risk_free_rate,
            exchange_rate_volatility,
            correlation,
        }
    }
}

impl AsObservable for QuantoForwardPerformanceEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoForwardPerformanceEuropeanEngine {
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
        let (strike, maturity) = {
            let args = self.base.arguments();
            let Some(payoff) = args.payoff.as_ref() else {
                fail!("no payoff given");
            };
            let Some(exercise) = args.exercise.as_ref() else {
                fail!("no exercise given");
            };
            (payoff.strike(), exercise.last_date())
        };
        let spot = self.process.state_variable().current_link()?.value()?;
        if spot.is_nan() || spot <= 0.0 {
            fail!("negative or null underlying");
        }

        const EXCHANGE_RATE_ATM: Real = 1.0;
        let correlation = self.correlation.current_link()?.value()?;
        let dividend_yield = Handle::new(shared(QuantoTermStructure::new(
            self.process.dividend_yield(),
            self.process.risk_free_rate(),
            self.foreign_risk_free_rate.clone(),
            self.process.black_volatility(),
            strike,
            self.exchange_rate_volatility.clone(),
            EXCHANGE_RATE_ATM,
            correlation,
        )) as Shared<dyn YieldTermStructure>);
        let quanto_process = shared(GeneralizedBlackScholesProcess::new(
            self.process.state_variable(),
            dividend_yield,
            self.process.risk_free_rate(),
            self.process.black_volatility(),
        ));

        let (value, mut greeks, more_greeks) = {
            let mut original = AnalyticForwardPerformanceVanillaEngine::new(quanto_process);
            let original_results = original.calculate_from_arguments(self.base.arguments())?;
            (
                original_results.instrument.value,
                original_results.greeks,
                original_results.more_greeks,
            )
        };

        let fx_flat_vol = self
            .exchange_rate_volatility
            .current_link()?
            .black_vol_date(maturity, EXCHANGE_RATE_ATM, false)?;
        let original_div_rho = greeks.dividend_rho;

        if let (Some(rho), Some(div_rho)) = (greeks.rho, original_div_rho) {
            greeks.rho = Some(rho + div_rho);
            greeks.dividend_rho = Some(div_rho);
        } else {
            greeks.rho = None;
            greeks.dividend_rho = None;
        }
        if let (Some(vega), Some(div_rho)) = (greeks.vega, original_div_rho) {
            greeks.vega = Some(vega + correlation * fx_flat_vol * div_rho);
        }

        let (qvega, qrho, qlambda) = if let Some(div_rho) = original_div_rho {
            let eq_vol = self
                .process
                .black_volatility()
                .current_link()?
                .black_vol_date(maturity, spot, false)?;
            (
                Some(correlation * eq_vol * div_rho),
                Some(-div_rho),
                Some(fx_flat_vol * eq_vol * div_rho),
            )
        } else {
            (None, None, None)
        };

        let results = self.base.results_mut();
        results.instrument.value = value;
        results.greeks = greeks;
        results.more_greeks = more_greeks;
        let extras = &mut results.instrument.additional_results;
        let mut extra = |tag: &str, value: Option<Real>| {
            if let Some(value) = value {
                extras.insert(tag.to_string(), shared(value) as Shared<dyn Any>);
            }
        };
        extra("qvega", qvega);
        extra("qrho", qrho);
        extra("qlambda", qlambda);
        Ok(())
    }
}

/// Attach a [`QuantoForwardPerformanceEuropeanEngine`] to a forward vanilla option.
pub fn set_quanto_forward_performance_european_engine(
    option: &mut crate::instruments::ForwardVanillaOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine = shared_mut(QuantoForwardPerformanceEuropeanEngine::new(
        process,
        foreign_risk_free_rate,
        exchange_rate_volatility,
        correlation,
    )) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
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

    struct PerformanceRow {
        option_type: OptionType,
        moneyness: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        start: Time,
        t: Time,
        vol: Volatility,
        fxr: Rate,
        fxv: Volatility,
        corr: Real,
        result: Real,
    }

    #[rustfmt::skip]
    const HAUG_QUANTO_FORWARD_PERFORMANCE: &[PerformanceRow] = &[
        PerformanceRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.0, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 5.3280 / 150.0,
        },
        PerformanceRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.0, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 0.0816,
        },
        PerformanceRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 0.0201,
        },
        PerformanceRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 0.0672,
        },
    ];

    fn quanto_forward_performance_option(
        settings: Shared<Settings<Date>>,
        row: &PerformanceRow,
    ) -> ForwardVanillaOption {
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
            flat_rate(row.q),
            flat_rate(row.r),
            flat_vol(row.vol),
        ));
        let payoff = shared(PlainVanillaPayoff::new(row.option_type, 0.0))
            as Shared<dyn crate::instruments::StrikedTypePayoff>;
        let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
        let reset = today() + time_to_days(row.start);
        let mut option = ForwardVanillaOption::new(
            row.moneyness,
            reset,
            payoff,
            exercise,
            Shared::clone(&settings),
        );
        set_quanto_forward_performance_european_engine(
            &mut option,
            process,
            flat_rate(row.fxr),
            flat_vol(row.fxv),
            Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
        );
        option
    }

    /// `quantooption.cpp` `testForwardPerformanceValues` @ 1e-4.
    #[test]
    fn haug_quanto_forward_performance_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        const TOL: Real = 1.0e-4;

        for row in HAUG_QUANTO_FORWARD_PERFORMANCE {
            let mut option = quanto_forward_performance_option(Shared::clone(&settings), row);
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= TOL,
                "{:?} start={}: {calculated} vs {} (tol {TOL})",
                row.option_type,
                row.start,
                row.result
            );
        }
    }

    /// Performance quanto-forward NPV ≈ one hundredth of the plain forward quanto.
    #[test]
    fn performance_npv_is_one_hundredth_of_plain_forward_quanto() {
        use crate::pricingengines::forward::set_quanto_forward_european_engine;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let row = &HAUG_QUANTO_FORWARD_PERFORMANCE[0];

        let mut perf = quanto_forward_performance_option(Shared::clone(&settings), row);

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
            flat_rate(row.q),
            flat_rate(row.r),
            flat_vol(row.vol),
        ));
        let payoff = shared(PlainVanillaPayoff::new(row.option_type, 0.0))
            as Shared<dyn crate::instruments::StrikedTypePayoff>;
        let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
        let mut plain = ForwardVanillaOption::new(
            row.moneyness,
            today() + time_to_days(row.start),
            payoff,
            exercise,
            settings,
        );
        set_quanto_forward_european_engine(
            &mut plain,
            process,
            flat_rate(row.fxr),
            flat_vol(row.fxv),
            Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
        );

        let p_npv = perf.npv().unwrap();
        let f_npv = plain.npv().unwrap();
        assert!(
            (p_npv * 100.0 - f_npv).abs() < 1e-10,
            "performance {p_npv} × 100 vs forward {f_npv}"
        );
    }
}
