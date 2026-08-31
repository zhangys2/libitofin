//! Analytic quanto forward vanilla engine.
//!
//! Port of `ql/pricingengines/quanto/quantoengine.hpp` specialised to
//! `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.

use std::any::Any;

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{
    ForwardVanillaArguments, ForwardVanillaEngineBase, OneAssetOptionResults,
};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::forward::forward_vanilla_calculate;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

/// `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.
pub struct QuantoForwardEuropeanEngine {
    base: ForwardVanillaEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoForwardEuropeanEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base = ForwardVanillaEngineBase::new(
            ForwardVanillaArguments::default(),
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

impl AsObservable for QuantoForwardEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoForwardEuropeanEngine {
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
        let arguments = self.base.arguments();
        let payoff = arguments.payoff.as_ref().unwrap();
        let exercise = arguments.exercise.as_ref().unwrap();
        let strike = payoff.strike();
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

        let forward_results = forward_vanilla_calculate(&quanto_process, arguments)?;
        let mut greeks = forward_results.greeks;
        let more_greeks = forward_results.more_greeks;
        let value = forward_results.instrument.value.unwrap_or(0.0);
        let maturity = exercise.last_date();
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
        results.instrument = forward_results.instrument;
        results.instrument.value = Some(value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::QuantoEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared_mut};
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

    struct ForwardRow {
        option_type: OptionType,
        moneyness: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        reset: Time,
        t: Time,
        vol: Volatility,
        fxr: Rate,
        fxv: Volatility,
        corr: Real,
        result: Real,
        tol: Real,
    }

    #[rustfmt::skip]
    const HAUG_FORWARD: &[ForwardRow] = &[
        ForwardRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, reset: 0.00, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 5.3280 / 1.5, tol: 1.0e-4,
        },
        ForwardRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, reset: 0.00, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 8.1636, tol: 1.0e-4,
        },
        ForwardRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, reset: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 2.0171, tol: 1.0e-4,
        },
        ForwardRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, reset: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 6.7296, tol: 1.0e-4,
        },
    ];

    fn quanto_forward_option(
        settings: Shared<Settings<Date>>,
        row: &ForwardRow,
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
        let reset = today() + time_to_days(row.reset);
        let mut option = ForwardVanillaOption::new(
            row.moneyness,
            reset,
            payoff,
            exercise,
            Shared::clone(&settings),
        );
        let engine = shared_mut(QuantoForwardEuropeanEngine::new(
            process,
            flat_rate(row.fxr),
            flat_vol(row.fxv),
            Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
        )) as SharedMut<dyn PricingEngine>;
        option.base_mut().set_pricing_engine(engine);
        option
    }

    /// `quantooption.cpp` `testForwardValues` @ 1e-4.
    #[test]
    fn haug_quanto_forward_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        for row in HAUG_FORWARD {
            let mut option = quanto_forward_option(Shared::clone(&settings), row);
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= row.tol,
                "{:?} reset={}: {calculated} vs Haug {} (tol {})",
                row.option_type,
                row.reset,
                row.result,
                row.tol
            );
        }
    }

    /// reset=0 forward quanto NPV matches `QuantoEuropeanEngine` at strike reset.
    #[test]
    fn zero_reset_matches_quanto_vanilla() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let row = &HAUG_FORWARD[0];
        let mut forward = quanto_forward_option(Shared::clone(&settings), row);

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
            flat_rate(row.q),
            flat_rate(row.r),
            flat_vol(row.vol),
        ));
        let payoff = shared(PlainVanillaPayoff::new(
            row.option_type,
            row.moneyness * row.spot,
        )) as Shared<dyn crate::instruments::StrikedTypePayoff>;
        let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
        let engine = shared_mut(QuantoEuropeanEngine::new(
            process,
            flat_rate(row.fxr),
            flat_vol(row.fxv),
            Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
        )) as SharedMut<dyn PricingEngine>;
        let mut vanilla = VanillaOption::new(payoff, exercise, settings);
        vanilla.base_mut().set_pricing_engine(engine);

        let f_npv = forward.npv().unwrap();
        let v_npv = vanilla.npv().unwrap();
        assert!(
            (f_npv - v_npv).abs() < 1e-10,
            "forward {f_npv} vs vanilla quanto {v_npv}"
        );
    }
}
