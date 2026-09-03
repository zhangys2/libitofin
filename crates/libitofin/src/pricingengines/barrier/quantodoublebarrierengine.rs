//! Analytic quanto double-barrier engine.
//!
//! Port of `QuantoEngine<DoubleBarrierOption, AnalyticDoubleBarrierEngine>`:
//! the inner Ikeda–Kunitomo engine runs on a process whose dividend curve is a
//! [`QuantoTermStructure`](crate::termstructures::yields::QuantoTermStructure).

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{DoubleBarrierArguments, StrikedTypePayoff};

use super::AnalyticDoubleBarrierEngine;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

type QuantoDoubleBarrierEngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// `QuantoEngine<DoubleBarrierOption, AnalyticDoubleBarrierEngine>`.
pub struct QuantoDoubleBarrierEngine {
    base: QuantoDoubleBarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoDoubleBarrierEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base = QuantoDoubleBarrierEngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
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

impl AsObservable for QuantoDoubleBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoDoubleBarrierEngine {
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
        let strike = {
            let arguments = self.base.arguments();
            let Some(payoff) = arguments.payoff.as_ref() else {
                fail!("no payoff given");
            };
            payoff.strike()
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

        let value = {
            let mut original = AnalyticDoubleBarrierEngine::new(quanto_process);
            let original_results = original.calculate_from_arguments(self.base.arguments())?;
            original_results.value
        };

        self.base.results_mut().value = value;
        Ok(())
    }
}

/// Attach a [`QuantoDoubleBarrierEngine`] to a double-barrier option.
pub fn set_quanto_double_barrier_engine(
    option: &mut crate::instruments::DoubleBarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine = shared_mut(QuantoDoubleBarrierEngine::new(
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
    use crate::instruments::{DoubleBarrierOption, DoubleBarrierType, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::set_analytic_double_barrier_engine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::Shared;
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

    struct DoubleBarrierRow {
        barrier_type: DoubleBarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        option_type: OptionType,
        spot: Real,
        strike: Real,
        q: Rate,
        r: Rate,
        t: Time,
        vol: Volatility,
        fxr: Rate,
        fxv: Volatility,
        corr: Real,
        result: Real,
    }

    /// `quantooption.cpp` `testDoubleBarrierValues` @ 1e-4.
    #[rustfmt::skip]
    #[allow(clippy::approx_constant)]
    const QUANTO_DOUBLE_BARRIER: &[DoubleBarrierRow] = &[
        DoubleBarrierRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, spot: 100.0, strike: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 3.4623,
        },
        DoubleBarrierRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, spot: 100.0, strike: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 0.5236,
        },
        DoubleBarrierRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, spot: 100.0, strike: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 1.1320,
        },
        DoubleBarrierRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, spot: 100.0, strike: 102.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 2.6313,
        },
        DoubleBarrierRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, spot: 100.0, strike: 102.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 1.9305,
        },
    ];

    #[test]
    fn haug_quanto_double_barrier_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        for row in QUANTO_DOUBLE_BARRIER {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
                flat_rate(row.q),
                flat_rate(row.r),
                flat_vol(row.vol),
            ));
            let payoff = PlainVanillaPayoff::new(row.option_type, row.strike);
            let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
            let mut option = DoubleBarrierOption::new(
                row.barrier_type,
                row.barrier_lo,
                row.barrier_hi,
                0.0,
                payoff,
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_quanto_double_barrier_engine(
                &mut option,
                process,
                flat_rate(row.fxr),
                flat_vol(row.fxv),
                Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= 1.0e-4,
                "{:?} {:?} lo={} hi={}: {calculated} vs Haug {} (tol 1e-4)",
                row.barrier_type,
                row.option_type,
                row.barrier_lo,
                row.barrier_hi,
                row.result,
            );
        }
    }

    /// Quanto double-barrier ≡ plain double-barrier with quanto-adjusted dividend yield.
    #[test]
    fn quanto_double_barrier_matches_adjusted_dividend_double_barrier() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let row = &QUANTO_DOUBLE_BARRIER[0];
        let q_adj = row.q + (row.r - row.fxr + row.corr * row.vol * row.fxv);
        let payoff = PlainVanillaPayoff::new(row.option_type, row.strike);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today() + time_to_days(row.t)));

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
            flat_rate(row.q),
            flat_rate(row.r),
            flat_vol(row.vol),
        ));
        let mut quanto = DoubleBarrierOption::new(
            row.barrier_type,
            row.barrier_lo,
            row.barrier_hi,
            0.0,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_quanto_double_barrier_engine(
            &mut quanto,
            process,
            flat_rate(row.fxr),
            flat_vol(row.fxv),
            Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
        );

        let adj_process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
            flat_rate(q_adj),
            flat_rate(row.r),
            flat_vol(row.vol),
        ));
        let mut plain = DoubleBarrierOption::new(
            row.barrier_type,
            row.barrier_lo,
            row.barrier_hi,
            0.0,
            PlainVanillaPayoff::new(row.option_type, row.strike),
            exercise,
            settings,
        )
        .unwrap();
        set_analytic_double_barrier_engine(&mut plain, adj_process);

        let q_npv = quanto.npv().unwrap();
        let p_npv = plain.npv().unwrap();
        assert!(
            (q_npv - p_npv).abs() < 1e-12,
            "quanto {q_npv} vs adjusted-dividend double barrier {p_npv}"
        );
    }
}
