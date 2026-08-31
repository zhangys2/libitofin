//! Analytic quanto barrier engine.
//!
//! Port of `ql/pricingengines/quanto/quantoengine.hpp` specialised to
//! `QuantoEngine<BarrierOption, AnalyticBarrierEngine>`: the inner Haug
//! barrier engine is run on a process whose dividend curve is a
//! [`QuantoTermStructure`](crate::termstructures::yields::QuantoTermStructure).
//!
//! `AnalyticBarrierEngine` fills NPV only, so the quanto greek adjustments in
//! `quantoengine.hpp:130-168` are null here (matching C++ when the inner
//! engine leaves `dividendRho` unset).

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{AnalyticBarrierEngine, BarrierArguments, StrikedTypePayoff};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// `QuantoEngine<BarrierOption, AnalyticBarrierEngine>`.
pub struct QuantoBarrierEngine {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoBarrierEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
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

impl AsObservable for QuantoBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoBarrierEngine {
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
            let mut original = AnalyticBarrierEngine::new(quanto_process);
            let original_results = original.calculate_from_arguments(self.base.arguments())?;
            original_results.value
        };

        self.base.results_mut().value = value;
        Ok(())
    }
}

/// Attach a [`QuantoBarrierEngine`] to a barrier option.
pub fn set_quanto_barrier_engine(
    option: &mut crate::instruments::BarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine = shared_mut(QuantoBarrierEngine::new(
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
    use crate::instruments::{
        BarrierOption, BarrierType, PlainVanillaPayoff, set_analytic_barrier_engine,
    };
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
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

    /// One row of `quantooption.cpp` `testBarrierValues` (tol 0.5).
    struct HaugRow {
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
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
        tol: Real,
    }

    #[rustfmt::skip]
    const HAUG_QUANTO_BARRIER: &[HaugRow] = &[
        HaugRow {
            barrier_type: BarrierType::DownOut, barrier: 95.0, rebate: 3.0,
            option_type: OptionType::Call, spot: 100.0, strike: 90.0,
            q: 0.04, r: 0.0212, t: 0.50, vol: 0.25, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 8.247, tol: 0.5,
        },
        HaugRow {
            barrier_type: BarrierType::DownOut, barrier: 95.0, rebate: 3.0,
            option_type: OptionType::Put, spot: 100.0, strike: 90.0,
            q: 0.04, r: 0.0212, t: 0.50, vol: 0.25, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 2.274, tol: 0.5,
        },
        HaugRow {
            barrier_type: BarrierType::DownIn, barrier: 95.0, rebate: 0.0,
            option_type: OptionType::Put, spot: 100.0, strike: 90.0,
            q: 0.04, r: 0.0212, t: 0.50, vol: 0.25, fxr: 0.05, fxv: 0.2, corr: 0.3,
            result: 2.85, tol: 0.5,
        },
    ];

    #[test]
    fn haug_quanto_barrier_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        for row in HAUG_QUANTO_BARRIER {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
                flat_rate(row.q),
                flat_rate(row.r),
                flat_vol(row.vol),
            ));
            let payoff = PlainVanillaPayoff::new(row.option_type, row.strike);
            let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
            let mut option = BarrierOption::with_rebate(
                row.barrier_type,
                row.barrier,
                row.rebate,
                payoff,
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_quanto_barrier_engine(
                &mut option,
                process,
                flat_rate(row.fxr),
                flat_vol(row.fxv),
                Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= row.tol,
                "{:?} {:?}: {calculated} vs Haug {} (tol {})",
                row.barrier_type,
                row.option_type,
                row.result,
                row.tol
            );
        }
    }

    /// Quanto barrier ≡ plain barrier with `q + (r_d − r_f + ρ σ σ_fx)`.
    #[test]
    fn quanto_barrier_matches_adjusted_dividend_barrier() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let (
            barrier_type,
            barrier,
            rebate,
            option_type,
            spot,
            strike,
            q,
            r,
            t,
            vol,
            fxr,
            fxv,
            corr,
        ) = (
            BarrierType::DownOut,
            95.0,
            3.0,
            OptionType::Call,
            100.0,
            90.0,
            0.04,
            0.0212,
            0.50,
            0.25,
            0.05,
            0.2,
            0.3,
        );
        let q_adj = q + (r - fxr + corr * vol * fxv);
        let payoff = PlainVanillaPayoff::new(option_type, strike);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today() + time_to_days(t)));

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            flat_rate(q),
            flat_rate(r),
            flat_vol(vol),
        ));
        let mut quanto = BarrierOption::with_rebate(
            barrier_type,
            barrier,
            rebate,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_quanto_barrier_engine(
            &mut quanto,
            process,
            flat_rate(fxr),
            flat_vol(fxv),
            Handle::new(shared(SimpleQuote::new(corr)) as Shared<dyn Quote>),
        );

        let adj_process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            flat_rate(q_adj),
            flat_rate(r),
            flat_vol(vol),
        ));
        let mut plain = BarrierOption::with_rebate(
            barrier_type,
            barrier,
            rebate,
            PlainVanillaPayoff::new(option_type, strike),
            exercise,
            settings,
        )
        .unwrap();
        set_analytic_barrier_engine(&mut plain, adj_process);

        let q_npv = quanto.npv().unwrap();
        let p_npv = plain.npv().unwrap();
        assert!(
            (q_npv - p_npv).abs() < 1e-12,
            "quanto {q_npv} vs adjusted-dividend barrier {p_npv}"
        );
    }
}
