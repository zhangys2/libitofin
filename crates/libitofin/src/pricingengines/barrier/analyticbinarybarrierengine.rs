//! Analytic American binary-barrier engine (`analyticbinarybarrierengine`).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::{EuropeanExercise, ExerciseType};
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{
    AssetOrNothingPayoff, BarrierArguments, BarrierOption, BarrierType, CashOrNothingPayoff,
    StrikedTypePayoff,
};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

pub struct AnalyticBinaryBarrierEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticBinaryBarrierEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticBinaryBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticBinaryBarrierEngine {
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
        let (exercise, payoff, barrier, barrier_type) = {
            let args = self.base.arguments();
            require!(args.binary_payoff.is_some(), "non-striked payoff given");
            (
                Shared::clone(args.exercise.as_ref().expect("validated")),
                Shared::clone(args.binary_payoff.as_ref().expect("validated")),
                args.barrier.expect("validated"),
                args.barrier_type.expect("validated"),
            )
        };
        require!(
            exercise.exercise_type() == ExerciseType::American,
            "non-American exercise given"
        );
        require!(exercise.payoff_at_expiry(), "payoff must be at expiry");
        let vol = self.process.black_volatility().current_link()?;
        require!(
            exercise.dates()[0] <= vol.reference_date()?,
            "American option with window exercise not handled yet"
        );
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(barrier > 0.0, "positive barrier value required");

        let down = spot <= barrier;
        let up = spot >= barrier;
        if matches!(
            (barrier_type, down, up),
            (BarrierType::DownOut, true, _) | (BarrierType::UpOut, _, true)
        ) {
            self.base.results_mut().value = Some(0.0);
            return Ok(());
        }
        if matches!(
            (barrier_type, down, up),
            (BarrierType::DownIn, true, _) | (BarrierType::UpIn, _, true)
        ) {
            let mut euro = AnalyticEuropeanEngine::new(Shared::clone(&self.process));
            self.base.results_mut().value = euro
                .calculate_from_arguments(
                    payoff,
                    shared(EuropeanExercise::new(exercise.last_date())),
                )?
                .instrument
                .value;
            return Ok(());
        }

        let last = exercise.last_date();
        let variance = vol.black_variance_date(last, payoff.strike(), false)?;
        let discount = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount_date(last, false)?;
        let value = payoff_at_expiry(
            &self.process,
            &*payoff,
            last,
            barrier,
            barrier_type,
            spot,
            variance,
            discount,
        )?;
        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

pub fn set_analytic_binary_barrier_engine(
    option: &mut BarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticBinaryBarrierEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[allow(clippy::too_many_arguments)]
fn payoff_at_expiry(
    process: &GeneralizedBlackScholesProcess,
    payoff: &dyn StrikedTypePayoff,
    last: crate::time::date::Date,
    barrier: Real,
    barrier_type: BarrierType,
    spot: Real,
    variance: Real,
    discount: Real,
) -> QlResult<Real> {
    let q_disc = process
        .dividend_yield()
        .current_link()?
        .discount_date(last, false)?;
    require!(spot > 0.0, "positive spot value required");
    require!(discount > 0.0, "positive discount required");
    require!(q_disc > 0.0, "positive dividend discount required");
    require!(variance >= 0.0, "negative variance not allowed");
    require!(barrier > 0.0, "positive barrier value required");

    let option_type = payoff.option_type();
    let strike = payoff.strike();
    let dynamic = payoff as &dyn Any;
    let mut mu = (q_disc / discount).ln() / variance - 0.5;
    let mut k = 0.0;
    if let Some(coo) = dynamic.downcast_ref::<CashOrNothingPayoff>() {
        k = coo.cash_payoff();
    }
    if dynamic.downcast_ref::<AssetOrNothingPayoff>().is_some() {
        mu += 1.0;
        k = spot * q_disc / discount;
    }

    let std_dev = variance.sqrt();
    let log_s_x = (spot / strike).ln();
    let log_s_h = (spot / barrier).ln();
    let log_h_s = (barrier / spot).ln();
    let log_h2_sx = (barrier * barrier / (spot * strike)).ln();
    let h_s_2mu = (barrier / spot).powf(2.0 * mu);
    let eta = match barrier_type {
        BarrierType::DownIn | BarrierType::DownOut => 1.0,
        BarrierType::UpIn | BarrierType::UpOut => -1.0,
    };
    let phi = match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    };

    let (cum_x1, cum_x2, cum_y1, cum_y2) = if variance >= f64::EPSILON {
        let n = CumulativeNormalDistribution::standard();
        (
            n.value(phi * (log_s_x / std_dev + mu * std_dev)),
            n.value(phi * (log_s_h / std_dev + mu * std_dev)),
            n.value(eta * (log_h2_sx / std_dev + mu * std_dev)),
            n.value(eta * (log_h_s / std_dev + mu * std_dev)),
        )
    } else {
        (
            if log_s_x > 0.0 { 1.0 } else { 0.0 },
            if log_s_h > 0.0 { 1.0 } else { 0.0 },
            if log_h2_sx > 0.0 { 1.0 } else { 0.0 },
            if log_h_s > 0.0 { 1.0 } else { 0.0 },
        )
    };

    let k_ge_h = strike >= barrier;
    let alpha = match (barrier_type, option_type, k_ge_h) {
        (BarrierType::DownIn, OptionType::Call, true) => h_s_2mu * cum_y1,
        (BarrierType::DownIn, OptionType::Call, false) => cum_x1 - cum_x2 + h_s_2mu * cum_y2,
        (BarrierType::DownIn, OptionType::Put, true) => cum_x2 + h_s_2mu * (-cum_y1 + cum_y2),
        (BarrierType::DownIn, OptionType::Put, false) => cum_x1,
        (BarrierType::UpIn, OptionType::Call, true) => cum_x1,
        (BarrierType::UpIn, OptionType::Call, false) => cum_x2 + h_s_2mu * (-cum_y1 + cum_y2),
        (BarrierType::UpIn, OptionType::Put, true) => cum_x1 - cum_x2 + h_s_2mu * cum_y2,
        (BarrierType::UpIn, OptionType::Put, false) => h_s_2mu * cum_y1,
        (BarrierType::DownOut, OptionType::Call, true) => cum_x1 - h_s_2mu * cum_y1,
        (BarrierType::DownOut, OptionType::Call, false) => cum_x2 - h_s_2mu * cum_y2,
        (BarrierType::DownOut, OptionType::Put, true) => {
            cum_x1 - cum_x2 + h_s_2mu * (cum_y1 - cum_y2)
        }
        (BarrierType::DownOut, OptionType::Put, false) => 0.0,
        (BarrierType::UpOut, OptionType::Call, true) => 0.0,
        (BarrierType::UpOut, OptionType::Call, false) => {
            cum_x1 - cum_x2 + h_s_2mu * (cum_y1 - cum_y2)
        }
        (BarrierType::UpOut, OptionType::Put, true) => cum_x2 - h_s_2mu * cum_y2,
        (BarrierType::UpOut, OptionType::Put, false) => cum_x1 - h_s_2mu * cum_y1,
    };
    Ok(discount * k * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::AmericanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{self, Call, Put};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use BarrierType::{DownIn, DownOut, UpIn, UpOut};

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }
    fn yts(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    struct Market {
        spot: Shared<SimpleQuote>,
        process: Shared<BlackScholesMertonProcess>,
        settings: Shared<Settings<Date>>,
    }

    fn market() -> Market {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let spot = shared(SimpleQuote::new(100.0));
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(Shared::clone(&spot) as Shared<dyn Quote>),
            yts(0.0),
            yts(0.10),
            Handle::new(
                shared(BlackConstantVol::new(today(), None, 0.20, Actual360::new()))
                    as Shared<dyn BlackVolTermStructure>,
            ),
        ));
        Market {
            spot,
            process,
            settings,
        }
    }

    #[rustfmt::skip]
    const ROWS: &[(BarrierType, Real, OptionType, Real, Real, Real)] = &[
        (DownIn, 15.0, Call, 102.0, 105.0, 4.9289),
        (DownIn, 15.0, Call, 98.0, 105.0, 6.2150),
        (UpIn, 15.0, Call, 102.0, 95.0, 5.8926),
        (UpIn, 15.0, Call, 98.0, 95.0, 7.4519),
        (DownIn, 15.0, Put, 102.0, 105.0, 4.4314),
        (DownIn, 15.0, Put, 98.0, 105.0, 3.1454),
        (UpIn, 15.0, Put, 102.0, 95.0, 5.3297),
        (UpIn, 15.0, Put, 98.0, 95.0, 3.7704),
        (DownOut, 15.0, Call, 102.0, 105.0, 4.8758),
        (DownOut, 15.0, Call, 98.0, 105.0, 4.9081),
        (UpOut, 15.0, Call, 102.0, 95.0, 0.0000),
        (UpOut, 15.0, Call, 98.0, 95.0, 0.0407),
        (DownOut, 15.0, Put, 102.0, 105.0, 0.0323),
        (DownOut, 15.0, Put, 98.0, 105.0, 0.0000),
        (UpOut, 15.0, Put, 102.0, 95.0, 3.0461),
        (UpOut, 15.0, Put, 98.0, 95.0, 3.0054),
        (DownIn, 15.0, Call, 98.0, 95.0, 7.4926),
        (DownOut, 15.0, Call, 98.0, 99.0, 0.0000),
        (DownIn, 0.0, Call, 102.0, 105.0, 37.2782),
        (DownIn, 0.0, Call, 98.0, 105.0, 45.8530),
        (UpIn, 0.0, Call, 102.0, 95.0, 44.5294),
        (UpIn, 0.0, Call, 98.0, 95.0, 54.9262),
        (DownIn, 0.0, Put, 102.0, 105.0, 27.5644),
        (DownIn, 0.0, Put, 98.0, 105.0, 18.9896),
        (UpIn, 0.0, Put, 102.0, 95.0, 33.1723),
        (UpIn, 0.0, Put, 98.0, 95.0, 22.7755),
        (DownOut, 0.0, Call, 102.0, 105.0, 39.9391),
        (DownOut, 0.0, Call, 98.0, 105.0, 40.1574),
        (UpOut, 0.0, Call, 102.0, 95.0, 0.0000),
        (UpOut, 0.0, Call, 98.0, 95.0, 0.2676),
        (DownOut, 0.0, Put, 102.0, 105.0, 0.2183),
        (DownOut, 0.0, Put, 98.0, 105.0, 0.0000),
        (UpOut, 0.0, Put, 102.0, 95.0, 17.2983),
        (UpOut, 0.0, Put, 98.0, 95.0, 17.0306),
    ];

    #[test]
    fn haug_cash_and_asset_or_nothing_barrier_values() {
        let m = market();
        let expiry = today() + 180;
        for &(bt, cash, ty, k, s, expected) in ROWS {
            m.spot.set_value(s);
            let payoff: Shared<dyn StrikedTypePayoff> = if cash > 0.0 {
                shared(CashOrNothingPayoff::new(ty, k, cash))
            } else {
                shared(AssetOrNothingPayoff::new(ty, k))
            };
            let mut option = BarrierOption::with_striked_payoff(
                bt,
                100.0,
                0.0,
                payoff,
                shared(AmericanExercise::new(today(), expiry, true).unwrap()),
                Shared::clone(&m.settings),
            )
            .unwrap();
            set_analytic_binary_barrier_engine(&mut option, Shared::clone(&m.process));
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 1.0e-4,
                "{bt:?} {ty:?} cash={cash} K={k} S={s}: {calculated} vs Haug {expected}"
            );
        }
    }
}
