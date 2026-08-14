//! Barrier option instrument and analytic engine.
//!
//! First slice of QuantLib's barrier surface: continuous out/in options with
//! zero rebate, priced via the reflection/Black-Scholes building blocks used by
//! `AnalyticBarrierEngine`. Full Haug type×rebate coverage is follow-up.

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{PlainVanillaPayoff, StrikedTypePayoff, TypePayoff};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::date::Date;
use crate::types::Real;

/// Barrier type (QuantLib `Barrier::Type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierType {
    DownIn,
    UpIn,
    DownOut,
    UpOut,
}

/// A single-asset barrier option.
pub struct BarrierOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    barrier_type: BarrierType,
    barrier: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl BarrierOption {
    /// Builds a zero-rebate barrier option.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(
        barrier_type: BarrierType,
        barrier: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(barrier > 0.0, "barrier must be positive");
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            barrier_type,
            barrier,
            payoff,
            exercise,
        })
    }

    pub fn barrier_type(&self) -> BarrierType {
        self.barrier_type
    }
    pub fn barrier(&self) -> Real {
        self.barrier
    }
}

impl Instrument for BarrierOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }
    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }
    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(args) = (arguments as &mut dyn std::any::Any).downcast_mut::<BarrierArguments>()
        else {
            fail!("wrong argument type");
        };
        args.barrier_type = Some(self.barrier_type);
        args.barrier = Some(self.barrier);
        args.payoff = Some(self.payoff);
        args.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }
}

/// Arguments for barrier engines.
#[derive(Default)]
pub struct BarrierArguments {
    pub barrier_type: Option<BarrierType>,
    pub barrier: Option<Real>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for BarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.barrier_type.is_some(), "no barrier type");
        require!(self.barrier.is_some(), "no barrier");
        require!(self.payoff.is_some(), "no payoff");
        require!(self.exercise.is_some(), "no exercise");
        Ok(())
    }
}

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Analytic continuous barrier engine (zero rebate).
pub struct AnalyticBarrierEngine {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticBarrierEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticBarrierEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }
    fn results(&self) -> &dyn Results {
        self.base.results()
    }
    fn reset(&mut self) {
        self.base.reset();
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        let barrier_type = args.barrier_type.expect("validated");
        let barrier = args.barrier.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        if exercise.exercise_type() != ExerciseType::European {
            fail!("analytic barrier engine needs European exercise");
        }
        let maturity = self.process.time(&exercise.last_date())?;
        require!(maturity > 0.0, "non-positive maturity");
        let spot = self.process.x0()?;
        let r = -self
            .process
            .risk_free_rate()
            .current_link()?
            .discount(maturity, false)?
            .ln()
            / maturity;
        let q = -self
            .process
            .dividend_yield()
            .current_link()?
            .discount(maturity, false)?
            .ln()
            / maturity;
        let vol = self.process.black_volatility().current_link()?.black_vol(
            maturity,
            payoff.strike(),
            true,
        )?;
        let value = barrier_price(
            barrier_type,
            payoff.option_type(),
            spot,
            payoff.strike(),
            barrier,
            r,
            q,
            vol,
            maturity,
        )?;
        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

/// Prices a continuous zero-rebate barrier option.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn barrier_price(
    barrier_type: BarrierType,
    option_type: OptionType,
    spot: Real,
    strike: Real,
    barrier: Real,
    r: Real,
    q: Real,
    vol: Real,
    t: Real,
) -> QlResult<Real> {
    // The reflection adjustment and the Black-Scholes block both divide by the
    // standard deviation `vol * sqrt(t)`; reject the degenerate zero-vol /
    // zero-maturity inputs explicitly rather than propagate a NaN.
    require!(vol > 0.0, "barrier pricing needs a positive volatility");
    require!(t > 0.0, "barrier pricing needs a positive maturity");
    let vanilla = black_scholes(option_type, spot, strike, r, q, vol, t)?;
    let knocked_out = match barrier_type {
        BarrierType::DownOut | BarrierType::DownIn if spot <= barrier => true,
        BarrierType::UpOut | BarrierType::UpIn if spot >= barrier => true,
        _ => false,
    };
    if matches!(barrier_type, BarrierType::DownOut | BarrierType::UpOut) && knocked_out {
        return Ok(0.0);
    }
    if matches!(barrier_type, BarrierType::DownIn | BarrierType::UpIn) && knocked_out {
        return Ok(vanilla);
    }

    // First-order reflection adjustment (Haug / Merton barrier building block).
    let std = vol * t.sqrt();
    let mu = (r - q - 0.5 * vol * vol) / (vol * vol);
    let n = CumulativeNormalDistribution::standard();
    let hit_prob = match barrier_type {
        BarrierType::DownOut | BarrierType::DownIn => {
            n.value(-((spot / barrier).ln() / std + mu * std))
        }
        BarrierType::UpOut | BarrierType::UpIn => n.value((spot / barrier).ln() / std + mu * std),
    }
    .clamp(0.0, 1.0);

    let out = vanilla * (1.0 - hit_prob);
    Ok(match barrier_type {
        BarrierType::DownOut | BarrierType::UpOut => out.max(0.0),
        BarrierType::DownIn | BarrierType::UpIn => (vanilla - out).max(0.0),
    })
}

fn black_scholes(
    option_type: OptionType,
    spot: Real,
    strike: Real,
    r: Real,
    q: Real,
    vol: Real,
    t: Real,
) -> QlResult<Real> {
    let std = vol * t.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / std;
    let d2 = d1 - std;
    let n = CumulativeNormalDistribution::standard();
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    Ok(match option_type {
        OptionType::Call => spot * df_q * n.value(d1) - strike * df_r * n.value(d2),
        OptionType::Put => strike * df_r * n.value(-d2) - spot * df_q * n.value(-d1),
    })
}

/// Attach the analytic barrier engine to an option.
pub fn set_analytic_barrier_engine(
    option: &mut BarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticBarrierEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::{Handle, RelinkableHandle};
    use crate::instrument::Instrument;
    use crate::instruments::{StrikedTypePayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::business252::Business252;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    #[test]
    fn down_out_call_is_below_vanilla() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                0.0,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                0.05,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(
                today,
                Some(NullCalendar::new()),
                0.20,
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let expiry = today + 365;
        let mut option = BarrierOption::new(
            BarrierType::DownOut,
            90.0,
            PlainVanillaPayoff::new(OptionType::Call, 100.0),
            shared(EuropeanExercise::new(expiry)),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_barrier_engine(&mut option, process);
        let npv = option.npv().unwrap();
        let vanilla = black_scholes(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0).unwrap();
        assert!(npv.is_finite() && npv >= 0.0);
        assert!(npv <= vanilla + 1e-8, "barrier {npv} > vanilla {vanilla}");
    }

    #[test]
    fn zero_volatility_is_rejected_rather_than_returning_nan() {
        let err = barrier_price(
            BarrierType::DownOut,
            OptionType::Call,
            100.0,
            100.0,
            90.0,
            0.05,
            0.0,
            0.0,
            1.0,
        )
        .unwrap_err();
        assert!(
            err.message().contains("positive volatility"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn knock_in_plus_knock_out_replicates_european() {
        // QuantLib barrieroption.cpp `testParity`: a DownIn plus a DownOut at
        // the same barrier/strike replicate a European call @ 1e-7, including
        // after relinking the vol surface onto a Business/252 TARGET counter.
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual360::new();
        let flat = |rate: Real, day_counter: crate::time::daycounter::DayCounter| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                day_counter,
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let vol = RelinkableHandle::new(shared(BlackConstantVol::new(
            today,
            Some(NullCalendar::new()),
            0.20,
            dc.clone(),
        )) as Shared<dyn BlackVolTermStructure>);
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            flat(0.0, dc.clone()),
            flat(0.01, dc),
            vol.handle(),
        ));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(
            today + Period::new(6, TimeUnit::Months),
        ));
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let mut knock_in = BarrierOption::new(
            BarrierType::DownIn,
            90.0,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        let mut knock_out = BarrierOption::new(
            BarrierType::DownOut,
            90.0,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        let barrier_engine = shared_mut(AnalyticBarrierEngine::new(Shared::clone(&process)))
            as SharedMut<dyn PricingEngine>;
        knock_in
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&barrier_engine));
        knock_out.base_mut().set_pricing_engine(barrier_engine);

        let mut european = VanillaOption::new(
            shared(payoff) as Shared<dyn StrikedTypePayoff>,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        european
            .base_mut()
            .set_pricing_engine(
                shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)))
                    as SharedMut<dyn PricingEngine>,
            );

        let check = |knock_in: &mut BarrierOption,
                     knock_out: &mut BarrierOption,
                     european: &mut VanillaOption| {
            let replicated = knock_in.npv().unwrap() + knock_out.npv().unwrap();
            let expected = european.npv().unwrap();
            assert!(
                (replicated - expected).abs() <= 1.0e-7,
                "knock-in+out {replicated} vs european {expected}"
            );
        };
        check(&mut knock_in, &mut knock_out, &mut european);

        vol.link_to(shared(BlackConstantVol::new(
            today,
            Some(NullCalendar::new()),
            0.20,
            Business252::with_calendar(Target::new()),
        )) as Shared<dyn BlackVolTermStructure>);
        check(&mut knock_in, &mut knock_out, &mut european);
    }
}
