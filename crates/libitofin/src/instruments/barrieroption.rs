//! Barrier option instrument and analytic engine.
//!
//! Port of QuantLib's `ql/pricingengines/barrier/analyticbarrierengine`:
//! continuous knock-in/out options priced with Haug's A–F formulae (*Option
//! pricing formulas*, McGraw-Hill, p.69), including cash rebate terms `E`/`F`.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{PlainVanillaPayoff, StrikedTypePayoff, TypePayoff};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::{DividendSchedule, FdBlackScholesBarrierEngine};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::{Quote, SimpleQuote};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Volatility};

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
    rebate: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl BarrierOption {
    /// Builds a zero-rebate barrier option.
    pub fn new(
        barrier_type: BarrierType,
        barrier: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        Self::with_rebate(barrier_type, barrier, 0.0, payoff, exercise, settings)
    }

    /// Builds a barrier option with a cash rebate paid if the barrier is hit
    /// (knock-out) or never hit (knock-in), matching QuantLib's constructor.
    #[allow(clippy::neg_cmp_op_on_partial_ord, clippy::too_many_arguments)]
    pub fn with_rebate(
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
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
            rebate,
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
    pub fn rebate(&self) -> Real {
        self.rebate
    }

    /// Black implied volatility such that the analytic barrier price equals
    /// `target_value` (`BarrierOption::impliedVolatility` with an empty
    /// dividend schedule).
    ///
    /// QuantLib defaults: `max_evaluations = 100`, `min_vol = 1e-7`,
    /// `max_vol = 4.0`.
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        &self,
        target_value: Real,
        process: Shared<GeneralizedBlackScholesProcess>,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
    ) -> QlResult<Volatility> {
        self.implied_volatility_with_dividends(
            target_value,
            process,
            Vec::new(),
            accuracy,
            max_evaluations,
            min_vol,
            max_vol,
        )
    }

    /// Black implied volatility such that the barrier price equals
    /// `target_value` (`BarrierOption::impliedVolatility` with dividends).
    ///
    /// Empty `dividends` uses [`AnalyticBarrierEngine`]; a non-empty schedule
    /// uses [`FdBlackScholesBarrierEngine`] with QuantLib's default 100×100
    /// Douglas grid.
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility_with_dividends(
        &self,
        target_value: Real,
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
    ) -> QlResult<Volatility> {
        if self.is_expired()? {
            fail!("option expired");
        }
        if self.exercise.exercise_type() != ExerciseType::European {
            fail!("engine not available for non-European barrier option");
        }

        let src_vol = process.black_volatility().current_link()?;
        let vol = shared(SimpleQuote::new(0.0));
        let vol_ts = BlackConstantVol::with_quote(
            src_vol.reference_date()?,
            src_vol.calendar(),
            Handle::new(Shared::clone(&vol) as Shared<dyn Quote>),
            src_vol.require_day_counter()?,
        );
        let cloned = shared(GeneralizedBlackScholesProcess::new(
            process.state_variable(),
            process.dividend_yield(),
            process.risk_free_rate(),
            Handle::new(shared(vol_ts) as Shared<dyn BlackVolTermStructure>),
        ));
        if dividends.is_empty() {
            self.solve_implied_volatility(
                target_value,
                vol,
                AnalyticBarrierEngine::new(cloned),
                accuracy,
                max_evaluations,
                min_vol,
                max_vol,
            )
        } else {
            self.solve_implied_volatility(
                target_value,
                vol,
                FdBlackScholesBarrierEngine::with_dividends(cloned, dividends),
                accuracy,
                max_evaluations,
                min_vol,
                max_vol,
            )
        }
    }

    fn solve_implied_volatility(
        &self,
        target_value: Real,
        vol: Shared<SimpleQuote>,
        mut engine: impl PricingEngine,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
    ) -> QlResult<Volatility> {
        self.setup_arguments(engine.arguments_mut())?;
        engine.arguments_mut().validate()?;

        let failure = RefCell::new(None);
        let objective = |x: Volatility| {
            vol.set_value(x);
            match engine.calculate() {
                Ok(()) => match engine
                    .results()
                    .as_instrument_results()
                    .and_then(|r| r.value)
                {
                    Some(value) => value - target_value,
                    None => {
                        failure.borrow_mut().get_or_insert_with(|| {
                            crate::errors::QlError::new(
                                "no results returned from pricing engine",
                                file!(),
                                line!(),
                            )
                        });
                        Real::NAN
                    }
                },
                Err(error) => {
                    failure.borrow_mut().get_or_insert(error);
                    Real::NAN
                }
            }
        };

        let guess = 0.5 * (min_vol + max_vol);
        let mut solver = Brent::new().with_max_evaluations(max_evaluations);
        let root = solver.solve_bracketed(objective, accuracy, guess, min_vol, max_vol);
        match failure.into_inner() {
            Some(error) => Err(error),
            None => root,
        }
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
        args.rebate = Some(self.rebate);
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
    pub rebate: Option<Real>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for BarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.barrier_type.is_some(), "no barrier type");
        require!(self.barrier.is_some(), "no barrier");
        require!(self.rebate.is_some(), "no rebate");
        require!(self.payoff.is_some(), "no payoff");
        require!(self.exercise.is_some(), "no exercise");
        Ok(())
    }
}

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Analytic continuous barrier engine (Haug A–F, including rebate).
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
        let rebate = args.rebate.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        if exercise.exercise_type() != ExerciseType::European {
            fail!("analytic barrier engine needs European exercise");
        }
        let strike = payoff.strike();
        require!(strike > 0.0, "strike must be positive");
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(!triggered(barrier_type, spot, barrier), "barrier touched");

        let maturity_date = exercise.last_date();
        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let rfdc = risk_free.require_day_counter()?;
        let divdc = dividend.require_day_counter()?;
        // Date-based extracts, matching QuantLib `AnalyticBarrierEngine`:
        // `stdDeviation` is `sqrt(blackVariance(date))`, not `vol * sqrt(process.time())`,
        // so knock-in/out parity still holds when the vol day counter differs.
        let r = risk_free
            .zero_rate_date(
                maturity_date,
                rfdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q = dividend
            .zero_rate_date(
                maturity_date,
                divdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let df_r = risk_free.discount_date(maturity_date, false)?;
        let df_q = dividend.discount_date(maturity_date, false)?;
        let vol = vol_ts.black_vol_date(maturity_date, strike, false)?;
        let std_dev = vol_ts
            .black_variance_date(maturity_date, strike, false)?
            .sqrt();
        let value = haug_barrier_price(
            barrier_type,
            payoff.option_type(),
            HaugInputs {
                spot,
                strike,
                barrier,
                rebate,
                vol,
                std_dev,
                r,
                q,
                df_r,
                df_q,
            },
        )?;
        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

/// Prices a continuous zero-rebate barrier option from flat `r`, `q`, `vol`.
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
    require!(vol > 0.0, "barrier pricing needs a positive volatility");
    require!(t > 0.0, "barrier pricing needs a positive maturity");
    require!(spot > 0.0, "negative or null underlying given");
    require!(strike > 0.0, "strike must be positive");
    haug_barrier_price(
        barrier_type,
        option_type,
        HaugInputs {
            spot,
            strike,
            barrier,
            rebate: 0.0,
            vol,
            std_dev: vol * t.sqrt(),
            r,
            q,
            df_r: (-r * t).exp(),
            df_q: (-q * t).exp(),
        },
    )
}

struct HaugInputs {
    spot: Real,
    strike: Real,
    barrier: Real,
    rebate: Real,
    vol: Real,
    std_dev: Real,
    r: Real,
    /// Kept for QuantLib `dividendYield()` parity; `mu` uses discounts instead.
    #[allow(dead_code)]
    q: Real,
    df_r: Real,
    df_q: Real,
}

impl HaugInputs {
    /// Drift parameter in Haug's A–F formulae.
    ///
    /// QuantLib uses `(r − q) / σ² − 1/2`. That equals
    /// `ln(df_q / df_r) / variance − 1/2` when the vol and yield day counters
    /// agree. When they differ (the Business/252 arm of `testParity`), the
    /// discount form keeps `A()` identical to [`AnalyticEuropeanEngine`], so a
    /// knock-in plus a knock-out still replicate the European.
    fn mu(&self) -> Real {
        (self.df_q / self.df_r).ln() / (self.std_dev * self.std_dev) - 0.5
    }

    fn mu_sigma(&self) -> Real {
        (1.0 + self.mu()) * self.std_dev
    }

    fn n(x: Real) -> Real {
        CumulativeNormalDistribution::standard().value(x)
    }

    fn a(&self, phi: Real) -> Real {
        let x1 = (self.spot / self.strike).ln() / self.std_dev + self.mu_sigma();
        phi * (self.spot * self.df_q * Self::n(phi * x1)
            - self.strike * self.df_r * Self::n(phi * (x1 - self.std_dev)))
    }

    fn b(&self, phi: Real) -> Real {
        let x2 = (self.spot / self.barrier).ln() / self.std_dev + self.mu_sigma();
        phi * (self.spot * self.df_q * Self::n(phi * x2)
            - self.strike * self.df_r * Self::n(phi * (x2 - self.std_dev)))
    }

    #[allow(clippy::float_cmp)] // QuantLib: N1/N2 == 0.0 guards 0 × ∞ → NaN
    fn c(&self, eta: Real, phi: Real) -> Real {
        let hs = self.barrier / self.spot;
        let pow_hs0 = hs.powf(2.0 * self.mu());
        let pow_hs1 = pow_hs0 * hs * hs;
        let y1 = (self.barrier * hs / self.strike).ln() / self.std_dev + self.mu_sigma();
        let n1 = Self::n(eta * y1);
        let n2 = Self::n(eta * (y1 - self.std_dev));
        phi * (self.spot * self.df_q * if n1 == 0.0 { 0.0 } else { pow_hs1 * n1 }
            - self.strike * self.df_r * if n2 == 0.0 { 0.0 } else { pow_hs0 * n2 })
    }

    #[allow(clippy::float_cmp)] // QuantLib: N1/N2 == 0.0 guards 0 × ∞ → NaN
    fn d(&self, eta: Real, phi: Real) -> Real {
        let hs = self.barrier / self.spot;
        let pow_hs0 = hs.powf(2.0 * self.mu());
        let pow_hs1 = pow_hs0 * hs * hs;
        let y2 = (self.barrier / self.spot).ln() / self.std_dev + self.mu_sigma();
        let n1 = Self::n(eta * y2);
        let n2 = Self::n(eta * (y2 - self.std_dev));
        phi * (self.spot * self.df_q * if n1 == 0.0 { 0.0 } else { pow_hs1 * n1 }
            - self.strike * self.df_r * if n2 == 0.0 { 0.0 } else { pow_hs0 * n2 })
    }

    #[allow(clippy::float_cmp)] // QuantLib: N2 == 0.0 guards 0 × ∞ → NaN
    fn e(&self, eta: Real) -> Real {
        if self.rebate <= 0.0 {
            return 0.0;
        }
        let pow_hs0 = (self.barrier / self.spot).powf(2.0 * self.mu());
        let x2 = (self.spot / self.barrier).ln() / self.std_dev + self.mu_sigma();
        let y2 = (self.barrier / self.spot).ln() / self.std_dev + self.mu_sigma();
        let n1 = Self::n(eta * (x2 - self.std_dev));
        let n2 = Self::n(eta * (y2 - self.std_dev));
        self.rebate * self.df_r * (n1 - if n2 == 0.0 { 0.0 } else { pow_hs0 * n2 })
    }

    #[allow(clippy::float_cmp)] // QuantLib: N1/N2 == 0.0 guards 0 × ∞ → NaN
    fn f(&self, eta: Real) -> Real {
        if self.rebate <= 0.0 {
            return 0.0;
        }
        let m = self.mu();
        let lambda = (m * m + 2.0 * self.r / (self.vol * self.vol)).sqrt();
        let hs = self.barrier / self.spot;
        let pow_plus = hs.powf(m + lambda);
        let pow_minus = hs.powf(m - lambda);
        let z = (self.barrier / self.spot).ln() / self.std_dev + lambda * self.std_dev;
        let n1 = Self::n(eta * z);
        let n2 = Self::n(eta * (z - 2.0 * lambda * self.std_dev));
        self.rebate
            * ((if n1 == 0.0 { 0.0 } else { pow_plus * n1 })
                + (if n2 == 0.0 { 0.0 } else { pow_minus * n2 }))
    }
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn haug_barrier_price(
    barrier_type: BarrierType,
    option_type: OptionType,
    p: HaugInputs,
) -> QlResult<Real> {
    require!(p.vol > 0.0, "barrier pricing needs a positive volatility");
    require!(
        p.std_dev > 0.0,
        "barrier pricing needs a positive volatility"
    );
    let k = p.strike;
    let h = p.barrier;
    let value = match option_type {
        OptionType::Call => match barrier_type {
            BarrierType::DownIn if k >= h => p.c(1.0, 1.0) + p.e(1.0),
            BarrierType::DownIn => p.a(1.0) - p.b(1.0) + p.d(1.0, 1.0) + p.e(1.0),
            BarrierType::UpIn if k >= h => p.a(1.0) + p.e(-1.0),
            BarrierType::UpIn => p.b(1.0) - p.c(-1.0, 1.0) + p.d(-1.0, 1.0) + p.e(-1.0),
            BarrierType::DownOut if k >= h => p.a(1.0) - p.c(1.0, 1.0) + p.f(1.0),
            BarrierType::DownOut => p.b(1.0) - p.d(1.0, 1.0) + p.f(1.0),
            BarrierType::UpOut if k >= h => p.f(-1.0),
            BarrierType::UpOut => p.a(1.0) - p.b(1.0) + p.c(-1.0, 1.0) - p.d(-1.0, 1.0) + p.f(-1.0),
        },
        OptionType::Put => match barrier_type {
            BarrierType::DownIn if k >= h => p.b(-1.0) - p.c(1.0, -1.0) + p.d(1.0, -1.0) + p.e(1.0),
            BarrierType::DownIn => p.a(-1.0) + p.e(1.0),
            BarrierType::UpIn if k >= h => p.a(-1.0) - p.b(-1.0) + p.d(-1.0, -1.0) + p.e(-1.0),
            BarrierType::UpIn => p.c(-1.0, -1.0) + p.e(-1.0),
            BarrierType::DownOut if k >= h => {
                p.a(-1.0) - p.b(-1.0) + p.c(1.0, -1.0) - p.d(1.0, -1.0) + p.f(1.0)
            }
            BarrierType::DownOut => p.f(1.0),
            BarrierType::UpOut if k >= h => p.b(-1.0) - p.d(-1.0, -1.0) + p.f(-1.0),
            BarrierType::UpOut => p.a(-1.0) - p.c(-1.0, -1.0) + p.f(-1.0),
        },
    };
    Ok(value)
}

/// QuantLib `BarrierOption::engine::triggered`: already across the barrier.
fn triggered(barrier_type: BarrierType, spot: Real, barrier: Real) -> bool {
    match barrier_type {
        BarrierType::DownIn | BarrierType::DownOut => spot < barrier,
        BarrierType::UpIn | BarrierType::UpOut => spot > barrier,
    }
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
    use crate::cashflows::dividend_vector;
    use crate::exercise::EuropeanExercise;
    use crate::handle::{Handle, RelinkableHandle};
    use crate::instrument::Instrument;
    use crate::instruments::{StrikedTypePayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::pricingengines::set_fd_black_scholes_barrier_engine;
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

    fn black_scholes(
        option_type: OptionType,
        spot: Real,
        strike: Real,
        r: Real,
        q: Real,
        vol: Real,
        t: Real,
    ) -> Real {
        let std = vol * t.sqrt();
        let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / std;
        let d2 = d1 - std;
        let n = CumulativeNormalDistribution::standard();
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        match option_type {
            OptionType::Call => spot * df_q * n.value(d1) - strike * df_r * n.value(d2),
            OptionType::Put => strike * df_r * n.value(-d2) - spot * df_q * n.value(-d1),
        }
    }

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
        let vanilla = black_scholes(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
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
            flat(0.01, dc.clone()),
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

    #[test]
    fn put_call_symmetry_for_knock_out() {
        // QuantLib barrieroption.cpp `testPutCallSymmetry`: a knock-out call
        // and the inverted-strike/barrier put (rates swapped) replicate @ 1e-4.
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual360::new();
        let spot_price = 100.0;
        let r = 0.01;
        let underlying = Handle::new(shared(SimpleQuote::new(spot_price)) as Shared<dyn Quote>);
        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let vol = Handle::new(shared(BlackConstantVol::new(
            today,
            Some(NullCalendar::new()),
            0.25,
            dc.clone(),
        )) as Shared<dyn BlackVolTermStructure>);
        let process_call = shared(BlackScholesMertonProcess::new(
            underlying.clone(),
            flat(0.0),
            flat(r),
            vol.clone(),
        ));
        let process_put = shared(BlackScholesMertonProcess::new(
            underlying,
            flat(r),
            flat(0.0),
            vol,
        ));
        let call_engine =
            shared_mut(AnalyticBarrierEngine::new(process_call)) as SharedMut<dyn PricingEngine>;
        let put_engine =
            shared_mut(AnalyticBarrierEngine::new(process_put)) as SharedMut<dyn PricingEngine>;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 360));

        // (call strike, call barrier, call type, put strike, put barrier, put type)
        let cases = [
            (
                90.0,
                95.0,
                BarrierType::DownOut,
                111.11111,
                105.26315,
                BarrierType::UpOut,
            ),
            (
                95.0,
                95.0,
                BarrierType::DownOut,
                105.26315,
                105.26315,
                BarrierType::UpOut,
            ),
            (
                100.0,
                95.0,
                BarrierType::DownOut,
                100.0,
                105.26315,
                BarrierType::UpOut,
            ),
            (
                105.0,
                95.0,
                BarrierType::DownOut,
                95.23809,
                105.26315,
                BarrierType::UpOut,
            ),
            (
                110.0,
                95.0,
                BarrierType::DownOut,
                90.90909,
                105.26315,
                BarrierType::UpOut,
            ),
            (
                90.0,
                120.0,
                BarrierType::UpOut,
                111.11111,
                83.33333,
                BarrierType::DownOut,
            ),
            (
                95.0,
                120.0,
                BarrierType::UpOut,
                105.26315,
                83.33333,
                BarrierType::DownOut,
            ),
            (
                100.0,
                120.0,
                BarrierType::UpOut,
                100.0,
                83.33333,
                BarrierType::DownOut,
            ),
            (
                105.0,
                120.0,
                BarrierType::UpOut,
                95.23809,
                83.33333,
                BarrierType::DownOut,
            ),
            (
                110.0,
                120.0,
                BarrierType::UpOut,
                90.90909,
                83.33333,
                BarrierType::DownOut,
            ),
        ];
        for (call_strike, call_barrier, call_type, put_strike, put_barrier, put_type) in cases {
            let mut call = BarrierOption::new(
                call_type,
                call_barrier,
                PlainVanillaPayoff::new(OptionType::Call, call_strike),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            let mut put = BarrierOption::new(
                put_type,
                put_barrier,
                PlainVanillaPayoff::new(OptionType::Put, put_strike),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            call.base_mut()
                .set_pricing_engine(SharedMut::clone(&call_engine));
            put.base_mut()
                .set_pricing_engine(SharedMut::clone(&put_engine));
            let put_value = put.npv().unwrap();
            let call_value = call.npv().unwrap();
            let scaled = (put_strike / spot_price) * call_value;
            let error = (put_value - scaled).abs();
            assert!(
                error <= 1.0e-4,
                "put-call symmetry failed: put {put_value} vs {scaled} * call {call_value} (error {error})"
            );
        }
    }

    #[test]
    fn european_haug_values() {
        // QuantLib barrieroption.cpp `testHaugValues` European rows (Haug 1998
        // p.72). Analytic engine only; American / FD / binomial stay follow-up.
        // Constants shared by every published row: rebate 3, S=100, q=0.04,
        // r=0.08, t=0.50, tolerance 1e-4.
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual360::new();
        let time_to_days = |t: Real| (t * 360.0).round() as i32;

        type Row = (BarrierType, Real, OptionType, Real, Real, Real);
        #[rustfmt::skip]
        const HAUG: &[Row] = &[
            // barrierType, barrier, type, strike, vol, result
            (BarrierType::DownOut,  95.0, OptionType::Call,  90.0, 0.25,  9.0246),
            (BarrierType::DownOut,  95.0, OptionType::Call, 100.0, 0.25,  6.7924),
            (BarrierType::DownOut,  95.0, OptionType::Call, 110.0, 0.25,  4.8759),
            (BarrierType::DownOut, 100.0, OptionType::Call,  90.0, 0.25,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Call, 100.0, 0.25,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Call, 110.0, 0.25,  3.0000),
            (BarrierType::UpOut,   105.0, OptionType::Call,  90.0, 0.25,  2.6789),
            (BarrierType::UpOut,   105.0, OptionType::Call, 100.0, 0.25,  2.3580),
            (BarrierType::UpOut,   105.0, OptionType::Call, 110.0, 0.25,  2.3453),
            (BarrierType::DownIn,   95.0, OptionType::Call,  90.0, 0.25,  7.7627),
            (BarrierType::DownIn,   95.0, OptionType::Call, 100.0, 0.25,  4.0109),
            (BarrierType::DownIn,   95.0, OptionType::Call, 110.0, 0.25,  2.0576),
            (BarrierType::DownIn,  100.0, OptionType::Call,  90.0, 0.25, 13.8333),
            (BarrierType::DownIn,  100.0, OptionType::Call, 100.0, 0.25,  7.8494),
            (BarrierType::DownIn,  100.0, OptionType::Call, 110.0, 0.25,  3.9795),
            (BarrierType::UpIn,    105.0, OptionType::Call,  90.0, 0.25, 14.1112),
            (BarrierType::UpIn,    105.0, OptionType::Call, 100.0, 0.25,  8.4482),
            (BarrierType::UpIn,    105.0, OptionType::Call, 110.0, 0.25,  4.5910),
            (BarrierType::DownOut,  95.0, OptionType::Call,  90.0, 0.30,  8.8334),
            (BarrierType::DownOut,  95.0, OptionType::Call, 100.0, 0.30,  7.0285),
            (BarrierType::DownOut,  95.0, OptionType::Call, 110.0, 0.30,  5.4137),
            (BarrierType::DownOut, 100.0, OptionType::Call,  90.0, 0.30,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Call, 100.0, 0.30,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Call, 110.0, 0.30,  3.0000),
            (BarrierType::UpOut,   105.0, OptionType::Call,  90.0, 0.30,  2.6341),
            (BarrierType::UpOut,   105.0, OptionType::Call, 100.0, 0.30,  2.4389),
            (BarrierType::UpOut,   105.0, OptionType::Call, 110.0, 0.30,  2.4315),
            (BarrierType::DownIn,   95.0, OptionType::Call,  90.0, 0.30,  9.0093),
            (BarrierType::DownIn,   95.0, OptionType::Call, 100.0, 0.30,  5.1370),
            (BarrierType::DownIn,   95.0, OptionType::Call, 110.0, 0.30,  2.8517),
            (BarrierType::DownIn,  100.0, OptionType::Call,  90.0, 0.30, 14.8816),
            (BarrierType::DownIn,  100.0, OptionType::Call, 100.0, 0.30,  9.2045),
            (BarrierType::DownIn,  100.0, OptionType::Call, 110.0, 0.30,  5.3043),
            (BarrierType::UpIn,    105.0, OptionType::Call,  90.0, 0.30, 15.2098),
            (BarrierType::UpIn,    105.0, OptionType::Call, 100.0, 0.30,  9.7278),
            (BarrierType::UpIn,    105.0, OptionType::Call, 110.0, 0.30,  5.8350),
            (BarrierType::DownOut,  95.0, OptionType::Put,   90.0, 0.25,  2.2798),
            (BarrierType::DownOut,  95.0, OptionType::Put,  100.0, 0.25,  2.2947),
            (BarrierType::DownOut,  95.0, OptionType::Put,  110.0, 0.25,  2.6252),
            (BarrierType::DownOut, 100.0, OptionType::Put,   90.0, 0.25,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Put,  100.0, 0.25,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Put,  110.0, 0.25,  3.0000),
            (BarrierType::UpOut,   105.0, OptionType::Put,   90.0, 0.25,  3.7760),
            (BarrierType::UpOut,   105.0, OptionType::Put,  100.0, 0.25,  5.4932),
            (BarrierType::UpOut,   105.0, OptionType::Put,  110.0, 0.25,  7.5187),
            (BarrierType::DownIn,   95.0, OptionType::Put,   90.0, 0.25,  2.9586),
            (BarrierType::DownIn,   95.0, OptionType::Put,  100.0, 0.25,  6.5677),
            (BarrierType::DownIn,   95.0, OptionType::Put,  110.0, 0.25, 11.9752),
            (BarrierType::DownIn,  100.0, OptionType::Put,   90.0, 0.25,  2.2845),
            (BarrierType::DownIn,  100.0, OptionType::Put,  100.0, 0.25,  5.9085),
            (BarrierType::DownIn,  100.0, OptionType::Put,  110.0, 0.25, 11.6465),
            (BarrierType::UpIn,    105.0, OptionType::Put,   90.0, 0.25,  1.4653),
            (BarrierType::UpIn,    105.0, OptionType::Put,  100.0, 0.25,  3.3721),
            (BarrierType::UpIn,    105.0, OptionType::Put,  110.0, 0.25,  7.0846),
            (BarrierType::DownOut,  95.0, OptionType::Put,   90.0, 0.30,  2.4170),
            (BarrierType::DownOut,  95.0, OptionType::Put,  100.0, 0.30,  2.4258),
            (BarrierType::DownOut,  95.0, OptionType::Put,  110.0, 0.30,  2.6246),
            (BarrierType::DownOut, 100.0, OptionType::Put,   90.0, 0.30,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Put,  100.0, 0.30,  3.0000),
            (BarrierType::DownOut, 100.0, OptionType::Put,  110.0, 0.30,  3.0000),
            (BarrierType::UpOut,   105.0, OptionType::Put,   90.0, 0.30,  4.2293),
            (BarrierType::UpOut,   105.0, OptionType::Put,  100.0, 0.30,  5.8032),
            (BarrierType::UpOut,   105.0, OptionType::Put,  110.0, 0.30,  7.5649),
            (BarrierType::DownIn,   95.0, OptionType::Put,   90.0, 0.30,  3.8769),
            (BarrierType::DownIn,   95.0, OptionType::Put,  100.0, 0.30,  7.7989),
            (BarrierType::DownIn,   95.0, OptionType::Put,  110.0, 0.30, 13.3078),
            (BarrierType::DownIn,  100.0, OptionType::Put,   90.0, 0.30,  3.3328),
            (BarrierType::DownIn,  100.0, OptionType::Put,  100.0, 0.30,  7.2636),
            (BarrierType::DownIn,  100.0, OptionType::Put,  110.0, 0.30, 12.9713),
            (BarrierType::UpIn,    105.0, OptionType::Put,   90.0, 0.30,  2.0658),
            (BarrierType::UpIn,    105.0, OptionType::Put,  100.0, 0.30,  4.4226),
            (BarrierType::UpIn,    105.0, OptionType::Put,  110.0, 0.30,  8.3686),
        ];

        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        for &(barrier_type, barrier, option_type, strike, vol, expected) in HAUG {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
                flat(0.04),
                flat(0.08),
                Handle::new(shared(BlackConstantVol::new(
                    today,
                    Some(NullCalendar::new()),
                    vol,
                    dc.clone(),
                )) as Shared<dyn BlackVolTermStructure>),
            ));
            let mut option = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                3.0,
                PlainVanillaPayoff::new(option_type, strike),
                shared(EuropeanExercise::new(today + time_to_days(0.50))),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_barrier_engine(&mut option, process);
            let calculated = option.npv().unwrap();
            let error = (calculated - expected).abs();
            assert!(
                error <= 1.0e-4,
                "{barrier_type:?} {option_type:?} H={barrier} K={strike} v={vol}: \
                 {calculated} vs Haug {expected} (error {error})"
            );
        }
    }

    fn flat_bs_process(
        today: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
        dc: crate::time::daycounter::DayCounter,
    ) -> Shared<BlackScholesMertonProcess> {
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(
                today,
                Some(NullCalendar::new()),
                vol,
                dc,
            )) as Shared<dyn BlackVolTermStructure>),
        ))
    }

    #[test]
    fn babsiri_knock_in_calls() {
        // QuantLib barrieroption.cpp `testBabsiriValues` analytic arm
        // (El Babsiri–Noel, Journal of Derivatives, Winter 1998). MC follow-up.
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual360::new();
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 360));
        #[rustfmt::skip]
        let cases = [
            // type, vol, strike, barrier, call value
            (BarrierType::DownIn, 0.10, 100.0,  90.0,  0.07187),
            (BarrierType::DownIn, 0.15, 100.0,  90.0,  0.60638),
            (BarrierType::DownIn, 0.20, 100.0,  90.0,  1.64005),
            (BarrierType::DownIn, 0.25, 100.0,  90.0,  2.98495),
            (BarrierType::DownIn, 0.30, 100.0,  90.0,  4.50952),
            (BarrierType::UpIn,   0.10, 100.0, 110.0,  4.79148),
            (BarrierType::UpIn,   0.15, 100.0, 110.0,  7.08268),
            (BarrierType::UpIn,   0.20, 100.0, 110.0,  9.11008),
            (BarrierType::UpIn,   0.25, 100.0, 110.0, 11.06148),
            (BarrierType::UpIn,   0.30, 100.0, 110.0, 12.98351),
        ];
        for (barrier_type, vol, strike, barrier, expected) in cases {
            let process = flat_bs_process(today, 100.0, 0.02, 0.05, vol, dc.clone());
            let mut option = BarrierOption::new(
                barrier_type,
                barrier,
                PlainVanillaPayoff::new(OptionType::Call, strike),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_barrier_engine(&mut option, process);
            let calculated = option.npv().unwrap();
            let error = (calculated - expected).abs();
            assert!(
                error <= 1.0e-5,
                "{barrier_type:?} K={strike} H={barrier} v={vol}: \
                 {calculated} vs Babsiri {expected} (error {error})"
            );
        }
    }

    #[test]
    fn beaglehole_down_out_call() {
        // QuantLib barrieroption.cpp `testBeagleholeValues` analytic arm
        // (Beaglehole–Dybvig–Zhou, FAJ Jan/Feb 1997). MC follow-up.
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = flat_bs_process(today, 50.0, 0.0, 1.1_f64.ln(), 0.50, Actual360::new());
        let mut option = BarrierOption::new(
            BarrierType::DownOut,
            45.0,
            PlainVanillaPayoff::new(OptionType::Call, 50.0),
            shared(EuropeanExercise::new(today + 360)),
            settings,
        )
        .unwrap();
        set_analytic_barrier_engine(&mut option, process);
        let calculated = option.npv().unwrap();
        let expected = 5.477;
        let error = (calculated - expected).abs();
        assert!(
            error <= 1.0e-3,
            "DownOut K=50 H=45: {calculated} vs Beaglehole {expected} (error {error})"
        );
    }

    #[test]
    fn low_volatility_matches_zero_vol_limits() {
        // QuantLib barrieroption.cpp `testLowVolatility`: vol = 1e-7 must not
        // yield NaN, and prices stay within 0.5 of the deterministic limit.
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let dc = Actual365Fixed::new();
        #[rustfmt::skip]
        let cases = [
            // strike, type, barrier, barrier type, rebate, r, q, expected
            (105.0, OptionType::Put,  107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 3.0),
            (109.0, OptionType::Put,  107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 7.0),
            (100.0, OptionType::Put,  107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 0.0),
            ( 99.0, OptionType::Put,  101.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 4.0),
            (105.0, OptionType::Put,  101.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 4.0),
            (105.0, OptionType::Call, 107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 0.0),
            (109.0, OptionType::Call, 107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 0.0),
            (100.0, OptionType::Call, 107.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 2.0),
            (105.0, OptionType::Call, 101.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 4.0),
            ( 99.0, OptionType::Call, 101.0, BarrierType::UpOut,   4.0, 0.03, 0.01, 4.0),
            (105.0, OptionType::Put,  107.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 4.0),
            (109.0, OptionType::Put,  107.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 4.0),
            (105.0, OptionType::Put,  101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 2.0),
            (100.0, OptionType::Put,  101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 0.0),
            (102.0, OptionType::Put,  101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 0.0),
            (105.0, OptionType::Call, 107.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 4.0),
            (109.0, OptionType::Call, 107.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 4.0),
            (105.0, OptionType::Call, 101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 0.0),
            (100.0, OptionType::Call, 101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 3.0),
            (102.0, OptionType::Call, 101.0, BarrierType::UpIn,    4.0, 0.03, 0.00, 1.0),
            ( 91.0, OptionType::Put,   93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 0.0),
            ( 95.0, OptionType::Put,   93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 0.0),
            (100.0, OptionType::Put,   93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 2.0),
            ( 97.0, OptionType::Put,   99.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 4.0),
            (101.0, OptionType::Put,   99.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 4.0),
            ( 91.0, OptionType::Call,  93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 7.0),
            ( 95.0, OptionType::Call,  93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 3.0),
            (100.0, OptionType::Call,  93.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 0.0),
            ( 95.0, OptionType::Call,  99.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 4.0),
            (101.0, OptionType::Call,  99.0, BarrierType::DownOut, 4.0, 0.01, 0.03, 4.0),
            ( 91.0, OptionType::Put,   93.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 4.0),
            ( 95.0, OptionType::Put,   93.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 4.0),
            (100.0, OptionType::Put,   99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 3.0),
            ( 95.0, OptionType::Put,   99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 0.0),
            ( 95.0, OptionType::Put,   99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 0.0),
            ( 91.0, OptionType::Call,  93.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 4.0),
            ( 95.0, OptionType::Call,  93.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 4.0),
            ( 98.0, OptionType::Call,  99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 0.0),
            (100.0, OptionType::Call,  99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 0.0),
            ( 95.0, OptionType::Call,  99.0, BarrierType::DownIn,  4.0, 0.01, 0.04, 2.0),
        ];
        for (strike, option_type, barrier, barrier_type, rebate, r, q, expected) in cases {
            let process = flat_bs_process(today, 100.0, q, r, 1e-7, dc.clone());
            let mut option = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                rebate,
                PlainVanillaPayoff::new(option_type, strike),
                shared(EuropeanExercise::new(expiry)),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_barrier_engine(&mut option, process);
            let calculated = option.npv().unwrap();
            let error = (calculated - expected).abs();
            assert!(
                calculated.is_finite() && error <= 0.5,
                "{barrier_type:?} {option_type:?} K={strike} H={barrier} r={r} q={q}: \
                 {calculated} vs {expected} (error {error})"
            );
        }
    }

    #[test]
    fn implied_volatility_reproduces_target_without_dividends() {
        // QuantLib barrieroption.cpp `testImpliedVolatility` no-dividend arm.
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual365Fixed::new();
        let process = flat_bs_process(today, 100.0, 0.0, 0.05, 0.0, dc.clone());
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(
            today + Period::new(1, TimeUnit::Years),
        ));
        let payoff = PlainVanillaPayoff::new(OptionType::Put, 105.0);
        #[rustfmt::skip]
        let cases = [
            (BarrierType::DownOut,  90.0, 1.0),
            (BarrierType::UpOut,   110.0, 1.0),
            (BarrierType::DownIn,   90.0, 5.0),
            (BarrierType::UpIn,    110.0, 5.0),
        ];
        for (barrier_type, barrier, target) in cases {
            let option = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            let implied = option
                .implied_volatility(target, Shared::clone(&process), 1e-6, 100, 1e-7, 4.0)
                .unwrap();
            let priced = flat_bs_process(today, 100.0, 0.0, 0.05, implied, dc.clone());
            let mut check = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_barrier_engine(&mut check, priced);
            let calculated = check.npv().unwrap();
            let error = (calculated - target).abs();
            assert!(
                error <= 1.0e-5,
                "{barrier_type:?} H={barrier}: NPV {calculated} vs target {target} \
                 (implied vol {implied}, error {error})"
            );
        }
    }

    #[test]
    fn implied_volatility_reproduces_target_with_dividends() {
        // QuantLib barrieroption.cpp `testImpliedVolatility` discrete-dividend arm.
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual365Fixed::new();
        let process = flat_bs_process(today, 100.0, 0.0, 0.05, 0.0, dc.clone());
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(
            today + Period::new(1, TimeUnit::Years),
        ));
        let payoff = PlainVanillaPayoff::new(OptionType::Put, 105.0);
        let dividends =
            dividend_vector(&[today + Period::new(6, TimeUnit::Months)], &[10.0]).unwrap();
        #[rustfmt::skip]
        let cases = [
            (BarrierType::DownOut,  90.0, 8.0),
            (BarrierType::UpOut,   110.0, 12.0),
            (BarrierType::DownIn,   90.0, 9.0),
            (BarrierType::UpIn,    110.0, 8.0),
        ];
        for (barrier_type, barrier, target) in cases {
            let option = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            let implied = option
                .implied_volatility_with_dividends(
                    target,
                    Shared::clone(&process),
                    dividends.clone(),
                    1e-6,
                    100,
                    1e-7,
                    4.0,
                )
                .unwrap();
            let priced = flat_bs_process(today, 100.0, 0.0, 0.05, implied, dc.clone());
            let mut check = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_fd_black_scholes_barrier_engine(
                &mut check,
                shared_mut(FdBlackScholesBarrierEngine::with_dividends(
                    priced,
                    dividends.clone(),
                )),
            );
            let calculated = check.npv().unwrap();
            let error = (calculated - target).abs();
            eprintln!(
                "{barrier_type:?} H={barrier}: implied={implied:.8} NPV={calculated:.8} \
                 target={target} error={error:.2e}"
            );
            assert!(
                error <= 1.0e-5,
                "{barrier_type:?} H={barrier}: NPV {calculated} vs target {target} \
                 (implied vol {implied}, error {error})"
            );
        }
    }
}
