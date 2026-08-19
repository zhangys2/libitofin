//! Binomial-tree engine for barrier options.
//!
//! Port of `ql/pricingengines/barrier/binomialbarrierengine.hpp` together with
//! `discretizedbarrieroption.{hpp,cpp}` (`DiscretizedBarrierOption` only).
//! Cox–Ross–Rubinstein steps are optionally lengthened by the Boyle–Lau
//! barrier-alignment heuristic. `DiscretizedDermanKaniBarrierOption` stays
//! follow-up.

use crate::discretizedasset::{DiscretizedAsset, DiscretizedAssetBase};
use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{BarrierArguments, BarrierType, PlainVanillaPayoff, StrikedTypePayoff};
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::comparison::close;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::binomialtree::CoxRossRubinstein;
use crate::methods::lattices::lattice::Lattice;
use crate::methods::lattices::treelattice::{TreeLattice1D, TreeLatticeImpl};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::binomialvanillaengine::DiscretizedVanillaOption;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time, Volatility};

use super::fdblackscholesbarrierengine::triggered;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// A constant-coefficient Black–Scholes binomial lattice over a
/// [`CoxRossRubinstein`] tree (`bsmlattice.hpp` `BlackScholesLattice`).
struct BlackScholesLattice {
    tree: CoxRossRubinstein,
    grid: TimeGrid,
    risk_free_rate: Real,
}

impl TreeLatticeImpl for BlackScholesLattice {
    type Tree = CoxRossRubinstein;

    fn tree(&self) -> &CoxRossRubinstein {
        &self.tree
    }

    fn discount(&self, i: Size, _index: Size) -> Real {
        (-self.risk_free_rate * self.grid.dt(i)).exp()
    }
}

/// Discretized barrier option (`discretizedbarrieroption.hpp:41`).
struct DiscretizedBarrierOption {
    base: DiscretizedAssetBase,
    barrier_type: BarrierType,
    barrier: Real,
    rebate: Real,
    payoff: PlainVanillaPayoff,
    exercise_type: ExerciseType,
    stopping_times: Vec<Time>,
    vanilla: DiscretizedVanillaOption,
}

impl DiscretizedBarrierOption {
    fn new(
        arguments: &BarrierArguments,
        process: &GeneralizedBlackScholesProcess,
        grid: &TimeGrid,
    ) -> QlResult<Self> {
        let exercise = arguments.exercise.as_ref().expect("validated");
        require!(
            !exercise.dates().is_empty(),
            "specify at least one stopping date"
        );
        let payoff = arguments.payoff.expect("validated");
        let payoff_shared: Shared<dyn StrikedTypePayoff> = shared(payoff);
        let vanilla = DiscretizedVanillaOption::new(payoff_shared, exercise, process, grid)?;
        let mut stopping_times = Vec::with_capacity(exercise.dates().len());
        for date in exercise.dates() {
            let time = process.time(date)?;
            let index = grid.closest_index(time);
            stopping_times.push(grid.times()[index]);
        }
        Ok(DiscretizedBarrierOption {
            base: DiscretizedAssetBase::default(),
            barrier_type: arguments.barrier_type.expect("validated"),
            barrier: arguments.barrier.expect("validated"),
            rebate: arguments.rebate.expect("validated"),
            payoff,
            exercise_type: exercise.exercise_type(),
            stopping_times,
            vanilla,
        })
    }

    fn stopping_time(&self) -> bool {
        let now = self.time();
        match self.exercise_type {
            ExerciseType::American => {
                now <= self.stopping_times[1] && now >= self.stopping_times[0]
            }
            ExerciseType::European => self.is_on_time(self.stopping_times[0]),
            ExerciseType::Bermudan => self.stopping_times.iter().any(|&t| self.is_on_time(t)),
        }
    }

    fn check_barrier(&self, optvalues: &mut Array, grid: &Array, vanilla: &Array) {
        let end_time = self.is_on_time(*self.stopping_times.last().expect("non-empty"));
        let stopping_time = self.stopping_time();
        for j in 0..optvalues.size() {
            let spot = grid[j];
            match self.barrier_type {
                BarrierType::DownIn => {
                    if spot <= self.barrier {
                        if stopping_time {
                            optvalues[j] = vanilla[j].max(self.payoff.value(spot));
                        } else {
                            optvalues[j] = vanilla[j];
                        }
                    } else if end_time {
                        optvalues[j] = self.rebate;
                    }
                }
                BarrierType::DownOut => {
                    if spot <= self.barrier {
                        optvalues[j] = self.rebate;
                    } else if stopping_time {
                        optvalues[j] = optvalues[j].max(self.payoff.value(spot));
                    }
                }
                BarrierType::UpIn => {
                    if spot >= self.barrier {
                        if stopping_time {
                            optvalues[j] = vanilla[j].max(self.payoff.value(spot));
                        } else {
                            optvalues[j] = vanilla[j];
                        }
                    } else if end_time {
                        optvalues[j] = self.rebate;
                    }
                }
                BarrierType::UpOut => {
                    if spot >= self.barrier {
                        optvalues[j] = self.rebate;
                    } else if stopping_time {
                        optvalues[j] = optvalues[j].max(self.payoff.value(spot));
                    }
                }
            }
        }
    }
}

impl DiscretizedAsset for DiscretizedBarrierOption {
    fn base(&self) -> &DiscretizedAssetBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut DiscretizedAssetBase {
        &mut self.base
    }

    fn as_asset_mut(&mut self) -> &mut dyn DiscretizedAsset {
        self
    }

    fn reset(&mut self, size: Size) -> QlResult<()> {
        let method = self.require_method()?;
        let t = self.time();
        self.vanilla.initialize(method, t)?;
        *self.values_mut() = Array::filled(size, 0.0);
        self.adjust_values()
    }

    fn mandatory_times(&self) -> Vec<Time> {
        self.stopping_times.clone()
    }

    fn post_adjust_values_impl(&mut self) -> QlResult<()> {
        if matches!(self.barrier_type, BarrierType::DownIn | BarrierType::UpIn) {
            self.vanilla.rollback(self.time())?;
        }
        let method = self.require_method()?;
        let grid = method.grid(self.time())?;
        let vanilla = self.vanilla.values().clone();
        let mut values = self.values().clone();
        self.check_barrier(&mut values, &grid, &vanilla);
        *self.values_mut() = values;
        Ok(())
    }
}

/// Boyle–Lau step count so a CRR node sits on the barrier
/// (`binomialbarrierengine.hpp`, Journal of Derivatives 1/1994).
fn boyle_lau_steps(
    time_steps: Size,
    max_time_steps: Size,
    spot: Real,
    barrier: Real,
    vol: Volatility,
    maturity: Time,
) -> Size {
    if max_time_steps <= time_steps || spot <= 0.0 || barrier <= 0.0 {
        return time_steps;
    }
    let ratio = if spot > barrier {
        spot / barrier
    } else {
        barrier / spot
    };
    let divisor = ratio.ln().powi(2);
    if close(divisor, 0.0) {
        return time_steps;
    }
    let mut optimum_steps = time_steps;
    for i in 1..time_steps {
        let optimum = ((i * i) as Real * vol * vol * maturity / divisor) as Size;
        if time_steps < optimum {
            optimum_steps = optimum;
            break;
        }
    }
    optimum_steps.min(max_time_steps)
}

/// Cox–Ross–Rubinstein binomial engine for barrier options.
pub struct BinomialBarrierEngine {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Size,
    max_time_steps: Size,
}

impl BinomialBarrierEngine {
    /// `BinomialBarrierEngine(process, timeSteps)` with Boyle–Lau enabled
    /// (`maxTimeSteps = 0` → `max(1000, 5·timeSteps)`).
    ///
    /// # Errors
    ///
    /// Fails when `time_steps` is zero.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Size,
    ) -> QlResult<Self> {
        Self::with_max_time_steps(process, time_steps, 0)
    }

    /// `BinomialBarrierEngine(process, timeSteps, maxTimeSteps)`.
    ///
    /// `max_time_steps == 0` applies the C++ default cap. `max_time_steps ==
    /// time_steps` disables Boyle–Lau.
    ///
    /// # Errors
    ///
    /// Fails when `time_steps` is zero, or when `max_time_steps` is neither
    /// zero nor at least `time_steps`.
    pub fn with_max_time_steps(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Size,
        max_time_steps: Size,
    ) -> QlResult<Self> {
        require!(
            time_steps > 0,
            "timeSteps must be positive, {time_steps} not allowed"
        );
        require!(
            max_time_steps == 0 || max_time_steps >= time_steps,
            "maxTimeSteps must be zero or greater than or equal to timeSteps, \
             {max_time_steps} not allowed"
        );
        let max_time_steps = if max_time_steps == 0 {
            1000.max(time_steps * 5)
        } else {
            max_time_steps
        };
        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(process.observable());
        Ok(BinomialBarrierEngine {
            base,
            process,
            time_steps,
            max_time_steps,
        })
    }
}

impl AsObservable for BinomialBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BinomialBarrierEngine {
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
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        let barrier_type = args.barrier_type.expect("validated");
        let barrier = args.barrier.expect("validated");
        require!(payoff.strike() > 0.0, "strike must be positive");

        let s0 = self.process.x0()?;
        require!(s0 > 0.0, "negative or null underlying given");
        require!(!triggered(barrier_type, s0, barrier), "barrier touched");

        let maturity_date = exercise.last_date();
        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let rfdc = risk_free.require_day_counter()?;
        let divdc = dividend.require_day_counter()?;
        let v = vol_ts.black_vol_date(maturity_date, s0, false)?;
        let r = risk_free
            .zero_rate_date(
                maturity_date,
                rfdc.clone(),
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
        let reference_date = risk_free.reference_date()?;
        let maturity = rfdc.year_fraction(reference_date, maturity_date);
        require!(
            maturity > 0.0,
            "the binomial engine needs a positive maturity"
        );

        let steps = boyle_lau_steps(
            self.time_steps,
            self.max_time_steps,
            s0,
            barrier,
            v,
            maturity,
        );
        let tree = CoxRossRubinstein::new(s0, r, q, v, maturity, steps)?;
        let grid = TimeGrid::new(maturity, steps)?;
        let bsl = BlackScholesLattice {
            tree,
            grid: grid.clone(),
            risk_free_rate: r,
        };
        let lattice: Shared<dyn Lattice> = shared(TreeLattice1D::new(bsl, grid.clone())?);

        let mut option = DiscretizedBarrierOption::new(args, &self.process, &grid)?;
        option.initialize(Shared::clone(&lattice), maturity)?;
        option.rollback(0.0)?;
        let value = option.present_value()?;

        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

/// Attach the binomial barrier engine to an option.
pub fn set_binomial_barrier_engine(
    option: &mut crate::instruments::BarrierOption,
    engine: SharedMut<BinomialBarrierEngine>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::BarrierOption;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::Quote;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    const STEPS: Size = 400;
    const BOYLE_LAU_TOL: Real = 1.1e-2;
    const HAUG_DAYS: i32 = 180;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat_bs_process(
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<BlackScholesMertonProcess> {
        let dc = Actual360::new();
        let t = today();
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                t,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                t,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(
                shared(BlackConstantVol::new(t, Some(NullCalendar::new()), vol, dc))
                    as Shared<dyn BlackVolTermStructure>,
            ),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn npv(
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
        option_type: OptionType,
        strike: Real,
        exercise: Shared<dyn Exercise>,
        spot: Real,
        vol: Real,
    ) -> QlResult<Real> {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut option = BarrierOption::with_rebate(
            barrier_type,
            barrier,
            rebate,
            PlainVanillaPayoff::new(option_type, strike),
            exercise,
            settings,
        )?;
        set_binomial_barrier_engine(
            &mut option,
            shared_mut(BinomialBarrierEngine::new(
                flat_bs_process(spot, 0.04, 0.08, vol),
                STEPS,
            )?),
        );
        option.npv()
    }

    fn european(days: i32) -> Shared<dyn Exercise> {
        shared(EuropeanExercise::new(today() + days))
    }

    fn american(days: i32) -> Shared<dyn Exercise> {
        shared(AmericanExercise::new(today(), today() + days, false).unwrap())
    }

    /// `barrieroption.cpp` `testHaugValues` American knock-out / knock-in
    /// rows (Haug VBA, 400 steps). Shared market: S=100, q=0.04, r=0.08,
    /// t=0.50, v=0.25.
    type HaugAmerican = (BarrierType, Real, Real, OptionType, Real, Real);
    #[rustfmt::skip]
    const HAUG_AMERICAN: &[HaugAmerican] = &[
        (BarrierType::DownOut,  95.0, 0.0, OptionType::Call,  90.0, 10.4655),
        (BarrierType::DownOut,  95.0, 0.0, OptionType::Call, 100.0,  4.5159),
        (BarrierType::DownOut,  95.0, 0.0, OptionType::Call, 110.0,  2.5971),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Call,  90.0,  3.0000),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Call, 100.0,  3.0000),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Call, 110.0,  3.0000),
        (BarrierType::UpOut,   105.0, 0.0, OptionType::Call,  90.0, 11.8076),
        (BarrierType::UpOut,   105.0, 0.0, OptionType::Call, 100.0,  3.3993),
        (BarrierType::UpOut,   105.0, 3.0, OptionType::Call, 110.0,  2.3457),
        (BarrierType::DownOut,  95.0, 3.0, OptionType::Put,   90.0,  2.2795),
        (BarrierType::DownOut,  95.0, 0.0, OptionType::Put,  100.0,  3.3512),
        (BarrierType::DownOut,  95.0, 0.0, OptionType::Put,  110.0, 11.5773),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Put,   90.0,  3.0000),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Put,  100.0,  3.0000),
        (BarrierType::DownOut, 100.0, 3.0, OptionType::Put,  110.0,  3.0000),
        (BarrierType::UpOut,   105.0, 0.0, OptionType::Put,   90.0,  1.4763),
        (BarrierType::UpOut,   105.0, 0.0, OptionType::Put,  100.0,  3.3001),
        (BarrierType::UpOut,   105.0, 0.0, OptionType::Put,  110.0, 10.0000),
        (BarrierType::DownIn,   95.0, 3.0, OptionType::Call,  90.0,  7.7615),
        (BarrierType::DownIn,   95.0, 3.0, OptionType::Call, 100.0,  4.0118),
        (BarrierType::DownIn,   95.0, 3.0, OptionType::Call, 110.0,  2.0544),
        (BarrierType::DownIn,  100.0, 3.0, OptionType::Call,  90.0, 13.8308),
        (BarrierType::UpIn,    105.0, 3.0, OptionType::Call,  90.0, 14.1150),
        (BarrierType::UpIn,    105.0, 3.0, OptionType::Call, 110.0,  4.5900),
    ];

    #[test]
    fn american_haug_binomial_matches_quantlib_oracle() {
        let days = HAUG_DAYS;
        let mut max_error = 0.0;
        let mut worst = String::new();
        for &(barrier_type, barrier, rebate, option_type, strike, expected) in HAUG_AMERICAN {
            let calculated = npv(
                barrier_type,
                barrier,
                rebate,
                option_type,
                strike,
                american(days),
                100.0,
                0.25,
            )
            .unwrap();
            let error = (calculated - expected).abs();
            if error > max_error {
                max_error = error;
                worst = format!(
                    "{barrier_type:?} {option_type:?} H={barrier} K={strike} R={rebate}: \
                     {calculated} vs {expected}"
                );
            }
            eprintln!(
                "{barrier_type:?} {option_type:?} H={barrier} K={strike}: \
                 calculated={calculated:.8} expected={expected:.4} diff={error:.2e}"
            );
            assert!(
                calculated.is_finite() && error <= BOYLE_LAU_TOL,
                "{barrier_type:?} {option_type:?} H={barrier} K={strike}: \
                 {calculated} vs Haug {expected} (error {error})"
            );
        }
        eprintln!("haug american binomial max error {max_error:.2e} at {worst}");
    }

    #[test]
    fn european_haug_down_out_call_stays_inside_boyle_lau_tol() {
        // One European row from the analytic table, priced on the same 400-step
        // Boyle–Lau tree QuantLib uses in `testHaugValues`.
        let calculated = npv(
            BarrierType::DownOut,
            95.0,
            3.0,
            OptionType::Call,
            90.0,
            european(HAUG_DAYS),
            100.0,
            0.25,
        )
        .unwrap();
        let expected = 9.0246;
        let error = (calculated - expected).abs();
        eprintln!("european DownOut call: {calculated:.8} vs {expected} diff={error:.2e}");
        assert!(
            error <= BOYLE_LAU_TOL,
            "{calculated} vs {expected} ({error})"
        );
    }

    #[test]
    fn binomial_barrier_rejects_zero_spot_and_triggered() {
        let days = HAUG_DAYS;
        let err = npv(
            BarrierType::DownOut,
            95.0,
            3.0,
            OptionType::Call,
            100.0,
            european(days),
            0.0,
            0.25,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("negative or null underlying"),
            "zero spot: {err}"
        );

        let err = npv(
            BarrierType::DownOut,
            101.0,
            3.0,
            OptionType::Call,
            100.0,
            european(days),
            100.0,
            0.25,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("barrier touched"), "triggered: {err}");
    }

    #[test]
    fn boyle_lau_lengthens_steps_when_the_barrier_is_off_the_tree() {
        let bumped = boyle_lau_steps(400, 2000, 100.0, 95.0, 0.25, 0.5);
        assert!(
            bumped >= 400,
            "Boyle–Lau must not shrink the requested step count, got {bumped}"
        );
        assert_eq!(boyle_lau_steps(400, 400, 100.0, 95.0, 0.25, 0.5), 400);
    }

    #[test]
    fn constructor_rejects_zero_steps() {
        let err = match BinomialBarrierEngine::new(flat_bs_process(100.0, 0.0, 0.05, 0.2), 0) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("zero steps must be rejected"),
        };
        assert!(err.contains("timeSteps must be positive"), "{err}");
    }

    #[test]
    fn american_exercise_is_accepted() {
        let days = HAUG_DAYS;
        let value = npv(
            BarrierType::DownOut,
            95.0,
            0.0,
            OptionType::Call,
            90.0,
            american(days),
            100.0,
            0.25,
        )
        .unwrap();
        assert!(value.is_finite() && value > 0.0, "{value}");
    }
}
