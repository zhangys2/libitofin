//! Finite-difference Tsiveriotis–Fernandes engine for convertible bonds.
//!
//! Equity `E` is discounted at the frozen risk-free rate. Cash `C` is
//! discounted at `r + s`, or at `r + λ(S)(1-R)` when a stock-dependent hazard
//! is supplied. The price is `U = E + C`. Call, put, coupon, and conversion
//! are a joint projection after both fields have stepped; the blended tree
//! discount is not used.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{BondResults, CallabilityType, ConvertibleBondArguments};
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::comparison::close_enough;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::CubicNaturalSpline;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher, process_helper,
};
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::solvers::{FdmBackwardSolver, FdmSchemeDesc};
use crate::methods::finitedifferences::{
    BoundarySide, DirichletBoundary, TimeDependentDirichletBoundary,
};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time};

const MESHER_EPS: Real = 0.0001;
const MESHER_SCALE: Real = 1.5;
const CONCENTRATION: Real = 0.01;
/// Events closer than this share a time layer. Matches the Douglas guard
/// against a step that would land on a negative time.
const TIME_MATCH: Time = 1.0e-8;

/// No-default cash-gamma P&L on a solved convertible grid.
#[derive(Clone, Copy, Debug)]
pub struct ConvertibleCashGamma {
    /// Revaluation minus the delta hedge, plus the coupon and financing.
    pub wealth: Real,
    /// `½ Γ (ΔS)² + Θ_B Δt`, with credit left inside `Θ`.
    pub continuation: Real,
    /// Coupons dated in `(0, Δt]`.
    pub coupon: Real,
}

enum Credit {
    Spread(Handle<dyn Quote>),
    Hazard {
        lambda0: Real,
        recovery: Real,
        alpha: Real,
        reference_spot: Real,
    },
}

struct SolvedGrid {
    log_spots: Vec<Real>,
    layers: Vec<(Time, Vec<Real>)>,
    coupons: Vec<(Time, Real)>,
    spot: Real,
    dividend_yield: Real,
    maturity: Time,
}

struct Frozen {
    spot: Real,
    rate: Real,
    dividend: Real,
    vol: Real,
    maturity: Time,
    settlement: Date,
    day_counter: DayCounter,
    spot_quote: Handle<dyn Quote>,
}

struct Schedule {
    exercise: ExerciseType,
    exercise_times: Vec<Time>,
    ratio: Real,
    redemption: Real,
    calls: Vec<(Time, Real)>,
    puts: Vec<(Time, Real)>,
    coupons: Vec<(Time, Real)>,
}

/// One-dimensional finite-difference engine for a convertible bond.
///
/// `r`, `q`, and `σ` are frozen at the conversion exercise's last date, the
/// same sample `BinomialConvertibleEngine` uses, and then held flat for the
/// whole solve. A sloped yield curve or a strike-dependent volatility on the
/// process is not read.
pub struct FdConvertibleEngine {
    base: GenericEngine<ConvertibleBondArguments, BondResults>,
    process: Shared<GeneralizedBlackScholesProcess>,
    x_grid: Size,
    time_steps: Size,
    credit: Credit,
    solved: Option<SolvedGrid>,
}

impl FdConvertibleEngine {
    /// Prices with a constant credit spread `s`, the binomial spread quote.
    ///
    /// `s` is the cash spread: equivalent to `α = 0` and `λ0 (1-R) = s`.
    ///
    /// # Errors
    ///
    /// Fails when the grid has fewer than three nodes or `time_steps` is zero.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        x_grid: Size,
        time_steps: Size,
        credit_spread: Handle<dyn Quote>,
    ) -> QlResult<Self> {
        Self::build(process, x_grid, time_steps, Credit::Spread(credit_spread))
    }

    /// Prices with `λ(S) = λ0 (S/S0)^{-α}` for `S > 0`.
    ///
    /// Cash is discounted at `r + λ(S)(1-R)`. At `α = 0` the spread is the
    /// constant `λ0 (1-R)`.
    ///
    /// # Errors
    ///
    /// Fails on a degenerate grid, a non-positive reference spot, a negative
    /// intensity, a recovery outside `[0, 1]`, or a non-finite exponent.
    pub fn with_hazard(
        process: Shared<GeneralizedBlackScholesProcess>,
        x_grid: Size,
        time_steps: Size,
        lambda0: Real,
        recovery: Real,
        alpha: Real,
        reference_spot: Real,
    ) -> QlResult<Self> {
        require!(
            reference_spot > 0.0,
            "hazard reference spot must be positive"
        );
        require!(lambda0 >= 0.0, "hazard intensity must be non-negative");
        require!(
            (0.0..=1.0).contains(&recovery),
            "recovery must lie in [0, 1], {recovery} not allowed"
        );
        require!(alpha.is_finite(), "hazard exponent must be finite");
        Self::build(
            process,
            x_grid,
            time_steps,
            Credit::Hazard {
                lambda0,
                recovery,
                alpha,
                reference_spot,
            },
        )
    }

    fn build(
        process: Shared<GeneralizedBlackScholesProcess>,
        x_grid: Size,
        time_steps: Size,
        credit: Credit,
    ) -> QlResult<Self> {
        require!(
            x_grid >= 3,
            "the convertible grid needs at least three points, {x_grid} given"
        );
        require!(
            time_steps > 0,
            "time steps must be positive, {time_steps} not allowed"
        );
        let base = GenericEngine::new(ConvertibleBondArguments::default(), BondResults::default());
        base.register_with(process.observable());
        if let Credit::Spread(spread) = &credit {
            spread.register_observer(&base.observer());
        }
        Ok(Self {
            base,
            process,
            x_grid,
            time_steps,
            credit,
            solved: None,
        })
    }

    /// Total value `U` at solver time `time` and spot `spot`.
    ///
    /// # Errors
    ///
    /// Fails when [`calculate`](PricingEngine::calculate) has not run, or when
    /// `spot` lies outside the mesh.
    pub fn value_at(&self, time: Time, spot: Real) -> QlResult<Real> {
        let grid = self.grid()?;
        grid.value_at(time, spot)
    }

    /// `∂U/∂S` at the pricing spot.
    pub fn delta(&self) -> QlResult<Real> {
        Ok(self.spot_greeks()?.1)
    }

    /// `∂²U/∂S²` at the pricing spot.
    pub fn gamma(&self) -> QlResult<Real> {
        Ok(self.spot_greeks()?.2)
    }

    /// `∂U/∂t` at the pricing spot, the secant over [`theta_step`](Self::theta_step).
    pub fn theta(&self) -> QlResult<Real> {
        let grid = self.grid()?;
        let step = grid.theta_step()?;
        let now = grid.value_at(0.0, grid.spot)?;
        let later = grid.value_at(step, grid.spot)?;
        Ok((later - now) / step)
    }

    /// Solver time of the first layer after valuation. [`theta`](Self::theta)
    /// is the secant across this step.
    pub fn theta_step(&self) -> QlResult<Time> {
        self.grid()?.theta_step()
    }

    /// No-default wealth change for a move to `spot_prime` over `dt`.
    ///
    /// `ΔΠ = [U(S', t+Δt) - U(S, t)] - Δ (S' - S) + coupon + (r_rebate - q) Δ S Δt - r_fund U Δt`.
    /// [`ConvertibleCashGamma::continuation`] is `½ Γ (ΔS)² + Θ_B Δt` with
    /// `Θ_B = Θ - r_fund U + (r_rebate - q) S Δ`. There is no default-jump term.
    ///
    /// # Errors
    ///
    /// Fails when the grid is missing, `dt` is not inside `(0, maturity]`, or
    /// either spot is outside the mesh.
    pub fn cash_gamma(
        &self,
        dt: Time,
        spot_prime: Real,
        r_rebate: Real,
        r_fund: Real,
    ) -> QlResult<ConvertibleCashGamma> {
        let grid = self.grid()?;
        require!(dt > 0.0, "cash-gamma interval must be positive");
        require!(
            dt <= grid.maturity,
            "cash-gamma interval {dt} extends past maturity {}",
            grid.maturity
        );
        let spot = grid.spot;
        let (value, delta, gamma) = grid.greeks_at(0.0, spot)?;
        let later = grid.value_at(dt, spot_prime)?;
        let theta = self.theta()?;
        let coupon = grid.coupon_between(0.0, dt);
        let move_s = spot_prime - spot;
        let carry = (r_rebate - grid.dividend_yield) * delta * spot * dt;
        let wealth = (later - value) - delta * move_s + coupon + carry - r_fund * value * dt;
        let theta_carry = theta - r_fund * value + (r_rebate - grid.dividend_yield) * spot * delta;
        let continuation = 0.5 * gamma * move_s * move_s + theta_carry * dt;
        Ok(ConvertibleCashGamma {
            wealth,
            continuation,
            coupon,
        })
    }

    fn grid(&self) -> QlResult<&SolvedGrid> {
        match &self.solved {
            Some(grid) => Ok(grid),
            None => Err(crate::errors::QlError::new(
                "convertible finite-difference grid is empty; calculate first",
                file!(),
                line!(),
            )),
        }
    }

    fn spot_greeks(&self) -> QlResult<(Real, Real, Real)> {
        let grid = self.grid()?;
        grid.greeks_at(0.0, grid.spot)
    }

    fn cash_spread(&self, spot: Real, quoted: Real) -> Real {
        match &self.credit {
            Credit::Spread(_) => quoted,
            Credit::Hazard {
                lambda0,
                recovery,
                alpha,
                reference_spot,
            } => lambda0 * (spot / reference_spot).powf(-*alpha) * (1.0 - recovery),
        }
    }
}

impl AsObservable for FdConvertibleEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdConvertibleEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
        self.solved = None;
    }

    fn calculate(&mut self) -> QlResult<()> {
        self.solved = None;
        let frozen = freeze(self.base.arguments(), &self.process)?;
        reject_unsupported(self.base.arguments())?;
        let schedule = Schedule::build(self.base.arguments(), &frozen)?;
        let quoted = match &self.credit {
            Credit::Spread(handle) => handle.current_link()?.value()?,
            Credit::Hazard { .. } => 0.0,
        };
        require!(quoted.is_finite(), "credit spread must be finite");

        let solved = solve(
            self,
            &frozen,
            &schedule,
            quoted,
            self.x_grid,
            self.time_steps,
        )?;
        let value = solved.value_at(0.0, solved.spot)?;
        require!(
            value.is_finite(),
            "floating-point overflow on the convertible grid"
        );

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.settlement_value = Some(value);
        self.solved = Some(solved);
        Ok(())
    }
}

fn reject_unsupported(arguments: &ConvertibleBondArguments) -> QlResult<()> {
    if arguments.callability_triggers.iter().any(Option::is_some) {
        fail!("soft calls are not supported by the finite-difference convertible engine");
    }
    for flow in &arguments.cashflows {
        if flow
            .as_coupon()
            .is_some_and(|coupon| coupon.fixing_date().is_some())
        {
            fail!("floating coupons are not supported by the finite-difference convertible engine");
        }
    }
    Ok(())
}

fn freeze(
    arguments: &ConvertibleBondArguments,
    process: &GeneralizedBlackScholesProcess,
) -> QlResult<Frozen> {
    arguments.validate()?;
    let exercise = arguments.exercise.as_ref().expect("validated exercise");
    let maturity_date = exercise.last_date();
    let settlement = arguments
        .settlement_date
        .expect("validated settlement date");

    let risk_free = process.risk_free_rate().current_link()?;
    let dividend_ts = process.dividend_yield().current_link()?;
    let vol_ts = process.black_volatility().current_link()?;
    let day_counter = risk_free.require_day_counter()?;
    let dividend_counter = dividend_ts.require_day_counter()?;

    let spot = process.x0()?;
    require!(spot > 0.0, "negative or null underlying");
    let vol = vol_ts.black_vol_date(maturity_date, spot, true)?;
    let rate = risk_free
        .zero_rate_date(
            maturity_date,
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::NoFrequency,
            true,
        )?
        .rate();
    let dividend = dividend_ts
        .zero_rate_date(
            maturity_date,
            dividend_counter,
            Compounding::Continuous,
            Frequency::NoFrequency,
            true,
        )?
        .rate();
    let maturity = day_counter.year_fraction(settlement, maturity_date);
    require!(
        maturity > 0.0,
        "the convertible engine needs a positive maturity"
    );

    Ok(Frozen {
        spot,
        rate,
        dividend,
        vol,
        maturity,
        settlement,
        day_counter,
        spot_quote: process.state_variable(),
    })
}

impl Schedule {
    fn build(arguments: &ConvertibleBondArguments, frozen: &Frozen) -> QlResult<Self> {
        let exercise = arguments.exercise.as_ref().expect("validated exercise");
        let mut exercise_times = Vec::with_capacity(exercise.dates().len());
        for &date in exercise.dates() {
            exercise_times.push(frozen.day_counter.year_fraction(frozen.settlement, date));
        }
        match exercise.exercise_type() {
            ExerciseType::American => {
                require!(
                    exercise_times.len() >= 2,
                    "American exercise needs an earliest and a latest date"
                );
            }
            ExerciseType::European | ExerciseType::Bermudan => {
                require!(!exercise_times.is_empty(), "exercise has no dates");
            }
        }

        let mut calls = Vec::new();
        let mut puts = Vec::new();
        for i in 0..arguments.callability_dates.len() {
            let time = frozen
                .day_counter
                .year_fraction(frozen.settlement, arguments.callability_dates[i]);
            match arguments.callability_types[i] {
                CallabilityType::Call => calls.push((time, arguments.callability_prices[i])),
                CallabilityType::Put => puts.push((time, arguments.callability_prices[i])),
            }
        }

        let mut coupons = Vec::new();
        if arguments.cashflows.len() > 1 {
            for flow in &arguments.cashflows[..arguments.cashflows.len() - 1] {
                if flow.date() > frozen.settlement {
                    coupons.push((
                        frozen
                            .day_counter
                            .year_fraction(frozen.settlement, flow.date()),
                        flow.amount()?,
                    ));
                }
            }
        }

        Ok(Self {
            exercise: exercise.exercise_type(),
            exercise_times,
            ratio: arguments.conversion_ratio,
            redemption: arguments.redemption,
            calls,
            puts,
            coupons,
        })
    }

    fn conversion_alive(&self, time: Time) -> bool {
        match self.exercise {
            ExerciseType::American => {
                let start = self.exercise_times[0].max(0.0);
                let end = self.exercise_times[1];
                time >= start && time <= end
            }
            ExerciseType::European => same_time(time, self.exercise_times[0]),
            ExerciseType::Bermudan => self
                .exercise_times
                .iter()
                .any(|&exercise| same_time(time, exercise)),
        }
    }

    fn mandatory_times(&self, maturity: Time) -> Vec<Time> {
        let mut events = Vec::new();
        events.extend(self.calls.iter().map(|(time, _)| *time));
        events.extend(self.puts.iter().map(|(time, _)| *time));
        events.extend(self.coupons.iter().map(|(time, _)| *time));
        match self.exercise {
            ExerciseType::American => {
                events.push(self.exercise_times[0].max(0.0));
                events.push(self.exercise_times[1]);
            }
            ExerciseType::European | ExerciseType::Bermudan => {
                events.extend_from_slice(&self.exercise_times);
            }
        }
        events.retain(|time| *time > 0.0 && *time < maturity && time.is_finite());
        events
    }
}

fn same_time(left: Time, right: Time) -> bool {
    (left - right).abs() <= TIME_MATCH || close_enough(left, right)
}

fn time_layers(maturity: Time, steps: Size, events: &[Time]) -> Vec<Time> {
    let mut times: Vec<Time> = (0..=steps)
        .map(|i| maturity * i as Real / steps as Real)
        .collect();
    for &event in events {
        if times.iter().any(|&time| same_time(time, event)) {
            continue;
        }
        times.push(event);
    }
    times.sort_by(Real::total_cmp);
    let mut merged = Vec::with_capacity(times.len());
    for time in times {
        if merged
            .last()
            .is_none_or(|previous| !same_time(*previous, time))
        {
            merged.push(time);
        }
    }
    merged
}

fn straight_bond(
    time: Time,
    rate: Real,
    spread: Real,
    coupons: &[(Time, Real)],
    redemption: Real,
    maturity: Time,
) -> Real {
    let yield_rate = rate + spread;
    let mut value = redemption * (-yield_rate * (maturity - time)).exp();
    for &(coupon_time, amount) in coupons {
        if coupon_time > time && !same_time(coupon_time, time) {
            value += amount * (-yield_rate * (coupon_time - time)).exp();
        }
    }
    value
}

fn coupon_at(coupons: &[(Time, Real)], time: Time) -> Real {
    coupons
        .iter()
        .filter(|(coupon_time, _)| same_time(*coupon_time, time))
        .map(|(_, amount)| *amount)
        .sum()
}

fn project(schedule: &Schedule, spots: &[Real], time: Time, equity: &mut Array, cash: &mut Array) {
    let alive = schedule.conversion_alive(time);
    for &(call_time, price) in &schedule.calls {
        if !same_time(call_time, time) {
            continue;
        }
        for i in 0..spots.len() {
            let total = equity[i] + cash[i];
            let conversion = schedule.ratio * spots[i];
            if alive {
                let hold = price.max(conversion);
                if total > hold {
                    if conversion >= price {
                        equity[i] = conversion;
                        cash[i] = 0.0;
                    } else {
                        equity[i] = 0.0;
                        cash[i] = price;
                    }
                }
            } else if total > price {
                equity[i] = 0.0;
                cash[i] = price;
            }
        }
    }
    for &(put_time, price) in &schedule.puts {
        if !same_time(put_time, time) {
            continue;
        }
        for i in 0..spots.len() {
            if price > equity[i] + cash[i] {
                equity[i] = 0.0;
                cash[i] = price;
            }
        }
    }
    let coupon = coupon_at(&schedule.coupons, time);
    if coupon != 0.0 {
        for node in cash.iter_mut() {
            *node += coupon;
        }
    }
    if alive {
        for i in 0..spots.len() {
            let conversion = schedule.ratio * spots[i];
            if equity[i] + cash[i] <= conversion {
                equity[i] = conversion;
                cash[i] = 0.0;
            }
        }
    }
}

struct LowerBoundary {
    rate: Real,
    spread: Real,
    maturity: Time,
}

fn enforce_boundaries(
    schedule: &Schedule,
    spots: &[Real],
    time: Time,
    lower: &LowerBoundary,
    equity: &mut Array,
    cash: &mut Array,
) {
    let last = spots.len() - 1;
    equity[0] = 0.0;
    cash[0] = straight_bond(
        time,
        lower.rate,
        lower.spread,
        &schedule.coupons,
        schedule.redemption,
        lower.maturity,
    ) + coupon_at(&schedule.coupons, time);
    equity[last] = schedule.ratio * spots[last];
    cash[last] = 0.0;
}

fn flat_curve(
    reference: Date,
    rate: Real,
    day_counter: DayCounter,
) -> Handle<dyn YieldTermStructure> {
    Handle::new(
        shared(crate::termstructures::yields::FlatForward::with_rate(
            reference,
            rate,
            day_counter,
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>,
    )
}

fn concentration_point(redemption: Real, ratio: Real) -> Option<(Real, Real)> {
    let strike = redemption / ratio;
    if strike.is_finite() && strike > 0.0 && strike < 1.0e6 {
        Some((strike, CONCENTRATION))
    } else {
        None
    }
}

fn solve(
    engine: &FdConvertibleEngine,
    frozen: &Frozen,
    schedule: &Schedule,
    quoted_spread: Real,
    x_grid: Size,
    time_steps: Size,
) -> QlResult<SolvedGrid> {
    let equity_curve = flat_curve(frozen.settlement, frozen.rate, frozen.day_counter.clone());
    let dividend_curve = flat_curve(
        frozen.settlement,
        frozen.dividend,
        frozen.day_counter.clone(),
    );
    let equity_process = process_helper(
        frozen.spot_quote.clone(),
        equity_curve,
        dividend_curve,
        frozen.vol,
    )?;
    let mesher_1d = fdm_black_scholes_mesher(
        x_grid,
        &equity_process,
        frozen.maturity,
        frozen.spot,
        None,
        None,
        MESHER_EPS,
        MESHER_SCALE,
        concentration_point(schedule.redemption, schedule.ratio),
        &[],
        0.0,
    )?;
    let mesher = shared(FdmMesherComposite::new(vec![mesher_1d]));
    let mesher_dyn = Shared::clone(&mesher) as Shared<dyn FdmMesher>;
    let log_spots: Vec<Real> = mesher.locations(0).iter().copied().collect();
    let spots: Vec<Real> = log_spots.iter().copied().map(Real::exp).collect();

    let equity_op = shared_mut(FdmBlackScholesOp::new(
        Shared::clone(&mesher_dyn),
        &equity_process,
        frozen.spot,
        0,
    )?) as SharedMut<dyn FdmLinearOpComposite>;

    let cash_op = match &engine.credit {
        Credit::Spread(_) => {
            let cash_process = process_helper(
                frozen.spot_quote.clone(),
                flat_curve(
                    frozen.settlement,
                    frozen.rate + quoted_spread,
                    frozen.day_counter.clone(),
                ),
                flat_curve(
                    frozen.settlement,
                    frozen.dividend + quoted_spread,
                    frozen.day_counter.clone(),
                ),
                frozen.vol,
            )?;
            shared_mut(FdmBlackScholesOp::new(
                Shared::clone(&mesher_dyn),
                &cash_process,
                frozen.spot,
                0,
            )?) as SharedMut<dyn FdmLinearOpComposite>
        }
        Credit::Hazard { .. } => {
            let mut extra = Array::with_size(spots.len());
            for (i, &spot) in spots.iter().enumerate() {
                extra[i] = engine.cash_spread(spot, quoted_spread);
            }
            shared_mut(
                FdmBlackScholesOp::new(
                    Shared::clone(&mesher_dyn),
                    &equity_process,
                    frozen.spot,
                    0,
                )?
                .with_extra_discount(extra)?,
            ) as SharedMut<dyn FdmLinearOpComposite>
        }
    };

    let spread_low = engine.cash_spread(spots[0], quoted_spread);
    let ratio = schedule.ratio;
    let upper_spot = spots[spots.len() - 1];
    let coupons = schedule.coupons.clone();
    let redemption = schedule.redemption;
    let maturity = frozen.maturity;
    let rate = frozen.rate;
    let equity_bc: Vec<Shared<dyn crate::methods::finitedifferences::BoundaryCondition>> = vec![
        shared(DirichletBoundary::new(BoundarySide::Lower, 0.0))
            as Shared<dyn crate::methods::finitedifferences::BoundaryCondition>,
        shared(DirichletBoundary::new(
            BoundarySide::Upper,
            ratio * upper_spot,
        )) as Shared<dyn crate::methods::finitedifferences::BoundaryCondition>,
    ];
    let cash_bc: Vec<Shared<dyn crate::methods::finitedifferences::BoundaryCondition>> = vec![
        TimeDependentDirichletBoundary::new(BoundarySide::Lower, move |time| {
            straight_bond(time, rate, spread_low, &coupons, redemption, maturity)
        }) as Shared<dyn crate::methods::finitedifferences::BoundaryCondition>,
        shared(DirichletBoundary::new(BoundarySide::Upper, 0.0))
            as Shared<dyn crate::methods::finitedifferences::BoundaryCondition>,
    ];

    let mut equity_solver =
        FdmBackwardSolver::new(equity_op, equity_bc, None, FdmSchemeDesc::douglas());
    let mut cash_solver = FdmBackwardSolver::new(cash_op, cash_bc, None, FdmSchemeDesc::douglas());

    let mut equity = Array::filled(spots.len(), 0.0);
    let mut cash = Array::filled(spots.len(), schedule.redemption);
    let lower = LowerBoundary {
        rate: frozen.rate,
        spread: spread_low,
        maturity,
    };
    project(schedule, &spots, maturity, &mut equity, &mut cash);
    enforce_boundaries(schedule, &spots, maturity, &lower, &mut equity, &mut cash);

    let mut layers = vec![(maturity, total_value(&equity, &cash))];
    let times = time_layers(maturity, time_steps, &schedule.mandatory_times(maturity));
    for window in times.windows(2).rev() {
        let from = window[1];
        let to = window[0];
        if same_time(from, to) {
            continue;
        }
        equity_solver.rollback(&mut equity, from, to, 1, 0)?;
        cash_solver.rollback(&mut cash, from, to, 1, 0)?;
        project(schedule, &spots, to, &mut equity, &mut cash);
        enforce_boundaries(schedule, &spots, to, &lower, &mut equity, &mut cash);
        layers.push((to, total_value(&equity, &cash)));
    }
    layers.sort_by(|left, right| left.0.total_cmp(&right.0));

    Ok(SolvedGrid {
        log_spots,
        layers,
        coupons: schedule.coupons.clone(),
        spot: frozen.spot,
        dividend_yield: frozen.dividend,
        maturity,
    })
}

fn total_value(equity: &Array, cash: &Array) -> Vec<Real> {
    equity
        .iter()
        .zip(cash.iter())
        .map(|(equity_i, cash_i)| equity_i + cash_i)
        .collect()
}

impl SolvedGrid {
    fn theta_step(&self) -> QlResult<Time> {
        require!(
            self.layers.len() >= 2,
            "the convertible grid has no time step"
        );
        let step = self.layers[1].0 - self.layers[0].0;
        require!(step > 0.0, "the convertible grid has a zero time step");
        Ok(step)
    }

    fn value_at(&self, time: Time, spot: Real) -> QlResult<Real> {
        require!(spot > 0.0, "spot must be positive");
        require!(!self.layers.is_empty(), "the convertible grid is empty");
        if time <= self.layers[0].0 || same_time(time, self.layers[0].0) {
            return interpolate(&self.log_spots, &self.layers[0].1, spot.ln());
        }
        let last = self.layers.len() - 1;
        if time >= self.layers[last].0 || same_time(time, self.layers[last].0) {
            return interpolate(&self.log_spots, &self.layers[last].1, spot.ln());
        }
        let upper = self.layers.partition_point(|(layer, _)| *layer < time);
        if same_time(self.layers[upper].0, time) {
            return interpolate(&self.log_spots, &self.layers[upper].1, spot.ln());
        }
        let (earlier_time, earlier) = &self.layers[upper - 1];
        let (later_time, later) = &self.layers[upper];
        let weight = (time - earlier_time) / (later_time - earlier_time);
        let left = interpolate(&self.log_spots, earlier, spot.ln())?;
        let right = interpolate(&self.log_spots, later, spot.ln())?;
        Ok(left + weight * (right - left))
    }

    fn greeks_at(&self, time: Time, spot: Real) -> QlResult<(Real, Real, Real)> {
        require!(spot > 0.0, "spot must be positive");
        let values = self.layer_near(time)?;
        let spline = CubicNaturalSpline::new(self.log_spots.clone(), values)?;
        let x = spot.ln();
        let value = spline.value(x)?;
        let d_log = spline.derivative(x)?;
        let d_log_2 = spline.second_derivative(x)?;
        let delta = d_log / spot;
        let gamma = (d_log_2 - d_log) / (spot * spot);
        Ok((value, delta, gamma))
    }

    fn layer_near(&self, time: Time) -> QlResult<Vec<Real>> {
        require!(!self.layers.is_empty(), "the convertible grid is empty");
        let mut best = 0usize;
        for (i, (layer, _)) in self.layers.iter().enumerate() {
            if (layer - time).abs() < (self.layers[best].0 - time).abs() {
                best = i;
            }
        }
        Ok(self.layers[best].1.clone())
    }

    fn coupon_between(&self, start: Time, end: Time) -> Real {
        self.coupons
            .iter()
            .filter(|(time, _)| *time > start && (*time < end || same_time(*time, end)))
            .map(|(_, amount)| *amount)
            .sum()
    }
}

fn interpolate(log_spots: &[Real], values: &[Real], log_spot: Real) -> QlResult<Real> {
    let spline = CubicNaturalSpline::new(log_spots.to_vec(), values.to_vec())?;
    spline.value(log_spot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::{
        ConvertibleFixedCouponBond, ConvertibleZeroCouponBond, FixedRateBond, PlainVanillaPayoff,
        StrikedTypePayoff, VanillaOption, ZeroCouponBond,
    };
    use crate::option::OptionType;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::bond::DiscountingBondEngine;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::volatility::BlackVolTermStructure;
    use crate::termstructures::yields::{FlatForward, ZeroSpreadedTermStructure};
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;

    struct Market {
        settings: Shared<Settings<Date>>,
        issue_date: Date,
        maturity_date: Date,
        process: Shared<BlackScholesMertonProcess>,
        credit_spread: Handle<dyn Quote>,
        risk_free: Handle<dyn YieldTermStructure>,
        conversion_ratio: Real,
        redemption: Real,
        settlement_days: Size,
        spot: Real,
    }

    fn market() -> Market {
        let today = Date::new(15, Month::January, 2020);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let calendar = Target::new();
        let day_counter = Actual360::new();
        let settlement_days = 3usize;
        let fixing_days = 2i32;
        let mut issue_date = calendar.advance(
            today,
            fixing_days,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let maturity_date = calendar.advance(
            issue_date,
            10,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );
        issue_date = calendar.advance(
            maturity_date,
            -10,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );

        let spot = 50.0;
        let spot_quote = shared(SimpleQuote::new(spot));
        let dividend_quote = shared(SimpleQuote::new(0.02));
        let rate_quote = shared(SimpleQuote::new(0.05));
        let vol_quote = shared(SimpleQuote::new(0.15));
        let quote_handle =
            |quote: &Shared<SimpleQuote>| Handle::new(Shared::clone(quote) as Shared<dyn Quote>);
        let flat = |quote: &Shared<SimpleQuote>| {
            Handle::new(shared(FlatForward::moving(
                0,
                NullCalendar::new(),
                quote_handle(quote),
                day_counter.clone(),
                Compounding::Continuous,
                Frequency::Annual,
                Shared::clone(&settings),
            )) as Shared<dyn YieldTermStructure>)
        };
        let risk_free = flat(&rate_quote);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot_quote),
            flat(&dividend_quote),
            risk_free.clone(),
            Handle::new(shared(BlackConstantVol::moving_with_quote(
                0,
                NullCalendar::new(),
                quote_handle(&vol_quote),
                day_counter,
                Shared::clone(&settings),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        Market {
            settings,
            issue_date,
            maturity_date,
            process,
            credit_spread: Handle::new(shared(SimpleQuote::new(0.005)) as Shared<dyn Quote>),
            risk_free,
            conversion_ratio: 100.0 / spot,
            redemption: 100.0,
            settlement_days,
            spot,
        }
    }

    fn discount_curve(market: &Market) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(ZeroSpreadedTermStructure::new(
            market.risk_free.clone(),
            market.credit_spread.clone(),
        )) as Shared<dyn YieldTermStructure>)
    }

    fn price_convertible(
        market: &Market,
        exercise: Shared<dyn Exercise>,
        ratio: Real,
        coupons: Vec<Real>,
        frequency: Frequency,
        settlement_days: u32,
        mesh: (Size, Size, Handle<dyn Quote>),
    ) -> QlResult<(Real, SharedMut<FdConvertibleEngine>)> {
        let (x_grid, time_steps, spread) = mesh;
        let engine = shared_mut(FdConvertibleEngine::new(
            Shared::clone(&market.process),
            x_grid,
            time_steps,
            spread,
        )?);
        let schedule = MakeSchedule::new()
            .from(market.issue_date)
            .to(market.maturity_date)
            .with_frequency(frequency)
            .with_calendar(Target::new())
            .backwards()
            .build();
        if coupons.is_empty() {
            let mut bond = ConvertibleZeroCouponBond::new(
                exercise,
                ratio,
                Vec::new(),
                market.issue_date,
                settlement_days,
                Actual360::new(),
                schedule,
                market.redemption,
                Shared::clone(&market.settings),
            )?;
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
            Ok((bond.npv()?, engine))
        } else {
            let mut bond = ConvertibleFixedCouponBond::new(
                exercise,
                ratio,
                Vec::new(),
                market.issue_date,
                settlement_days,
                coupons,
                Actual360::new(),
                schedule,
                market.redemption,
                Shared::clone(&market.settings),
            )?;
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
            Ok((bond.npv()?, engine))
        }
    }

    #[test]
    fn zero_spread_european_zero_matches_discounted_redemption_plus_call() {
        let market = market();
        let settlement_days = 0u32;
        let spread = Handle::new(shared(SimpleQuote::new(0.0)) as Shared<dyn Quote>);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.maturity_date));
        let (calculated, _) = price_convertible(
            &market,
            Shared::clone(&exercise),
            market.conversion_ratio,
            Vec::new(),
            Frequency::Once,
            settlement_days,
            (201, 400, spread),
        )
        .unwrap();

        let strike = market.redemption / market.conversion_ratio;
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, strike));
        let mut option = VanillaOption::new(
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&market.settings),
        );
        option
            .base_mut()
            .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(Shared::clone(
                &market.process,
            ))) as SharedMut<dyn PricingEngine>);
        let discount = market
            .risk_free
            .current_link()
            .unwrap()
            .discount_date(market.maturity_date, false)
            .unwrap();
        let expected =
            market.redemption * discount + market.conversion_ratio * option.npv().unwrap();
        let tolerance = 5.0e-2;
        assert!(
            (calculated - expected).abs() < tolerance,
            "zero convertible {calculated} vs discounted redemption + call {expected}"
        );
    }

    #[test]
    fn vanishing_conversion_ratio_matches_the_straight_bond() {
        let market = market();
        let ratio = 1.0e-16;
        let european: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.maturity_date));
        let american: Shared<dyn Exercise> =
            shared(AmericanExercise::new(market.issue_date, market.maturity_date, false).unwrap());

        let (eu_zero, _) = price_convertible(
            &market,
            Shared::clone(&european),
            ratio,
            Vec::new(),
            Frequency::Once,
            market.settlement_days as u32,
            (101, 120, market.credit_spread.clone()),
        )
        .unwrap();
        let (am_zero, _) = price_convertible(
            &market,
            Shared::clone(&american),
            ratio,
            Vec::new(),
            Frequency::Once,
            market.settlement_days as u32,
            (101, 120, market.credit_spread.clone()),
        )
        .unwrap();

        let mut zero = ZeroCouponBond::new(
            market.settlement_days as u32,
            Target::new(),
            100.0,
            market.maturity_date,
            BusinessDayConvention::Following,
            market.redemption,
            Some(market.issue_date),
            Shared::clone(&market.settings),
        )
        .unwrap();
        let bond_engine = shared_mut(DiscountingBondEngine::new(
            discount_curve(&market),
            None,
            Shared::clone(&market.settings),
        ));
        zero.bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&bond_engine) as SharedMut<dyn PricingEngine>);
        let expected_zero = zero.bond_mut().settlement_value().unwrap();
        assert!(
            (eu_zero - expected_zero).abs() < 1.0e-2,
            "eu zero convertible {eu_zero} vs vanilla {expected_zero}"
        );
        assert!(
            (am_zero - expected_zero).abs() < 1.0e-2,
            "am zero convertible {am_zero} vs vanilla {expected_zero}"
        );

        let (eu_fixed, _) = price_convertible(
            &market,
            Shared::clone(&european),
            ratio,
            vec![0.05],
            Frequency::Annual,
            market.settlement_days as u32,
            (101, 160, market.credit_spread.clone()),
        )
        .unwrap();
        let (am_fixed, _) = price_convertible(
            &market,
            american,
            ratio,
            vec![0.05],
            Frequency::Annual,
            market.settlement_days as u32,
            (101, 160, market.credit_spread.clone()),
        )
        .unwrap();
        let schedule = MakeSchedule::new()
            .from(market.issue_date)
            .to(market.maturity_date)
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut fixed = FixedRateBond::new(
            market.settlement_days as u32,
            100.0,
            schedule,
            vec![0.05],
            Actual360::new(),
            BusinessDayConvention::Following,
            market.redemption,
            Some(market.issue_date),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&market.settings),
        )
        .unwrap();
        fixed
            .bond_mut()
            .base_mut()
            .set_pricing_engine(bond_engine as SharedMut<dyn PricingEngine>);
        let expected_fixed = fixed.bond_mut().settlement_value().unwrap();
        assert!(
            (eu_fixed - expected_fixed).abs() < 2.0e-2,
            "eu fixed convertible {eu_fixed} vs vanilla {expected_fixed}"
        );
        assert!(
            (am_fixed - expected_fixed).abs() < 2.0e-2,
            "am fixed convertible {am_fixed} vs vanilla {expected_fixed}"
        );
    }

    #[test]
    fn at_the_money_fixed_coupon_exceeds_the_straight_bond() {
        let market = market();
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.maturity_date));
        let (convertible, _) = price_convertible(
            &market,
            exercise,
            market.conversion_ratio,
            vec![0.05],
            Frequency::Annual,
            market.settlement_days as u32,
            (121, 160, market.credit_spread.clone()),
        )
        .unwrap();
        let schedule = MakeSchedule::new()
            .from(market.issue_date)
            .to(market.maturity_date)
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut straight = FixedRateBond::new(
            market.settlement_days as u32,
            100.0,
            schedule,
            vec![0.05],
            Actual360::new(),
            BusinessDayConvention::Following,
            market.redemption,
            Some(market.issue_date),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&market.settings),
        )
        .unwrap();
        straight
            .bond_mut()
            .base_mut()
            .set_pricing_engine(shared_mut(DiscountingBondEngine::new(
                discount_curve(&market),
                None,
                Shared::clone(&market.settings),
            )) as SharedMut<dyn PricingEngine>);
        let bond = straight.bond_mut().settlement_value().unwrap();
        assert!(
            convertible > bond + 0.5,
            "ATM convertible {convertible} should exceed the straight bond {bond}"
        );
    }

    #[test]
    fn partial_conversion_mesh_refines_by_less_than_the_previous_step() {
        let market = market();
        let european = EuropeanExercise::new(market.maturity_date);
        let price_at = |x_grid, time_steps| {
            price_convertible(
                &market,
                shared(european) as Shared<dyn Exercise>,
                market.conversion_ratio,
                Vec::new(),
                Frequency::Once,
                0,
                (x_grid, time_steps, market.credit_spread.clone()),
            )
            .unwrap()
            .0
        };
        let coarse = price_at(51, 40);
        let middle = price_at(101, 80);
        let fine = price_at(201, 160);
        let coarse_step = (middle - coarse).abs();
        let fine_step = (fine - middle).abs();
        assert!(
            fine_step < coarse_step,
            "mesh steps {coarse_step} then {fine_step} (prices {coarse}, {middle}, {fine})"
        );
    }

    #[test]
    fn zero_alpha_hazard_matches_the_constant_spread_price() {
        let market = market();
        let recovery = 0.5;
        let lambda0 = 0.01;
        let spread = lambda0 * (1.0 - recovery);
        let spread_handle = Handle::new(shared(SimpleQuote::new(spread)) as Shared<dyn Quote>);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.maturity_date));
        let (constant, _) = price_convertible(
            &market,
            Shared::clone(&exercise),
            market.conversion_ratio,
            vec![0.05],
            Frequency::Annual,
            market.settlement_days as u32,
            (81, 80, spread_handle),
        )
        .unwrap();

        let engine = shared_mut(
            FdConvertibleEngine::with_hazard(
                Shared::clone(&market.process),
                81,
                80,
                lambda0,
                recovery,
                0.0,
                market.spot,
            )
            .unwrap(),
        );
        let schedule = MakeSchedule::new()
            .from(market.issue_date)
            .to(market.maturity_date)
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut bond = ConvertibleFixedCouponBond::new(
            exercise,
            market.conversion_ratio,
            Vec::new(),
            market.issue_date,
            market.settlement_days as u32,
            vec![0.05],
            Actual360::new(),
            schedule,
            market.redemption,
            Shared::clone(&market.settings),
        )
        .unwrap();
        bond.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        let hazard = bond.npv().unwrap();
        assert!(
            (hazard - constant).abs() < 1.0e-8,
            "α=0 hazard {hazard} vs constant spread {constant}"
        );
    }

    #[test]
    fn positive_alpha_is_cheaper_at_a_low_spot_and_the_mesh_tightens() {
        let market = market();
        let recovery = 0.5;
        let lambda0 = 0.01;
        let low_spot = 20.0;
        let price_at = |x_grid, time_steps, alpha| {
            let engine = shared_mut(
                FdConvertibleEngine::with_hazard(
                    Shared::clone(&market.process),
                    x_grid,
                    time_steps,
                    lambda0,
                    recovery,
                    alpha,
                    market.spot,
                )
                .unwrap(),
            );
            let exercise: Shared<dyn Exercise> =
                shared(EuropeanExercise::new(market.maturity_date));
            let mut bond = ConvertibleZeroCouponBond::new(
                exercise,
                market.conversion_ratio,
                Vec::new(),
                market.issue_date,
                0,
                Actual360::new(),
                MakeSchedule::new()
                    .from(market.issue_date)
                    .to(market.maturity_date)
                    .with_frequency(Frequency::Once)
                    .with_calendar(Target::new())
                    .backwards()
                    .build(),
                market.redemption,
                Shared::clone(&market.settings),
            )
            .unwrap();
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
            bond.npv().unwrap();
            engine.borrow().value_at(0.0, low_spot).unwrap()
        };

        let flat = price_at(101, 80, 0.0);
        let sloped = price_at(101, 80, 1.0);
        assert!(
            sloped < flat,
            "hazard price {sloped} at spot {low_spot} should be below the α=0 price {flat}"
        );

        let coarse = price_at(51, 40, 1.0);
        let middle = price_at(101, 80, 1.0);
        let fine = price_at(201, 160, 1.0);
        let coarse_step = (middle - coarse).abs();
        let fine_step = (fine - middle).abs();
        assert!(
            fine_step < coarse_step,
            "hazard mesh steps {coarse_step} then {fine_step} (prices {coarse}, {middle}, {fine})"
        );
    }

    #[test]
    fn cash_gamma_revaluation_matches_the_carry_adjusted_theta() {
        let market = market();
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.maturity_date));
        let (_, engine) = price_convertible(
            &market,
            exercise,
            market.conversion_ratio,
            Vec::new(),
            Frequency::Once,
            0,
            (161, 200, market.credit_spread.clone()),
        )
        .unwrap();
        let engine = engine.borrow();
        let dt = engine.theta_step().unwrap();
        let report = engine
            .cash_gamma(dt, market.spot + 0.05, 0.05, 0.04)
            .unwrap();
        let remainder = (report.wealth - report.continuation - report.coupon).abs();
        assert!(
            remainder < 1.0e-3,
            "cash-gamma remainder {remainder} (wealth {}, continuation {}, coupon {})",
            report.wealth,
            report.continuation,
            report.coupon
        );
    }
}
