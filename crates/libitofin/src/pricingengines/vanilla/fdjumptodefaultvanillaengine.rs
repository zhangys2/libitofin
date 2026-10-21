//! Finite-difference engine for European calls and puts on a Carr–Madan stock.
//!
//! The stock follows local volatility and jumps to zero with intensity
//! `λ(t, S) = f(t) S^p` (working-paper eq. 1). The price `U` solves
//!
//! `U_t + (r − q + λ) S U_S + ½ σ² S² U_SS − (r + λ) U + λ g = 0`.
//!
//! A call has `g = 0` and terminal `(S − K)+`. A European put pays `K` at
//! expiry after default, so `g(t) = K D(T) / D(t)`; the shift `U = P − g`
//! removes the source and leaves the terminal `−min(S, K)`. American exercise
//! needs the source term and is rejected. Convertibles are a different credit
//! convention and are not this engine.

use crate::errors::QlResult;
use crate::exercise::{EuropeanExercise, Exercise, ExerciseType};
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher_with_quanto,
};
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::solvers::{
    FdmBlackScholesSolver, FdmSchemeDesc, FdmSolverDesc,
};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::models::equity::jumptodefault::PowerHazard;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::LocalVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Real, Size, Time};
use crate::utilities::null::Null;

/// European call and put engine under Carr–Madan jump-to-default.
pub struct FdJumpToDefaultVanillaEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    /// Dupire (or other) local volatility. `None` uses the process Black vol
    /// for the generator; pass the process local vol to match
    /// [`FdBlackScholesVanillaEngine`](super::FdBlackScholesVanillaEngine)'s
    /// local-vol branch. The process Black vol is what sizes the mesh.
    local_vol: Option<Handle<dyn LocalVolTermStructure>>,
    hazard: Shared<PowerHazard>,
    t_grid: Size,
    x_grid: Size,
    scheme_desc: FdmSchemeDesc,
    illegal_local_vol_overwrite: Real,
}

impl FdJumpToDefaultVanillaEngine {
    /// European engine. `local_vol` is the diffusion coefficient `σ(S, t)`;
    /// `None` samples the process Black variance instead. `hazard` is
    /// `f(t) S^p`. The grid and scheme match
    /// [`FdBlackScholesVanillaEngine::with_params`](super::FdBlackScholesVanillaEngine::with_params)
    /// (no damping).
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        local_vol: Option<Handle<dyn LocalVolTermStructure>>,
        hazard: Shared<PowerHazard>,
        t_grid: Size,
        x_grid: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            local_vol,
            hazard,
            t_grid,
            x_grid,
            scheme_desc,
            illegal_local_vol_overwrite: -Real::null(),
        }
    }

    /// Replaces a failed local-vol sample, as on the Black–Scholes engine.
    /// A negative value, the default, lets the failure surface.
    pub fn with_illegal_local_vol_overwrite(mut self, overwrite: Real) -> Self {
        self.illegal_local_vol_overwrite = overwrite;
        self
    }

    /// Fills the arguments and returns the NPV.
    pub fn price(
        &mut self,
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
    ) -> QlResult<Real> {
        {
            let args = self.base.arguments_mut();
            args.payoff = Some(payoff);
            args.exercise = Some(exercise);
        }
        self.calculate()?;
        match self.base.results().instrument.value {
            Some(value) => Ok(value),
            None => fail!("no results returned"),
        }
    }

    /// Survival probability `V(T)` implied by this intensity and diffusion.
    ///
    /// The value of a claim that pays 1 at `T` if no default has occurred
    /// (`g = 0`, terminal value 1), divided by the risk-free discount `D(T)`.
    /// Working-paper eq. 2.
    ///
    /// # Errors
    ///
    /// Returns an error when `maturity` is non-finite or not strictly positive,
    /// or the discount is not strictly positive.
    pub fn survival_probability(&self, maturity: Time) -> QlResult<Real> {
        require!(
            maturity.is_finite() && maturity > 0.0,
            "survival maturity must be a positive finite time, got {maturity}"
        );
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        // European with no coupons ignores the exercise date; the rollback
        // length is `maturity`.
        let reference = self
            .process
            .risk_free_rate()
            .current_link()?
            .reference_date()?;
        let exercise = EuropeanExercise::new(reference);
        let solved = self.solve(spot, maturity, spot, shared(UnitPayoff), &exercise)?;
        let discount = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount(maturity, true)?;
        require!(
            discount > 0.0,
            "survival probability needs a positive discount, got {discount}"
        );
        Ok(solved.value / discount)
    }

    fn solve(
        &self,
        strike: Real,
        maturity: Time,
        spot: Real,
        terminal: Shared<dyn Payoff>,
        exercise: &dyn Exercise,
    ) -> QlResult<GridValue> {
        let equity = fdm_black_scholes_mesher_with_quanto(
            self.x_grid,
            &self.process,
            maturity,
            strike,
            None,
            None,
            0.0001,
            1.5,
            Some((strike, 0.1)),
            &[],
            None,
            0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;
        let calculator: Shared<dyn FdmInnerValueCalculator> =
            shared(fdm_log_inner_value(terminal, Shared::clone(&mesher_dyn), 0));

        let r_ts = self.process.risk_free_rate().current_link()?;
        let conditions = FdmStepConditionComposite::vanilla_composite(
            &[],
            exercise,
            Shared::clone(&mesher_dyn),
            Shared::clone(&calculator),
            r_ts.reference_date()?,
            r_ts.require_day_counter()?,
        )?;
        let solver_desc = FdmSolverDesc {
            mesher: Shared::clone(&mesher_dyn),
            bc_set: Vec::new(),
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: 0,
        };

        let wrapped;
        let process_ref: &GeneralizedBlackScholesProcess = if let Some(local_vol) = &self.local_vol
        {
            wrapped = GeneralizedBlackScholesProcess::with_local_vol(
                self.process.state_variable(),
                self.process.dividend_yield(),
                self.process.risk_free_rate(),
                self.process.black_volatility(),
                local_vol.clone(),
            );
            &wrapped
        } else {
            &self.process
        };
        let op = shared_mut(
            FdmBlackScholesOp::with_local_vol(
                mesher_dyn,
                process_ref,
                strike,
                self.local_vol.is_some(),
                self.illegal_local_vol_overwrite,
                0,
            )?
            .with_jump_to_default(Shared::clone(&self.hazard)),
        ) as SharedMut<dyn FdmLinearOpComposite>;
        let solver = FdmBlackScholesSolver::from_operator(solver_desc, self.scheme_desc, op);
        Ok(GridValue {
            value: solver.value_at(spot)?,
            delta: solver.delta_at(spot)?,
            gamma: solver.gamma_at(spot)?,
            theta: solver.theta_at(spot)?,
        })
    }
}

/// NPV and spot greeks of one rollback.
struct GridValue {
    value: Real,
    delta: Real,
    gamma: Real,
    theta: Real,
}

/// Terminal condition of the shifted put, `−min(S, K)`.
struct ShiftedPutTerminal {
    strike: Real,
}

impl Payoff for ShiftedPutTerminal {
    fn name(&self) -> String {
        "JumpToDefaultPut".to_string()
    }

    fn description(&self) -> String {
        format!("-min(S, {})", self.strike)
    }

    fn value(&self, price: Real) -> Real {
        -price.min(self.strike)
    }
}

/// Claim that pays one at expiry whether or not the spot moved.
struct UnitPayoff;

impl Payoff for UnitPayoff {
    fn name(&self) -> String {
        "Unit".to_string()
    }

    fn description(&self) -> String {
        "1".to_string()
    }

    fn value(&self, _price: Real) -> Real {
        1.0
    }
}

/// `K D(T) / D(t)` and its finite-difference theta, using the same step the
/// one-dimensional solver snapshots (`fdm1dimsolver`).
fn put_shift(
    yield_curve: &dyn YieldTermStructure,
    strike: Real,
    maturity: Time,
) -> QlResult<(Real, Real)> {
    let discount_t = yield_curve.discount(maturity, true)?;
    let discount_0 = yield_curve.discount(0.0, true)?;
    require!(
        discount_0 > 0.0,
        "put shift needs a positive discount at the reference, got {discount_0}"
    );
    let cash = strike * discount_t / discount_0;
    // European, no dividends: the first stopping time is the maturity.
    let theta_t = 0.99 * Real::min(1.0 / 365.0, maturity);
    let theta = if theta_t == 0.0 {
        0.0
    } else {
        let discount_theta = yield_curve.discount(theta_t, true)?;
        let cash_theta = cash * discount_0 / discount_theta;
        (cash_theta - cash) / theta_t
    };
    Ok((cash, theta))
}

impl AsObservable for FdJumpToDefaultVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdJumpToDefaultVanillaEngine {
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
        let Some(exercise) = arguments.exercise.as_ref() else {
            fail!("no exercise given");
        };
        let Some(payoff) = arguments.payoff.as_ref() else {
            fail!("no payoff given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "jump-to-default vanilla engine supports European exercise only; an American put needs a source term"
        );
        let strike = payoff.strike();
        require!(strike >= 0.0, "strike must be non-negative");
        let maturity = self.process.time(&exercise.last_date())?;
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

        let (terminal, shift): (Shared<dyn Payoff>, Option<(Real, Real)>) =
            match payoff.option_type() {
                OptionType::Call => (Shared::clone(payoff) as Shared<dyn Payoff>, None),
                OptionType::Put => {
                    let yield_curve = self.process.risk_free_rate().current_link()?;
                    (
                        shared(ShiftedPutTerminal { strike }) as Shared<dyn Payoff>,
                        Some(put_shift(&*yield_curve, strike, maturity)?),
                    )
                }
            };
        let solved = self.solve(strike, maturity, spot, terminal, &**exercise)?;
        let (cash, cash_theta) = shift.unwrap_or((0.0, 0.0));
        let results = self.base.results_mut();
        results.instrument.value = Some(solved.value + cash);
        results.greeks = Greeks {
            delta: Some(solved.delta),
            gamma: Some(solved.gamma),
            theta: Some(solved.theta + cash_theta),
            ..Greeks::default()
        };
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::exercise::{AmericanExercise, EuropeanExercise};
    use crate::instruments::PlainVanillaPayoff;
    use crate::interestrate::Compounding;
    use crate::patterns::observable::AsObservable;
    use crate::pricingengines::black_formula;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::termstructures::volatility::LocalVolTermStructure;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::{TermStructure, TermStructureBase};
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::date::{Date, Month};
    use crate::time::daycounter::DayCounter;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Rate, Volatility};

    const SPOT: Real = 100.0;
    const RATE: Rate = 0.05;
    const DIVIDEND: Rate = 0.01;
    const VOL: Volatility = 0.20;

    fn today() -> Date {
        Date::new(11, Month::February, 2018)
    }

    fn flat_rate(reference: Date, rate: Rate, dc: DayCounter) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference,
            rate,
            dc,
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn process(vol: Volatility) -> Shared<GeneralizedBlackScholesProcess> {
        let dc = Actual365Fixed::new();
        let reference = today();
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(SPOT)) as Shared<dyn Quote>),
            flat_rate(reference, DIVIDEND, dc.clone()),
            flat_rate(reference, RATE, dc.clone()),
            Handle::new(shared(BlackConstantVol::new(reference, None, vol, dc))
                as Shared<dyn BlackVolTermStructure>),
        ))
    }

    fn expiry(years: Real) -> Date {
        today() + Period::new((years * 365.0).round() as i32, TimeUnit::Days)
    }

    fn year_fraction(years: Real) -> Time {
        Actual365Fixed::new().year_fraction(today(), expiry(years))
    }

    /// Black–Scholes call with rate `r + λ` and dividend `q`.
    fn jumped_black_call(strike: Real, maturity: Time, lambda: Real) -> Real {
        let forward = SPOT * ((RATE + lambda - DIVIDEND) * maturity).exp();
        let discount = (-(RATE + lambda) * maturity).exp();
        black_formula(
            OptionType::Call,
            strike,
            forward,
            VOL * maturity.sqrt(),
            discount,
            0.0,
        )
        .unwrap()
    }

    fn engine(
        hazard: Shared<PowerHazard>,
        local_vol: Option<Handle<dyn LocalVolTermStructure>>,
        t_grid: Size,
        x_grid: Size,
    ) -> FdJumpToDefaultVanillaEngine {
        FdJumpToDefaultVanillaEngine::new(
            process(VOL),
            local_vol,
            hazard,
            t_grid,
            x_grid,
            FdmSchemeDesc::douglas(),
        )
    }

    fn price_vanilla(
        hazard: Shared<PowerHazard>,
        local_vol: Option<Handle<dyn LocalVolTermStructure>>,
        option_type: OptionType,
        strike: Real,
        maturity_years: Real,
        t_grid: Size,
        x_grid: Size,
    ) -> Real {
        let mut engine = engine(hazard, local_vol, t_grid, x_grid);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry(maturity_years)));
        engine.price(payoff, exercise).unwrap()
    }

    #[test]
    fn american_exercise_is_rejected() {
        let mut engine = engine(
            shared(PowerHazard::constant(0.0, 0.0).unwrap()),
            None,
            20,
            20,
        );
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, SPOT));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::over(today(), expiry(1.0)).unwrap());
        let error = engine.price(payoff, exercise).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("European"),
            "rejection should name the exercise limit, got {message}"
        );
    }

    #[test]
    fn constant_hazard_call_matches_black_scholes_with_rate_plus_intensity() {
        let lambda = 0.03;
        let hazard = shared(PowerHazard::constant(lambda, 0.0).unwrap());
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        for years in [0.25, 1.0, 3.0] {
            let maturity = year_fraction(years);
            for moneyness in [0.5, 0.75, 1.0, 1.25, 1.5] {
                let strike = SPOT * moneyness;
                let expected = jumped_black_call(strike, maturity, lambda);
                let priced = price_vanilla(
                    Shared::clone(&hazard),
                    None,
                    OptionType::Call,
                    strike,
                    years,
                    800,
                    800,
                );
                let scale = expected.abs().max(1.0);
                let error = (priced - expected).abs() / scale;
                assert!(
                    error <= 1e-4,
                    "K={strike} T={maturity}: {priced} vs {expected} (rel {error})"
                );
            }
        }
    }

    #[test]
    fn zero_hazard_matches_the_black_scholes_local_vol_engine() {
        use super::super::FdBlackScholesVanillaEngine;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let proc = process(VOL);
        let local_vol = proc.local_volatility().unwrap();
        let strike = 105.0;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry(1.0)));
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, strike));

        let mut reference = FdBlackScholesVanillaEngine::with_local_vol(
            Shared::clone(&proc),
            Vec::new(),
            80,
            80,
            0,
            FdmSchemeDesc::douglas(),
            true,
            -Real::null(),
        );
        let expected = reference
            .price(Shared::clone(&payoff), Shared::clone(&exercise))
            .unwrap();

        let mut jumped = FdJumpToDefaultVanillaEngine::new(
            proc,
            Some(local_vol),
            shared(PowerHazard::constant(0.0, 0.0).unwrap()),
            80,
            80,
            FdmSchemeDesc::douglas(),
        );
        let priced = jumped.price(payoff, exercise).unwrap();
        assert!((priced - expected).abs() <= 1e-10, "{priced} vs {expected}");
    }

    /// Local volatility that depends on the spot, so parity is not the flat
    /// Black–Scholes identity in disguise.
    struct SlopedLocalVol {
        base: TermStructureBase,
        spot: Real,
    }

    impl SlopedLocalVol {
        fn new(reference: Date, day_counter: DayCounter, spot: Real) -> Self {
            Self {
                base: TermStructureBase::with_reference_date(reference, None, Some(day_counter)),
                spot,
            }
        }
    }

    impl AsObservable for SlopedLocalVol {
        fn observable(&self) -> &Observable {
            self.base.observable()
        }
    }

    impl TermStructure for SlopedLocalVol {
        fn base(&self) -> &TermStructureBase {
            &self.base
        }

        fn max_date(&self) -> Date {
            Date::max_date()
        }
    }

    impl crate::termstructures::volatility::VolatilityTermStructure for SlopedLocalVol {
        fn business_day_convention(&self) -> BusinessDayConvention {
            BusinessDayConvention::Following
        }

        fn min_strike(&self) -> Rate {
            Rate::MIN
        }

        fn max_strike(&self) -> Rate {
            Rate::MAX
        }
    }

    impl LocalVolTermStructure for SlopedLocalVol {
        fn local_vol_impl(&self, _t: Time, level: Real) -> QlResult<Volatility> {
            Ok(0.20 * (level / self.spot).powf(-0.2))
        }
    }

    fn sloped_surface() -> Handle<dyn LocalVolTermStructure> {
        Handle::new(
            shared(SlopedLocalVol::new(today(), Actual365Fixed::new(), SPOT))
                as Shared<dyn LocalVolTermStructure>,
        )
    }

    #[test]
    fn put_call_parity_holds_for_a_negative_power() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let years = 1.0;
        let maturity = year_fraction(years);
        let strike = 100.0;
        // λ(S0) = 0.04.
        let scale = 0.04 * SPOT.powf(1.5);
        let hazard = shared(PowerHazard::constant(scale, -1.5).unwrap());
        let local_vol = sloped_surface();
        let call = price_vanilla(
            Shared::clone(&hazard),
            Some(local_vol.clone()),
            OptionType::Call,
            strike,
            years,
            1200,
            1200,
        );
        let put = price_vanilla(
            hazard,
            Some(local_vol),
            OptionType::Put,
            strike,
            years,
            1200,
            1200,
        );
        let forward_diff = SPOT * (-DIVIDEND * maturity).exp() - strike * (-RATE * maturity).exp();
        let gap = (call - put - forward_diff).abs();
        assert!(
            gap <= 1e-6 * SPOT,
            "parity gap {gap} (call {call}, put {put}, forward {forward_diff})"
        );
    }

    #[test]
    fn mesh_errors_shrink_at_least_first_order() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let scale = 0.03 * SPOT.powf(1.5);
        let hazard = shared(PowerHazard::constant(scale, -1.5).unwrap());
        let local_vol = sloped_surface();
        let mut prices = Vec::new();
        for n in [80, 160, 320] {
            prices.push(price_vanilla(
                Shared::clone(&hazard),
                Some(local_vol.clone()),
                OptionType::Call,
                SPOT,
                1.0,
                n,
                n,
            ));
        }
        let first = (prices[0] - prices[1]).abs();
        let second = (prices[1] - prices[2]).abs();
        let order = if second == 0.0 {
            Real::INFINITY
        } else {
            (first / second).ln() / 2.0_f64.ln()
        };
        assert!(
            second <= first * 0.5 * 1.25,
            "observed order {order}: prices {prices:?}, differences {first} then {second}"
        );
        eprintln!("carr-madan mesh order {order} on grids 80/160/320");
    }

    /// Both the call and the put rise with `f`. Put–call parity does not
    /// depend on `f`, so the two premiums move together; the constant-hazard
    /// reduction to Black–Scholes with rate `r + λ` is increasing in `λ`.
    #[test]
    fn call_and_put_rise_as_the_scale_grows() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let local_vol = sloped_surface();
        let mut calls = Vec::new();
        let mut puts = Vec::new();
        for lambda_at_spot in [0.0, 0.05, 0.12] {
            let hazard =
                shared(PowerHazard::constant(lambda_at_spot * SPOT.powf(1.5), -1.5).unwrap());
            calls.push(price_vanilla(
                Shared::clone(&hazard),
                Some(local_vol.clone()),
                OptionType::Call,
                SPOT,
                1.0,
                120,
                120,
            ));
            puts.push(price_vanilla(
                hazard,
                Some(local_vol.clone()),
                OptionType::Put,
                SPOT,
                1.0,
                120,
                120,
            ));
        }
        assert!(calls[0] < calls[1] && calls[1] < calls[2], "{calls:?}");
        assert!(puts[0] < puts[1] && puts[1] < puts[2], "{puts:?}");
    }

    #[test]
    fn survival_of_a_constant_hazard_is_exponential() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let lambda = 0.04;
        let engine = engine(
            shared(PowerHazard::constant(lambda, 0.0).unwrap()),
            None,
            200,
            80,
        );
        for years in [0.25, 1.0, 3.0] {
            let maturity = year_fraction(years);
            let survival = engine.survival_probability(maturity).unwrap();
            let expected = (-lambda * maturity).exp();
            assert!(
                (survival - expected).abs() <= 1e-5,
                "T={maturity}: {survival} vs {expected}"
            );
        }
    }
}
