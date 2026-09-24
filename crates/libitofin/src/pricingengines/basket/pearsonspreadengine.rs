//! Pearson (1995) spread option pricing via 1-D numerical integration.
//!
//! Port of `ql/pricingengines/basket/pearsonspreadengine.{hpp,cpp}` and
//! `ql/pricingengines/basket/spreadblackscholesvanillaengine.{hpp,cpp}`.
//!
//! References:
//! Neil D. Pearson, "An Efficient Approach for Pricing Spread Options",
//! Journal of Derivatives, 3 (1995), pp. 76–91.

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::instruments::{
    BasketArguments, BasketOption, BasketPayoff, BasketResults, StrikedTypePayoff, TypePayoff,
};
use crate::math::distributions::normal::NormalDistribution;
use crate::math::integrals::Integrator;
use crate::math::integrals::lobatto::GaussLobattoIntegral;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::blackcalculator::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

/// Evaluates Pearson's (1995) spread option formula via 1-D numerical integration on forwards
/// with default integration tolerance, iteration budget, and truncation bounds.
#[allow(clippy::too_many_arguments)]
pub fn pearson_spread_option_value(
    forward1: Real,
    forward2: Real,
    strike: Real,
    option_type: OptionType,
    variance1: Real,
    variance2: Real,
    discount: Real,
    rho: Real,
) -> QlResult<Real> {
    pearson_spread_option_value_with_config(
        forward1,
        forward2,
        strike,
        option_type,
        variance1,
        variance2,
        discount,
        rho,
        PearsonSpreadEngine::DEFAULT_INTEGRATION_TOLERANCE,
        PearsonSpreadEngine::DEFAULT_MAX_INTEGRATION_ITERATIONS,
        PearsonSpreadEngine::DEFAULT_N_STD,
    )
}

/// Evaluates Pearson's (1995) spread option formula via 1-D numerical integration on forwards
/// with custom integration parameters.
#[allow(clippy::too_many_arguments)]
pub fn pearson_spread_option_value_with_config(
    forward1: Real,
    forward2: Real,
    strike: Real,
    option_type: OptionType,
    variance1: Real,
    variance2: Real,
    discount: Real,
    rho: Real,
    integration_tolerance: Real,
    max_integration_iterations: usize,
    n_std: Real,
) -> QlResult<Real> {
    require!(
        (-1.0..=1.0).contains(&rho),
        "correlation must be in [-1, 1]"
    );
    require!(
        forward1 > 0.0 && forward2 > 0.0,
        "forwards must be positive"
    );
    require!(
        variance1 >= 0.0 && variance2 >= 0.0,
        "variances must be non-negative"
    );
    require!(discount > 0.0, "discount must be positive");
    require!(n_std > 0.0, "n_std must be positive");

    let sigma1 = variance1.sqrt();
    let sigma2 = variance2.sqrt();

    // Under the forward measure, the joint dynamics are:
    //   ln F1 ~ N(mu1, sigma1^2),  ln F2 ~ N(mu2, sigma2^2) with correlation rho.
    // Condition on z = standard normal driving F2:
    //   F2(z) = f2 * exp(-0.5 * sigma2^2 + sigma2 * z)
    // Conditional on z, F1 is log-normal with:
    //   conditional mean of ln F1:  mu1_cond = ln(f1) - 0.5*sigma1^2 + rho*sigma1*z
    //   conditional variance:       sigma1_cond^2 = sigma1^2 * (1 - rho^2)
    // The conditional spread option value is Black(F1_cond, K + F2(z), sigma1_cond).
    let sigma1_cond = sigma1 * (1.0 - rho * rho).max(0.0).sqrt();
    let phi = NormalDistribution::standard();

    let integrand = |z: Real| -> Real {
        let f2_z = forward2 * (-0.5 * variance2 + sigma2 * z).exp();
        let effective_strike = f2_z + strike;
        let f1_cond = forward1 * (rho * sigma1 * z - 0.5 * rho * rho * variance1).exp();

        if effective_strike <= 0.0 {
            let val = match option_type {
                OptionType::Call => (f1_cond - effective_strike).max(0.0),
                OptionType::Put => 0.0,
            };
            return phi.value(z) * val;
        }

        let black =
            match BlackCalculator::new(option_type, effective_strike, f1_cond, sigma1_cond, 1.0) {
                Ok(b) => b.value(),
                Err(_) => 0.0,
            };
        phi.value(z) * black
    };

    let integrator = GaussLobattoIntegral::new(max_integration_iterations, integration_tolerance)?;
    let undiscounted = integrator.integrate(integrand, -n_std, n_std)?;

    Ok(discount * undiscounted)
}

/// Pricing engine for spread options using Pearson (1995) 1-D numerical integration.
pub struct PearsonSpreadEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    integration_tolerance: Real,
    max_integration_iterations: usize,
    n_std: Real,
}

impl PearsonSpreadEngine {
    pub const DEFAULT_INTEGRATION_TOLERANCE: Real = 1e-10;
    pub const DEFAULT_MAX_INTEGRATION_ITERATIONS: usize = 10_000;
    pub const DEFAULT_N_STD: Real = 8.0;

    /// Creates a new `PearsonSpreadEngine` with default integration settings.
    pub fn new(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
    ) -> QlResult<Self> {
        Self::with_config(
            process1,
            process2,
            rho,
            Self::DEFAULT_INTEGRATION_TOLERANCE,
            Self::DEFAULT_MAX_INTEGRATION_ITERATIONS,
            Self::DEFAULT_N_STD,
        )
    }

    /// Creates a new `PearsonSpreadEngine` with custom integration parameters.
    pub fn with_config(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
        integration_tolerance: Real,
        max_integration_iterations: usize,
        n_std: Real,
    ) -> QlResult<Self> {
        require!(
            (-1.0..=1.0).contains(&rho),
            "correlation must be in [-1, 1]"
        );
        require!(n_std > 0.0, "n_std must be positive");
        let base = EngineBase::new(BasketArguments::default(), BasketResults::default());
        base.register_with(process1.observable());
        base.register_with(process2.observable());
        Ok(Self {
            base,
            process1,
            process2,
            rho,
            integration_tolerance,
            max_integration_iterations,
            n_std,
        })
    }

    pub fn rho(&self) -> Real {
        self.rho
    }

    pub fn integration_tolerance(&self) -> Real {
        self.integration_tolerance
    }

    pub fn max_integration_iterations(&self) -> usize {
        self.max_integration_iterations
    }

    pub fn n_std(&self) -> Real {
        self.n_std
    }
}

impl AsObservable for PearsonSpreadEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for PearsonSpreadEngine {
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
        let exercise = arguments.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European exercise"
        );
        let payoff_wrapper = arguments.payoff.as_ref().expect("validated");
        let spread_payoff = match payoff_wrapper {
            BasketPayoff::Spread(p) => p,
            _ => return Err(QlError::new("spread payoff expected", file!(), line!())),
        };
        let vanilla_payoff = spread_payoff.base_payoff();
        let strike = vanilla_payoff.strike();
        let option_type = vanilla_payoff.option_type();

        let maturity = exercise.last_date();
        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let (df1, df2) = (
            self.process1
                .risk_free_rate()
                .current_link()?
                .discount_date(maturity, false)?,
            self.process2
                .risk_free_rate()
                .current_link()?
                .discount_date(maturity, false)?,
        );

        let q1 = self
            .process1
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;
        let q2 = self
            .process2
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;

        let (forward1, forward2) = (s1 * q1 / df1, s2 * q2 / df2);

        let (vol1, vol2) = (
            self.process1.black_volatility().current_link()?,
            self.process2.black_volatility().current_link()?,
        );
        let var1 = vol1.black_variance_date(maturity, forward1, false)?;
        let var2 = vol2.black_variance_date(maturity, forward2, false)?;

        let value = pearson_spread_option_value_with_config(
            forward1,
            forward2,
            strike,
            option_type,
            var1,
            var2,
            df1,
            self.rho,
            self.integration_tolerance,
            self.max_integration_iterations,
            self.n_std,
        )?;
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`PearsonSpreadEngine`] with default parameters to `option`.
pub fn set_pearson_engine(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) -> QlResult<()> {
    let engine = shared_mut(PearsonSpreadEngine::new(process1, process2, rho)?)
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

/// Attaches [`PearsonSpreadEngine`] with custom parameters to `option`.
pub fn set_pearson_engine_with_config(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    integration_tolerance: Real,
    max_integration_iterations: usize,
    n_std: Real,
) -> QlResult<()> {
    let engine = shared_mut(PearsonSpreadEngine::with_config(
        process1,
        process2,
        rho,
        integration_tolerance,
        max_integration_iterations,
        n_std,
    )?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{MinBasketPayoff, PlainVanillaPayoff, SpreadBasketPayoff};
    use crate::interestrate::Compounding;
    use crate::pricingengines::basket::set_bjerksund_stensland_engine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn flat_rate(d: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            d,
            Handle::new(shared(SimpleQuote::new(r))),
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn make_process_365(
        date: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let vol_ts: Handle<dyn BlackVolTermStructure> =
            Handle::new(shared(BlackConstantVol::with_quote(
                date,
                None,
                Handle::new(shared(SimpleQuote::new(vol))),
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>);
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot))),
            flat_rate(date, q),
            flat_rate(date, r),
            vol_ts,
        ))
    }

    /// Replicates QuantLib `testPearsonSpreadEngine` from `basketoption.cpp:2616`:
    /// 1. Put-call parity: (C - P) / df = F1 - F2 - K @ tol 1e-10
    /// 2. Exchange option (K = 0): Pearson matches Bjerksund-Stensland @ tol 1e-6
    #[test]
    fn test_pearson_spread_engine() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(12, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let rho = 0.75;
        let f1 = 100.0;
        let f2 = 110.0;
        let r = 0.05;
        let v1 = 0.25;
        let v2 = 0.35;

        let p1 = make_process_365(today, f1, r, r, v1);
        let p2 = make_process_365(today, f2, r, r, v2);

        // 1. Put-call parity: (C - P) / df = F1 - F2 - K
        let strike = 5.0;
        let mut call_option = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_pearson_engine(&mut call_option, p1.clone(), p2.clone(), rho).unwrap();
        let call_npv = call_option.npv().unwrap();

        let mut put_option = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_pearson_engine(&mut put_option, p1.clone(), p2.clone(), rho).unwrap();
        let put_npv = put_option.npv().unwrap();

        let t = Actual365Fixed::new().year_fraction(today, maturity);
        let df = (-r * t).exp();
        let fwd = (call_npv - put_npv) / df;
        let expected_fwd = f1 - f2 - strike;
        let diff = (fwd - expected_fwd).abs();
        assert!(
            diff < 1e-10,
            "failed put-call parity: calc={fwd}, exp={expected_fwd}, diff={diff}"
        );

        // 2. Exchange option (K = 0): should match Bjerksund-Stensland within 1e-6
        let exchange_strike = 0.0;
        let mut exchange_option = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, exchange_strike)),
            exercise,
            Shared::clone(&settings),
        );
        set_pearson_engine(&mut exchange_option, p1.clone(), p2.clone(), rho).unwrap();
        let pearson_exchange = exchange_option.npv().unwrap();

        set_bjerksund_stensland_engine(&mut exchange_option, p1, p2, rho).unwrap();
        let bs_exchange = exchange_option.npv().unwrap();

        let ex_diff = (pearson_exchange - bs_exchange).abs();
        assert!(
            ex_diff < 1e-6,
            "failed exchange option match: Pearson={pearson_exchange}, BS={bs_exchange}, diff={ex_diff}"
        );
    }

    /// Parity verification with distinct dividend yields q1 != q2 != r.
    #[test]
    fn test_pearson_dividend_yield_parity() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(6, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let s1 = 120.0;
        let s2 = 100.0;
        let r = 0.06;
        let q1 = 0.02;
        let q2 = 0.04;
        let v1 = 0.30;
        let v2 = 0.20;
        let rho = -0.40;
        let strike = 12.0;

        let p1 = make_process_365(today, s1, q1, r, v1);
        let p2 = make_process_365(today, s2, q2, r, v2);

        let mut call_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_pearson_engine(&mut call_opt, p1.clone(), p2.clone(), rho).unwrap();
        let c = call_opt.npv().unwrap();

        let mut put_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
            exercise,
            Shared::clone(&settings),
        );
        set_pearson_engine(&mut put_opt, p1, p2, rho).unwrap();
        let p = put_opt.npv().unwrap();

        let t = Actual365Fixed::new().year_fraction(today, maturity);
        let df = (-r * t).exp();
        let f1 = s1 * (-q1 * t).exp() / df;
        let f2 = s2 * (-q2 * t).exp() / df;
        let implied_fwd = (c - p) / df;
        let expected_fwd = f1 - f2 - strike;

        assert!((implied_fwd - expected_fwd).abs() < 1e-10);
    }

    /// Validation checks: rejected inputs and engine error cases.
    #[test]
    fn test_pearson_validation_rejects_invalid_inputs() {
        assert!(
            pearson_spread_option_value(
                100.0,
                100.0,
                10.0,
                OptionType::Call,
                0.04,
                0.04,
                0.95,
                1.5,
            )
            .is_err(),
            "rho > 1 should fail"
        );
        assert!(
            pearson_spread_option_value(
                -10.0,
                100.0,
                10.0,
                OptionType::Call,
                0.04,
                0.04,
                0.95,
                0.5,
            )
            .is_err(),
            "forward1 <= 0 should fail"
        );
        assert!(
            pearson_spread_option_value(100.0, 0.0, 10.0, OptionType::Call, 0.04, 0.04, 0.95, 0.5,)
                .is_err(),
            "forward2 <= 0 should fail"
        );
        assert!(
            pearson_spread_option_value(
                100.0,
                100.0,
                10.0,
                OptionType::Call,
                -0.04,
                0.04,
                0.95,
                0.5,
            )
            .is_err(),
            "negative variance should fail"
        );
        assert!(
            pearson_spread_option_value(
                100.0,
                100.0,
                10.0,
                OptionType::Call,
                0.04,
                0.04,
                -0.95,
                0.5,
            )
            .is_err(),
            "negative discount should fail"
        );
        assert!(
            pearson_spread_option_value_with_config(
                100.0,
                100.0,
                10.0,
                OptionType::Call,
                0.04,
                0.04,
                0.95,
                0.5,
                1e-10,
                10_000,
                0.0,
            )
            .is_err(),
            "n_std <= 0 should fail"
        );

        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let p1 = make_process_365(today, 100.0, 0.05, 0.05, 0.20);
        let p2 = make_process_365(today, 100.0, 0.05, 0.05, 0.20);

        let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 0.0));
        let american_ex: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 365, false).unwrap());
        let mut amer_opt = BasketOption::new(payoff, american_ex, Shared::clone(&settings));
        set_pearson_engine(&mut amer_opt, p1.clone(), p2.clone(), 0.5).unwrap();
        assert!(
            amer_opt
                .npv()
                .unwrap_err()
                .to_string()
                .contains("not an European exercise")
        );

        let min_payoff = MinBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 0.0));
        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 365));
        let mut min_opt = BasketOption::new(min_payoff, euro_ex, settings);
        set_pearson_engine(&mut min_opt, p1, p2, 0.5).unwrap();
        assert!(
            min_opt
                .npv()
                .unwrap_err()
                .to_string()
                .contains("spread payoff expected")
        );
    }

    /// Tests that negative strikes where effective_strike can become negative preserve call-put parity.
    #[test]
    fn test_pearson_negative_strike_parity() {
        let f1 = 100.0;
        let f2 = 110.0;
        let strike = -30.0;
        let v1 = 0.25;
        let v2 = 0.35;
        let df = 0.95;
        let rho = 0.60;

        let call =
            pearson_spread_option_value(f1, f2, strike, OptionType::Call, v1, v2, df, rho).unwrap();
        let put =
            pearson_spread_option_value(f1, f2, strike, OptionType::Put, v1, v2, df, rho).unwrap();

        let parity = (call - put) / df;
        let expected = f1 - f2 - strike;
        assert!(
            (parity - expected).abs() < 1e-10,
            "failed negative strike parity: calc={parity}, exp={expected}"
        );
    }

    /// Tests zero residual vol (rho = 1, K = 0, v1 = v2) yields exact discounted intrinsic payoff.
    #[test]
    fn test_pearson_residual_vol_zero() {
        let f1 = 100.0;
        let f2 = 90.0;
        let strike = 0.0;
        let v = 0.25;
        let df = 0.95;
        let rho = 1.0;

        let call =
            pearson_spread_option_value(f1, f2, strike, OptionType::Call, v, v, df, rho).unwrap();
        let expected = df * (f1 - f2);
        assert!(
            (call - expected).abs() < 1e-10,
            "failed residual vol zero: calc={call}, exp={expected}"
        );
    }

    /// Tests zero volatility on both assets yields deterministic discounted payoff.
    #[test]
    fn test_pearson_zero_vol_deterministic() {
        let f1 = 105.0;
        let f2 = 100.0;
        let strike = 2.0;
        let df = 0.95;
        let rho = 0.50;

        let call = pearson_spread_option_value(f1, f2, strike, OptionType::Call, 0.0, 0.0, df, rho)
            .unwrap();
        let expected_call = df * (f1 - f2 - strike);
        assert!((call - expected_call).abs() < 1e-10);

        let put = pearson_spread_option_value(f1, f2, strike, OptionType::Put, 0.0, 0.0, df, rho)
            .unwrap();
        assert!(put.abs() < 1e-10);
    }
}
