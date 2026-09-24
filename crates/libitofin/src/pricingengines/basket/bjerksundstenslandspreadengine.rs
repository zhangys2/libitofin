//! Bjerksund and Stensland (2014) closed-form pricing engine for 2D European spread options.
//!
//! Port of `ql/pricingengines/basket/bjerksundstenslandspreadengine.{hpp,cpp}` and
//! `ql/pricingengines/basket/spreadblackscholesvanillaengine.{hpp,cpp}`.
//!
//! References:
//! P. Bjerksund and G. Stensland, "Closed form spread option valuation",
//! Quantitative Finance, 14 (2014), pp. 1785–1794.

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::instruments::{
    BasketArguments, BasketOption, BasketPayoff, BasketResults, StrikedTypePayoff, TypePayoff,
};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

/// Evaluates Bjerksund & Stensland (2014) closed-form spread option formula on forwards.
#[allow(clippy::too_many_arguments)]
pub fn bjerksund_stensland_spread_option_value(
    forward1: Real,
    forward2: Real,
    strike: Real,
    option_type: OptionType,
    variance1: Real,
    variance2: Real,
    discount: Real,
    rho: Real,
) -> QlResult<Real> {
    require!(
        (-1.0..=1.0).contains(&rho),
        "correlation must be in [-1, 1]"
    );
    require!(
        forward1 > 0.0 && forward2 > 0.0,
        "forwards must be positive"
    );
    let a = forward2 + strike;
    require!(a > 0.0, "forward2 + strike must be positive");
    require!(
        variance1 >= 0.0 && variance2 >= 0.0,
        "variances must be non-negative"
    );
    require!(discount > 0.0, "discount must be positive");

    let b = forward2 / a;
    let (s1, s2) = (variance1.sqrt(), variance2.sqrt());
    let stdev = (variance1 + b * b * variance2 - 2.0 * rho * b * s1 * s2)
        .max(0.0)
        .sqrt();
    require!(stdev > 0.0, "stdev must be positive");

    let lfa = (forward1 / a).ln();
    let d1 = (lfa + (0.5 * variance1 + 0.5 * b * b * variance2 - b * rho * s1 * s2)) / stdev;
    let d2 = (lfa + (-0.5 * variance1 + variance2 * b * (0.5 * b - 1.0) + rho * s1 * s2)) / stdev;
    let d3 = (lfa + (-0.5 * variance1 + 0.5 * b * b * variance2)) / stdev;

    let phi = CumulativeNormalDistribution::standard();
    let cp = match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    };

    Ok(discount
        * cp
        * (forward1 * phi.value(cp * d1)
            - forward2 * phi.value(cp * d2)
            - strike * phi.value(cp * d3)))
}

/// 2D European spread option pricing engine using Bjerksund & Stensland (2014) closed-form formula.
pub struct BjerksundStenslandSpreadEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
}

impl BjerksundStenslandSpreadEngine {
    pub fn new(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
    ) -> QlResult<Self> {
        require!(
            (-1.0..=1.0).contains(&rho),
            "correlation must be in [-1, 1]"
        );
        let base = EngineBase::new(BasketArguments::default(), BasketResults::default());
        base.register_with(process1.observable());
        base.register_with(process2.observable());
        Ok(Self {
            base,
            process1,
            process2,
            rho,
        })
    }

    pub fn rho(&self) -> Real {
        self.rho
    }

    pub fn process1(&self) -> &Shared<GeneralizedBlackScholesProcess> {
        &self.process1
    }

    pub fn process2(&self) -> &Shared<GeneralizedBlackScholesProcess> {
        &self.process2
    }
}

impl AsObservable for BjerksundStenslandSpreadEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BjerksundStenslandSpreadEngine {
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
        let (strike, option_type) = (vanilla_payoff.strike(), vanilla_payoff.option_type());

        let maturity = exercise.last_date();
        let (s1, s2) = (self.process1.x0()?, self.process2.x0()?);
        require!(s1 > 0.0 && s2 > 0.0, "underlying prices must be positive");

        let (rf1, rf2) = (
            self.process1.risk_free_rate().current_link()?,
            self.process2.risk_free_rate().current_link()?,
        );
        let (df1, df2) = (
            rf1.discount_date(maturity, false)?,
            rf2.discount_date(maturity, false)?,
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

        let value = bjerksund_stensland_spread_option_value(
            forward1,
            forward2,
            strike,
            option_type,
            var1,
            var2,
            df1,
            self.rho,
        )?;
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`BjerksundStenslandSpreadEngine`] to `option`.
pub fn set_bjerksund_stensland_engine(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) -> QlResult<()> {
    let engine = shared_mut(BjerksundStenslandSpreadEngine::new(
        process1, process2, rho,
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

    /// Tests `testBjerksundStenslandSpreadEngine` from QuantLib `basketoption.cpp:1074`.
    ///
    /// Checks against PyFENG 0.2.6 reference value `17.850835947276213` and put-call parity within 100 * EPSILON.
    #[test]
    fn test_bjerksund_stensland_reference_put_and_parity() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2024);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(12, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let rho = 0.75;
        let (f1, f2, r) = (100.0, 110.0, 0.05);

        // In QL testBjerksundStenslandSpreadEngine, BlackProcess is used with forward quote (q = r = 0.05).
        let p1 = make_process_365(today, f1, r, r, 0.25);
        let p2 = make_process_365(today, f2, r, r, 0.35);

        let strike = 5.0;
        let mut put_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_bjerksund_stensland_engine(&mut put_opt, p1.clone(), p2.clone(), rho).unwrap();
        let put_npv = put_opt.npv().unwrap();

        let expected_put_npv = 17.850835947276213;
        let tol = f64::EPSILON * 100.0;
        assert!((put_npv - expected_put_npv).abs() <= tol);

        let mut call_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
            exercise,
            Shared::clone(&settings),
        );
        set_bjerksund_stensland_engine(&mut call_opt, p1, p2, rho).unwrap();
        let call_npv = call_opt.npv().unwrap();

        let t = Actual365Fixed::new().year_fraction(today, maturity);
        let df = (-r * t).exp();
        let parity_diff = ((call_npv - put_npv) / df - (f1 - f2 - strike)).abs();
        assert!(parity_diff <= tol);
    }

    /// Verifies that with K = 0 (exchange option), Bjerksund-Stensland matches Margrabe formula.
    #[test]
    fn test_bjerksund_stensland_exchange_option_k_zero() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 365));
        let rho = 0.75;
        let (f1, f2, r, v1, v2) = (100.0, 110.0, 0.05, 0.25, 0.35);

        let p1 = make_process_365(today, f1, r, r, v1);
        let p2 = make_process_365(today, f2, r, r, v2);

        let mut opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 0.0)),
            exercise,
            Shared::clone(&settings),
        );
        set_bjerksund_stensland_engine(&mut opt, p1, p2, rho).unwrap();
        let calc_val = opt.npv().unwrap();

        // Margrabe formula comparison:
        let t = 1.0;
        let sigma = (v1 * v1 * t + v2 * v2 * t - 2.0 * rho * v1 * v2 * t).sqrt();
        let d1 = ((f1 / f2).ln() + 0.5 * sigma * sigma) / sigma;
        let d2 = d1 - sigma;
        let phi = CumulativeNormalDistribution::standard();
        let expected_val = (-r * t).exp() * (f1 * phi.value(d1) - f2 * phi.value(d2));

        assert!((calc_val - expected_val).abs() < 1e-12);
    }

    /// Parity verification with distinct dividend yields q1 != q2 != r.
    #[test]
    fn test_bjerksund_stensland_dividend_yield_parity() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 2024);
        settings.set_evaluation_date(today);

        let maturity = today + 180;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let (s1, s2, q1, q2, r, v1, v2, rho, strike) =
            (122.0, 120.0, 0.04, 0.02, 0.08, 0.20, 0.25, -0.4, 4.0);

        let p1 = make_process_365(today, s1, q1, r, v1);
        let p2 = make_process_365(today, s2, q2, r, v2);

        let mut call_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_bjerksund_stensland_engine(&mut call_opt, p1.clone(), p2.clone(), rho).unwrap();
        let call_npv = call_opt.npv().unwrap();

        let mut put_opt = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
            exercise,
            Shared::clone(&settings),
        );
        set_bjerksund_stensland_engine(&mut put_opt, p1, p2, rho).unwrap();
        let put_npv = put_opt.npv().unwrap();

        let t = Actual365Fixed::new().year_fraction(today, maturity);
        let df = (-r * t).exp();
        let f1 = s1 * (-q1 * t).exp() / df;
        let f2 = s2 * (-q2 * t).exp() / df;

        let parity_diff = call_npv - put_npv;
        let expected_parity = df * (f1 - f2 - strike);
        assert!((parity_diff - expected_parity).abs() < 1e-10);
    }

    #[test]
    fn test_bjerksund_stensland_validation_rejects_invalid_inputs() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 2024);
        settings.set_evaluation_date(today);

        let p1 = make_process_365(today, 122.0, 0.05, 0.05, 0.20);
        let p2 = make_process_365(today, 120.0, 0.05, 0.05, 0.20);

        assert!(BjerksundStenslandSpreadEngine::new(p1.clone(), p2.clone(), 1.5).is_err());
        assert!(BjerksundStenslandSpreadEngine::new(p1.clone(), p2.clone(), -1.5).is_err());

        let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0));
        let american_ex: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 36, false).unwrap());
        let mut opt_american = BasketOption::new(payoff, american_ex, Shared::clone(&settings));
        set_bjerksund_stensland_engine(&mut opt_american, p1.clone(), p2.clone(), 0.5).unwrap();
        assert!(
            opt_american
                .npv()
                .unwrap_err()
                .to_string()
                .contains("not an European exercise")
        );

        let min_payoff = MinBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0));
        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 36));
        let mut opt_min = BasketOption::new(min_payoff, euro_ex, Shared::clone(&settings));
        set_bjerksund_stensland_engine(&mut opt_min, p1, p2, 0.5).unwrap();
        assert!(
            opt_min
                .npv()
                .unwrap_err()
                .to_string()
                .contains("spread payoff expected")
        );
    }
}
