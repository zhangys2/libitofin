//! Kirk (1995) approximation pricing engine for 2D European spread options.
//!
//! Port of `ql/pricingengines/basket/kirkengine.{hpp,cpp}` and
//! `ql/pricingengines/basket/spreadblackscholesvanillaengine.{hpp,cpp}`.
//!
//! Kirk's approximation prices European spread options with payoff:
//! `max(S_1 - S_2 - K, 0)` for Call (and `max(K - (S_1 - S_2), 0)` for Put)
//! by approximating `S_2 + K` as a lognormal variable with adjusted volatility.

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::instruments::{
    BasketArguments, BasketOption, BasketPayoff, BasketResults, StrikedTypePayoff, TypePayoff,
};
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

/// Evaluates Kirk's (1995) spread option formula on forwards.
#[allow(clippy::too_many_arguments)]
pub fn kirk_spread_option_value(
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
    require!(forward1 > 0.0, "forward1 must be positive");
    require!(forward2 > 0.0, "forward2 must be positive");
    let denom = forward2 + strike;
    require!(denom > 0.0, "forward2 + strike must be positive");
    require!(variance1 >= 0.0, "variance1 must be non-negative");
    require!(variance2 >= 0.0, "variance2 must be non-negative");
    require!(discount > 0.0, "discount must be positive");

    let f = forward1 / denom;
    let ratio = forward2 / denom;
    let v_sq =
        variance1 + variance2 * ratio * ratio - 2.0 * rho * (variance1 * variance2).sqrt() * ratio;
    let v = v_sq.max(0.0).sqrt();

    let black = BlackCalculator::new(option_type, 1.0, f, v, discount)?;
    Ok(denom * black.value())
}

/// 2D European spread option pricing engine using Kirk's (1995) approximation.
pub struct KirkEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
}

impl KirkEngine {
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
}

impl AsObservable for KirkEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for KirkEngine {
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

        let rf1 = self.process1.risk_free_rate().current_link()?;
        let rf2 = self.process2.risk_free_rate().current_link()?;
        let risk_free_discount1 = rf1.discount_date(maturity, false)?;
        let risk_free_discount2 = rf2.discount_date(maturity, false)?;

        let q_disc1 = self
            .process1
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;
        let q_disc2 = self
            .process2
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;

        let forward1 = s1 * q_disc1 / risk_free_discount1;
        let forward2 = s2 * q_disc2 / risk_free_discount2;

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, forward1, false)?;
        let variance2 = vol2.black_variance_date(maturity, forward2, false)?;

        let df = risk_free_discount1;

        let value = kirk_spread_option_value(
            forward1,
            forward2,
            strike,
            option_type,
            variance1,
            variance2,
            df,
            self.rho,
        )?;

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`KirkEngine`] to `option`.
pub fn set_kirk_engine(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) -> QlResult<()> {
    let engine =
        shared_mut(KirkEngine::new(process1, process2, rho)?) as SharedMut<dyn PricingEngine>;
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
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn make_process_360(
        date: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let q_ts: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::new(
            date,
            quote_handle(&shared(SimpleQuote::new(q))),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let r_ts: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::new(
            date,
            quote_handle(&shared(SimpleQuote::new(r))),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let vol_ts: Handle<dyn BlackVolTermStructure> =
            Handle::new(shared(BlackConstantVol::with_quote(
                date,
                None,
                quote_handle(&shared(SimpleQuote::new(vol))),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>);
        shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(spot))),
            q_ts,
            r_ts,
            vol_ts,
        ))
    }

    /// 18-row `basketoption.cpp` `testEuroTwoValues` spread oracle (Haug p. 59-60 @ 1e-3).
    #[test]
    fn test_euro_two_values_spread() {
        type Row = (
            Real, // strike
            Real, // s1
            Real, // s2
            Real, // r
            Real, // t
            Real, // v1
            Real, // v2
            Real, // rho
            Real, // expected result
        );

        #[rustfmt::skip]
        let rows: [Row; 18] = [
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.20, -0.5,  4.7530),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.20,  0.0,  3.7970),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.20,  0.5,  2.5537),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.25, 0.20, -0.5,  5.4275),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.25, 0.20,  0.0,  4.3712),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.25, 0.20,  0.5,  3.0086),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.25, -0.5,  5.4061),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.25,  0.0,  4.3451),
            (3.0, 122.0, 120.0, 0.10, 0.1, 0.20, 0.25,  0.5,  2.9723),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.20, -0.5, 10.7517),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.20,  0.0,  8.7020),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.20,  0.5,  6.0257),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.25, 0.20, -0.5, 12.1941),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.25, 0.20,  0.0,  9.9340),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.25, 0.20,  0.5,  7.0067),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.25, -0.5, 12.1483),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.25,  0.0,  9.8780),
            (3.0, 122.0, 120.0, 0.10, 0.5, 0.20, 0.25,  0.5,  6.9284),
        ];

        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        for (strike, s1, s2, r, t, v1, v2, rho, expected) in rows {
            let p1 = make_process_360(today, s1, r, r, v1);
            let p2 = make_process_360(today, s2, r, r, v2);

            let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike));
            let exercise_date = today + (t * 360.0).round() as i32;
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(exercise_date));
            let mut option = BasketOption::new(payoff, exercise, Shared::clone(&settings));
            set_kirk_engine(&mut option, p1, p2, rho).unwrap();

            let calculated = option.npv().unwrap();
            let err = (calculated - expected).abs();
            assert!(
                err <= 1.0e-3,
                "Failed on row strike={strike}, s1={s1}, s2={s2}, r={r}, t={t}, v1={v1}, v2={v2}, rho={rho}: calc={calculated}, exp={expected}, err={err}"
            );
        }
    }

    /// 15-row `basketoption.cpp` `testStrangSplittingSpreadEngineVsMathematica` (Kirk NPV column @ 100*EPSILON).
    #[test]
    fn test_kirk_reference_values_mathematica() {
        type Row = (
            Real, // T
            Real, // K
            Real, // vol1
            Real, // rho
            Real, // kirkNPV
        );

        #[rustfmt::skip]
        let test_cases: [Row; 15] = [
            (5.0, 20.0, 0.1,  0.6, 15.39520956886349),
            (10., 20.0, 0.1,  0.6, 22.91537136258191),
            (20., 20.0, 0.1,  0.6, 33.6985901856974),
            (1.0, 20.0, 0.3,  0.6, 10.9751711157804),
            (2.0, 20.0, 0.3,  0.6, 15.68896063758723),
            (3.0, 20.0, 0.3,  0.6, 19.33110275816226),
            (4.0, 20.0, 0.3,  0.6, 22.40185479100672),
            (5.0, 20.0, 0.3,  0.6, 25.09737848235137),
            (1.0, 10.0, 0.3,  0.6, 16.10447007803242),
            (1.0, 40.0, 0.3,  0.6,  4.65751918957598),
            (1.0, 60.0, 0.3,  0.6,  1.83735906790182),
            (1.0, 20.0, 0.5,  0.6, 18.79838447214884),
            (1.0, 20.0, 0.3, -0.9, 20.17112122874686),
            (1.0, 20.0, 0.3,  0.0, 15.38036208157481),
            (2.0, 20.0, 0.3, -0.5, 25.80847626931109),
        ];

        let s1 = 110.0;
        let s2 = 90.0;
        let r = 0.05;
        let vol2 = 0.20;

        let settings = shared(Settings::new());
        let today = Date::new(27, Month::May, 2024);
        settings.set_evaluation_date(today);

        for (t, strike, vol1, rho, expected) in test_cases {
            let r_ts: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::new(
                today,
                quote_handle(&shared(SimpleQuote::new(r))),
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            ))
                as Shared<dyn YieldTermStructure>);

            let maturity_date = today + (t * 365.0).round() as i32;
            let dr = r_ts
                .current_link()
                .unwrap()
                .discount_date(maturity_date, false)
                .unwrap();
            let f1 = s1 / dr;
            let f2 = s2 / dr;

            let vol_ts1: Handle<dyn BlackVolTermStructure> =
                Handle::new(shared(BlackConstantVol::with_quote(
                    today,
                    None,
                    quote_handle(&shared(SimpleQuote::new(vol1))),
                    Actual365Fixed::new(),
                )) as Shared<dyn BlackVolTermStructure>);

            let vol_ts2: Handle<dyn BlackVolTermStructure> =
                Handle::new(shared(BlackConstantVol::with_quote(
                    today,
                    None,
                    quote_handle(&shared(SimpleQuote::new(vol2))),
                    Actual365Fixed::new(),
                )) as Shared<dyn BlackVolTermStructure>);

            // In QL testStrangSplittingSpreadEngineVsMathematica, BlackProcess is used with forward quote and rTS
            let p1 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(f1))),
                r_ts.clone(),
                r_ts.clone(),
                vol_ts1,
            ));
            let p2 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(f2))),
                r_ts.clone(),
                r_ts.clone(),
                vol_ts2,
            ));

            let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike));
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity_date));
            let mut option = BasketOption::new(payoff, exercise, Shared::clone(&settings));
            set_kirk_engine(&mut option, p1, p2, rho).unwrap();

            let calculated = option.npv().unwrap();
            let rel_diff = ((calculated - expected) / expected).abs();
            assert!(
                rel_diff <= 100.0 * f64::EPSILON,
                "Failed on T={t}, K={strike}, vol1={vol1}, rho={rho}: calc={calculated}, exp={expected}, rel_diff={rel_diff}"
            );
        }
    }

    #[test]
    fn test_kirk_rejects_non_european_exercise() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        let p1 = make_process_360(today, 122.0, 0.10, 0.10, 0.20);
        let p2 = make_process_360(today, 120.0, 0.10, 0.10, 0.20);

        let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 36, false).unwrap());
        let mut option = BasketOption::new(payoff, exercise, Shared::clone(&settings));
        let res = set_kirk_engine(&mut option, p1, p2, 0.5);
        assert!(res.is_ok());
        let err = option.npv().unwrap_err();
        assert!(err.to_string().contains("not an European exercise"));
    }

    #[test]
    fn test_kirk_rejects_non_spread_payoff() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        let p1 = make_process_360(today, 122.0, 0.10, 0.10, 0.20);
        let p2 = make_process_360(today, 120.0, 0.10, 0.10, 0.20);

        let payoff = MinBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 36));
        let mut option = BasketOption::new(payoff, exercise, Shared::clone(&settings));
        let res = set_kirk_engine(&mut option, p1, p2, 0.5);
        assert!(res.is_ok());
        let err = option.npv().unwrap_err();
        assert!(err.to_string().contains("spread payoff expected"));
    }

    #[test]
    fn test_kirk_rejects_invalid_correlation() {
        let today = Date::new(15, Month::May, 1998);
        let p1 = make_process_360(today, 122.0, 0.10, 0.10, 0.20);
        let p2 = make_process_360(today, 120.0, 0.10, 0.10, 0.20);

        assert!(KirkEngine::new(p1.clone(), p2.clone(), 1.5).is_err());
        assert!(KirkEngine::new(p1, p2, -1.5).is_err());
    }

    #[test]
    fn test_kirk_put_and_dividend_yield_parity() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        // 1. Futures market (q = r = 0.10, Haug first row market): Call ≈ 4.753, Put ≈ 5.743
        let pf1 = make_process_360(today, 122.0, 0.10, 0.10, 0.20);
        let pf2 = make_process_360(today, 120.0, 0.10, 0.10, 0.20);
        let ex01: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 36));

        let mut fut_call = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0)),
            Shared::clone(&ex01),
            Shared::clone(&settings),
        );
        set_kirk_engine(&mut fut_call, pf1.clone(), pf2.clone(), -0.5).unwrap();
        let fut_c = fut_call.npv().unwrap();
        assert!((fut_c - 4.7530).abs() < 1e-4);

        let mut fut_put = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, 3.0)),
            Shared::clone(&ex01),
            Shared::clone(&settings),
        );
        set_kirk_engine(&mut fut_put, pf1, pf2, -0.5).unwrap();
        let fut_p = fut_put.npv().unwrap();
        assert!((fut_p - 5.74305).abs() < 1e-3);
        let df01 = (-0.10_f64 * 0.1).exp();
        assert!(((fut_c - fut_p) - df01 * (122.0 - 120.0 - 3.0)).abs() < 1e-10);

        // 2. Equity market with q = 0, r = 0.10: Call ≈ 4.8152, Put ≈ 5.7853
        let pe1 = make_process_360(today, 122.0, 0.0, 0.10, 0.20);
        let pe2 = make_process_360(today, 120.0, 0.0, 0.10, 0.20);

        let mut eq_call = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 3.0)),
            Shared::clone(&ex01),
            Shared::clone(&settings),
        );
        set_kirk_engine(&mut eq_call, pe1.clone(), pe2.clone(), -0.5).unwrap();
        let eq_c = eq_call.npv().unwrap();
        assert!((eq_c - 4.8152).abs() < 1e-3);

        let mut eq_put = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, 3.0)),
            Shared::clone(&ex01),
            Shared::clone(&settings),
        );
        set_kirk_engine(&mut eq_put, pe1, pe2, -0.5).unwrap();
        let eq_p = eq_put.npv().unwrap();
        assert!((eq_p - 5.7853).abs() < 1e-3);
        assert!(((eq_c - eq_p) - (122.0 - 120.0 - df01 * 3.0)).abs() < 1e-10);

        // 3. General q1 != q2 != r dividend yield parity check at t = 0.5
        let (s1, s2, r, q1, q2, v1, v2, rho, strike, t): (
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
        ) = (122.0, 120.0, 0.10, 0.04, 0.02, 0.20, 0.20, -0.5, 3.0, 0.5);

        let pq1 = make_process_360(today, s1, q1, r, v1);
        let pq2 = make_process_360(today, s2, q2, r, v2);

        let exercise_date = today + (t * 360.0).round() as i32;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(exercise_date));

        let call_payoff =
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike));
        let mut call_opt = BasketOption::new(
            call_payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_kirk_engine(&mut call_opt, pq1.clone(), pq2.clone(), rho).unwrap();
        let call_npv = call_opt.npv().unwrap();

        let put_payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike));
        let mut put_opt = BasketOption::new(put_payoff, exercise, Shared::clone(&settings));
        set_kirk_engine(&mut put_opt, pq1, pq2, rho).unwrap();
        let put_npv = put_opt.npv().unwrap();

        let df = (-r * t).exp();
        let f1 = s1 * (-q1 * t).exp() / df;
        let f2 = s2 * (-q2 * t).exp() / df;
        let parity_diff = call_npv - put_npv;
        let expected_parity = df * (f1 - f2 - strike);
        assert!(
            (parity_diff - expected_parity).abs() < 1e-10,
            "Put-call parity violated: C={call_npv}, P={put_npv}, diff={parity_diff}, exp={expected_parity}"
        );
    }
}
