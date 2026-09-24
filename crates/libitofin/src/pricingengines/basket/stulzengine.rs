//! 2D European basket engine for options on the minimum or maximum of two assets (Stulz 1982).
//!
//! Port of `ql/pricingengines/basket/stulzengine.{hpp,cpp}`.

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::instruments::{
    BasketArguments, BasketOption, BasketPayoff, BasketResults, StrikedTypePayoff, TypePayoff,
};
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionWe04DP;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

fn euro_two_asset_min_basket_call(
    forward1: Real,
    forward2: Real,
    strike: Real,
    risk_free_discount: Real,
    variance1: Real,
    variance2: Real,
    rho: Real,
) -> QlResult<Real> {
    let std_dev1 = variance1.sqrt();
    let std_dev2 = variance2.sqrt();

    let variance = variance1 + variance2 - 2.0 * rho * std_dev1 * std_dev2;
    let std_dev = variance.sqrt();

    let mod_rho1 = ((rho * std_dev2 - std_dev1) / std_dev).clamp(-1.0, 1.0);
    let mod_rho2 = ((rho * std_dev1 - std_dev2) / std_dev).clamp(-1.0, 1.0);

    let d1 = ((forward1 / forward2).ln() + 0.5 * variance) / std_dev;

    let (alfa, beta, gamma) = if strike != 0.0 {
        let biv_c_norm = BivariateCumulativeNormalDistributionWe04DP::new(rho.clamp(-1.0, 1.0))?;
        let biv_c_norm_mod1 = BivariateCumulativeNormalDistributionWe04DP::new(mod_rho1)?;
        let biv_c_norm_mod2 = BivariateCumulativeNormalDistributionWe04DP::new(mod_rho2)?;

        let d1_1 = ((forward1 / strike).ln() + 0.5 * variance1) / std_dev1;
        let d1_2 = ((forward2 / strike).ln() + 0.5 * variance2) / std_dev2;

        let alfa = biv_c_norm_mod1.value(d1_1, -d1);
        let beta = biv_c_norm_mod2.value(d1_2, d1 - std_dev);
        let gamma = biv_c_norm.value(d1_1 - std_dev1, d1_2 - std_dev2);
        (alfa, beta, gamma)
    } else {
        let cum = CumulativeNormalDistribution::standard();
        let alfa = cum.value(-d1);
        let beta = cum.value(d1 - std_dev);
        let gamma = 1.0;
        (alfa, beta, gamma)
    };

    Ok(risk_free_discount * (forward1 * alfa + forward2 * beta - strike * gamma))
}

fn euro_two_asset_max_basket_call(
    forward1: Real,
    forward2: Real,
    strike: Real,
    risk_free_discount: Real,
    variance1: Real,
    variance2: Real,
    rho: Real,
) -> QlResult<Real> {
    let black1 = black_formula(
        OptionType::Call,
        strike,
        forward1,
        variance1.sqrt(),
        risk_free_discount,
        0.0,
    )?;
    let black2 = black_formula(
        OptionType::Call,
        strike,
        forward2,
        variance2.sqrt(),
        risk_free_discount,
        0.0,
    )?;
    let min_call = euro_two_asset_min_basket_call(
        forward1,
        forward2,
        strike,
        risk_free_discount,
        variance1,
        variance2,
        rho,
    )?;
    Ok(black1 + black2 - min_call)
}

/// 2D European Basket pricing engine for min/max basket options due to Stulz (1982).
pub struct StulzEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
}

impl StulzEngine {
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
}

impl AsObservable for StulzEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for StulzEngine {
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
            "not an European Option"
        );
        let payoff_wrapper = arguments.payoff.as_ref().expect("validated");
        let (is_max, vanilla_payoff) = match payoff_wrapper {
            BasketPayoff::Max(p) => (true, p.base_payoff()),
            BasketPayoff::Min(p) => (false, p.base_payoff()),
            _ => return Err(QlError::new("unknown basket type", file!(), line!())),
        };
        let strike = vanilla_payoff.strike();
        let option_type = vanilla_payoff.option_type();

        let maturity = exercise.last_date();
        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, strike, false)?;
        let variance2 = vol2.black_variance_date(maturity, strike, false)?;

        let rf = self.process1.risk_free_rate().current_link()?;
        let risk_free_discount = rf.discount_date(maturity, false)?;

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

        let forward1 = s1 * q_disc1 / risk_free_discount;
        let forward2 = s2 * q_disc2 / risk_free_discount;

        let value = if is_max {
            match option_type {
                OptionType::Call => euro_two_asset_max_basket_call(
                    forward1,
                    forward2,
                    strike,
                    risk_free_discount,
                    variance1,
                    variance2,
                    self.rho,
                )?,
                OptionType::Put => {
                    let call_zero = euro_two_asset_max_basket_call(
                        forward1,
                        forward2,
                        0.0,
                        risk_free_discount,
                        variance1,
                        variance2,
                        self.rho,
                    )?;
                    let call_k = euro_two_asset_max_basket_call(
                        forward1,
                        forward2,
                        strike,
                        risk_free_discount,
                        variance1,
                        variance2,
                        self.rho,
                    )?;
                    strike * risk_free_discount - call_zero + call_k
                }
            }
        } else {
            match option_type {
                OptionType::Call => euro_two_asset_min_basket_call(
                    forward1,
                    forward2,
                    strike,
                    risk_free_discount,
                    variance1,
                    variance2,
                    self.rho,
                )?,
                OptionType::Put => {
                    let call_zero = euro_two_asset_min_basket_call(
                        forward1,
                        forward2,
                        0.0,
                        risk_free_discount,
                        variance1,
                        variance2,
                        self.rho,
                    )?;
                    let call_k = euro_two_asset_min_basket_call(
                        forward1,
                        forward2,
                        strike,
                        risk_free_discount,
                        variance1,
                        variance2,
                        self.rho,
                    )?;
                    strike * risk_free_discount - call_zero + call_k
                }
            }
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`StulzEngine`] to `option`.
pub fn set_stulz_engine(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) -> QlResult<()> {
    let engine =
        shared_mut(StulzEngine::new(process1, process2, rho)?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{
        MaxBasketPayoff, MinBasketPayoff, PlainVanillaPayoff, SpreadBasketPayoff,
    };
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
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            reference,
            None,
            quote_handle(quote),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    /// Full 39-row `basketoption.cpp` `testEuroTwoValues` min/max oracle (Firth + Haug @ 1e-3 / 1e-4).
    #[test]
    fn test_euro_two_values_min_max() {
        type Row = (
            bool,       // is_max
            OptionType, // Call or Put
            Real,       // strike
            Real,       // s1
            Real,       // s2
            Real,       // q1
            Real,       // q2
            Real,       // r
            Real,       // t
            Real,       // v1
            Real,       // v2
            Real,       // rho
            Real,       // expected result
            Real,       // tolerance
        );

        #[rustfmt::skip]
        let rows: [Row; 39] = [
            // MinBasket Call (Firth)
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.90, 10.898, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.70,  8.483, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.50,  6.844, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30,  5.531, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.10,  4.413, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.50, 0.70, 0.00,  4.981, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.50, 0.30, 0.00,  4.159, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.50, 0.10, 0.00,  2.597, 1.0e-3),
            (false, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.50, 0.10, 0.50,  4.030, 1.0e-3),

            // MaxBasket Call (Firth)
            (true, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.90, 17.565, 1.0e-3),
            (true, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.70, 19.980, 1.0e-3),
            (true, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.50, 21.619, 1.0e-3),
            (true, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30, 22.932, 1.0e-3),
            (true, OptionType::Call, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.10, 24.049, 1.1e-3),
            (true, OptionType::Call, 100.0,  80.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30, 16.508, 1.0e-3),
            (true, OptionType::Call, 100.0,  80.0,  80.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30,  8.049, 1.0e-3),
            (true, OptionType::Call, 100.0,  80.0, 120.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30, 30.141, 1.0e-3),
            (true, OptionType::Call, 100.0, 120.0, 120.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30, 42.889, 1.0e-3),

            // MinBasket Put (Firth)
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.90, 11.369, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.70, 12.856, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.50, 13.890, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30, 14.741, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.10, 15.485, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 0.50, 0.30, 0.30, 0.10, 11.893, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 0.25, 0.30, 0.30, 0.10,  8.881, 1.0e-3),
            (false, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 2.00, 0.30, 0.30, 0.10, 19.268, 1.0e-3),

            // MaxBasket Put (Firth)
            (true, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.90,  7.339, 1.0e-3),
            (true, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.70,  5.853, 1.0e-3),
            (true, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.50,  4.818, 1.0e-3),
            (true, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.30,  3.967, 1.1e-3),
            (true, OptionType::Put, 100.0, 100.0, 100.0, 0.00, 0.00, 0.05, 1.00, 0.30, 0.30, 0.10,  3.223, 1.0e-3),

            // Min/Max Call/Put (Haug p. 58 q=0)
            (false, OptionType::Call, 98.0, 100.0, 105.0, 0.00, 0.00, 0.05, 0.50, 0.11, 0.16, 0.63,  4.8177, 1.0e-4),
            (true,  OptionType::Call, 98.0, 100.0, 105.0, 0.00, 0.00, 0.05, 0.50, 0.11, 0.16, 0.63, 11.6323, 1.0e-4),
            (false, OptionType::Put,  98.0, 100.0, 105.0, 0.00, 0.00, 0.05, 0.50, 0.11, 0.16, 0.63,  2.0376, 1.0e-4),
            (true,  OptionType::Put,  98.0, 100.0, 105.0, 0.00, 0.00, 0.05, 0.50, 0.11, 0.16, 0.63,  0.5731, 1.0e-4),

            // Min/Max Call/Put (Haug p. 58 q!=0)
            (false, OptionType::Call, 98.0, 100.0, 105.0, 0.06, 0.09, 0.05, 0.50, 0.11, 0.16, 0.63,  2.9340, 1.0e-4),
            (false, OptionType::Put,  98.0, 100.0, 105.0, 0.06, 0.09, 0.05, 0.50, 0.11, 0.16, 0.63,  3.5224, 1.0e-4),
            (true,  OptionType::Call, 98.0, 100.0, 105.0, 0.06, 0.09, 0.05, 0.50, 0.11, 0.16, 0.63,  8.0701, 1.0e-4),
            (true,  OptionType::Put,  98.0, 100.0, 105.0, 0.06, 0.09, 0.05, 0.50, 0.11, 0.16, 0.63,  1.2181, 1.0e-4),
        ];

        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        for (is_max, opt_type, strike, s1, s2, q1, q2, r, t, v1, v2, rho, expected, tol) in rows {
            let p1 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(s1))),
                flat_rate(today, &shared(SimpleQuote::new(q1))),
                flat_rate(today, &shared(SimpleQuote::new(r))),
                flat_vol(today, &shared(SimpleQuote::new(v1))),
            ));
            let p2 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(s2))),
                flat_rate(today, &shared(SimpleQuote::new(q2))),
                flat_rate(today, &shared(SimpleQuote::new(r))),
                flat_vol(today, &shared(SimpleQuote::new(v2))),
            ));

            let vanilla_payoff = PlainVanillaPayoff::new(opt_type, strike);
            let basket_payoff: BasketPayoff = if is_max {
                MaxBasketPayoff::new(vanilla_payoff).into()
            } else {
                MinBasketPayoff::new(vanilla_payoff).into()
            };

            let exercise_date = today + (t * 360.0).round() as i32;
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(exercise_date));
            let mut option = BasketOption::new(basket_payoff, exercise, Shared::clone(&settings));
            set_stulz_engine(&mut option, p1, p2, rho).unwrap();

            let calculated = option.npv().unwrap();
            let err = (calculated - expected).abs();
            assert!(
                err <= tol,
                "Failed on row is_max={is_max}, opt_type={opt_type:?}, strike={strike}, s1={s1}, s2={s2}, q1={q1}, q2={q2}, r={r}, t={t}, v1={v1}, v2={v2}, rho={rho}: calc={calculated}, exp={expected}, err={err}, tol={tol}"
            );
        }
    }

    #[test]
    fn test_stulz_rejects_non_european_exercise() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));

        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 360, false).unwrap());
        let mut option = BasketOption::new(
            MinBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 100.0)),
            exercise,
            Shared::clone(&settings),
        );
        set_stulz_engine(&mut option, p1, p2, 0.5).unwrap();
        assert!(option.npv().is_err());
    }

    #[test]
    fn test_stulz_rejects_non_min_max_payoff() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));

        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 360));
        let mut option = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 100.0)),
            exercise,
            Shared::clone(&settings),
        );
        set_stulz_engine(&mut option, p1, p2, 0.5).unwrap();
        assert!(option.npv().is_err());
    }

    #[test]
    fn test_stulz_rejects_out_of_bounds_correlation() {
        let today = Date::new(15, Month::May, 1998);
        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.05))),
            flat_vol(today, &shared(SimpleQuote::new(0.30))),
        ));

        assert!(StulzEngine::new(Shared::clone(&p1), Shared::clone(&p2), 1.5).is_err());
        assert!(StulzEngine::new(p1, p2, -1.5).is_err());
    }
}
