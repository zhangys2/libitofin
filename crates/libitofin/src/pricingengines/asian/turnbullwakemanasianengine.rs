//! Turnbull–Wakeman moment-matching Asian option engine.
//!
//! Port of `ql/pricingengines/asian/turnbullwakemanasianengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults, StrikedTypePayoff,
    TypePayoff,
};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::blackcalculator::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::types::Real;

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Turnbull–Wakeman two-moment Asian engine for discrete arithmetic averages.
pub struct TurnbullWakemanAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl TurnbullWakemanAsianEngine {
    /// `TurnbullWakemanAsianEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for TurnbullWakemanAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for TurnbullWakemanAsianEngine {
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
        let args = self.base.arguments();
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not a European Option"
        );
        require!(
            args.average_type == Some(AverageType::Arithmetic),
            "must be Arithmetic Average::Type"
        );

        let past_fixings = args.past_fixings.expect("validated");
        let fixing_dates = args.fixing_dates.clone();
        let future_fixings = fixing_dates.len();
        let running_accumulator = args.running_accumulator.expect("validated");
        let payoff = args.payoff.expect("validated");
        let m = past_fixings + future_fixings;
        require!(m > 0, "no fixings given");

        let accrued_average = if past_fixings != 0 {
            running_accumulator / m as Real
        } else {
            0.0
        };

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let discount = r_ts.discount_date(exercise.last_date(), false)?;
        let effective_strike = payoff.strike() - accrued_average;

        if effective_strike <= 0.0 {
            let (value, delta) = match payoff.option_type() {
                OptionType::Call => {
                    let spot = self.process.state_variable().current_link()?.value()?;
                    let mut s_a_hat = accrued_average;
                    for &fd in &fixing_dates {
                        let fwd = spot * q_ts.discount_date(fd, false)?
                            / r_ts.discount_date(fd, false)?;
                        s_a_hat += fwd / m as Real;
                    }
                    (
                        discount * (s_a_hat - payoff.strike()),
                        discount * (s_a_hat - accrued_average) / spot,
                    )
                }
                OptionType::Put => (0.0, 0.0),
            };
            let results = self.base.results_mut();
            results.instrument.value = Some(value);
            results.greeks.delta = Some(delta);
            results.greeks.gamma = Some(0.0);
            return Ok(());
        }

        require!(
            effective_strike > 0.0,
            "expected effectiveStrike to be positive"
        );

        let spot = self.process.state_variable().current_link()?.value()?;
        let mut forwards = Vec::with_capacity(future_fixings);
        let mut times = Vec::with_capacity(future_fixings);
        let mut spot_vars = Vec::with_capacity(future_fixings);
        let mut ea = 0.0;

        for &fd in &fixing_dates {
            let dividend_discount = q_ts.discount_date(fd, false)?;
            let risk_free_discount = r_ts.discount_date(fd, false)?;
            let forward = spot * dividend_discount / risk_free_discount;
            let t = vol_ts.time_from_reference(fd)?;
            let variance = vol_ts.black_variance(t, effective_strike, false)?;
            forwards.push(forward);
            times.push(t);
            spot_vars.push(variance);
            ea += forward;
        }
        ea /= m as Real;

        let n = forwards.len();
        let mut ea2 = 0.0;
        for i in 0..n {
            ea2 += forwards[i] * forwards[i] * spot_vars[i].exp();
            for j in 0..i {
                ea2 += 2.0 * forwards[i] * forwards[j] * spot_vars[j].exp();
            }
        }
        ea2 /= (m * m) as Real;

        let tn = *times.last().expect("future fixings non-empty when strike > 0");
        let sigma = (ea2 / (ea * ea)).ln().sqrt() / tn.sqrt();

        let black = BlackCalculator::new(
            payoff.option_type(),
            effective_strike,
            ea,
            sigma * tn.sqrt(),
            discount,
        )?;
        let value = black.value();
        let delta = black.delta(spot)?;
        let gamma = black.gamma(spot)?;

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks.delta = Some(delta);
        results.greeks.gamma = Some(gamma);
        Ok(())
    }
}

/// Attaches [`TurnbullWakemanAsianEngine`] to `option`.
pub fn set_turnbull_wakeman_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(TurnbullWakemanAsianEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::{DiscreteAveragingAsianOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::test_market::time_to_days;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{
        BlackConstantVol, BlackVarianceCurve, BlackVolTermStructure,
    };
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Real, Size, Volatility};

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::with_rate(
                reference,
                rate,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    fn flat_vol(reference: Date, vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::new(
                reference,
                None,
                vol,
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        )
    }

    /// `asianoptions.cpp` `testPastFixingsModelDependency`.
    #[test]
    fn past_fixings_model_dependency_guaranteed_exercise() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_ts = flat_rate(today, 0.03);
        let r_ts = flat_rate(today, 0.06);
        let vol_ts = flat_vol(today, 0.20);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            Handle::clone(&q_ts),
            Handle::clone(&r_ts),
            vol_ts,
        ));

        let fixing_dates = vec![
            today - Period::new(6, TimeUnit::Weeks),
            today - Period::new(2, TimeUnit::Weeks),
            today + Period::new(2, TimeUnit::Weeks),
            today + Period::new(6, TimeUnit::Weeks),
        ];
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today + Period::new(6, TimeUnit::Weeks)));
        let all_past = vec![100.0, 100.0];

        let mut call = DiscreteAveragingAsianOption::with_all_past_fixings(
            AverageType::Arithmetic,
            fixing_dates.clone(),
            PlainVanillaPayoff::new(OptionType::Call, 20.0),
            Shared::clone(&exercise),
            all_past.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        let mut put = DiscreteAveragingAsianOption::with_all_past_fixings(
            AverageType::Arithmetic,
            fixing_dates.clone(),
            PlainVanillaPayoff::new(OptionType::Put, 20.0),
            Shared::clone(&exercise),
            all_past,
            Shared::clone(&settings),
        )
        .unwrap();

        set_turnbull_wakeman_asian_engine(&mut call, Shared::clone(&process));
        set_turnbull_wakeman_asian_engine(&mut put, Shared::clone(&process));

        let q = q_ts.current_link().unwrap();
        let r = r_ts.current_link().unwrap();
        let expected_call = r.discount_date(exercise.last_date(), false).unwrap()
            * ((100.0
                + 100.0
                + 100.0 * q.discount_date(fixing_dates[2], false).unwrap()
                    / r.discount_date(fixing_dates[2], false).unwrap()
                + 100.0 * q.discount_date(fixing_dates[3], false).unwrap()
                    / r.discount_date(fixing_dates[3], false).unwrap())
                / fixing_dates.len() as Real
                - 20.0);

        let call_npv = call.npv().unwrap();
        let put_npv = put.npv().unwrap();
        assert_eq!(call_npv, expected_call);
        assert_eq!(put_npv, 0.0);

        let d_s = 0.001;
        let call_price = call_npv;
        let put_price = put_npv;
        let call_delta = call.delta().unwrap();
        let call_gamma = call.gamma().unwrap();
        let put_delta = put.delta().unwrap();
        let put_gamma = put.gamma().unwrap();

        let process_up = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0 + d_s))),
            Handle::clone(&q_ts),
            Handle::clone(&r_ts),
            flat_vol(today, 0.20),
        ));
        let process_down = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0 - d_s))),
            Handle::clone(&q_ts),
            Handle::clone(&r_ts),
            flat_vol(today, 0.20),
        ));

        set_turnbull_wakeman_asian_engine(&mut call, Shared::clone(&process_up));
        set_turnbull_wakeman_asian_engine(&mut put, Shared::clone(&process_up));
        let call_up = call.npv().unwrap();
        let put_up = put.npv().unwrap();

        set_turnbull_wakeman_asian_engine(&mut call, Shared::clone(&process_down));
        set_turnbull_wakeman_asian_engine(&mut put, Shared::clone(&process_down));
        let call_down = call.npv().unwrap();
        let put_down = put.npv().unwrap();

        let tol = 1.0e-8;
        let call_delta_bump = (call_up - call_down) / (2.0 * d_s);
        let call_gamma_bump = (call_up + call_down - 2.0 * call_price) / (d_s * d_s);
        let put_delta_bump = (put_up - put_down) / (2.0 * d_s);
        let put_gamma_bump = (put_up + put_down - 2.0 * put_price) / (d_s * d_s);

        assert!(
            (call_delta_bump - call_delta).abs() <= tol,
            "call delta analytic={call_delta} bump={call_delta_bump}"
        );
        assert!(
            (call_gamma_bump - call_gamma).abs() <= tol,
            "call gamma analytic={call_gamma} bump={call_gamma_bump}"
        );
        assert!(
            (put_delta_bump - put_delta).abs() <= tol,
            "put delta analytic={put_delta} bump={put_delta_bump}"
        );
        assert!(
            (put_gamma_bump - put_gamma).abs() <= tol,
            "put gamma analytic={put_gamma} bump={put_gamma_bump}"
        );
    }

    #[derive(Clone, Copy)]
    enum Slope {
        Flat,
        Up,
        Down,
    }

    struct Case {
        option_type: OptionType,
        strike: Real,
        slope: Slope,
        result: Real,
    }

    /// Haug Table 4-28 via `testTurnbullWakemanAsianEngine` (NPV @ 2.5e-3;
    /// δ/γ vs bump @ 1e-6).
    #[test]
    fn turnbull_wakeman_matches_haug_table() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            Case {
                option_type: OptionType::Call,
                strike: 80.0,
                slope: Slope::Flat,
                result: 19.5152,
            },
            Case {
                option_type: OptionType::Call,
                strike: 80.0,
                slope: Slope::Up,
                result: 19.5063,
            },
            Case {
                option_type: OptionType::Call,
                strike: 80.0,
                slope: Slope::Down,
                result: 19.5885,
            },
            Case {
                option_type: OptionType::Put,
                strike: 80.0,
                slope: Slope::Flat,
                result: 0.0090,
            },
            Case {
                option_type: OptionType::Put,
                strike: 80.0,
                slope: Slope::Up,
                result: 0.0001,
            },
            Case {
                option_type: OptionType::Put,
                strike: 80.0,
                slope: Slope::Down,
                result: 0.0823,
            },
            Case {
                option_type: OptionType::Call,
                strike: 90.0,
                slope: Slope::Flat,
                result: 10.1437,
            },
            Case {
                option_type: OptionType::Call,
                strike: 90.0,
                slope: Slope::Up,
                result: 9.8313,
            },
            Case {
                option_type: OptionType::Call,
                strike: 90.0,
                slope: Slope::Down,
                result: 10.7062,
            },
            Case {
                option_type: OptionType::Put,
                strike: 90.0,
                slope: Slope::Flat,
                result: 0.3906,
            },
            Case {
                option_type: OptionType::Put,
                strike: 90.0,
                slope: Slope::Up,
                result: 0.0782,
            },
            Case {
                option_type: OptionType::Put,
                strike: 90.0,
                slope: Slope::Down,
                result: 0.9531,
            },
            Case {
                option_type: OptionType::Call,
                strike: 100.0,
                slope: Slope::Flat,
                result: 3.2700,
            },
            Case {
                option_type: OptionType::Call,
                strike: 100.0,
                slope: Slope::Up,
                result: 2.2819,
            },
            Case {
                option_type: OptionType::Call,
                strike: 100.0,
                slope: Slope::Down,
                result: 4.3370,
            },
            Case {
                option_type: OptionType::Put,
                strike: 100.0,
                slope: Slope::Flat,
                result: 3.2700,
            },
            Case {
                option_type: OptionType::Put,
                strike: 100.0,
                slope: Slope::Up,
                result: 2.2819,
            },
            Case {
                option_type: OptionType::Put,
                strike: 100.0,
                slope: Slope::Down,
                result: 4.3370,
            },
            Case {
                option_type: OptionType::Call,
                strike: 110.0,
                slope: Slope::Flat,
                result: 0.5515,
            },
            Case {
                option_type: OptionType::Call,
                strike: 110.0,
                slope: Slope::Up,
                result: 0.1314,
            },
            Case {
                option_type: OptionType::Call,
                strike: 110.0,
                slope: Slope::Down,
                result: 1.2429,
            },
            Case {
                option_type: OptionType::Put,
                strike: 110.0,
                slope: Slope::Flat,
                result: 10.3046,
            },
            Case {
                option_type: OptionType::Put,
                strike: 110.0,
                slope: Slope::Up,
                result: 9.8845,
            },
            Case {
                option_type: OptionType::Put,
                strike: 110.0,
                slope: Slope::Down,
                result: 10.9960,
            },
            Case {
                option_type: OptionType::Call,
                strike: 120.0,
                slope: Slope::Flat,
                result: 0.0479,
            },
            Case {
                option_type: OptionType::Call,
                strike: 120.0,
                slope: Slope::Up,
                result: 0.0016,
            },
            Case {
                option_type: OptionType::Call,
                strike: 120.0,
                slope: Slope::Down,
                result: 0.2547,
            },
            Case {
                option_type: OptionType::Put,
                strike: 120.0,
                slope: Slope::Flat,
                result: 19.5541,
            },
            Case {
                option_type: OptionType::Put,
                strike: 120.0,
                slope: Slope::Up,
                result: 19.5078,
            },
            Case {
                option_type: OptionType::Put,
                strike: 120.0,
                slope: Slope::Down,
                result: 19.7609,
            },
        ];

        let first = 1.0 / 52.0;
        let expiry = 0.5;
        let fixings: Size = 26;
        let risk_free = 0.05;
        let base_vol = 0.2;
        let vol_slope = 0.005;
        let npv_tol = 2.5e-3;
        let greek_tol = 1.0e-6;
        let d_s = 0.001;

        for (i, case) in cases.iter().enumerate() {
            let dt = (expiry - first) / (fixings as Real - 1.0);
            let mut fixing_dates = Vec::with_capacity(fixings);
            fixing_dates.push(today + time_to_days(first));
            for j in 1..fixings {
                fixing_dates.push(today + time_to_days(j as Real * dt + first));
            }
            let maturity = today + time_to_days(expiry);

            let q_ts = flat_rate(today, risk_free);
            let r_ts = flat_rate(today, risk_free);
            let vol_ts: Handle<dyn BlackVolTermStructure> = match case.slope {
                Slope::Flat => flat_vol(today, base_vol),
                Slope::Up => {
                    let vols: Vec<Volatility> = (0..fixings)
                        .map(|k| {
                            base_vol - (fixings as Real - 1.0) * vol_slope + k as Real * vol_slope
                        })
                        .collect();
                    Handle::new(
                        shared(
                            BlackVarianceCurve::new(
                                today,
                                &fixing_dates,
                                &vols,
                                Actual360::new(),
                                true,
                            )
                            .unwrap(),
                        ) as Shared<dyn BlackVolTermStructure>,
                    )
                }
                Slope::Down => {
                    let vols: Vec<Volatility> = (0..fixings)
                        .map(|k| {
                            base_vol + (fixings as Real - 1.0) * vol_slope - k as Real * vol_slope
                        })
                        .collect();
                    Handle::new(
                        shared(
                            BlackVarianceCurve::new(
                                today,
                                &fixing_dates,
                                &vols,
                                Actual360::new(),
                                false,
                            )
                            .unwrap(),
                        ) as Shared<dyn BlackVolTermStructure>,
                    )
                }
            };

            let spot = shared(SimpleQuote::new(100.0));
            let process = shared(BlackScholesMertonProcess::new(
                quote_handle(&spot),
                q_ts,
                r_ts,
                Handle::clone(&vol_ts),
            ));

            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Arithmetic,
                0.0,
                0,
                fixing_dates,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                shared(EuropeanExercise::new(maturity)),
                Shared::clone(&settings),
            )
            .unwrap();
            set_turnbull_wakeman_asian_engine(&mut option, Shared::clone(&process));

            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= npv_tol,
                "case {i}: expected {}, got {calculated}",
                case.result
            );

            let delta = option.delta().unwrap();
            let gamma = option.gamma().unwrap();

            let process_up = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(100.0 + d_s))),
                flat_rate(today, risk_free),
                flat_rate(today, risk_free),
                Handle::clone(&vol_ts),
            ));
            let process_down = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(100.0 - d_s))),
                flat_rate(today, risk_free),
                flat_rate(today, risk_free),
                Handle::clone(&vol_ts),
            ));

            set_turnbull_wakeman_asian_engine(&mut option, process_up);
            let up = option.npv().unwrap();
            set_turnbull_wakeman_asian_engine(&mut option, process_down);
            let down = option.npv().unwrap();

            let delta_bump = (up - down) / (2.0 * d_s);
            let gamma_bump = (up + down - 2.0 * calculated) / (d_s * d_s);
            assert!(
                (delta_bump - delta).abs() <= greek_tol,
                "case {i} delta: analytic={delta} bump={delta_bump}"
            );
            assert!(
                (gamma_bump - gamma).abs() <= greek_tol,
                "case {i} gamma: analytic={gamma} bump={gamma_bump}"
            );
        }
    }
}
