//! Vecer engine for continuous arithmetic-average Asian options.
//!
//! Port of `ql/experimental/exoticoptions/continuousarithmeticasianvecerengine.{hpp,cpp}`
//! (see <http://www.stat.columbia.edu/~vecer/asian-vecer.pdf>).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults,
    StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::methods::finitedifferences::TridiagonalOperator;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time};

type EngineBase = GenericEngine<ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults>;

/// Vecer continuous arithmetic Asian engine (FD on the control-variate state).
pub struct ContinuousArithmeticAsianVecerEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    current_average: Handle<dyn Quote>,
    start_date: Date,
    time_steps: Size,
    asset_steps: Size,
    z_min: Real,
    z_max: Real,
}

impl ContinuousArithmeticAsianVecerEngine {
    /// `ContinuousArithmeticAsianVecerEngine(process, currentAverage, startDate, ...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        current_average: Handle<dyn Quote>,
        start_date: Date,
        time_steps: Size,
        asset_steps: Size,
        z_min: Real,
        z_max: Real,
    ) -> Self {
        let base = EngineBase::new(
            ContinuousAveragingAsianArguments::default(),
            ContinuousAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        current_average.register_observer(&base.observer());
        Self {
            base,
            process,
            current_average,
            start_date,
            time_steps,
            asset_steps,
            z_min,
            z_max,
        }
    }

    /// Defaults: 100 time / asset steps, `z ∈ [-1, 1]`.
    pub fn with_defaults(
        process: Shared<GeneralizedBlackScholesProcess>,
        current_average: Handle<dyn Quote>,
        start_date: Date,
    ) -> Self {
        Self::new(process, current_average, start_date, 100, 100, -1.0, 1.0)
    }

    /// Replication of the average by holding this amount in assets.
    fn cont_strategy(t: Time, t1: Time, t2: Time, v: Real, r: Real) -> QlResult<Real> {
        const EPS: Real = 0.00001;
        require!(t1 <= t2, "Average Start must be before Average End");
        if (t - t2).abs() < EPS {
            return Ok(0.0);
        }
        if t < t1 {
            if (r - v).abs() >= EPS {
                Ok((v * (t - t2)).exp() * (1.0 - ((v - r) * (t2 - t1)).exp())
                    / ((r - v) * (t2 - t1)))
            } else {
                Ok((v * (t - t2)).exp())
            }
        } else if (r - v).abs() >= EPS {
            Ok((v * (t - t2)).exp() * (1.0 - ((v - r) * (t2 - t)).exp())
                / ((r - v) * (t2 - t1)))
        } else {
            Ok((v * (t - t2)).exp() * (t2 - t) / (t2 - t1))
        }
    }
}

impl AsObservable for ContinuousArithmeticAsianVecerEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for ContinuousArithmeticAsianVecerEngine {
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
        require!(
            args.average_type == Some(AverageType::Arithmetic),
            "not an Arithmetic average option"
        );
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );
        let payoff = args.payoff.expect("validated");

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let s0 = self.process.state_variable().current_link()?.value()?;

        let maturity = exercise.last_date();
        let strike = payoff.strike();
        require!(
            self.z_min <= 0.0 && self.z_max >= 0.0,
            "strike (0 for vecer fixed strike asian)  not on Grid"
        );

        let sigma = vol_ts.black_vol_date(maturity, strike, false)?;
        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let r = r_ts
            .zero_rate_date(
                maturity,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q = q_ts
            .zero_rate_date(
                maturity,
                divdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();

        let today = r_ts.reference_date()?;
        require!(
            self.start_date >= today,
            "Seasoned Asian not yet implemented"
        );

        let t = rfdc.year_fraction(today, maturity);
        let t1 = rfdc.year_fraction(today, self.start_date);
        let t2 = t;

        let value = if (t2 - t1) < 0.001 {
            let payoff_shared: Shared<dyn StrikedTypePayoff> = shared(payoff);
            let mut european = AnalyticEuropeanEngine::new(Shared::clone(&self.process));
            let results = european.calculate_from_arguments(payoff_shared, Shared::clone(&exercise))?;
            results
                .instrument
                .value
                .expect("analytic european NPV")
        } else {
            let theta = 0.5; // Crank–Nicolson
            let z0 = Self::cont_strategy(0.0, t1, t2, q, r)? - (-r * t).exp() * strike / s0;
            require!(
                z0 >= self.z_min && z0 <= self.z_max,
                "spot not on grid"
            );

            let h = (self.z_max - self.z_min) / self.asset_steps as Real;
            let k = t / self.time_steps as Real;
            let sigma2 = sigma * sigma;

            let n_grid = self.asset_steps + 1;
            let mut s_vec = Array::with_size(n_grid);
            for i in 0..n_grid {
                s_vec[i] = self.z_min + i as Real * h;
            }

            let mut gamma_op = TridiagonalOperator::with_size(n_grid)?;
            gamma_op.set_first_row(0.0, 0.0);
            gamma_op.set_mid_rows(1.0 / (h * h), -2.0 / (h * h), 1.0 / (h * h));
            gamma_op.set_last_row(0.0, 0.0);

            let upper_d = gamma_op.upper_diagonal().clone();
            let lower_d = gamma_op.lower_diagonal().clone();
            let dia = gamma_op.diagonal().clone();

            let mut u = Array::with_size(n_grid);
            for i in 0..n_grid {
                u[i] = s_vec[i].max(0.0); // call payoff
            }

            for j in 1..=self.time_steps {
                if theta != 1.0 {
                    for i in 1..=n_grid - 2 {
                        let vecer_term = s_vec[i]
                            - (-q * (t - (j - 1) as Real * k)).exp()
                                * Self::cont_strategy(t - (j - 1) as Real * k, t1, t2, q, r)?;
                        let coef = 0.5 * sigma2 * vecer_term * vecer_term;
                        gamma_op.set_mid_row(
                            i,
                            coef * lower_d[i - 1],
                            coef * dia[i],
                            coef * upper_d[i],
                        )?;
                    }
                    let identity = TridiagonalOperator::identity(n_grid)?;
                    let explicit_scale = (1.0 - theta) * k * &gamma_op;
                    let mut explicit_part = &identity + &explicit_scale;
                    explicit_part.set_first_row(1.0, 0.0);
                    explicit_part.set_last_row(-1.0, 1.0);
                    u = explicit_part.apply_to(&u)?;
                    u[self.asset_steps] = u[self.asset_steps - 1] + h;
                    u[0] = 0.0;
                }

                if theta != 0.0 {
                    for i in 1..=n_grid - 2 {
                        let vecer_term = s_vec[i]
                            - (-q * (t - j as Real * k)).exp()
                                * Self::cont_strategy(t - j as Real * k, t1, t2, q, r)?;
                        let coef = 0.5 * sigma2 * vecer_term * vecer_term;
                        gamma_op.set_mid_row(
                            i,
                            coef * lower_d[i - 1],
                            coef * dia[i],
                            coef * upper_d[i],
                        )?;
                    }
                    let identity = TridiagonalOperator::identity(n_grid)?;
                    let implicit_scale = theta * k * &gamma_op;
                    let mut implicit_part = &identity - &implicit_scale;
                    implicit_part.set_first_row(1.0, 0.0);
                    implicit_part.set_last_row(-1.0, 1.0);
                    let mut rhs = u.clone();
                    rhs[0] = 0.0;
                    rhs[self.asset_steps] = h;
                    u = implicit_part.solve_for(&rhs)?;
                }
            }

            // DownRounding(0): floor toward −∞
            let lower_i = ((z0 - self.z_min) / h).floor() as Size;
            let pv = u[lower_i] + (u[lower_i + 1] - u[lower_i]) * (z0 - s_vec[lower_i]) / h;
            let mut value = s0 * pv;

            if payoff.option_type() == OptionType::Put {
                let expected_average = if r == q {
                    s0
                } else {
                    s0 * (((r - q) * t2).exp() - ((r - q) * t1).exp()) / ((r - q) * (t2 - t1))
                };
                let asian_forward = (-r * t2).exp() * (expected_average - strike);
                value -= asian_forward;
            }
            // current_average is registered for notifications; unseasoned path does not read it.
            let _ = self.current_average.is_empty();
            value
        };

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`ContinuousArithmeticAsianVecerEngine`] to `option`.
#[allow(clippy::too_many_arguments)]
pub fn set_continuous_arithmetic_asian_vecer_engine(
    option: &mut crate::instruments::ContinuousAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    current_average: Handle<dyn Quote>,
    start_date: Date,
    time_steps: Size,
    asset_steps: Size,
    z_min: Real,
    z_max: Real,
) {
    let engine = shared_mut(ContinuousArithmeticAsianVecerEngine::new(
        process,
        current_average,
        start_date,
        time_steps,
        asset_steps,
        z_min,
        z_max,
    )) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::instruments::{ContinuousAveragingAsianOption, PlainVanillaPayoff};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::date::SerialNumber;
    use crate::types::{Natural, Volatility};

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
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

    struct Case {
        spot: Real,
        risk_free_rate: Real,
        volatility: Real,
        strike: Real,
        length: Natural,
        result: Real,
        tolerance: Real,
    }

    /// `asianoptions.cpp` `testVecerEngine`.
    #[test]
    fn continuous_arithmetic_asian_vecer_matches_published() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            Case {
                spot: 1.9,
                risk_free_rate: 0.05,
                volatility: 0.5,
                strike: 2.0,
                length: 1,
                result: 0.193174,
                tolerance: 1.0e-5,
            },
            Case {
                spot: 2.0,
                risk_free_rate: 0.05,
                volatility: 0.5,
                strike: 2.0,
                length: 1,
                result: 0.246416,
                tolerance: 1.0e-5,
            },
            Case {
                spot: 2.1,
                risk_free_rate: 0.05,
                volatility: 0.5,
                strike: 2.0,
                length: 1,
                result: 0.306220,
                tolerance: 1.0e-4,
            },
            Case {
                spot: 2.0,
                risk_free_rate: 0.02,
                volatility: 0.1,
                strike: 2.0,
                length: 1,
                result: 0.055986,
                tolerance: 2.0e-4,
            },
            Case {
                spot: 2.0,
                risk_free_rate: 0.18,
                volatility: 0.3,
                strike: 2.0,
                length: 1,
                result: 0.218388,
                tolerance: 1.0e-4,
            },
            Case {
                spot: 2.0,
                risk_free_rate: 0.0125,
                volatility: 0.25,
                strike: 2.0,
                length: 2,
                result: 0.172269,
                tolerance: 1.0e-4,
            },
            Case {
                spot: 2.0,
                risk_free_rate: 0.05,
                volatility: 0.5,
                strike: 2.0,
                length: 2,
                result: 0.350095,
                tolerance: 2.0e-4,
            },
        ];

        let q = flat_rate(today, 0.0);
        let time_steps = 200;
        let asset_steps = 200;

        for case in &cases {
            let spot = shared(SimpleQuote::new(case.spot));
            let r = flat_rate(today, case.risk_free_rate);
            let sigma = flat_vol(today, case.volatility);
            let process = shared(BlackScholesMertonProcess::new(
                quote_handle(&spot),
                q.clone(),
                r,
                sigma,
            ));

            let maturity = today + (case.length as SerialNumber * 360);
            let exercise = shared(EuropeanExercise::new(maturity));
            let payoff = PlainVanillaPayoff::new(OptionType::Call, case.strike);
            let average = shared(SimpleQuote::new(0.0));

            let mut option = ContinuousAveragingAsianOption::new(
                AverageType::Arithmetic,
                payoff,
                Shared::clone(&exercise) as Shared<dyn Exercise>,
                Shared::clone(&settings),
            )
            .unwrap();
            set_continuous_arithmetic_asian_vecer_engine(
                &mut option,
                Shared::clone(&process) as Shared<GeneralizedBlackScholesProcess>,
                quote_handle(&average),
                today,
                time_steps,
                asset_steps,
                -1.0,
                1.0,
            );

            let calculated = option.npv().unwrap();
            let error = (calculated - case.result).abs();
            assert!(
                error <= case.tolerance,
                "spot={} r={} vol={} T={}y: calculated={calculated} expected={} error={error} tol={}",
                case.spot,
                case.risk_free_rate,
                case.volatility,
                case.length,
                case.result,
                case.tolerance
            );
        }
    }
}
