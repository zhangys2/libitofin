//! Analytic double-barrier binary (cash-or-nothing) option engine.
//!
//! Port of `ql/pricingengines/barrier/analyticdoublebarrierbinaryengine.{hpp,cpp}`:
//! implements C.H. Hui series ("One-Touch Double Barrier Binary Option Values",
//! *Applied Financial Economics* 6/1996), described in E.G. Haug,
//! *Complete guide to option pricing formulas* 2nd Ed., McGraw-Hill 2007, p.180.
//!
//! The Knock-In part of KIKO and KOKI options pays at hit, while Double Knock-In
//! pays at end. This engine requires European exercise for Double Knock options
//! (Knock-In / Knock-Out), and American exercise for KIKO / KOKI.

use std::any::Any;
use std::f64::consts::PI;

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{
    CashOrNothingPayoff, DoubleBarrierArguments, DoubleBarrierOption, DoubleBarrierType,
    StrikedTypePayoff,
};
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::Real;

type DoubleBarrierEngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// Analytic pricing engine for double barrier binary (cash-or-nothing) options.
pub struct AnalyticDoubleBarrierBinaryEngine {
    base: DoubleBarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    max_iterations: usize,
    required_convergence: Real,
}

impl AnalyticDoubleBarrierBinaryEngine {
    /// Builds the engine with QuantLib default convergence settings:
    /// 100 iterations for KnockIn / KnockOut, 1000 iterations for KIKO / KOKI,
    /// and required convergence tolerance `1e-8`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_options(process, 0, 1e-8)
    }

    /// Builds the engine with custom maximum iterations and convergence tolerance.
    /// If `max_iterations == 0`, defaults are used (100 for expiry, 1000 for KIKO/KOKI).
    pub fn with_options(
        process: Shared<GeneralizedBlackScholesProcess>,
        max_iterations: usize,
        required_convergence: Real,
    ) -> Self {
        let base = DoubleBarrierEngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            max_iterations,
            required_convergence,
        }
    }
}

impl AsObservable for AnalyticDoubleBarrierBinaryEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDoubleBarrierBinaryEngine {
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
        let barrier_type = arguments
            .barrier_type
            .ok_or_else(|| QlError::new("no double-barrier type", file!(), line!()))?;

        let exercise = arguments
            .exercise
            .as_ref()
            .ok_or_else(|| QlError::new("no exercise given", file!(), line!()))?;

        match barrier_type {
            DoubleBarrierType::KIKO | DoubleBarrierType::KOKI => {
                require!(
                    exercise.exercise_type() == ExerciseType::American,
                    "KIKO/KOKI options must have American exercise"
                );
                let vol_ts = self.process.black_volatility().current_link()?;
                let dates = exercise.dates();
                require!(
                    !dates.is_empty() && dates[0] <= vol_ts.reference_date()?,
                    "American option with window exercise not handled yet"
                );
            }
            DoubleBarrierType::KnockIn | DoubleBarrierType::KnockOut => {
                require!(
                    exercise.exercise_type() == ExerciseType::European,
                    "non-European exercise given"
                );
            }
        }

        let (cash_payoff, strike) = if let Some(ref binary) = arguments.binary_payoff {
            if let Some(coo) = (binary.as_ref() as &dyn Any).downcast_ref::<CashOrNothingPayoff>() {
                (coo.cash_payoff(), coo.strike())
            } else {
                fail!("a cash-or-nothing payoff must be given");
            }
        } else {
            fail!("a cash-or-nothing payoff must be given");
        };

        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

        let barrier_lo = arguments
            .barrier_lo
            .ok_or_else(|| QlError::new("no low barrier given", file!(), line!()))?;
        let barrier_hi = arguments
            .barrier_hi
            .ok_or_else(|| QlError::new("no high barrier given", file!(), line!()))?;

        require!(barrier_lo > 0.0, "positive low barrier value required");
        require!(barrier_hi > 0.0, "positive high barrier value required");
        require!(barrier_lo < barrier_hi, "barrier_lo must be < barrier_hi");

        // Degenerate cases
        match barrier_type {
            DoubleBarrierType::KnockOut => {
                if spot <= barrier_lo || spot >= barrier_hi {
                    self.set_terminal_result(0.0);
                    return Ok(());
                }
            }
            DoubleBarrierType::KnockIn => {
                if spot <= barrier_lo || spot >= barrier_hi {
                    self.set_terminal_result(cash_payoff);
                    return Ok(());
                }
            }
            DoubleBarrierType::KIKO => {
                if spot >= barrier_hi {
                    self.set_terminal_result(0.0);
                    return Ok(());
                } else if spot <= barrier_lo {
                    self.set_terminal_result(cash_payoff);
                    return Ok(());
                }
            }
            DoubleBarrierType::KOKI => {
                if spot <= barrier_lo {
                    self.set_terminal_result(0.0);
                    return Ok(());
                } else if spot >= barrier_hi {
                    self.set_terminal_result(cash_payoff);
                    return Ok(());
                }
            }
        }

        let maturity = exercise.last_date();
        let vol_ts = self.process.black_volatility().current_link()?;
        let variance = vol_ts.black_variance_date(maturity, strike, false)?;
        require!(variance >= 0.0, "negative variance not allowed");

        let residual_time = self.process.time(&maturity)?;
        require!(residual_time > 0.0, "expiration time must be > 0");

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;

        let r = r_ts
            .zero_rate(
                residual_time,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q = q_ts
            .zero_rate(
                residual_time,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();

        let discount = r_ts.discount(residual_time, false)?;
        require!(discount > 0.0, "positive discount required");

        let max_iter = if self.max_iterations > 0 {
            Some(self.max_iterations)
        } else {
            None
        };
        let req_conv = Some(self.required_convergence);

        let value = analytic_double_barrier_binary_value(
            barrier_type,
            barrier_lo,
            barrier_hi,
            cash_payoff,
            spot,
            variance,
            residual_time,
            r,
            q,
            discount,
            max_iter,
            req_conv,
        )?;

        let results = self.base.results_mut();
        results.value = Some(value);

        Ok(())
    }
}

impl AnalyticDoubleBarrierBinaryEngine {
    fn set_terminal_result(&mut self, value: Real) {
        let results = self.base.results_mut();
        results.value = Some(value);
    }
}

/// Standalone evaluation of double-barrier binary option price (Hui 1996 / Haug 2007).
#[allow(clippy::too_many_arguments)]
pub fn analytic_double_barrier_binary_value(
    barrier_type: DoubleBarrierType,
    barrier_lo: Real,
    barrier_hi: Real,
    cash: Real,
    spot: Real,
    variance: Real,
    residual_time: Real,
    r: Real,
    q: Real,
    discount: Real,
    max_iterations: Option<usize>,
    required_convergence: Option<Real>,
) -> QlResult<Real> {
    require!(spot > 0.0, "positive spot value required");
    require!(variance >= 0.0, "negative variance not allowed");
    require!(residual_time > 0.0, "expiration time must be > 0");
    require!(barrier_lo > 0.0, "positive low barrier value required");
    require!(barrier_hi > 0.0, "positive high barrier value required");

    let tol = required_convergence.unwrap_or(1e-8);
    let sigmaq = variance / residual_time;
    let b = r - q;

    match barrier_type {
        DoubleBarrierType::KnockOut | DoubleBarrierType::KnockIn => {
            require!(barrier_lo < barrier_hi, "barrier_lo must be < barrier_hi");
            require!(discount > 0.0, "positive discount required");

            let alpha = -0.5 * (2.0 * b / sigmaq - 1.0);
            let beta = -0.25 * (2.0 * b / sigmaq - 1.0).powi(2) - 2.0 * r / sigmaq;
            let z = (barrier_hi / barrier_lo).ln();
            let factor = 2.0 * PI * cash / (z * z);
            let lo_alpha = (spot / barrier_lo).powf(alpha);
            let hi_alpha = (spot / barrier_hi).powf(alpha);

            let max_iter = max_iterations.unwrap_or(100);
            let mut tot = 0.0;
            let mut last_term = 0.0;

            for i in 1..max_iter {
                let i_real = i as Real;
                let i_pi_over_z = i_real * PI / z;
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let term1 =
                    (lo_alpha - sign * hi_alpha) / (alpha * alpha + i_pi_over_z * i_pi_over_z);
                let term2 = (i_pi_over_z * (spot / barrier_lo).ln()).sin();
                let term3 = (-0.5 * (i_pi_over_z * i_pi_over_z - beta) * variance).exp();
                last_term = factor * i_real * term1 * term2 * term3;
                tot += last_term;
            }

            require!(
                last_term.abs() < tol,
                "serie did not converge sufficiently fast"
            );

            if barrier_type == DoubleBarrierType::KnockOut {
                Ok(tot.max(0.0))
            } else {
                Ok((cash * discount - tot).max(0.0))
            }
        }
        DoubleBarrierType::KIKO | DoubleBarrierType::KOKI => {
            let (lo, hi) = if barrier_type == DoubleBarrierType::KOKI {
                (barrier_hi, barrier_lo)
            } else {
                (barrier_lo, barrier_hi)
            };

            let alpha = -0.5 * (2.0 * b / sigmaq - 1.0);
            let beta = -0.25 * (2.0 * b / sigmaq - 1.0).powi(2) - 2.0 * r / sigmaq;
            let z = (hi / lo).ln();
            let log_s_l = (spot / lo).ln();

            let max_iter = max_iterations.unwrap_or(1000);
            let mut tot = 0.0;
            let mut last_term = 0.0;

            for i in 1..max_iter {
                let i_real = i as Real;
                let i_pi_over_z = i_real * PI / z;
                let factor = i_pi_over_z * i_pi_over_z - beta;
                let term1 =
                    (beta - i_pi_over_z * i_pi_over_z * (-0.5 * factor * variance).exp()) / factor;
                let term2 = (i_pi_over_z * log_s_l).sin();
                last_term = (2.0 / (i_real * PI)) * term1 * term2;
                tot += last_term;
            }

            tot += 1.0 - log_s_l / z;
            tot *= cash * (spot / lo).powf(alpha);

            require!(
                last_term.abs() < tol,
                "serie did not converge sufficiently fast"
            );

            Ok(tot.max(0.0))
        }
    }
}

/// Attaches an [`AnalyticDoubleBarrierBinaryEngine`] to `option`.
pub fn set_analytic_double_barrier_binary_engine(
    option: &mut DoubleBarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticDoubleBarrierBinaryEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{CashOrNothingPayoff, PlainVanillaPayoff, StrikedTypePayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::{Rate, Time, Volatility};

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    fn flat_rate(
        rate: Rate,
    ) -> Handle<dyn crate::termstructures::yieldtermstructure::YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<
                dyn crate::termstructures::yieldtermstructure::YieldTermStructure,
            >)
    }

    fn flat_vol(
        vol: Volatility,
    ) -> Handle<dyn crate::termstructures::volatility::BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::new(today(), None, vol, Actual360::new()))
                as Shared<dyn crate::termstructures::volatility::BlackVolTermStructure>,
        )
    }

    struct DoubleBinaryOptionData {
        barrier_type: DoubleBarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        cash: Real,
        s: Real,
        q: Real,
        r: Real,
        t: Real,
        v: Real,
        result: Real,
        tol: Real,
    }

    /// Replicates QuantLib `testHaugValues` from `doublebinaryoption.cpp:77-246`.
    /// 32 reference values from Haug 2nd Ed. p.181 + Haug VBA values,
    /// covering KnockOut, KIKO, KnockIn, KOKI across 4 barrier pairs and 4 vol levels,
    /// plus 8 degenerate cases (total 72 oracle cases).
    #[test]
    fn test_haug_values() {
        #[rustfmt::skip]
        let values = [
            // KnockOut
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 9.8716, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 8.9307, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 6.3272, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 1.9094, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 9.7961, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 7.2300, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 3.7100, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 0.4271, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 8.9054, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 3.6752, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 0.7960, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 0.0059, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 3.6323, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 0.0911, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 0.0002, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 0.0000, tol: 1e-4 },

            // KIKO
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 0.2402, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 1.4076, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 3.8160, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0075, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 0.9910, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 2.8098, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 4.6612, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.2656, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 2.7954, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 4.4024, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 4.9266, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 2.6285, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 4.7523, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 4.9096, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 4.9675, tol: 1e-4 },

            // KnockIn
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0042, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 0.9450, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 3.5486, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 7.9663, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0797, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 2.6458, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 6.1658, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 9.4486, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.9704, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 6.2006, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 9.0798, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 9.8699, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 6.2434, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 9.7847, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 9.8756, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 9.8758, tol: 1e-4 },

            // KOKI
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0041, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 0.7080, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 2.1581, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 80.00, barrier_hi: 120.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 4.2061, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0723, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 1.6663, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 3.3930, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 85.00, barrier_hi: 115.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 4.8679, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.7080, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 3.4424, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 4.7496, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 90.00, barrier_hi: 110.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 5.0475, tol: 1e-4 },

            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 3.6524, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.20, result: 5.1256, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.30, result: 5.0763, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 100.00, q: 0.02, r: 0.05, t: 0.25, v: 0.50, result: 5.0275, tol: 1e-4 },

            // Degenerate cases
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 80.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 110.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 80.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 10.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 110.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 10.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 80.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 10.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KIKO, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 110.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 80.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 0.0000, tol: 1e-4 },
            DoubleBinaryOptionData { barrier_type: DoubleBarrierType::KOKI, barrier_lo: 95.00, barrier_hi: 105.00, cash: 10.00, s: 110.00, q: 0.02, r: 0.05, t: 0.25, v: 0.10, result: 10.0000, tol: 1e-4 },
        ];

        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        for (idx, val) in values.iter().enumerate() {
            let spot = Handle::new(shared(SimpleQuote::new(val.s)) as Shared<dyn Quote>);
            let q_ts = flat_rate(val.q);
            let r_ts = flat_rate(val.r);
            let vol_ts = flat_vol(val.v);

            let process = shared(GeneralizedBlackScholesProcess::new(
                spot, q_ts, r_ts, vol_ts,
            ));

            let payoff: Shared<dyn StrikedTypePayoff> =
                shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, val.cash));

            let ex_date = today + time_to_days(val.t);

            let exercise: Shared<dyn Exercise> = match val.barrier_type {
                DoubleBarrierType::KIKO | DoubleBarrierType::KOKI => {
                    shared(AmericanExercise::new(today, ex_date, false).unwrap())
                }
                DoubleBarrierType::KnockIn | DoubleBarrierType::KnockOut => {
                    shared(EuropeanExercise::new(ex_date))
                }
            };

            let mut opt = DoubleBarrierOption::with_striked_payoff(
                val.barrier_type,
                val.barrier_lo,
                val.barrier_hi,
                0.0,
                payoff,
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();

            set_analytic_double_barrier_binary_engine(&mut opt, Shared::clone(&process));

            let calc = opt.npv().unwrap();
            let diff = (calc - val.result).abs();
            assert!(
                diff <= val.tol,
                "Row {idx} ({:?}, lo={}, hi={}, v={}) failed: calc={calc}, exp={}, diff={diff}, tol={}",
                val.barrier_type,
                val.barrier_lo,
                val.barrier_hi,
                val.v,
                val.result,
                val.tol
            );
        }
    }

    #[test]
    fn test_in_out_parity() {
        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        let s = 100.0;
        let q = 0.03;
        let r = 0.05;
        let v = 0.25;
        let cash = 10.0;
        let blo = 80.0;
        let bhi = 120.0;

        let spot = Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>);
        let q_ts = flat_rate(q);
        let r_ts = flat_rate(r);
        let vol_ts = flat_vol(v);

        let process = shared(GeneralizedBlackScholesProcess::new(
            spot, q_ts, r_ts, vol_ts,
        ));
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, cash));

        let ex_date = today + 90;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(ex_date));

        let mut ko_opt = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockOut,
            blo,
            bhi,
            0.0,
            payoff.clone(),
            exercise.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut ko_opt, Shared::clone(&process));

        let mut ki_opt = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockIn,
            blo,
            bhi,
            0.0,
            payoff,
            exercise,
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut ki_opt, Shared::clone(&process));

        let ko_val = ko_opt.npv().unwrap();
        let ki_val = ki_opt.npv().unwrap();

        let df = (-r * 90.0 / 360.0_f64).exp();
        let expected_sum = cash * df;
        let actual_sum = ko_val + ki_val;

        assert!(
            (actual_sum - expected_sum).abs() < 1e-10,
            "KnockOut + KnockIn parity failed: ko={ko_val}, ki={ki_val}, sum={actual_sum}, exp={expected_sum}"
        );
    }

    #[test]
    fn test_validation_rejects_invalid_inputs() {
        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        let spot = Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>);
        let q_ts = flat_rate(0.02);
        let r_ts = flat_rate(0.05);
        let vol_ts = flat_vol(0.20);

        let process = shared(GeneralizedBlackScholesProcess::new(
            spot, q_ts, r_ts, vol_ts,
        ));
        let cash_payoff: Shared<dyn StrikedTypePayoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, 10.0));
        let plain_payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);

        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 90));
        let amer_ex: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 90, false).unwrap());

        // 1. Non-cash-or-nothing payoff (e.g. plain vanilla)
        let mut opt_plain = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            plain_payoff,
            euro_ex.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut opt_plain, Shared::clone(&process));
        assert!(opt_plain.npv().is_err());

        // 2. American exercise for KnockOut (requires European)
        let mut opt_amer_ko = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            cash_payoff.clone(),
            amer_ex.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut opt_amer_ko, Shared::clone(&process));
        assert!(opt_amer_ko.npv().is_err());

        // 3. European exercise for KIKO (requires American)
        let mut opt_euro_kiko = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KIKO,
            80.0,
            120.0,
            0.0,
            cash_payoff.clone(),
            euro_ex.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut opt_euro_kiko, Shared::clone(&process));
        assert!(opt_euro_kiko.npv().is_err());

        // 4. European exercise for KOKI (requires American)
        let mut opt_euro_koki = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KOKI,
            80.0,
            120.0,
            0.0,
            cash_payoff,
            euro_ex,
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_double_barrier_binary_engine(&mut opt_euro_koki, Shared::clone(&process));
        assert!(opt_euro_koki.npv().is_err());
    }

    #[test]
    fn test_binary_option_with_vanilla_engines_fails_gracefully() {
        use crate::pricingengines::barrier::analyticdoublebarrierengine::set_analytic_double_barrier_engine;

        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        let spot = Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>);
        let q_ts = flat_rate(0.02);
        let r_ts = flat_rate(0.05);
        let vol_ts = flat_vol(0.20);
        let process = shared(GeneralizedBlackScholesProcess::new(
            spot, q_ts, r_ts, vol_ts,
        ));

        let cash_payoff: Shared<dyn StrikedTypePayoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, 10.0));
        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 90));

        let mut opt = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            cash_payoff,
            euro_ex,
            Shared::clone(&settings),
        )
        .unwrap();

        set_analytic_double_barrier_engine(&mut opt, process);
        let err = opt.npv().unwrap_err();
        assert!(
            err.to_string().contains("no plain vanilla payoff given"),
            "Expected 'no plain vanilla payoff given' error, got: {err}"
        );
    }

    #[test]
    fn test_engine_reuse_no_stale_payoff() {
        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        let plain_payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 90));

        let opt_plain = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            plain_payoff,
            euro_ex.clone(),
            Shared::clone(&settings),
        )
        .unwrap();

        let mut args = DoubleBarrierArguments::default();
        opt_plain.setup_arguments(&mut args).unwrap();
        assert!(args.payoff.is_some());
        assert!(args.binary_payoff.is_some());

        let cash_payoff: Shared<dyn StrikedTypePayoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, 10.0));
        let opt_binary = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            cash_payoff,
            euro_ex,
            Shared::clone(&settings),
        )
        .unwrap();

        opt_binary.setup_arguments(&mut args).unwrap();
        assert!(
            args.payoff.is_none(),
            "Vanilla payoff must be None for binary option"
        );
        assert!(args.binary_payoff.is_some());
    }

    #[test]
    fn test_strike_dependent_variance_lookup() {
        let settings = shared(Settings::new());
        let today = today();
        settings.set_evaluation_date(today);

        let spot = Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>);
        let q_ts = flat_rate(0.02);
        let r_ts = flat_rate(0.05);
        let vol_ts = flat_vol(0.20);
        let process = shared(GeneralizedBlackScholesProcess::new(
            spot, q_ts, r_ts, vol_ts,
        ));

        // CashOrNothingPayoff with strike K = 105.0
        let strike = 105.0;
        let cash_payoff: Shared<dyn StrikedTypePayoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, strike, 10.0));
        let euro_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 90));

        let mut opt = DoubleBarrierOption::with_striked_payoff(
            DoubleBarrierType::KnockOut,
            80.0,
            120.0,
            0.0,
            cash_payoff,
            euro_ex,
            Shared::clone(&settings),
        )
        .unwrap();

        set_analytic_double_barrier_binary_engine(&mut opt, process);
        let npv = opt.npv().unwrap();
        assert!(npv > 0.0);
    }
}
