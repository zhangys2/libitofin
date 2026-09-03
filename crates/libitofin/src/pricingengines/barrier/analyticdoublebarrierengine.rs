//! Analytic double-barrier engine (Ikeda–Kunitomo series).
//!
//! Port of `ql/pricingengines/barrier/analyticdoublebarrierengine.{hpp,cpp}`:
//! Haug *Complete guide to option pricing formulas* 2nd Ed., p.156.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::Instrument;
use crate::instrument::InstrumentResults;
use crate::instruments::{
    DoubleBarrierArguments, DoubleBarrierType, StrikedTypePayoff, TypePayoff,
    double_barrier_triggered,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Time};

type DoubleBarrierEngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// Pricing engine for European double-barrier options (flat barriers, series sum).
pub struct AnalyticDoubleBarrierEngine {
    base: DoubleBarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    series: i32,
    normal: CumulativeNormalDistribution,
}

impl AnalyticDoubleBarrierEngine {
    /// Default Ikeda–Kunitomo series truncation (`series = 5`).
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_series(process, 5)
    }

    /// `AnalyticDoubleBarrierEngine(process, series)`.
    pub fn with_series(process: Shared<GeneralizedBlackScholesProcess>, series: i32) -> Self {
        let base = DoubleBarrierEngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            series,
            normal: CumulativeNormalDistribution::standard(),
        }
    }

    /// Fills the arguments and calculates; used by
    /// [`QuantoDoubleBarrierEngine`](super::QuantoDoubleBarrierEngine).
    pub(crate) fn calculate_from_arguments(
        &mut self,
        arguments: &DoubleBarrierArguments,
    ) -> QlResult<&InstrumentResults> {
        {
            let dest = self.base.arguments_mut();
            dest.barrier_type = arguments.barrier_type;
            dest.barrier_lo = arguments.barrier_lo;
            dest.barrier_hi = arguments.barrier_hi;
            dest.rebate = arguments.rebate;
            dest.payoff = arguments.payoff;
            dest.exercise = arguments.exercise.as_ref().map(Shared::clone);
        }
        PricingEngine::calculate(self)?;
        Ok(self.base.results())
    }
}

impl AsObservable for AnalyticDoubleBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDoubleBarrierEngine {
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
        let barrier_type = args.barrier_type.expect("validated");
        let barrier_lo = args.barrier_lo.expect("validated");
        let barrier_hi = args.barrier_hi.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        if exercise.exercise_type() != ExerciseType::European {
            fail!("this engine handles only european options");
        }
        let strike = payoff.strike();
        require!(strike > 0.0, "strike must be positive");
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(
            !double_barrier_triggered(spot, barrier_lo, barrier_hi),
            "barrier(s) already touched"
        );

        let maturity = exercise.last_date();
        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let residual_time = self.process.time(&maturity)?;
        let r = risk_free
            .zero_rate(
                residual_time,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q = dividend
            .zero_rate(
                residual_time,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let df_r = risk_free.discount(residual_time, false)?;
        let df_q = dividend.discount(residual_time, false)?;
        let vol = vol_ts.black_vol(residual_time, strike, false)?;
        let std_dev = (vol * vol * residual_time).sqrt();
        let b = r - q;
        let sig2 = vol * vol;

        let vanilla = {
            let forward = spot * df_q / df_r;
            let black = BlackCalculator::with_striked_payoff(&payoff, forward, std_dev, df_r)?;
            black.value().max(0.0)
        };

        let value = match payoff.option_type() {
            OptionType::Call => match barrier_type {
                DoubleBarrierType::KnockOut => call_ko(
                    &self.normal,
                    self.series,
                    spot,
                    strike,
                    barrier_lo,
                    barrier_hi,
                    b,
                    sig2,
                    residual_time,
                    std_dev,
                    df_r,
                    df_q,
                ),
                DoubleBarrierType::KnockIn => (vanilla
                    - call_ko(
                        &self.normal,
                        self.series,
                        spot,
                        strike,
                        barrier_lo,
                        barrier_hi,
                        b,
                        sig2,
                        residual_time,
                        std_dev,
                        df_r,
                        df_q,
                    ))
                .max(0.0),
            },
            OptionType::Put => match barrier_type {
                DoubleBarrierType::KnockOut => put_ko(
                    &self.normal,
                    self.series,
                    spot,
                    strike,
                    barrier_lo,
                    barrier_hi,
                    b,
                    sig2,
                    residual_time,
                    std_dev,
                    df_r,
                    df_q,
                ),
                DoubleBarrierType::KnockIn => (vanilla
                    - put_ko(
                        &self.normal,
                        self.series,
                        spot,
                        strike,
                        barrier_lo,
                        barrier_hi,
                        b,
                        sig2,
                        residual_time,
                        std_dev,
                        df_r,
                        df_q,
                    ))
                .max(0.0),
            },
        };

        self.base.results_mut().value = Some(value.max(0.0));
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn call_ko(
    normal: &CumulativeNormalDistribution,
    series: i32,
    underlying: Real,
    strike: Real,
    barrier_lo: Real,
    barrier_hi: Real,
    cost_of_carry: Rate,
    vol_squared: Real,
    residual_time: Time,
    std_dev: Real,
    risk_free_discount: Real,
    dividend_discount: Real,
) -> Real {
    let mu1 = 2.0 * cost_of_carry / vol_squared + 1.0;
    let bsigma = (cost_of_carry + vol_squared / 2.0) * residual_time / std_dev;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    for n in -series..=series {
        let n_f = f64::from(n);
        let l2n = barrier_lo.powf(2.0 * n_f);
        let u2n = barrier_hi.powf(2.0 * n_f);
        let d1 = ((underlying * u2n) / (strike * l2n)).ln() / std_dev + bsigma;
        let d2 = ((underlying * u2n) / (barrier_hi * l2n)).ln() / std_dev + bsigma;
        let d3 = (barrier_lo.powf(2.0 * n_f + 2.0) / (strike * underlying * u2n)).ln() / std_dev
            + bsigma;
        let d4 = (barrier_lo.powf(2.0 * n_f + 2.0) / (barrier_hi * underlying * u2n)).ln()
            / std_dev
            + bsigma;

        acc1 += (barrier_hi.powf(n_f) / barrier_lo.powf(n_f)).powf(mu1)
            * (phi(normal, d1) - phi(normal, d2))
            - (barrier_lo.powf(n_f + 1.0) / (barrier_hi.powf(n_f) * underlying)).powf(mu1)
                * (phi(normal, d3) - phi(normal, d4));
        acc2 += (barrier_hi.powf(n_f) / barrier_lo.powf(n_f)).powf(mu1 - 2.0)
            * (phi(normal, d1 - std_dev) - phi(normal, d2 - std_dev))
            - (barrier_lo.powf(n_f + 1.0) / (barrier_hi.powf(n_f) * underlying)).powf(mu1 - 2.0)
                * (phi(normal, d3 - std_dev) - phi(normal, d4 - std_dev));
    }
    let kov = underlying * dividend_discount * acc1 - strike * risk_free_discount * acc2;
    kov.max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn put_ko(
    normal: &CumulativeNormalDistribution,
    series: i32,
    underlying: Real,
    strike: Real,
    barrier_lo: Real,
    barrier_hi: Real,
    cost_of_carry: Rate,
    vol_squared: Real,
    residual_time: Time,
    std_dev: Real,
    risk_free_discount: Real,
    dividend_discount: Real,
) -> Real {
    let mu1 = 2.0 * cost_of_carry / vol_squared + 1.0;
    let bsigma = (cost_of_carry + vol_squared / 2.0) * residual_time / std_dev;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    for n in -series..=series {
        let n_f = f64::from(n);
        let l2n = barrier_lo.powf(2.0 * n_f);
        let u2n = barrier_hi.powf(2.0 * n_f);
        let y1 = ((underlying * u2n) / barrier_lo.powf(2.0 * n_f + 1.0)).ln() / std_dev + bsigma;
        let y2 = ((underlying * u2n) / (strike * l2n)).ln() / std_dev + bsigma;
        let y3 = (barrier_lo.powf(2.0 * n_f + 2.0) / (barrier_lo * underlying * u2n)).ln()
            / std_dev
            + bsigma;
        let y4 = (barrier_lo.powf(2.0 * n_f + 2.0) / (strike * underlying * u2n)).ln() / std_dev
            + bsigma;

        acc1 += (barrier_hi.powf(n_f) / barrier_lo.powf(n_f)).powf(mu1 - 2.0)
            * (phi(normal, y1 - std_dev) - phi(normal, y2 - std_dev))
            - (barrier_lo.powf(n_f + 1.0) / (barrier_hi.powf(n_f) * underlying)).powf(mu1 - 2.0)
                * (phi(normal, y3 - std_dev) - phi(normal, y4 - std_dev));
        acc2 += (barrier_hi.powf(n_f) / barrier_lo.powf(n_f)).powf(mu1)
            * (phi(normal, y1) - phi(normal, y2))
            - (barrier_lo.powf(n_f + 1.0) / (barrier_hi.powf(n_f) * underlying)).powf(mu1)
                * (phi(normal, y3) - phi(normal, y4));
    }
    let kov = strike * risk_free_discount * acc1 - underlying * dividend_discount * acc2;
    kov.max(0.0)
}

fn phi(normal: &CumulativeNormalDistribution, x: Real) -> Real {
    normal.value(x)
}

/// Attach an [`AnalyticDoubleBarrierEngine`] to a double-barrier option.
pub fn set_analytic_double_barrier_engine(
    option: &mut crate::instruments::DoubleBarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine =
        shared_mut(AnalyticDoubleBarrierEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{DoubleBarrierOption, DoubleBarrierType, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
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

    struct HaugRow {
        barrier_type: DoubleBarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        option_type: OptionType,
        strike: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        vol: Volatility,
        result: Real,
    }

    /// `doublebarrieroption.cpp` `testEuropeanHaugValues` Ikeda/Kunitomo @ 1e-4.
    #[rustfmt::skip]
    #[allow(clippy::approx_constant)]
    const HAUG_DOUBLE_BARRIER: &[HaugRow] = &[
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 4.3515,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 6.1644,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 7.0373,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 6.9853,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 7.9336,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 6.5088,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 4.3505,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 5.8500,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 5.7726,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 6.8082,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 6.3383,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 4.3841,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 4.3139,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 4.8293,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 3.7765,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 5.9697,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 4.0004,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 2.2563,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 3.7516,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 2.6387,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 1.4903,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 3.5805,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 1.5098,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 0.5635,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 1.2055,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 0.3098,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 0.0477,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 0.5537,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 0.0441,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 0.0011,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 1.8825,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 3.7855,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 5.7191,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 2.1374,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 4.7033,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 7.1683,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 1.8825,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 3.7845,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 5.6060,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 2.1374,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 4.6236,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 6.1062,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 1.8825,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 3.7014,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 4.6472,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 2.1325,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 3.8944,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 3.5868,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 1.8600,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 2.6866,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 2.0719,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 1.8883,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 1.7851,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 0.8244,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 0.9473,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 0.3449,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 0.0578,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 0.4555,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 0.0491,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockOut, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Put, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 0.0013,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 0.0000,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 0.0900,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 1.1537,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 0.0292,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 1.6487,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 50.0, barrier_hi: 150.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 5.7321,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 0.0010,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 0.4045,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 2.4184,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 0.2062,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 3.2439,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 60.0, barrier_hi: 140.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 7.8569,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 0.0376,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 1.4252,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 4.4145,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 1.0447,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 5.5818,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 70.0, barrier_hi: 130.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 9.9846,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 0.5999,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 3.6158,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 6.7007,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 3.4340,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 8.0724,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 80.0, barrier_hi: 120.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 11.6774,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.15, result: 3.1460,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.25, result: 5.9447,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.25, vol: 0.35, result: 8.1432,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.15, result: 6.4608,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.25, result: 9.5382,
        },
        HaugRow {
            barrier_type: DoubleBarrierType::KnockIn, barrier_lo: 90.0, barrier_hi: 110.0,
            option_type: OptionType::Call, strike: 100.0, spot: 100.0,
            q: 0.0, r: 0.1, t: 0.50, vol: 0.35, result: 12.2398,
        },
    ];

    #[test]
    fn european_haug_double_barrier_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        for row in HAUG_DOUBLE_BARRIER {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn crate::quotes::Quote>),
                flat_rate(row.q),
                flat_rate(row.r),
                flat_vol(row.vol),
            ));
            let payoff = PlainVanillaPayoff::new(row.option_type, row.strike);
            let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
            let mut option = DoubleBarrierOption::new(
                row.barrier_type,
                row.barrier_lo,
                row.barrier_hi,
                0.0,
                payoff,
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_double_barrier_engine(&mut option, process);
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= 1.0e-4,
                "{:?} {:?} lo={} hi={} v={}: {calculated} vs Haug {} (tol 1e-4)",
                row.barrier_type,
                row.option_type,
                row.barrier_lo,
                row.barrier_hi,
                row.vol,
                row.result,
            );
        }
    }
}
