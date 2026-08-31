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
                DoubleBarrierType::KnockIn => (vanilla - call_ko(
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
                DoubleBarrierType::KnockIn => (vanilla - put_ko(
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
    let engine = shared_mut(AnalyticDoubleBarrierEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}
