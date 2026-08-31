//! Analytic soft-barrier European engine.
//!
//! Port of `ql/pricingengines/barrier/analyticsoftbarrierengine.{hpp,cpp}`
//! (Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed., p.165).

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    BarrierOption, BarrierType, SoftBarrierArguments, SoftBarrierResults, StrikedTypePayoff,
    TypePayoff, set_analytic_barrier_engine,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::{BlackScholesMertonProcess, GeneralizedBlackScholesProcess};
use crate::quotes::{Quote, SimpleQuote};
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::termstructures::yields::FlatForward;
use crate::time::calendars::target::Target;
use crate::time::daycounters::actual360::Actual360;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time, Volatility};

type EngineBase = GenericEngine<SoftBarrierArguments, SoftBarrierResults>;

/// Pricing engine for European soft-barrier options (Hart–Ross / Haug).
pub struct AnalyticSoftBarrierEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticSoftBarrierEngine {
    /// `AnalyticSoftBarrierEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            SoftBarrierArguments::default(),
            SoftBarrierResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            f: CumulativeNormalDistribution::standard(),
        }
    }

    fn underlying(&self) -> QlResult<Real> {
        self.process.x0()
    }

    fn strike(&self) -> Real {
        self.base
            .arguments()
            .payoff
            .expect("validated")
            .strike()
    }

    fn residual_time(&self) -> QlResult<Time> {
        let exercise = self
            .base
            .arguments()
            .exercise
            .as_ref()
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn volatility(&self) -> QlResult<Volatility> {
        let t = self.residual_time()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        vol_ts.black_vol(t, self.strike(), true)
    }

    fn std_deviation(&self) -> QlResult<Real> {
        Ok(self.volatility()? * self.residual_time()?.sqrt())
    }

    fn barrier_lo(&self) -> Real {
        self.base.arguments().barrier_lo.expect("validated")
    }

    fn barrier_hi(&self) -> Real {
        self.base.arguments().barrier_hi.expect("validated")
    }

    fn risk_free_rate(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        Ok(r_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn risk_free_discount(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        r_ts.discount(t, false)
    }

    fn dividend_yield(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        Ok(q_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn dividend_discount(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        q_ts.discount(t, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_inputs(
        &self,
        s: Real,
        x: Real,
        r: Real,
        q: Real,
        t: Time,
        u: Real,
        l: Real,
        option_type: OptionType,
        barrier_type: BarrierType,
        sigma: Real,
    ) -> QlResult<()> {
        require!(s > 0.0, "Spot price must be > 0");
        require!(x > 0.0, "Strike price must be > 0");
        require!(t > 0.0, "Option must have time to maturity > 0");
        require!(sigma > 0.0, "Volatility must be > 0");
        match option_type {
            OptionType::Call | OptionType::Put => {}
        }
        require!(
            (-0.05..=1.0).contains(&r),
            "Interest rate must be between -5% and 100%"
        );
        require!(
            (-0.1..=1.0).contains(&q),
            "Dividend yield must be between -10% and 100%"
        );
        match barrier_type {
            BarrierType::DownIn
            | BarrierType::DownOut
            | BarrierType::UpIn
            | BarrierType::UpOut => {}
        }
        require!(u > 0.0 && l > 0.0, "Barrier levels must be positive");
        require!(
            u >= l,
            "Upper barrier must be greater than or equal to lower barrier"
        );
        Ok(())
    }

    fn vanilla_equivalent(&self) -> QlResult<Real> {
        let payoff = self.base.arguments().payoff.expect("validated");
        let forward = self.underlying()? * self.dividend_discount()? / self.risk_free_discount()?;
        let black = BlackCalculator::with_striked_payoff(
            &payoff as &dyn StrikedTypePayoff,
            forward,
            self.std_deviation()?,
            self.risk_free_discount()?,
        )?;
        Ok(black.value().max(0.0))
    }

    fn standard_barrier_equivalent(&self) -> QlResult<Real> {
        let args = self.base.arguments();
        let payoff = args.payoff.expect("validated");
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        let barrier_type = args.barrier_type.expect("validated");
        let barrier_hi = args.barrier_hi.expect("validated");

        let spot_val = self.underlying()?;
        let q_val = self.dividend_yield()?;
        let r_val = self.risk_free_rate()?;
        let vol_val = self.volatility()?;
        let dc = Actual360::new();
        let today = self
            .process
            .risk_free_rate()
            .current_link()?
            .reference_date()?;

        let spot = Handle::new(shared(SimpleQuote::new(spot_val)) as Shared<dyn Quote>);
        let q = Handle::new(
            shared(FlatForward::with_rate(
                today,
                q_val,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        );
        let r = Handle::new(
            shared(FlatForward::with_rate(
                today,
                r_val,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        );
        let vol = Handle::new(
            shared(BlackConstantVol::new(
                today,
                Some(Target::new()),
                vol_val,
                dc,
            )) as Shared<dyn BlackVolTermStructure>,
        );
        let process = shared(BlackScholesMertonProcess::new(spot, q, r, vol));

        // Soft barrier has no Settings on the instrument path for this helper;
        // reuse a temporary BarrierOption with a fresh settings object.
        let settings = shared(crate::settings::Settings::new());
        settings.set_evaluation_date(today);
        let mut temp = BarrierOption::with_rebate(
            barrier_type,
            barrier_hi,
            0.0,
            payoff,
            exercise,
            settings,
        )?;
        set_analytic_barrier_engine(&mut temp, process);
        Ok(temp.npv()?.max(0.0))
    }

    #[allow(clippy::too_many_arguments)]
    fn knock_in_value(
        &self,
        s: Real,
        x: Real,
        r: Real,
        sigma: Volatility,
        t: Time,
        u: Real,
        l: Real,
        b: Real,
        eta: Real,
    ) -> Real {
        let mu = (b + 0.5 * sigma * sigma) / (sigma * sigma);
        let sqrt_t = t.sqrt();
        let lambda1 = (-0.5 * sigma * sigma * t * (mu + 0.5) * (mu - 0.5)).exp();
        let lambda2 = (-0.5 * sigma * sigma * t * (mu - 0.5) * (mu - 1.5)).exp();
        let sx = s * x;
        let log_u2_sx = ((u * u) / sx).ln();
        let log_l2_sx = ((l * l) / sx).ln();

        let d1 = log_u2_sx / (sigma * sqrt_t) + mu * sigma * sqrt_t;
        let d2 = d1 - (mu + 0.5) * sigma * sqrt_t;
        let d3 = log_u2_sx / (sigma * sqrt_t) + (mu - 1.0) * sigma * sqrt_t;
        let d4 = d3 - (mu - 0.5) * sigma * sqrt_t;

        let e1 = log_l2_sx / (sigma * sqrt_t) + mu * sigma * sqrt_t;
        let e2 = e1 - (mu + 0.5) * sigma * sqrt_t;
        let e3 = log_l2_sx / (sigma * sqrt_t) + (mu - 1.0) * sigma * sqrt_t;
        let e4 = e3 - (mu - 0.5) * sigma * sqrt_t;

        let nd1 = self.f.value(eta * d1);
        let nd2 = self.f.value(eta * d2);
        let nd3 = self.f.value(eta * d3);
        let nd4 = self.f.value(eta * d4);
        let ne1 = self.f.value(eta * e1);
        let ne2 = self.f.value(eta * e2);
        let ne3 = self.f.value(eta * e3);
        let ne4 = self.f.value(eta * e4);

        let mut term1 = eta
            * s
            * ((b - r) * t).exp()
            * s.powf(-2.0 * mu)
            * sx.powf(mu + 0.5)
            / (2.0 * (mu + 0.5));
        term1 *= ((u * u) / sx).powf(mu + 0.5) * nd1
            - lambda1 * nd2
            - ((l * l) / sx).powf(mu + 0.5) * ne1
            + lambda1 * ne2;

        let mut term2 = eta
            * x
            * (-r * t).exp()
            * s.powf(-2.0 * (mu - 1.0))
            * sx.powf(mu - 0.5)
            / (2.0 * (mu - 0.5));
        term2 *= ((u * u) / sx).powf(mu - 0.5) * nd3
            - lambda2 * nd4
            - ((l * l) / sx).powf(mu - 0.5) * ne3
            + lambda2 * ne4;

        (1.0 / (u - l)) * (term1 - term2)
    }
}

impl AsObservable for AnalyticSoftBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticSoftBarrierEngine {
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
        let mut r = self.risk_free_rate()?;
        let q = self.dividend_yield()?;
        let sigma = self.volatility()?;
        let s = self.underlying()?;
        let x = self.strike();
        let u = self.barrier_hi();
        let l = self.barrier_lo();
        let t = self.residual_time()?;
        let barrier_type = self.base.arguments().barrier_type.expect("validated");
        let option_type = self
            .base
            .arguments()
            .payoff
            .expect("validated")
            .option_type();

        // Stability tweak for r ≈ q (avoids μ = 0.5 singularity).
        const EPSILON: Real = 1e-6;
        if (r - q).abs() < 1e-10 {
            r = q + EPSILON;
        }

        let eta = match option_type {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        };
        let b = r - q;

        self.validate_inputs(s, x, r, q, t, u, l, option_type, barrier_type, sigma)?;

        let is_knocked_in = matches!(
            (barrier_type, s <= l, s >= u),
            (BarrierType::DownIn, true, _) | (BarrierType::UpIn, _, true)
        );
        let is_knocked_out = matches!(
            (barrier_type, s <= l, s >= u),
            (BarrierType::DownOut, true, _) | (BarrierType::UpOut, _, true)
        );
        let is_single_barrier = (u - l).abs() < 1e-4;

        let value = if is_knocked_in {
            self.vanilla_equivalent()?
        } else if is_knocked_out {
            0.0
        } else if is_single_barrier {
            self.standard_barrier_equivalent()?
        } else {
            let w = self.knock_in_value(s, x, r, sigma, t, u, l, b, eta);
            match barrier_type {
                BarrierType::DownIn | BarrierType::UpIn => w,
                BarrierType::DownOut | BarrierType::UpOut => self.vanilla_equivalent()? - w,
            }
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticSoftBarrierEngine`] to `option`.
pub fn set_analytic_soft_barrier_engine(
    option: &mut crate::instruments::SoftBarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticSoftBarrierEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instruments::{PlainVanillaPayoff, SoftBarrierOption};
    use crate::settings::Settings;
    use crate::time::date::{Date, Month};
    use crate::types::{Rate, Time, Volatility};

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::new(
                reference,
                quote_handle(quote),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::with_quote(
                reference,
                None,
                quote_handle(quote),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        )
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    struct Case {
        barrier_type: BarrierType,
        option_type: OptionType,
        spot: Real,
        strike: Real,
        u: Real,
        l: Real,
        q: Rate,
        r: Rate,
        t: Time,
        v: Volatility,
        result: Real,
        tol: Real,
    }

    /// `softbarrieroption.cpp` `testSoftBarrierHaug`.
    #[test]
    fn soft_barrier_haug_matches_textbook() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);

        // Haug 2nd ed. p.166 DownOut calls (one tight-barrier/high-vol row omitted, as in QL).
        let cases = [
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 3.8075, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 90.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0175, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0529, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0648, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0708, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0744, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0768, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0785, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0798, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 0.5, v: 0.1, result: 4.0808, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 4.5263, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 90.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 5.5615, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.0394, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.2594, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.3740, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.4429, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.4889, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.5217, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.5463, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 0.5, v: 0.2, result: 6.5654, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 4.7297, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 90.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 6.2595, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 7.2496, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 7.8567, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.2253, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.4578, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.6142, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.7260, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.8099, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 0.5, v: 0.3, result: 8.8751, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 5.4187, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 90.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.0758, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.2641, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.3336, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.3685, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.3894, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.4034, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.4133, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.4208, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 1.0, v: 0.1, result: 6.4266, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 5.3614, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 90.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 6.9776, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 7.9662, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 8.5432, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 8.8822, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 9.0931, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 9.2343, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 9.3353, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 9.4110, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 1.0, v: 0.2, result: 9.4698, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 95.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 5.2300, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 85.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 8.7092, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 80.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 9.8118, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 75.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 10.5964, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 70.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 11.1476, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 65.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 11.5384, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 60.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 11.8228, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 55.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 12.0369, tol: 1e-4 },
            Case { barrier_type: BarrierType::DownOut, option_type: OptionType::Call, spot: 100.0, strike: 100.0, u: 95.0, l: 50.0, q: 0.05, r: 0.1, t: 1.0, v: 0.3, result: 12.2036, tol: 1e-4 },
        ];

        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.0));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        for (i, case) in cases.iter().enumerate() {
            spot.set_value(case.spot);
            q_rate.set_value(case.q);
            r_rate.set_value(case.r);
            vol.set_value(case.v);

            let exercise = shared(EuropeanExercise::new(today + time_to_days(case.t)));
            let mut option = SoftBarrierOption::new(
                case.barrier_type,
                case.l,
                case.u,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_soft_barrier_engine(&mut option, Shared::clone(&process));
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= case.tol,
                "case {i}: expected {}, got {calculated} (tol {})",
                case.result,
                case.tol
            );
        }
    }
}
