//! Replicating variance-swap engine.
//!
//! Port of `ql/pricingengines/forward/replicatingvarianceswapengine.hpp`
//! (Demeterfi, Derman, Kamal & Zou, 1999).

use crate::errors::QlResult;
use crate::exercise::{EuropeanExercise, Exercise};
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    PlainVanillaPayoff, StrikedTypePayoff, VarianceSwap, VarianceSwapArguments, VarianceSwapResults,
};
use crate::interestrate::Compounding;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::position::Position;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::Real;
use std::any::Any;

type EngineBase = GenericEngine<VarianceSwapArguments, VarianceSwapResults>;
type Weight = (Shared<dyn StrikedTypePayoff>, Real);

pub struct ReplicatingVarianceSwapEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    dk: Real,
    call_strikes: Vec<Real>,
    put_strikes: Vec<Real>,
}

impl ReplicatingVarianceSwapEngine {
    #[rustfmt::skip]
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>, call_strikes: Vec<Real>, put_strikes: Vec<Real>,
    ) -> QlResult<Self> {
        Self::with_dk(process, 5.0, call_strikes, put_strikes)
    }

    #[rustfmt::skip]
    pub fn with_dk(
        process: Shared<GeneralizedBlackScholesProcess>, dk: Real, call_strikes: Vec<Real>, put_strikes: Vec<Real>,
    ) -> QlResult<Self> {
        require!(!call_strikes.is_empty() && !put_strikes.is_empty(), "no strike(s) given");
        require!(put_strikes.iter().copied().fold(Real::INFINITY, Real::min) > 0.0, "min put strike must be positive");
        let min_call = call_strikes.iter().copied().fold(Real::INFINITY, Real::min);
        let max_put = put_strikes.iter().copied().fold(Real::NEG_INFINITY, Real::max);
        require!(min_call == max_put, "min call and max put strikes differ");
        let base = EngineBase::new(VarianceSwapArguments::default(), VarianceSwapResults::default());
        base.register_with(process.observable());
        Ok(Self { base, process, dk, call_strikes, put_strikes })
    }
}

#[rustfmt::skip]
impl AsObservable for ReplicatingVarianceSwapEngine {
    fn observable(&self) -> &Observable { self.base.observable() }
}

#[rustfmt::skip]
impl PricingEngine for ReplicatingVarianceSwapEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments { self.base.arguments_mut() }
    fn results(&self) -> &dyn Results { self.base.results() }
    fn reset(&mut self) { self.base.reset(); }

    #[rustfmt::skip]
    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        let maturity = args.maturity_date.expect("validated");
        let strike = args.strike.expect("validated");
        let notional = args.notional.expect("validated");
        let position = args.position.expect("validated");
        let mut weights = Vec::new();
        self.weights(self.call_strikes.clone(), OptionType::Call, &mut weights)?;
        self.weights(self.put_strikes.clone(), OptionType::Put, &mut weights)?;
        let variance = self.portfolio(&weights, maturity)?;
        let t = self.process.time(&maturity)?;
        let disc = self.process.risk_free_rate().current_link()?.discount(t, false)?;
        let sign = match position { Position::Long => 1.0, Position::Short => -1.0 };
        let results = self.base.results_mut();
        results.variance = Some(variance);
        results.instrument.value = Some(sign * disc * notional * (variance - strike));
        results.instrument.additional_results.insert("optionWeights".into(), shared(weights) as Shared<dyn Any>);
        Ok(())
    }
}

impl ReplicatingVarianceSwapEngine {
    #[rustfmt::skip]
    fn log_payoff(&self, strike: Real, f: Real, t: Real) -> Real {
        (2.0 / t) * ((strike - f) / f - (strike / f).ln())
    }

    #[rustfmt::skip]
    fn weights(&self, mut strikes: Vec<Real>, ty: OptionType, out: &mut Vec<Weight>) -> QlResult<()> {
        if strikes.is_empty() { return Ok(()); }
        match ty {
            OptionType::Call => { strikes.sort_by(|a, b| a.partial_cmp(b).unwrap()); strikes.push(strikes[strikes.len() - 1] + self.dk); }
            OptionType::Put => { strikes.sort_by(|a, b| b.partial_cmp(a).unwrap()); let last = strikes[strikes.len() - 1]; strikes.push((last - self.dk).max(0.0)); }
        }
        strikes.dedup();
        let f = strikes[0];
        let t = self.process.time(&self.base.arguments().maturity_date.expect("validated"))?;
        let mut prev = 0.0;
        for i in 0..strikes.len() - 1 {
            let slope = ((self.log_payoff(strikes[i + 1], f, t) - self.log_payoff(strikes[i], f, t)) / (strikes[i + 1] - strikes[i])).abs();
            let w = if i == 0 { slope } else { slope - prev };
            out.push((shared(PlainVanillaPayoff::new(ty, strikes[i])) as Shared<dyn StrikedTypePayoff>, w));
            prev = slope;
        }
        Ok(())
    }

    #[rustfmt::skip]
    fn portfolio(&self, weights: &[Weight], maturity: Date) -> QlResult<Real> {
        let exercise = shared(EuropeanExercise::new(maturity)) as Shared<dyn Exercise>;
        let mut euro = AnalyticEuropeanEngine::new(Shared::clone(&self.process));
        let mut options = 0.0;
        for (payoff, w) in weights {
            let Some(v) = euro.calculate_from_arguments(Shared::clone(payoff), Shared::clone(&exercise))?.instrument.value else {
                fail!("no value");
            };
            options += v * w;
        }
        let f = weights[0].0.strike();
        let t = self.process.time(&maturity)?;
        let r = self.process.risk_free_rate().current_link()?.zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, true)?.rate();
        let disc = self.process.risk_free_rate().current_link()?.discount(t, false)?;
        let s = self.process.x0()?;
        Ok(2.0 * r - 2.0 / t * ((s / disc - f) / f + (f / s).ln()) + options / disc)
    }
}

#[rustfmt::skip]
pub fn set_replicating_variance_swap_engine(swap: &mut VarianceSwap, process: Shared<GeneralizedBlackScholesProcess>, call_strikes: Vec<Real>, put_strikes: Vec<Real>) -> QlResult<()> {
    swap.base_mut().set_pricing_engine(shared_mut(ReplicatingVarianceSwapEngine::new(process, call_strikes, put_strikes)?) as SharedMut<dyn PricingEngine>);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::math::matrix::Matrix;
    use crate::option::OptionType;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::termstructures::volatility::{BlackVarianceSurface, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::NullCalendar;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    #[test]
    #[rustfmt::skip]
    fn replicating_variance_swap_derman() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);
        let qh = |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.05));
        let flat = |q: &Shared<SimpleQuote>| Handle::new(shared(FlatForward::new(today, qh(q), Actual365Fixed::new(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>);
        let data: [(OptionType, Real, Real); 19] = [
            (OptionType::Put, 50.0, 0.30), (OptionType::Put, 55.0, 0.29), (OptionType::Put, 60.0, 0.28),
            (OptionType::Put, 65.0, 0.27), (OptionType::Put, 70.0, 0.26), (OptionType::Put, 75.0, 0.25),
            (OptionType::Put, 80.0, 0.24), (OptionType::Put, 85.0, 0.23), (OptionType::Put, 90.0, 0.22),
            (OptionType::Put, 95.0, 0.21), (OptionType::Put, 100.0, 0.20), (OptionType::Call, 100.0, 0.20),
            (OptionType::Call, 105.0, 0.19), (OptionType::Call, 110.0, 0.18), (OptionType::Call, 115.0, 0.17),
            (OptionType::Call, 120.0, 0.16), (OptionType::Call, 125.0, 0.15), (OptionType::Call, 130.0, 0.14),
            (OptionType::Call, 135.0, 0.13),
        ];
        let mut calls = Vec::new(); let mut puts = Vec::new();
        let mut call_vols = Vec::new(); let mut put_vols = Vec::new();
        for (ty, k, v) in data {
            if ty == OptionType::Call { calls.push(k); call_vols.push(v); } else { puts.push(k); put_vols.push(v); }
        }
        let mut strikes = puts.clone();
        let mut vols = Matrix::with_size(data.len() - 1, 1);
        for (j, v) in put_vols.iter().enumerate() { vols[(j, 0)] = *v; }
        for (k, v) in call_vols.iter().enumerate().skip(1) {
            vols[(put_vols.len() - 1 + k, 0)] = *v;
            strikes.push(calls[k]);
        }
        let t: Real = 0.246575;
        let ex = today + (t * 365.0).round() as i32;
        let vol_ts = shared(BlackVarianceSurface::new(today, Some(NullCalendar::new()), &[ex], strikes, &vols, Actual365Fixed::new()).unwrap()) as Shared<dyn BlackVolTermStructure>;
        let process = shared(GeneralizedBlackScholesProcess::new(qh(&spot), flat(&q_rate), flat(&r_rate), Handle::new(vol_ts)));
        let mut swap = VarianceSwap::new(Position::Long, 0.04, 50000.0, today, ex, Shared::clone(&settings));
        set_replicating_variance_swap_engine(&mut swap, Shared::clone(&process), calls.clone(), puts.clone()).unwrap();
        let v = swap.variance().unwrap();
        let long = swap.npv().unwrap();
        assert!((v - 0.04189).abs() <= 1e-4, "variance {v} vs 0.04189");
        assert!((long - 93.271669134338).abs() <= 1e-4, "long npv {long}");
        assert!(swap.additional_results().unwrap().contains_key("optionWeights"));
        let mut short = VarianceSwap::new(Position::Short, 0.04, 50000.0, today, ex, Shared::clone(&settings));
        set_replicating_variance_swap_engine(&mut short, Shared::clone(&process), calls.clone(), puts.clone()).unwrap();
        assert!((short.npv().unwrap() + long).abs() <= 1e-8, "short != -long");
        q_rate.set_value(0.05);
        let mut qswap = VarianceSwap::new(Position::Long, 0.04, 50000.0, today, ex, Shared::clone(&settings));
        set_replicating_variance_swap_engine(&mut qswap, process, calls, puts).unwrap();
        let qv = qswap.variance().unwrap();
        assert!((qv - 0.042298936629).abs() <= 1e-4, "q=0.05 QL-hybrid variance {qv}");
    }
}
