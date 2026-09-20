//! Analytic two-asset barrier engine (Heynen–Kat / Haug).
//!
//! Port of `ql/pricingengines/barrier/analytictwoassetbarrierengine.{hpp,cpp}`.
//! `B` is identically 0 as in QuantLib.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    BarrierType, StrikedTypePayoff, TwoAssetBarrierArguments, TwoAssetBarrierResults, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<TwoAssetBarrierArguments, TwoAssetBarrierResults>;

/// Analytic engine for European two-asset barrier options.
pub struct AnalyticTwoAssetBarrierEngine {
    base: EngineBase,
    p1: Shared<GeneralizedBlackScholesProcess>,
    p2: Shared<GeneralizedBlackScholesProcess>,
    rho: Handle<dyn Quote>,
    n: CumulativeNormalDistribution,
}

impl AnalyticTwoAssetBarrierEngine {
    /// `AnalyticTwoAssetBarrierEngine(process1, process2, rho)`.
    pub fn new(
        p1: Shared<GeneralizedBlackScholesProcess>,
        p2: Shared<GeneralizedBlackScholesProcess>,
        rho: Handle<dyn Quote>,
    ) -> Self {
        let base = EngineBase::new(
            TwoAssetBarrierArguments::default(),
            TwoAssetBarrierResults::default(),
        );
        base.register_with(p1.observable());
        base.register_with(p2.observable());
        rho.register_observer(&base.observer());
        Self {
            base,
            p1,
            p2,
            rho,
            n: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticTwoAssetBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticTwoAssetBarrierEngine {
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
        let a = self.base.arguments();
        let payoff = a.payoff.expect("validated");
        let x = payoff.strike();
        require!(x > 0.0, "strike must be positive");
        let h = a.barrier.expect("validated");
        let bt = a.barrier_type.expect("validated");
        let last = a.exercise.as_ref().expect("validated").last_date();
        let s1 = self.p1.x0()?;
        let s2 = self.p2.x0()?;
        require!(s2 > 0.0, "negative or null underlying given");
        let touched = match bt {
            BarrierType::DownIn | BarrierType::DownOut => s2 < h,
            BarrierType::UpIn | BarrierType::UpOut => s2 > h,
        };
        require!(!touched, "barrier touched");
        let t = self.p1.time(&last)?;
        let sqrt_t = t.sqrt();
        let bv = |p: &GeneralizedBlackScholesProcess| -> QlResult<Real> {
            p.black_volatility().current_link()?.black_vol(t, x, true)
        };
        let zr = |ts: &Handle<dyn YieldTermStructure>| {
            ts.current_link()?
                .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)
                .map(|z| z.rate())
        };
        let sigma1 = bv(&self.p1)?;
        let sigma2 = bv(&self.p2)?;
        let r = zr(&self.p1.risk_free_rate())?;
        let q1 = zr(&self.p1.dividend_yield())?;
        let q2 = zr(&self.p2.dividend_yield())?;
        let rho = self.rho.current_link()?.value()?;
        let b1 = r - q1;
        let b2 = r - q2;
        let mu1 = b1 - sigma1 * sigma1 / 2.0;
        let mu2 = b2 - sigma2 * sigma2 / 2.0;
        let log_hs = (h / s2).ln();
        let d1 = ((s1 / x).ln() + (mu1 + sigma1 * sigma1) * t) / (sigma1 * sqrt_t);
        let d2 = d1 - sigma1 * sqrt_t;
        let vanilla = match payoff.option_type() {
            OptionType::Call => s1 * self.n.value(d1) - x * (-r * t).exp() * self.n.value(d2),
            OptionType::Put => x * (-r * t).exp() * self.n.value(-d2) - s1 * self.n.value(-d1),
        };
        let a_term = |eta: Real, phi: Real| -> QlResult<Real> {
            let d3 = d1 + (2.0 * rho * log_hs) / (sigma2 * sqrt_t);
            let d4 = d2 + (2.0 * rho * log_hs) / (sigma2 * sqrt_t);
            let e1 = (log_hs - (mu2 + rho * sigma1 * sigma2) * t) / (sigma2 * sqrt_t);
            let e2 = e1 + rho * sigma1 * sqrt_t;
            let e3 = e1 - (2.0 * log_hs) / (sigma2 * sqrt_t);
            let e4 = e2 - (2.0 * log_hs) / (sigma2 * sqrt_t);
            let mrho = -eta * phi * rho;
            let m = BivariateCumulativeNormalDistributionDr78::new(mrho)?;
            let w = eta
                * s1
                * ((b1 - r) * t).exp()
                * (m.value(eta * d1, phi * e1)
                    - ((2.0 * (mu2 + rho * sigma1 * sigma2) * log_hs) / (sigma2 * sigma2)).exp()
                        * m.value(eta * d3, phi * e3))
                - eta
                    * x
                    * (-r * t).exp()
                    * (m.value(eta * d2, phi * e2)
                        - ((2.0 * mu2 * log_hs) / (sigma2 * sigma2)).exp()
                            * m.value(eta * d4, phi * e4));
            Ok(w)
        };
        let value = match (payoff.option_type(), bt) {
            (OptionType::Call, BarrierType::DownOut) => a_term(1.0, -1.0)?,
            (OptionType::Call, BarrierType::UpOut) => a_term(1.0, 1.0)?,
            (OptionType::Call, BarrierType::DownIn) => vanilla - a_term(1.0, -1.0)?,
            (OptionType::Call, BarrierType::UpIn) => vanilla - a_term(1.0, 1.0)?,
            (OptionType::Put, BarrierType::DownOut) => a_term(-1.0, -1.0)?,
            (OptionType::Put, BarrierType::UpOut) => a_term(-1.0, 1.0)?,
            (OptionType::Put, BarrierType::DownIn) => vanilla - a_term(-1.0, -1.0)?,
            (OptionType::Put, BarrierType::UpIn) => vanilla - a_term(-1.0, 1.0)?,
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticTwoAssetBarrierEngine`] to `option`.
pub fn set_analytic_two_asset_barrier_engine(
    option: &mut crate::instruments::TwoAssetBarrierOption,
    p1: Shared<GeneralizedBlackScholesProcess>,
    p2: Shared<GeneralizedBlackScholesProcess>,
    rho: Handle<dyn Quote>,
) {
    let engine =
        shared_mut(AnalyticTwoAssetBarrierEngine::new(p1, p2, rho)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, TwoAssetBarrierOption};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    fn quote_h(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    fn flat_rate(today: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            today,
            quote_h(r),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(today: Date, v: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            today,
            None,
            quote_h(v),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    /// `twoassetbarrieroption.cpp` `testHaugValues` @ 4e-3.
    #[test]
    fn two_asset_barrier_haug_npv() {
        use BarrierType::{DownOut, UpOut};
        use OptionType::{Call, Put};
        type Row = (BarrierType, OptionType, Real, Real, Real, Real);
        #[rustfmt::skip]
        let rows: [Row; 4] = [
            (DownOut, Call, 95.0, 90.0, 0.5, 6.6592),
            (UpOut, Call, 105.0, 90.0, -0.5, 4.6670),
            (DownOut, Put, 95.0, 90.0, -0.5, 0.6184),
            (UpOut, Put, 105.0, 100.0, 0.0, 0.8246),
        ];
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let r = flat_rate(today, 0.08);
        for (bt, ty, h, k, rho, expected) in rows {
            #[rustfmt::skip]
            let mk = || shared(BlackScholesMertonProcess::new(
                quote_h(100.0), flat_rate(today, 0.0), Handle::clone(&r), flat_vol(today, 0.2),
            ));
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 180));
            let mut option = TwoAssetBarrierOption::new(
                bt,
                h,
                PlainVanillaPayoff::new(ty, k),
                exercise,
                Shared::clone(&settings),
            );
            set_analytic_two_asset_barrier_engine(&mut option, mk(), mk(), quote_h(rho));
            let got = option.npv().unwrap();
            assert!(
                (got - expected).abs() <= 4e-3,
                "{bt:?} {ty:?}: {expected} vs {got}"
            );
        }
    }
}
