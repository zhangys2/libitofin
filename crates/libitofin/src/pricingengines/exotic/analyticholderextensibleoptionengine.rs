//! Analytic holder-extensible option engine (Haug).
//! Port of `ql/pricingengines/exotic/analyticholderextensibleoptionengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    HolderExtensibleArguments, HolderExtensibleResults, StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<HolderExtensibleArguments, HolderExtensibleResults>;

/// Analytic engine for European holder-extensible options.
pub struct AnalyticHolderExtensibleOptionEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticHolderExtensibleOptionEngine {
    /// `AnalyticHolderExtensibleOptionEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            HolderExtensibleArguments::default(),
            HolderExtensibleResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticHolderExtensibleOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticHolderExtensibleOptionEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    #[rustfmt::skip]
    fn calculate(&mut self) -> QlResult<()> {
        let a = self.base.arguments();
        let payoff = a.payoff.expect("validated");
        let last1 = a.exercise.as_ref().expect("validated").last_date();
        let last2 = a.second_expiry.expect("validated");
        let x1 = payoff.strike();
        let x2 = a.second_strike.expect("validated");
        let prem = a.premium.expect("validated");
        let s = self.process.x0()?;
        require!(s > 0.0, "negative or null underlying given");
        let t1 = self.process.time(&last1)?;
        let t2 = self.process.time(&last2)?;
        let zr = |ts: &Handle<dyn YieldTermStructure>| {
            ts.current_link()?
                .zero_rate(t1, Compounding::Continuous, Frequency::NoFrequency, false)
                .map(|z| z.rate())
        };
        let r = zr(&self.process.risk_free_rate())?;
        let q = zr(&self.process.dividend_yield())?;
        let b = r - q;
        let vol = self.process.black_volatility().current_link()?.black_vol(t1, x1, true)?;
        let qy = self.process.dividend_yield().current_link()?;
        let rf = self.process.risk_free_rate().current_link()?;
        let df_q = |t: Real| qy.discount(t, false);
        let df_r = |t: Real| rf.discount(t, false);
        let bs_ext = |spot: Real, ty: OptionType| -> QlResult<(Real, Real)> {
            let t = t2 - t1;
            let disc = df_r(t)?;
            let growth = df_q(t)?;
            let bc = BlackCalculator::new(ty, x2, spot * growth / disc, vol * t.sqrt(), disc)?;
            Ok((bc.value(), bc.delta(spot)?))
        };
        let newton =
            |mut sv: Real, ty: OptionType, s_yi: Real, add: Real, s_di: Real| -> QlResult<Real> {
                for _ in 0..10_000 {
                    let (c, d) = bs_ext(sv, ty)?;
                    let yi = c - prem + s_yi * sv + add;
                    let di = d + s_di;
                    if yi.abs() <= 0.001 {
                        return Ok(sv);
                    }
                    sv -= yi / di;
                    require!(sv.is_finite() && sv > 0.0, "holder Newton left the domain");
                }
                fail!("holder-extensible Newton did not converge");
            };
        let (i1, i2) = match payoff.option_type() {
            OptionType::Call => {
                let i1 = if prem == 0.0 { 0.0 } else { newton(s, OptionType::Call, 0.0, 0.0, 0.0)? };
                let i2 = if prem < x1 - x2 * (-r * (t2 - t1)).exp() { Real::INFINITY } else { newton(s, OptionType::Call, -1.0, x1, -1.0)? };
                (i1, i2)
            }
            OptionType::Put => {
                let i1 = if x2 * (-r * (t2 - t1)).exp() - x1 - prem > 0.0 { 0.0 } else { newton(s, OptionType::Put, 1.0, -x1, 1.0)? };
                let i2 = if prem == 0.0 { Real::INFINITY } else { newton(s, OptionType::Put, 0.0, 0.0, 0.0)? };
                (i1, i2)
            }
        };
        let y = |i: Real| ((s / i).ln() + (b + vol * vol / 2.0) * t1) / (vol * t1.sqrt());
        let y1 = y(i2);
        let y2 = y(i1);
        let z1 = ((s / x2).ln() + (b + vol * vol / 2.0) * t2) / (vol * t2.sqrt());
        let z2 = ((s / x1).ln() + (b + vol * vol / 2.0) * t1) / (vol * t1.sqrt());
        let rho = (t1 / t2).sqrt();
        let n = CumulativeNormalDistribution::standard();
        let n2 = |lo: Real, hi: Real| n.value(hi) - n.value(lo);
        let m2 = |aa: Real, bb: Real, cc: Real, dd: Real| -> QlResult<Real> {
            let m = BivariateCumulativeNormalDistributionDr78::new(rho)?;
            Ok(m.value(bb, dd) - m.value(aa, dd) - m.value(bb, cc) + m.value(aa, cc))
        };
        let growth1 = df_q(t1)?;
        let disc1 = df_r(t1)?;
        let bsm = BlackCalculator::new(payoff.option_type(), x1, s * growth1 / disc1, vol * t1.sqrt(), disc1)?.value();
        let ninf = Real::NEG_INFINITY;
        let vs = vol * t1.sqrt();
        let vt = vol * t2.sqrt();
        let value = match payoff.option_type() {
            OptionType::Call => {
                bsm + s * ((b - r) * t2).exp() * m2(y1, y2, ninf, z1)?
                    - x2 * (-r * t2).exp() * m2(y1 - vs, y2 - vs, ninf, z1 - vt)?
                    - s * ((b - r) * t1).exp() * n2(y1, z2)
                    + x1 * (-r * t1).exp() * n2(y1 - vs, z2 - vs)
                    - prem * (-r * t1).exp() * n2(y1 - vs, y2 - vs)
            }
            OptionType::Put => {
                -s * ((b - r) * t1).exp() * n2(y2, Real::INFINITY)
                    + x1 * (-r * t1).exp() * n2(y2 - vs, Real::INFINITY)
                    - s * ((b - r) * t2).exp() * m2(-y2, -y1, ninf, -z1)?
                    + x2 * (-r * t2).exp() * m2(vs - y2, vs - y1, ninf, vt - z1)?
                    - prem * (-r * t1).exp() * n2(y1 - vs, y2 - vs)
            }
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticHolderExtensibleOptionEngine`] to `option`.
pub fn set_analytic_holder_extensible_option_engine(
    option: &mut crate::instruments::HolderExtensibleOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticHolderExtensibleOptionEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{HolderExtensibleOption, PlainVanillaPayoff};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    fn quote_h(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    #[rustfmt::skip]
    fn flat_rate(today: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(today, quote_h(r), Actual360::new(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>)
    }

    #[rustfmt::skip]
    fn flat_vol(today: Date, v: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(today, None, quote_h(v), Actual360::new())) as Shared<dyn BlackVolTermStructure>)
    }

    #[rustfmt::skip]
    fn price(ty: OptionType, q: Real) -> Real {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let r = flat_rate(today, 0.08);
        let p = shared(BlackScholesMertonProcess::new(
            quote_h(100.0), flat_rate(today, q), r, flat_vol(today, 0.25),
        ));
        let ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 180));
        let mut option = HolderExtensibleOption::new(
            1.0, today + 270, 105.0, PlainVanillaPayoff::new(ty, 100.0), ex, settings,
        );
        set_analytic_holder_extensible_option_engine(&mut option, p);
        option.npv().unwrap()
    }

    /// `extensibleoptions.cpp` `testAnalyticHolderExtensibleOptionEngine` @ 1e-4.
    #[test]
    #[rustfmt::skip]
    fn holder_extensible_haug_npv() {
        let call = price(OptionType::Call, 0.0);
        assert!((call - 9.4233).abs() <= 1e-4, "Haug call 9.4233 vs {call}");
        let put = price(OptionType::Put, 0.0);
        assert!((put - 7.2042).abs() <= 1e-4, "put 7.2042 vs {put}");
        let q_call = price(OptionType::Call, 0.05);
        assert!((q_call - 7.8299).abs() <= 1e-4, "q≠0 call 7.8299 vs {q_call}");
    }
}
