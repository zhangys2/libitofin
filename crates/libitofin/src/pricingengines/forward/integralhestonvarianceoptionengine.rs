//! Integral Heston variance-option engine.
//!
//! Port of `ql/experimental/varianceoption/integralhestonvarianceoptionengine.cpp`.
//! The oscillatory integral follows Bailey and Swarztrauber (SIAM J. Sci.
//! Comput. 15(5), 1994). A plain call uses the one-dimensional formula; every
//! other striked payoff uses the two-dimensional inversion. The engine rejects
//! a dividend yield, matching QuantLib.

use std::any::Any;

use crate::errors::QlResult;
use crate::fail;
use crate::instruments::{
    PlainVanillaPayoff, StrikedTypePayoff, TypePayoff, VarianceOptionArguments,
    VarianceOptionEngine,
};
use crate::interestrate::Compounding;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::HestonProcess;
use crate::require;
use crate::shared::Shared;
use crate::time::frequency::Frequency;
use crate::types::{Complex, Real, Time};

/// Prices a variance option from the Heston variance parameters.
pub struct IntegralHestonVarianceOptionEngine {
    base: VarianceOptionEngine,
    process: Shared<HestonProcess>,
}

impl IntegralHestonVarianceOptionEngine {
    /// Builds the engine. A dividend yield on `process` is rejected at pricing.
    pub fn new(process: Shared<HestonProcess>) -> Self {
        let base = VarianceOptionEngine::new(
            VarianceOptionArguments::default(),
            crate::instrument::InstrumentResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for IntegralHestonVarianceOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for IntegralHestonVarianceOptionEngine {
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
        require!(
            self.process.dividend_yield().is_empty(),
            "this engine does not manage dividend yields"
        );
        let (payoff, notional, maturity, settings) = {
            let args = self.base.arguments();
            args.validate()?;
            let Some(payoff) = args.payoff.as_ref() else {
                fail!("no strike given");
            };
            let Some(notional) = args.notional else {
                fail!("no notional given");
            };
            let Some(maturity) = args.maturity_date else {
                fail!("null maturity date given");
            };
            let Some(settings) = args.settings.as_ref() else {
                fail!("variance option settings missing");
            };
            (
                Shared::clone(payoff),
                notional,
                maturity,
                Shared::clone(settings),
            )
        };
        let Some(evaluation) = settings.evaluation_date() else {
            fail!("no evaluation date set");
        };
        let risk_free = self.process.risk_free_rate().current_link()?;
        let day_counter = risk_free.require_day_counter()?;
        let tau = day_counter.year_fraction(evaluation, maturity);
        require!(
            tau > 0.0,
            "variance option maturity must be after the evaluation date"
        );
        let r = risk_free
            .zero_rate_date(
                maturity,
                day_counter,
                Compounding::Continuous,
                Frequency::Annual,
                false,
            )?
            .rate();
        let epsilon = self.process.sigma();
        let chi = self.process.kappa();
        let theta = self.process.theta();
        let rho = self.process.rho();
        let v0 = self.process.v0();
        let unit = if let Some(vanilla) =
            (payoff.as_ref() as &dyn Any).downcast_ref::<PlainVanillaPayoff>()
            && vanilla.option_type() == OptionType::Call
        {
            ivop_one_dim(epsilon, chi, theta, rho, v0, vanilla.strike(), tau, r)?
        } else {
            ivop_two_dim(epsilon, chi, theta, rho, v0, tau, r, |s| payoff.value(s))?
        };
        self.base.results_mut().value = Some(unit * notional);
        Ok(())
    }
}

fn alt_sign(j: i32) -> Real {
    if j % 2 == 0 { 1.0 } else { -1.0 }
}

#[allow(clippy::too_many_arguments)]
fn ivop_one_dim(
    eps: Real,
    chi: Real,
    theta: Real,
    _rho: Real,
    v0: Real,
    strike: Real,
    tau: Time,
    rtax: Real,
) -> QlResult<Real> {
    let s = 2.0 * chi * theta / (eps * eps) - 1.0;
    require!(
        eps > 0.0 && s.is_finite() && s > 0.0,
        "this parameter must be greater than zero-> {s}"
    );
    let ss = s + 1.0;
    let pi2 = 2.0 * std::f64::consts::PI;
    let dstep = 256.0;
    let nris = pi2.sqrt() / dstep;
    let mm = (dstep * dstep) as i32;
    let ui = Complex::new(0.0, 1.0);
    let mut ff = vec![Complex::new(0.0, 0.0); mm as usize];
    for j in 0..mm {
        let xi = Complex::new((f64::from(j) - f64::from(mm) / 2.0) * nris, 0.0);
        let caux2 = chi * chi + 2.0 * eps * eps * xi * ui;
        let zita = 0.5 * caux2.sqrt();
        let decay = (-2.0 * tau * zita).exp();
        let beta = 0.5 * chi + zita + decay * (zita - 0.5 * chi);
        let gamma = 1.0 - decay;
        let mut phase = Complex::new(-ss * tau, 0.0) * (zita - 0.5 * chi);
        phase += Complex::new(ss, 0.0) * (2.0 * (zita / beta)).ln();
        phase += -v0 * ui * xi * (gamma / beta);
        let mut value = phase.exp();
        let radius = xi.norm();
        let contrib = if radius > 1.0e-6 {
            let carry = (ui * xi * strike).exp();
            -strike / (ui * xi) + (carry - 1.0) / (ui * xi * ui * xi)
        } else {
            Complex::new(strike * strike * 0.5, 0.0)
        };
        value *= contrib;
        ff[j as usize] = value;
    }
    let mut csum = Complex::new(0.0, 0.0);
    for j in 0..mm {
        let caux2 =
            -2.0 * std::f64::consts::PI * f64::from(mm) * f64::from(j) * 0.5 / f64::from(mm);
        csum += ff[j as usize] * alt_sign(j) * (ui * caux2).exp();
    }
    let sign_mm = Complex::new(alt_sign(mm), 0.0).sqrt();
    csum *= sign_mm * nris / pi2;
    let vero = theta * tau + (1.0 - (-chi * tau).exp()) * (v0 - theta) / chi - strike;
    csum += vero;
    let impart = csum.im;
    require!(
        impart.abs() <= 1e-12,
        "imaginary part option (must be zero) = {impart}"
    );
    let price = (-rtax * tau).exp() * csum.re;
    require!(price.is_finite(), "variance option price must be finite");
    Ok(price)
}

#[allow(clippy::too_many_arguments)]
fn ivop_two_dim(
    eps: Real,
    chi: Real,
    theta: Real,
    _rho: Real,
    v0: Real,
    tau: Time,
    rtax: Real,
    payoff: impl Fn(Real) -> Real,
) -> QlResult<Real> {
    let s = 2.0 * chi * theta / (eps * eps) - 1.0;
    require!(
        eps > 0.0 && s.is_finite() && s > 0.0,
        "this parameter must be greater than zero-> {s}"
    );
    let ss = s + 1.0;
    let pi2 = 2.0 * std::f64::consts::PI;
    let dstep = 64.0;
    let nris = pi2.sqrt() / dstep;
    let mm = (dstep * dstep) as i32;
    let ui = Complex::new(0.0, 1.0);
    let mut xiv = vec![0.0; mm as usize];
    let mut ivet = vec![0.0; mm as usize];
    let mut ff = vec![Complex::new(0.0, 0.0); mm as usize];
    for j in 0..mm {
        let shift = f64::from(j) - f64::from(mm) / 2.0;
        xiv[j as usize] = shift * nris;
        ivet[j as usize] = shift * pi2 / (f64::from(mm) * nris);
    }
    for j in 0..mm {
        let xi = Complex::new(xiv[j as usize], 0.0);
        let caux2 = chi * chi + 2.0 * eps * eps * xi * ui;
        let zita = 0.5 * caux2.sqrt();
        let decay = (-2.0 * tau * zita).exp();
        let beta = 0.5 * chi + zita + decay * (zita - 0.5 * chi);
        let gamma = 1.0 - decay;
        let mut phase = Complex::new(-ss * tau, 0.0) * (zita - 0.5 * chi);
        phase += Complex::new(ss, 0.0) * (2.0 * (zita / beta)).ln();
        phase += -v0 * ui * xi * (gamma / beta);
        ff[j as usize] = phase.exp();
    }
    let mut sumr = 0.0;
    for k in 0..mm {
        let payoff_val = payoff(-ivet[k as usize]);
        let dxi = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(mm) * ui;
        let mut csum = Complex::new(0.0, 0.0);
        for j in 0..mm {
            let z = -f64::from(j) * dxi;
            csum += ff[j as usize] * alt_sign(j) * z.exp();
        }
        csum *= alt_sign(k) * nris / pi2;
        sumr += payoff_val * csum.re;
    }
    sumr *= nris;
    let price = (-rtax * tau).exp() * sumr;
    require!(price.is_finite(), "variance option price must be finite");
    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VarianceOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn price(kind: crate::option::OptionType, strike: Real, years: Real, v0: Real) -> Real {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let r_quote = shared(SimpleQuote::new(0.0));
        let r_ts = shared(FlatForward::new(
            today,
            Handle::new(Shared::clone(&r_quote) as Shared<dyn Quote>),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>;
        let process = shared(HestonProcess::new(
            Handle::new(r_ts),
            Handle::<dyn YieldTermStructure>::empty(),
            Handle::new(shared(SimpleQuote::new(1.0)) as Shared<dyn Quote>),
            v0,
            2.0,
            0.01,
            0.1,
            -0.5,
        ));
        let days = (360.0 * years) as i32;
        let mut option = VarianceOption::new(
            shared(PlainVanillaPayoff::new(kind, strike)),
            1.0,
            today,
            today + days,
            settings,
        )
        .unwrap();
        option
            .base_mut()
            .set_pricing_engine(shared_mut(IntegralHestonVarianceOptionEngine::new(process))
                as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    #[test]
    fn test_integral_heston() {
        let call = price(Call, 0.05, 1.5, 2.0);
        assert!(
            (call - 0.9104619).abs() <= 1.0e-7,
            "call {call} vs 0.9104619"
        );
        let put = price(Put, 0.7, 1.0, 1.5);
        assert!((put - 0.0466796).abs() <= 1.0e-7, "put {put} vs 0.0466796");
    }

    #[test]
    fn evaluation_date_shortens_the_horizon() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let r_ts = shared(FlatForward::new(
            today,
            Handle::new(shared(SimpleQuote::new(0.0)) as Shared<dyn Quote>),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>;
        let process = shared(HestonProcess::new(
            Handle::new(r_ts),
            Handle::<dyn YieldTermStructure>::empty(),
            Handle::new(shared(SimpleQuote::new(1.0)) as Shared<dyn Quote>),
            2.0,
            2.0,
            0.01,
            0.1,
            -0.5,
        ));
        let mut option = VarianceOption::new(
            shared(PlainVanillaPayoff::new(Call, 0.05)),
            1.0,
            today,
            today + 540,
            Shared::clone(&settings),
        )
        .unwrap();
        option
            .base_mut()
            .set_pricing_engine(shared_mut(IntegralHestonVarianceOptionEngine::new(process))
                as SharedMut<dyn PricingEngine>);
        let at_inception = option.npv().unwrap();
        assert!((at_inception - 0.9104619).abs() <= 1.0e-7);
        settings.set_evaluation_date(today + 180);
        let rolled = option.npv().unwrap();
        assert!(
            (rolled - 0.8203413932).abs() <= 1.0e-6,
            "rolled horizon {rolled}"
        );
    }
}
