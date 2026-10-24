//! Analytic Heston engine for forward-starting European options (T=0 path).
//!
//! Port of `ql/experimental/forward/analytichestonforwardeuropeanengine.{hpp,cpp}`
//! restricted to `resetTime <= 1e-3` (`calculate` falls back to `calculateP1P2`).
//! The `tReset > 0` 2-D propagator (modified Bessel, Kruse 2003) is deferred.

use std::any::Any;
use std::f64::consts::PI;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instruments::{
    ForwardOptionArguments, OneAssetOptionResults, PlainVanillaPayoff, StrikedTypePayoff,
    TypePayoff,
};
use crate::math::integrals::gaussianquadratures::GaussianQuadrature;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::HestonChf;
use crate::processes::HestonProcess;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess;
use crate::types::{Complex, Real, Time};
use crate::{fail, require};

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// `AnalyticHestonForwardEuropeanEngine` T=0 analytic path.
pub struct AnalyticHestonForwardEuropeanEngine {
    base: ForwardEngineBase,
    process: Shared<HestonProcess>,
}

impl AnalyticHestonForwardEuropeanEngine {
    /// `AnalyticHestonForwardEuropeanEngine(process, integrationOrder = 144)`.
    /// The Gauss-Laguerre order is unused until the `tReset > 0` path lands.
    ///
    /// # Errors
    ///
    /// `sigma <= 0.1` (QuantLib constructor guard: propagator numerical issues).
    pub fn new(process: Shared<HestonProcess>) -> QlResult<Self> {
        require!(
            process.sigma() > 0.1,
            "Very low values (<~10%) for Heston Vol-of-Vol cause numerical issues \
             in this implementation of the propagator function, try using \
             MCForwardEuropeanHestonEngine Monte-Carlo engine instead"
        );
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Ok(Self { base, process })
    }

    fn p12_integrand(chf: &HestonChf, log_k: Real, tenor: Time, p1: bool, phi: Real) -> Real {
        let phi_right = 100.0;
        let phi_dash = (0.5 + 1e-8 + 0.5 * phi) * phi_right;
        let i = Complex::new(0.0, 1.0);
        let adj = if p1 {
            Complex::new(0.0, -1.0)
        } else {
            Complex::new(0.0, 0.0)
        };
        0.5 * phi_right
            * ((-phi_dash * log_k * i).exp() / (phi_dash * i)
                * chf.chf(Complex::new(phi_dash, 0.0) + adj, tenor))
            .re
    }

    fn calculate_p1_p2(
        chf: &HestonChf,
        tenor: Time,
        spot: Real,
        strike: Real,
        ratio: Real,
    ) -> QlResult<(Real, Real)> {
        let log_k = (strike * ratio / spot).ln();
        let integrator = GaussianQuadrature::legendre(128)?;
        let p1 = integrator.integrate(|phi| Self::p12_integrand(chf, log_k, tenor, true, phi));
        let p2 = integrator.integrate(|phi| Self::p12_integrand(chf, log_k, tenor, false, phi));
        Ok((0.5 + p1 / PI, 0.5 + p2 / PI))
    }
}

impl AsObservable for AnalyticHestonForwardEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticHestonForwardEuropeanEngine {
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
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("no exercise given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European option"
        );
        let Some(payoff) = args.payoff.as_ref() else {
            fail!("no payoff given");
        };
        let payoff: &dyn StrikedTypePayoff = &**payoff;
        let Some(payoff) = (payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("non plain vanilla payoff given");
        };
        let Some(moneyness) = args.moneyness else {
            fail!("null moneyness given");
        };
        let option_type = payoff.option_type();
        let reset_date = args.reset_date;
        let expiry_date = exercise.last_date();

        let reset_time = self.process.time(&reset_date)?;
        let expiry_time = self.process.time(&expiry_date)?;
        require!(reset_time >= 0.0, "Reset Date cannot be in the past");
        require!(expiry_time >= 0.0, "Expiry Date cannot be in the past");
        require!(
            reset_time <= 1e-3,
            "AnalyticHestonForwardEuropeanEngine tReset>0 (2-D propagator) is deferred"
        );

        let r = self.process.risk_free_rate().current_link()?;
        let q = self.process.dividend_yield().current_link()?;
        let expiry_dcf = r.discount(expiry_time, false)?;
        let reset_dcf = r.discount(reset_time, false)?;
        let expiry_div = q.discount(expiry_time, false)?;
        let reset_div = q.discount(reset_time, false)?;
        let expiry_ratio = expiry_dcf / expiry_div;
        let reset_ratio = reset_dcf / reset_div;
        let spot = self.process.s0().current_link()?.value()?;
        let tenor = expiry_time - reset_time;
        let chf = HestonChf::new(
            self.process.kappa(),
            self.process.theta(),
            self.process.sigma(),
            self.process.rho(),
            self.process.v0(),
        );
        let (p1, p2) = Self::calculate_p1_p2(&chf, tenor, spot, moneyness * spot, expiry_ratio)?;
        let fwd = spot / expiry_ratio;
        let value = match option_type {
            OptionType::Call => expiry_dcf * (fwd * p1 - moneyness * spot * p2 / reset_ratio),
            OptionType::Put => {
                expiry_dcf * (moneyness * spot * (1.0 - p2) / reset_ratio - fwd * (1.0 - p1))
            }
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, VanillaOption};
    use crate::models::HestonModel;
    use crate::pricingengines::vanilla::analytichestonengine::AnalyticHestonEngine;
    use crate::pricingengines::vanilla::test_market::{market, today};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn smile_heston() -> Shared<HestonProcess> {
        let mkt = market();
        let sigma_bs = 0.245;
        mkt.set(100.0, 0.04, 0.01, sigma_bs);
        shared(HestonProcess::new(
            mkt.process.risk_free_rate(),
            mkt.process.dividend_yield(),
            mkt.process.state_variable(),
            sigma_bs * sigma_bs,
            1.0,
            0.08,
            0.39,
            -0.93,
        ))
    }

    /// `forwardoption.cpp` `testHestonMCPrices` Test 2 analytic arm:
    /// `reset = today` vs vanilla `AnalyticHestonEngine(model, 96)` @ 5e-4.
    #[test]
    fn forward_heston_analytic_t0_vs_vanilla() {
        let mkt = market();
        let sigma_bs = 0.245;
        mkt.set(100.0, 0.04, 0.01, sigma_bs);
        let heston = shared(HestonProcess::new(
            mkt.process.risk_free_rate(),
            mkt.process.dividend_yield(),
            mkt.process.state_variable(),
            sigma_bs * sigma_bs,
            1.0,
            0.08,
            0.39,
            -0.93,
        ));
        let model = HestonModel::new(Shared::clone(&heston)).unwrap();
        let exercise = shared(EuropeanExercise::new(
            today() + Period::new(1, TimeUnit::Years),
        ));
        let analytic = shared_mut(AnalyticHestonEngine::new(SharedMut::clone(&model), 96).unwrap())
            as SharedMut<dyn PricingEngine>;
        let forward =
            shared_mut(AnalyticHestonForwardEuropeanEngine::new(Shared::clone(&heston)).unwrap())
                as SharedMut<dyn PricingEngine>;
        let payoff_fwd =
            |t| shared(PlainVanillaPayoff::new(t, 0.0)) as Shared<dyn StrikedTypePayoff>;
        for option_type in [OptionType::Call, OptionType::Put] {
            for m in [0.8, 0.9, 1.0, 1.1, 1.2] {
                let vanilla_payoff = shared(PlainVanillaPayoff::new(option_type, 100.0 * m))
                    as Shared<dyn StrikedTypePayoff>;
                let mut vanilla = VanillaOption::new(
                    vanilla_payoff,
                    Shared::clone(&exercise) as _,
                    Shared::clone(&mkt.settings),
                );
                vanilla
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&analytic));
                let vanilla_npv = vanilla.npv().unwrap();
                let mut option = ForwardVanillaOption::new(
                    m,
                    today(),
                    payoff_fwd(option_type),
                    Shared::clone(&exercise) as _,
                    Shared::clone(&mkt.settings),
                );
                option
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&forward));
                let fwd_npv = option.npv().unwrap();
                let error = (vanilla_npv - fwd_npv).abs() / 100.0;
                assert!(
                    error <= 5e-4,
                    "{option_type:?} m={m}: vanilla={vanilla_npv} forward={fwd_npv} rel={error}"
                );
            }
        }
    }

    #[test]
    fn factory_rejects_low_vol_of_vol() {
        let mkt = market();
        mkt.set(100.0, 0.04, 0.01, 0.245);
        let heston = shared(HestonProcess::new(
            mkt.process.risk_free_rate(),
            mkt.process.dividend_yield(),
            mkt.process.state_variable(),
            0.06,
            1.0,
            0.06,
            0.05,
            -0.5,
        ));
        assert!(AnalyticHestonForwardEuropeanEngine::new(heston).is_err());
    }

    #[test]
    fn t_reset_positive_is_deferred() {
        let mkt = market();
        let sigma_bs = 0.245;
        mkt.set(100.0, 0.04, 0.01, sigma_bs);
        let heston = smile_heston();
        let mut option = ForwardVanillaOption::new(
            1.0,
            today() + Period::new(6, TimeUnit::Months),
            shared(PlainVanillaPayoff::new(OptionType::Call, 0.0)) as _,
            shared(EuropeanExercise::new(
                today() + Period::new(1, TimeUnit::Years),
            )) as _,
            Shared::clone(&mkt.settings),
        );
        option.base_mut().set_pricing_engine(shared_mut(
            AnalyticHestonForwardEuropeanEngine::new(heston).unwrap(),
        ) as SharedMut<dyn PricingEngine>);
        assert!(option.npv().is_err());
    }
}
