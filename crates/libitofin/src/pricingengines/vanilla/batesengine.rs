//! Analytic Bates engine (Heston + log-normal jumps via Gatheral Fourier).
//!
//! Port of `ql/pricingengines/vanilla/batesengine.{hpp,cpp}`: [`BatesEngine`]
//! prices European plain-vanilla options under the Bates jump-diffusion SV
//! model by extending the Heston Gatheral characteristic function with the
//! jump `addOnTerm` (`batesengine.cpp:39-54`).
//!
//! Deferred: `BatesDetJumpEngine`, `BatesDoubleExpEngine`,
//! `BatesDoubleExpDetJumpEngine`, and the Gauss-Lobatto constructor.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    OneAssetOptionEngine, OneAssetOptionResults, OptionArguments, PlainVanillaPayoff,
    StrikedTypePayoff, TypePayoff,
};
use crate::models::equity::BatesModel;
use crate::models::model::CalibratedModelHolder;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::vanilla::analytichestonengine::{Integration, price_vanilla_gatheral};
use crate::require;
use crate::shared::SharedMut;
use crate::stochasticprocess::StochasticProcess;
use crate::types::{Complex, Real, Size, Time};

/// Bates jump characteristic-function addon
/// (`BatesEngine::addOnTerm`, `batesengine.cpp:39-54`).
pub fn bates_add_on_term(
    phi: Real,
    t: Time,
    j: Size,
    lambda: Real,
    nu: Real,
    delta: Real,
) -> Complex {
    let delta2 = 0.5 * delta * delta;
    let i = if j == 1 { 1.0 } else { 0.0 };
    let g = Complex::new(i, phi);
    t * lambda
        * ((nu * g + delta2 * g * g).exp()
            - Complex::new(1.0, 0.0)
            - g * ((nu + delta2).exp() - 1.0))
}

/// Analytic Bates pricing engine (`batesengine.hpp:106`): Gatheral Fourier
/// with log-normal jump `addOnTerm`.
pub struct BatesEngine {
    base: OneAssetOptionEngine,
    model: SharedMut<BatesModel>,
    integration: Integration,
}

impl BatesEngine {
    /// `BatesEngine(model, integrationOrder)` (`batesengine.cpp:26-30`):
    /// Gauss-Laguerre Gatheral path (default order 144).
    ///
    /// # Errors
    ///
    /// Propagates [`Integration::gauss_laguerre`] failure.
    pub fn new(model: SharedMut<BatesModel>, integration_order: Size) -> QlResult<BatesEngine> {
        let integration = Integration::gauss_laguerre(integration_order)?;
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        Ok(BatesEngine {
            base,
            model,
            integration,
        })
    }

    /// Default integration order 144 (`batesengine.hpp:109`).
    pub fn with_default_order(model: SharedMut<BatesModel>) -> QlResult<BatesEngine> {
        BatesEngine::new(model, 144)
    }
}

impl AsObservable for BatesEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BatesEngine {
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
        let Some(exercise) = &arguments.exercise else {
            fail!("no exercise given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European option"
        );
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let payoff: &dyn StrikedTypePayoff = &**payoff;
        let Some(payoff) = (payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("non plain vanilla payoff given");
        };
        let payoff = *payoff;
        let maturity_date = exercise.last_date();

        let model = self.model.borrow();
        let process = model.process();
        let kappa = model.kappa();
        let sigma = model.sigma();
        let theta = model.theta();
        let rho = model.rho();
        let v0 = model.v0();
        let lambda = model.lambda();
        let nu = model.nu();
        let delta = model.delta();
        drop(model);

        let spot = process.s0().current_link()?.value()?;
        if spot.is_nan() || spot <= 0.0 {
            fail!("negative or null underlying given");
        }

        let dividend_discount = process
            .dividend_yield()
            .current_link()?
            .discount_date(maturity_date, false)?;
        let risk_free_discount_date = process
            .risk_free_rate()
            .current_link()?
            .discount_date(maturity_date, false)?;
        let fwd = spot * dividend_discount / risk_free_discount_date;

        let time = process.time(&maturity_date)?;
        let dr = process
            .risk_free_rate()
            .current_link()?
            .discount(time, false)?;

        let add_on =
            move |phi: Real, t: Time, j: Size| bates_add_on_term(phi, t, j, lambda, nu, delta);

        let value = price_vanilla_gatheral(
            &self.integration,
            kappa,
            theta,
            sigma,
            v0,
            rho,
            spot,
            payoff.strike(),
            time,
            fwd,
            dr,
            payoff.option_type(),
            add_on,
        )?;

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::VanillaOption;
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::blackformula::black_formula;
    use crate::processes::BatesProcess;
    use crate::quotes::make_quote_handle;
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actualactual::{ActualActual, Convention as ActActConvention};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    /// ORACLE `testAnalyticVsBlack` (`test-suite/batesmodel.cpp:84-137`): tiny
    /// jumps + near-zero vol-of-vol → Black @ 2e-7.
    #[test]
    fn analytic_vs_black() {
        let settlement = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(settlement);

        let day_counter = ActualActual::with_convention(ActActConvention::ISDA);
        let exercise_date = settlement + Period::new(6, TimeUnit::Months);

        let risk_free = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.1,
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let dividend = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.04,
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let s0 = make_quote_handle(32.0).handle();

        let year_fraction = day_counter.year_fraction(settlement, exercise_date);
        let forward_price = 32.0 * ((0.1 - 0.04) * year_fraction).exp();
        let expected = black_formula(
            OptionType::Put,
            30.0,
            forward_price,
            (0.05 * year_fraction).sqrt(),
            (-0.1 * year_fraction).exp(),
            0.0,
        )
        .unwrap();

        let process = shared(BatesProcess::new(
            risk_free, dividend, s0, 0.05,   // v0
            5.0,    // kappa
            0.05,   // theta
            1.0e-4, // sigma
            0.0,    // rho
            0.0001, // lambda
            0.0,    // nu
            0.0001, // delta
        ));
        let model = BatesModel::new(process).unwrap();

        let payoff =
            shared(PlainVanillaPayoff::new(OptionType::Put, 30.0)) as Shared<dyn StrikedTypePayoff>;
        let exercise = shared(EuropeanExercise::new(exercise_date)) as Shared<dyn Exercise>;
        let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&settings));
        let engine =
            shared_mut(BatesEngine::new(model, 64).unwrap()) as SharedMut<dyn PricingEngine>;
        option.base_mut().set_pricing_engine(engine);

        let calculated = option.npv().unwrap();
        let error = (calculated - expected).abs();
        assert!(
            error <= 2.0e-7,
            "failed to reproduce Black price with BatesEngine: calculated={calculated} \
             expected={expected} error={error}"
        );
    }

    /// Tiny-λ Bates Gatheral should match pure-Heston Gatheral on the same SV
    /// parameters (identity pin; jump CF → 0).
    #[test]
    fn tiny_lambda_matches_heston_gatheral() {
        use crate::models::HestonModel;
        use crate::pricingengines::vanilla::analytichestonengine::{
            AnalyticHestonEngine, ComplexLogFormula,
        };
        use crate::processes::HestonProcess;
        use crate::time::daycounters::actual360::Actual360;

        let settlement = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(settlement);
        let expiry = settlement + Period::new(1, TimeUnit::Years);

        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                settlement,
                rate,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let s0 = make_quote_handle(100.0).handle();
        let (v0, kappa, theta, sigma, rho) = (0.04, 1.5, 0.04, 0.3, -0.5);

        let heston_process = shared(HestonProcess::new(
            flat(0.05),
            flat(0.02),
            s0.clone(),
            v0,
            kappa,
            theta,
            sigma,
            rho,
        ));
        let heston_model = HestonModel::new(heston_process).unwrap();
        let heston_engine = shared_mut(
            AnalyticHestonEngine::with_complex_log(heston_model, ComplexLogFormula::Gatheral, 128)
                .unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let bates_process = shared(BatesProcess::new(
            flat(0.05),
            flat(0.02),
            s0,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            1e-12, // λ ≈ 0
            0.0,
            1e-4,
        ));
        let bates_model = BatesModel::new(bates_process).unwrap();
        let bates_engine =
            shared_mut(BatesEngine::new(bates_model, 128).unwrap()) as SharedMut<dyn PricingEngine>;

        let mk = |engine: SharedMut<dyn PricingEngine>| {
            let payoff = shared(PlainVanillaPayoff::new(OptionType::Call, 100.0))
                as Shared<dyn StrikedTypePayoff>;
            let exercise = shared(EuropeanExercise::new(expiry)) as Shared<dyn Exercise>;
            let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&settings));
            option.base_mut().set_pricing_engine(engine);
            option.npv().unwrap()
        };

        let heston_npv = mk(heston_engine);
        let bates_npv = mk(bates_engine);
        assert!(
            (heston_npv - bates_npv).abs() < 1e-8,
            "λ→0 Bates {bates_npv} vs Heston Gatheral {heston_npv}"
        );
    }
}
