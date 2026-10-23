//! Monte Carlo prices of European calls and puts on an SLV-JD stock.
//!
//! Prices read the calibrated particle snapshots: a call pays `(S_T - K)+`
//! only on survival, and a put adds the claim `K` paid when the stock has
//! jumped to zero. An equity default swap triggers on that jump or on a
//! spot barrier.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    OneAssetOptionEngine, OneAssetOptionResults, OptionArguments, PlainVanillaPayoff,
    StrikedTypePayoff, TypePayoff,
};
use crate::models::equity::jumptodefault::{defaultable_option_value, SlvJtdCalibrationResult};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::require;
use crate::shared::{shared, Shared};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Rate, Real, Time};

/// European engine on a calibrated SLV-JD particle set.
pub struct McSlvJumpToDefaultVanillaEngine {
    base: OneAssetOptionEngine,
    calibration: Shared<SlvJtdCalibrationResult>,
}

impl McSlvJumpToDefaultVanillaEngine {
    /// Engine that prices from `calibration` without resimulating.
    pub fn new(calibration: Shared<SlvJtdCalibrationResult>) -> Self {
        Self {
            base: OneAssetOptionEngine::new(
                OptionArguments::default(),
                OneAssetOptionResults::default(),
            ),
            calibration,
        }
    }

    /// The calibration the engine prices from.
    pub fn calibration(&self) -> &Shared<SlvJtdCalibrationResult> {
        &self.calibration
    }

    /// Price a European call or put.
    ///
    /// Returns `(npv, survival component, default-floor component)`. The
    /// expiry is snapped to the nearest calibrated time.
    ///
    /// # Errors
    ///
    /// Returns an error when `expiry` is off the calibration grid or the
    /// strike is not positive.
    pub fn price_components(
        &self,
        option_type: OptionType,
        strike: Real,
        expiry: Time,
    ) -> QlResult<(Real, Real, Real)> {
        let index = self.snapshot_index(expiry)?;
        let snap = &self.calibration.snapshots[index];
        let discount = self
            .calibration
            .process
            .slv()
            .risk_free_rate()
            .current_link()?
            .discount(snap.time, true)?;
        defaultable_option_value(
            option_type,
            strike,
            discount,
            &snap.spots,
            &snap.survival_weights,
        )
    }

    /// Par spread of an equity default swap.
    ///
    /// Protection pays `1 - eds_recovery` if the spot has traded at or below
    /// `barrier_ratio * S_0` or the name has jumped to default. Returns
    /// `(par spread, total trigger probability, credit-only default probability)`.
    ///
    /// # Errors
    ///
    /// Returns an error when the maturity is off the grid, the barrier ratio
    /// is negative, or recovery is outside `[0, 1)`.
    pub fn price_equity_default_swap(
        &self,
        maturity: Time,
        barrier_ratio: Real,
        eds_recovery: Real,
    ) -> QlResult<(Rate, Real, Real)> {
        require!(
            maturity.is_finite() && maturity > 0.0,
            "equity default swap maturity must be positive, got {maturity}"
        );
        require!(
            barrier_ratio.is_finite() && barrier_ratio >= 0.0,
            "equity default swap barrier ratio must be non-negative, got {barrier_ratio}"
        );
        require!(
            eds_recovery.is_finite() && (0.0..1.0).contains(&eds_recovery),
            "equity default swap recovery must lie in [0, 1), got {eds_recovery}"
        );
        let index = self.snapshot_index(maturity)?;
        let spot0 = self
            .calibration
            .process
            .slv()
            .s0()
            .current_link()?
            .value()?;
        let barrier = barrier_ratio * spot0;
        let risk_free = self
            .calibration
            .process
            .slv()
            .risk_free_rate()
            .current_link()?;
        let n = self.calibration.snapshots[0].spots.len();
        let mut still_above = vec![true; n];
        let mut times = Vec::new();
        let mut eds_survival = Vec::new();
        let mut credit_survival = 0.0;
        for snap in self.calibration.snapshots.iter().take(index + 1) {
            let mut alive = 0.0;
            let mut survived = 0.0;
            for ((spot, weight), above) in snap
                .spots
                .iter()
                .zip(&snap.survival_weights)
                .zip(still_above.iter_mut())
            {
                if *spot <= barrier {
                    *above = false;
                }
                survived += weight;
                if *above {
                    alive += weight;
                }
            }
            let scale = n as Real;
            times.push(snap.time);
            eds_survival.push(alive / scale);
            credit_survival = survived / scale;
        }
        let (protection, annuity) =
            protection_and_annuity(&times, &eds_survival, &*risk_free, 1.0 - eds_recovery)?;
        require!(
            annuity > 0.0,
            "equity default swap risky annuity is not positive"
        );
        Ok((
            protection / annuity,
            1.0 - eds_survival[eds_survival.len() - 1],
            1.0 - credit_survival,
        ))
    }

    fn snapshot_index(&self, expiry: Time) -> QlResult<usize> {
        require!(
            expiry.is_finite() && expiry >= 0.0,
            "option expiry must be non-negative, got {expiry}"
        );
        let times = &self.calibration.time_grid;
        require!(!times.is_empty(), "calibration time grid is empty");
        let mut best = 0;
        let mut best_gap = Real::MAX;
        for (index, time) in times.iter().enumerate() {
            let gap = (time - expiry).abs();
            if gap < best_gap {
                best = index;
                best_gap = gap;
            }
        }
        let step = if times.len() > 1 {
            times[1] - times[0]
        } else {
            0.0
        };
        require!(
            best_gap <= 0.5 * step + 1.0e-10,
            "expiry {expiry} is off the calibration grid (nearest {})",
            times[best]
        );
        Ok(best)
    }
}

fn protection_and_annuity(
    times: &[Time],
    survival: &[Real],
    risk_free: &dyn YieldTermStructure,
    loss_given_default: Real,
) -> QlResult<(Real, Real)> {
    let mut protection = 0.0;
    let mut annuity = 0.0;
    for i in 0..times.len() - 1 {
        let dt = times[i + 1] - times[i];
        let mid = 0.5 * (times[i] + times[i + 1]);
        let discount = risk_free.discount(mid, true)?;
        let survival_mid = 0.5 * (survival[i] + survival[i + 1]);
        protection += loss_given_default * discount * (survival[i] - survival[i + 1]);
        annuity += discount * survival_mid * dt;
    }
    Ok((protection, annuity))
}

impl AsObservable for McSlvJumpToDefaultVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for McSlvJumpToDefaultVanillaEngine {
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
            "McSlvJumpToDefaultVanillaEngine requires European exercise"
        );
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let Some(payoff) = (&**payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("McSlvJumpToDefaultVanillaEngine requires a plain vanilla payoff");
        };
        let risk_free = self
            .calibration
            .process
            .slv()
            .risk_free_rate()
            .current_link()?;
        let expiry = risk_free.time_from_reference(exercise.last_date())?;
        let (value, survival_component, default_floor) =
            self.price_components(payoff.option_type(), payoff.strike(), expiry)?;
        let index = self.snapshot_index(expiry)?;
        let weights = &self.calibration.snapshots[index].survival_weights;
        let survival_probability = weights.iter().sum::<Real>() / weights.len() as Real;
        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        let extras = &mut results.instrument.additional_results;
        extras.insert(
            "survivalComponent".into(),
            shared(survival_component) as Shared<dyn Any>,
        );
        extras.insert(
            "defaultFloor".into(),
            shared(default_floor) as Shared<dyn Any>,
        );
        extras.insert(
            "survivalProbability".into(),
            shared(survival_probability) as Shared<dyn Any>,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instruments::PlainVanillaPayoff;
    use crate::interestrate::Compounding;
    use crate::models::equity::jumptodefault::{
        calibrate_slv_jump_to_default, MomentGrid, SlvJtdParticleConfig,
    };
    use crate::processes::HestonProcess;
    use crate::quotes::make_quote_handle;
    use crate::shared::shared;
    use crate::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
    use crate::termstructures::credit::flathazardrate::FlatHazardRate;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn calibration(hazard: Real, exponent: Real) -> Shared<SlvJtdCalibrationResult> {
        let rate = 0.03;
        let yield_curve = |value| {
            Handle::new(shared(FlatForward::with_rate(
                today(),
                value,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let black = Handle::new(shared(BlackConstantVol::new(
            today(),
            None,
            0.22,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>);
        let heston = shared(HestonProcess::new(
            yield_curve(rate),
            yield_curve(0.01),
            make_quote_handle(100.0).handle(),
            1.0,
            1.6,
            1.0,
            0.35,
            -0.45,
        ));
        let survival = Handle::new(shared(FlatHazardRate::with_rate(
            today(),
            hazard,
            Actual365Fixed::new(),
        )) as Shared<dyn DefaultProbabilityTermStructure>);
        let config = SlvJtdParticleConfig {
            n_particles: 200,
            time_steps_per_year: 20,
            n_strike_knots: 11,
            moment_grid: MomentGrid::new(8.0, 16).unwrap(),
            ..SlvJtdParticleConfig::default()
        };
        shared(
            calibrate_slv_jump_to_default(black, heston, survival, exponent, 1.0, 1.0, &config)
                .unwrap(),
        )
    }

    #[test]
    fn put_call_parity_holds_across_the_strike_range() {
        let calibration = calibration(0.02, -1.25);
        let engine = McSlvJumpToDefaultVanillaEngine::new(Shared::clone(&calibration));
        let spot = 100.0;
        let expiry = 1.0;
        let risk_free = calibration
            .process
            .slv()
            .risk_free_rate()
            .current_link()
            .unwrap();
        let dividend = calibration
            .process
            .slv()
            .dividend_yield()
            .current_link()
            .unwrap();
        let discount = risk_free.discount(expiry, true).unwrap();
        let dividend_discount = dividend.discount(expiry, true).unwrap();
        for moneyness in [0.65, 0.80, 1.00, 1.15, 1.30] {
            let strike = moneyness * spot;
            let (call, _, _) = engine
                .price_components(OptionType::Call, strike, expiry)
                .unwrap();
            let (put, _, _) = engine
                .price_components(OptionType::Put, strike, expiry)
                .unwrap();
            let parity = spot * dividend_discount - strike * discount;
            assert!(
                (call - put - parity).abs() <= 1e-11 * spot,
                "K={strike} call-put {} vs {parity}",
                call - put
            );
        }
    }

    #[test]
    fn equity_default_swap_contains_the_credit_event_and_the_limit_is_the_cds() {
        let calibration = calibration(0.03, -1.0);
        let engine = McSlvJumpToDefaultVanillaEngine::new(Shared::clone(&calibration));
        let maturity = 1.0;
        let recovery = 0.40;
        let (spread, trigger, credit) = engine
            .price_equity_default_swap(maturity, 0.30, recovery)
            .unwrap();
        assert!(trigger + 1e-14 >= credit, "{trigger} vs {credit}");
        assert!(spread.is_finite() && trigger.is_finite() && credit.is_finite());

        let (limit, limit_trigger, limit_credit) = engine
            .price_equity_default_swap(maturity, 0.0, recovery)
            .unwrap();
        assert!((limit_trigger - limit_credit).abs() <= 1e-14);
        let cds = cds_par_spread(&calibration, maturity, recovery);
        assert!(
            (limit - cds).abs() <= 5.0e-5,
            "barrier-free EDS {limit} vs CDS {cds}"
        );
    }

    fn cds_par_spread(
        calibration: &SlvJtdCalibrationResult,
        maturity: Time,
        recovery: Real,
    ) -> Real {
        let risk_free = calibration
            .process
            .slv()
            .risk_free_rate()
            .current_link()
            .unwrap();
        let mut times = Vec::new();
        let mut survival = Vec::new();
        for (time, probability) in calibration
            .time_grid
            .iter()
            .zip(&calibration.survival_probs_mc)
        {
            if *time > maturity + 1e-12 {
                break;
            }
            times.push(*time);
            survival.push(*probability);
        }
        let (protection, annuity) =
            protection_and_annuity(&times, &survival, &*risk_free, 1.0 - recovery).unwrap();
        protection / annuity
    }

    #[test]
    fn pricing_engine_rejects_american_exercise_and_fills_the_vanilla_results() {
        let calibration = calibration(0.02, 0.0);
        let mut engine = McSlvJumpToDefaultVanillaEngine::new(Shared::clone(&calibration));
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today() + 365));
        {
            let args = engine.base.arguments_mut();
            args.payoff = Some(Shared::clone(&payoff));
            args.exercise = Some(exercise);
        }
        engine.calculate().unwrap();
        let (expected, survival, floor) = engine
            .price_components(OptionType::Call, 100.0, 1.0)
            .unwrap();
        let value = engine.base.results().instrument.value.unwrap();
        assert!((value - expected).abs() <= 1e-14);
        let extras = &engine.base.results().instrument.additional_results;
        let component = extras["survivalComponent"]
            .as_ref()
            .downcast_ref::<Real>()
            .copied()
            .unwrap();
        let reported_floor = extras["defaultFloor"]
            .as_ref()
            .downcast_ref::<Real>()
            .copied()
            .unwrap();
        assert!((component - survival).abs() <= 1e-14);
        assert!((reported_floor - floor).abs() <= 1e-14);
        assert!(floor.abs() <= 1e-12 || reported_floor >= 0.0);

        let american = crate::exercise::AmericanExercise::over(today(), today() + 365).unwrap();
        engine.base.arguments_mut().exercise = Some(shared(american) as Shared<dyn Exercise>);
        assert!(engine
            .calculate()
            .unwrap_err()
            .to_string()
            .contains("European"));
    }
}
