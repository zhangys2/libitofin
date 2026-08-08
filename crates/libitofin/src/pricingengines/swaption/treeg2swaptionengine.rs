//! Numerical lattice engine for swaptions under G2++.
//!
//! Twin of [`TreeSwaptionEngine`](super::TreeSwaptionEngine) bound to
//! [`G2`](crate::models::shortrate::G2): builds a [`DiscretizedSwaption`], fits a
//! [`TimeGrid`] over its mandatory times, grows [`G2::tree`](crate::models::shortrate::G2::tree)
//! on that grid, then rolls the swaption back to the first exercise
//! (`treeswaptionengine.cpp:51` with a two-factor model).
//!
//! Kept as a concrete twin (rather than a generic `ShortRateModel` engine) to
//! match the Hull–White / Jamshidian precedent and the existing
//! [`FdG2SwaptionEngine`](super::FdG2SwaptionEngine) house style.

use crate::discretizedasset::DiscretizedAsset;
use crate::errors::QlResult;
use crate::instrument::InstrumentResults;
use crate::instruments::{SettlementMethod, SwaptionArguments, SwaptionEngine};
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::lattice::Lattice;
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::G2;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared};
use crate::time::date::Date;
use crate::types::{Size, Time};
use crate::{fail, require};

use super::DiscretizedSwaption;

/// Numerical lattice engine for swaptions under [`G2`]
/// (`treeswaptionengine.hpp:44` with `TwoFactorModel::tree`).
pub struct TreeG2SwaptionEngine {
    base: SwaptionEngine,
    model: SharedMut<G2>,
    time_steps: Size,
    settings: Shared<Settings<Date>>,
}

impl TreeG2SwaptionEngine {
    /// Builds the engine over a G2 `model` with a fixed step count
    /// (`treeswaptionengine.cpp:26` + `latticeshortratemodelengine.hpp:56`).
    ///
    /// # Errors
    /// `time_steps` must be positive (`latticeshortratemodelengine.hpp:60`).
    pub fn new(
        model: SharedMut<G2>,
        time_steps: Size,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<TreeG2SwaptionEngine> {
        require!(
            time_steps > 0,
            "timeSteps must be positive, {time_steps} not allowed"
        );
        let base = SwaptionEngine::new(SwaptionArguments::default(), InstrumentResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        Ok(TreeG2SwaptionEngine {
            base,
            model,
            time_steps,
            settings,
        })
    }
}

impl AsObservable for TreeG2SwaptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for TreeG2SwaptionEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    /// `calculate()` (`treeswaptionengine.cpp:51`): guard the settlement method,
    /// take the reference date / day counter from the model's term structure,
    /// build the [`DiscretizedSwaption`], grow the G2 tree over its mandatory
    /// times, then roll back to the first exercise and read the present value.
    fn calculate(&mut self) -> QlResult<()> {
        require!(
            self.base.arguments().settlement_method != SettlementMethod::ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with TreeG2SwaptionEngine"
        );

        let model = self.model.borrow();
        let (reference_date, day_counter) = {
            let curve = model.term_structure().current_link()?;
            (curve.reference_date()?, curve.require_day_counter()?)
        };

        let (mut swaption, stopping_times) = {
            let args = self.base.arguments();
            let Some(exercise) = args.exercise.as_ref() else {
                fail!("exercise not set");
            };
            let stopping_times: Vec<Time> = exercise
                .dates()
                .iter()
                .map(|&date| day_counter.year_fraction(reference_date, date))
                .collect();
            let swaption =
                DiscretizedSwaption::new(args, reference_date, &day_counter, &self.settings)?;
            (swaption, stopping_times)
        };

        let times = swaption.mandatory_times();
        let grid = TimeGrid::with_mandatory_times(&times, self.time_steps)?;
        let lattice: Shared<dyn Lattice> = shared(model.tree(grid)?);
        drop(model);

        let Some(&last) = stopping_times.last() else {
            fail!("swaption has no exercise dates");
        };
        swaption.initialize(Shared::clone(&lattice), last)?;

        let Some(next_exercise) = stopping_times.iter().copied().find(|&t| t >= 0.0) else {
            fail!("swaption has no non-negative exercise time");
        };
        swaption.rollback(next_exercise)?;

        self.base.results_mut().value = Some(swaption.present_value()?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    use crate::exercise::{BermudanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::indexes::IborIndex;
    use crate::indexes::InterestRateIndex;
    use crate::indexes::ibor::Euribor;
    use crate::instrument::Instrument;
    use crate::instruments::{
        FixedVsFloatingSwap, SettlementType, SwapType, Swaption, VanillaSwap,
    };
    use crate::interestrate::Compounding;
    use crate::pricingengines::DiscountingSwapEngine;
    use crate::pricingengines::swaption::G2SwaptionEngine;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;
    use crate::types::Real;

    fn settings_at(today: Date) -> Shared<Settings<Date>> {
        let settings = shared(Settings::<Date>::new());
        settings.set_evaluation_date(today);
        settings
    }

    fn flat_curve(settlement: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            settlement,
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    #[test]
    fn rejects_non_positive_time_steps() {
        let settings = settings_at(Date::new(15, Month::September, 2016));
        let g2 = G2::new(
            flat_curve(Date::new(19, Month::September, 2016), 0.05),
            0.1,
            0.01,
            0.2,
            0.01,
            -0.5,
        )
        .unwrap();
        let err = TreeG2SwaptionEngine::new(g2, 0, settings)
            .err()
            .expect("time_steps == 0 must be rejected");
        assert_eq!(err.message(), "timeSteps must be positive, 0 not allowed");
    }

    #[test]
    fn rejects_par_yield_cash_settlement() {
        let settings = settings_at(Date::new(15, Month::September, 2016));
        let g2 = G2::new(
            flat_curve(Date::new(19, Month::September, 2016), 0.05),
            0.1,
            0.01,
            0.2,
            0.01,
            -0.5,
        )
        .unwrap();
        let mut engine = TreeG2SwaptionEngine::new(g2, 50, settings).unwrap();
        let args = (engine.arguments_mut() as &mut dyn Any)
            .downcast_mut::<SwaptionArguments>()
            .expect("engine carries SwaptionArguments");
        args.settlement_method = SettlementMethod::ParYieldCurve;
        assert_eq!(
            engine.calculate().unwrap_err().message(),
            "cash settled (ParYieldCurve) swaptions not priced with TreeG2SwaptionEngine"
        );
    }

    /// Port of `bermudanswaption.cpp` `testCachedG2Values` — tree half.
    ///
    /// Same CommonVars fixture as the FDM half in `FdG2SwaptionEngine` tests.
    /// Expecteds are the at-par coupon branch (`Settings::using_at_par_coupons`
    /// defaults to `true`); tol `0.005` matches QuantLib.
    #[test]
    fn cached_g2_tree_bermudan_values() {
        let today = Date::new(15, Month::September, 2016);
        let settlement = Date::new(19, Month::September, 2016);
        let settings = settings_at(today);
        assert!(
            settings.using_at_par_coupons(),
            "oracle expects the at-par coupon branch"
        );

        let calendar = Target::new();
        let curve = flat_curve(settlement, 0.04875825);
        let index: Shared<IborIndex> =
            shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));

        let make_swap = |fixed_rate: Real| -> SharedMut<FixedVsFloatingSwap> {
            let start = calendar.advance_by_period(
                settlement,
                Period::new(1, TimeUnit::Years),
                BusinessDayConvention::Following,
                false,
            );
            let maturity = calendar.advance_by_period(
                start,
                Period::new(5, TimeUnit::Years),
                BusinessDayConvention::Following,
                false,
            );
            let fixed_schedule = MakeSchedule::new()
                .from(start)
                .to(maturity)
                .with_frequency(Frequency::Annual)
                .with_calendar(calendar.clone())
                .with_convention(BusinessDayConvention::Unadjusted)
                .with_termination_date_convention(BusinessDayConvention::Unadjusted)
                .forwards()
                .end_of_month(false)
                .build();
            let float_schedule = MakeSchedule::new()
                .from(start)
                .to(maturity)
                .with_frequency(Frequency::Semiannual)
                .with_calendar(calendar.clone())
                .with_convention(BusinessDayConvention::ModifiedFollowing)
                .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
                .forwards()
                .end_of_month(false)
                .build();
            let floating_dc = index.day_counter().clone();
            shared_mut(
                VanillaSwap::new(
                    SwapType::Payer,
                    1000.0,
                    fixed_schedule,
                    fixed_rate,
                    Thirty360::with_convention(Convention::BondBasis),
                    float_schedule,
                    Shared::clone(&index),
                    0.0,
                    floating_dc,
                    None,
                    Shared::clone(&settings),
                )
                .unwrap()
                .into_fixed_vs_floating(),
            )
        };

        let discounting = shared_mut(DiscountingSwapEngine::new(
            curve.clone(),
            None,
            None,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        let atm_swap = make_swap(0.0);
        atm_swap
            .borrow_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&discounting));
        let atm_rate = atm_swap.borrow_mut().fair_rate().unwrap();

        let moneyness = [0.5, 0.75, 1.0, 1.25, 1.5];
        let expected = [103.248, 54.6726, 20.1685, 5.44118, 1.12737];
        let tol = 0.005;

        let g2 = G2::new(curve, 0.1, 0.01, 0.2, 0.013, -0.5).unwrap();
        let engine = shared_mut(
            TreeG2SwaptionEngine::new(SharedMut::clone(&g2), 50, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        for (i, &s) in moneyness.iter().enumerate() {
            let swap = make_swap(s * atm_rate);
            let exercise_dates: Vec<Date> = swap
                .borrow()
                .fixed_leg()
                .iter()
                .map(|flow| {
                    flow.as_coupon()
                        .expect("fixed leg carries coupons")
                        .accrual_start_date()
                })
                .collect();
            let mut swaption = Swaption::new(
                swap,
                shared(BermudanExercise::new(exercise_dates, false).unwrap())
                    as Shared<dyn Exercise>,
                SettlementType::Physical,
                SettlementMethod::PhysicalOTC,
                Shared::clone(&settings),
            );
            swaption
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let got = swaption.npv().unwrap();
            assert!(
                (got - expected[i]).abs() <= tol,
                "moneyness {s}: got {got}, expected {}, |diff|={}",
                expected[i],
                (got - expected[i]).abs()
            );
        }
    }

    #[test]
    fn bermudan_dominates_european_analytic() {
        // Same G2 CommonVars geometry; Bermudan tree ≥ European analytic G2.
        let today = Date::new(15, Month::September, 2016);
        let settlement = Date::new(19, Month::September, 2016);
        let settings = settings_at(today);
        let calendar = Target::new();
        let curve = flat_curve(settlement, 0.04875825);
        let index: Shared<IborIndex> =
            shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));

        let start = calendar.advance_by_period(
            settlement,
            Period::new(1, TimeUnit::Years),
            BusinessDayConvention::Following,
            false,
        );
        let maturity = calendar.advance_by_period(
            start,
            Period::new(5, TimeUnit::Years),
            BusinessDayConvention::Following,
            false,
        );
        let fixed_schedule = MakeSchedule::new()
            .from(start)
            .to(maturity)
            .with_frequency(Frequency::Annual)
            .with_calendar(calendar.clone())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .forwards()
            .end_of_month(false)
            .build();
        let float_schedule = MakeSchedule::new()
            .from(start)
            .to(maturity)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(calendar.clone())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build();
        let floating_dc = index.day_counter().clone();
        let swap = shared_mut(
            VanillaSwap::new(
                SwapType::Payer,
                1000.0,
                fixed_schedule,
                0.04,
                Thirty360::with_convention(Convention::BondBasis),
                float_schedule,
                index,
                0.0,
                floating_dc,
                None,
                Shared::clone(&settings),
            )
            .unwrap()
            .into_fixed_vs_floating(),
        );

        let g2 = G2::new(curve, 0.1, 0.01, 0.2, 0.013, -0.5).unwrap();

        let mut european = Swaption::new(
            SharedMut::clone(&swap),
            shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        european
            .base_mut()
            .set_pricing_engine(shared_mut(G2SwaptionEngine::new(
                SharedMut::clone(&g2),
                6.0,
                64,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        let euro_npv = european.npv().unwrap();

        let exercise_dates: Vec<Date> = swap
            .borrow()
            .fixed_leg()
            .iter()
            .map(|flow| {
                flow.as_coupon()
                    .expect("fixed leg carries coupons")
                    .accrual_start_date()
            })
            .collect();
        let mut bermudan = Swaption::new(
            swap,
            shared(BermudanExercise::new(exercise_dates, false).unwrap()) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        bermudan.base_mut().set_pricing_engine(shared_mut(
            TreeG2SwaptionEngine::new(g2, 50, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>);
        let berm_npv = bermudan.npv().unwrap();
        assert!(
            berm_npv + 1e-8 >= euro_npv,
            "bermudan tree {berm_npv} must dominate european analytic {euro_npv}"
        );
    }
}
