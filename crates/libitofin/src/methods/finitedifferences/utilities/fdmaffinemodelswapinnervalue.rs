//! Swap exercise payoff on a G2++ or Hull–White factor grid.
//!
//! Port of QuantLib's `FdmAffineModelSwapInnerValue<ModelType>`
//! (`ql/methods/finitedifferences/utilities/fdmaffinemodelswapinnervalue.{hpp,cpp}`):
//! at each exercise time the calculator re-links (or [`set_variable`]s)
//! [`FdmAffineModelTermStructure`]s for the discount and forecast curves, marks
//! the remaining swap coupons to market, and returns `max(0, NPV)`.
//!
//! ## Scope
//!
//! Vanilla Ibor swaps. G2 `getState` returns the two factor coordinates;
//! Hull–White `getState` returns the short rate `x + φ(t)` (C++ specializations
//! in `fdmaffinemodelswapinnervalue.cpp`). The Overnight-index rebuild branch
//! is deferred.
//!
//! ## Divergences from QuantLib
//!
//! - Concrete G2 / Hull–White types rather than a C++ class template.
//! - Exercise times are a `Vec<(Time, Date)>` with exact `Time` match (C++
//!   `std::map<Time, Date>`).
//! - Coupon `amount()` / discount failures panic with a message (C++ throws).

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::handle::RelinkableHandle;
use crate::indexes::InterestRateIndex;
use crate::instruments::{FixedVsFloatingSwap, SwapType, VanillaSwap};
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::operators::FdmLinearOpIterator;
use crate::models::shortrate::{AffineModel, G2, HullWhite};
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::types::{Real, Size, Time};

use super::fdmaffinemodeltermstructure::FdmAffineModelTermStructure;
use super::fdminnervaluecalculator::FdmInnerValueCalculator;

/// G2++ affine-model swap inner value (`fdmaffinemodelswapinnervalue.hpp`).
pub struct FdmAffineModelSwapInnerValue {
    dis_ts: RelinkableHandle<dyn YieldTermStructure>,
    fwd_ts: RelinkableHandle<dyn YieldTermStructure>,
    dis_affine: RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    fwd_affine: RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    dis_model: SharedMut<G2>,
    fwd_model: SharedMut<G2>,
    swap: FixedVsFloatingSwap,
    exercise_dates: Vec<(Time, Date)>,
    mesher: Shared<dyn FdmMesher>,
    direction: Size,
}

impl FdmAffineModelSwapInnerValue {
    /// `FdmAffineModelSwapInnerValue(disModel, fwdModel, swap, exerciseDates,
    /// mesher, direction)` — vanilla Ibor rebuild only.
    ///
    /// The floating index is cloned onto an empty [`RelinkableHandle`] that
    /// [`inner_value`](FdmInnerValueCalculator::inner_value) fills with an
    /// affine forecast curve at each exercise date.
    ///
    /// # Errors
    ///
    /// Propagates swap rebuild failures (empty schedule, bad nominal, …).
    pub fn new(
        dis_model: SharedMut<G2>,
        fwd_model: SharedMut<G2>,
        swap: &FixedVsFloatingSwap,
        exercise_dates: Vec<(Time, Date)>,
        mesher: Shared<dyn FdmMesher>,
        direction: Size,
    ) -> QlResult<FdmAffineModelSwapInnerValue> {
        let dis_ts = RelinkableHandle::empty();
        let fwd_ts = RelinkableHandle::empty();
        let settings = Shared::clone(swap.ibor_index().base().settings());
        let rebuilt = VanillaSwap::new(
            swap.swap_type(),
            swap.nominal()?,
            swap.fixed_schedule().clone(),
            swap.fixed_rate(),
            swap.fixed_day_count().clone(),
            swap.floating_schedule().clone(),
            shared(swap.ibor_index().clone_with(fwd_ts.handle())),
            swap.spread(),
            swap.floating_day_count().clone(),
            Some(swap.payment_convention()),
            settings,
        )?;

        Ok(FdmAffineModelSwapInnerValue {
            dis_ts,
            fwd_ts,
            dis_affine: RefCell::new(None),
            fwd_affine: RefCell::new(None),
            dis_model,
            fwd_model,
            swap: rebuilt.into_fixed_vs_floating(),
            exercise_dates,
            mesher,
            direction,
        })
    }

    /// G2 `getState`: mesher coordinates on `(direction, direction + 1)`.
    fn get_state(&self, iter: &FdmLinearOpIterator) -> Array {
        Array::from([
            self.mesher.location(iter, self.direction),
            self.mesher.location(iter, self.direction + 1),
        ])
    }

    fn exercise_date(&self, t: Time) -> Date {
        self.exercise_dates
            .iter()
            .find(|(time, _)| *time == t)
            .map(|(_, date)| *date)
            .unwrap_or_else(|| panic!("FdmAffineModelSwapInnerValue: no exercise date for t={t}"))
    }

    fn link_or_set(
        model: &SharedMut<G2>,
        rate: Array,
        exercise_date: Date,
        handle: &RelinkableHandle<dyn YieldTermStructure>,
        cached: &RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    ) -> QlResult<()> {
        let need_relink = handle.handle().is_empty()
            || handle.handle().current_link()?.reference_date()? != exercise_date;

        if need_relink {
            let discount = model.borrow().term_structure().current_link()?;
            let calendar = discount.calendar().unwrap_or_else(NullCalendar::new);
            let day_counter = discount.require_day_counter()?;
            let model_ref = discount.reference_date()?;
            let affine = shared(FdmAffineModelTermStructure::new(
                rate,
                calendar,
                day_counter,
                exercise_date,
                model_ref,
                SharedMut::clone(model) as SharedMut<dyn AffineModel>,
            ));
            *cached.borrow_mut() = Some(Shared::clone(&affine));
            handle.link_to(affine as Shared<dyn YieldTermStructure>);
        } else {
            cached
                .borrow()
                .as_ref()
                .expect("affine curve is cached whenever the handle is non-empty")
                .set_variable(rate);
        }
        Ok(())
    }

    fn npv_at(&self, exercise_date: Date) -> QlResult<Real> {
        let dis = self.dis_ts.handle().current_link()?;
        let mut npv = 0.0;
        for (j, leg) in [self.swap.fixed_leg(), self.swap.floating_leg()]
            .into_iter()
            .enumerate()
        {
            for flow in leg {
                let coupon = flow.as_coupon().expect("vanilla swap legs carry coupons");
                if coupon.accrual_start_date() >= exercise_date {
                    npv += flow.amount()? * dis.discount_date(flow.date(), false)?;
                }
            }
            if j == 0 {
                npv *= -1.0;
            }
        }
        if self.swap.swap_type() == SwapType::Receiver {
            npv *= -1.0;
        }
        Ok(npv.max(0.0))
    }
}

impl FdmInnerValueCalculator for FdmAffineModelSwapInnerValue {
    fn inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        let exercise_date = self.exercise_date(t);
        let dis_rate = self.get_state(iter);
        let fwd_rate = self.get_state(iter);

        Self::link_or_set(
            &self.dis_model,
            dis_rate,
            exercise_date,
            &self.dis_ts,
            &self.dis_affine,
        )
        .unwrap_or_else(|e| panic!("FdmAffineModelSwapInnerValue discount curve: {e}"));
        Self::link_or_set(
            &self.fwd_model,
            fwd_rate,
            exercise_date,
            &self.fwd_ts,
            &self.fwd_affine,
        )
        .unwrap_or_else(|e| panic!("FdmAffineModelSwapInnerValue forecast curve: {e}"));

        self.npv_at(exercise_date)
            .unwrap_or_else(|e| panic!("FdmAffineModelSwapInnerValue NPV: {e}"))
    }

    fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        self.inner_value(iter, t)
    }
}

/// Hull–White affine-model swap inner value
/// (`FdmAffineModelSwapInnerValue<HullWhite>`).
///
/// `getState` returns the short rate at the mesher coordinate
/// (`fdmaffinemodelswapinnervalue.cpp:31-36`), not the OU factor.
pub struct FdmHullWhiteSwapInnerValue {
    dis_ts: RelinkableHandle<dyn YieldTermStructure>,
    fwd_ts: RelinkableHandle<dyn YieldTermStructure>,
    dis_affine: RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    fwd_affine: RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    dis_model: SharedMut<HullWhite>,
    fwd_model: SharedMut<HullWhite>,
    swap: FixedVsFloatingSwap,
    exercise_dates: Vec<(Time, Date)>,
    mesher: Shared<dyn FdmMesher>,
    direction: Size,
}

impl FdmHullWhiteSwapInnerValue {
    /// `FdmAffineModelSwapInnerValue<HullWhite>(disModel, fwdModel, swap,
    /// exerciseDates, mesher, direction)` — vanilla Ibor rebuild only.
    ///
    /// # Errors
    ///
    /// Propagates swap rebuild failures (empty schedule, bad nominal, …).
    pub fn new(
        dis_model: SharedMut<HullWhite>,
        fwd_model: SharedMut<HullWhite>,
        swap: &FixedVsFloatingSwap,
        exercise_dates: Vec<(Time, Date)>,
        mesher: Shared<dyn FdmMesher>,
        direction: Size,
    ) -> QlResult<FdmHullWhiteSwapInnerValue> {
        let dis_ts = RelinkableHandle::empty();
        let fwd_ts = RelinkableHandle::empty();
        let settings = Shared::clone(swap.ibor_index().base().settings());
        let rebuilt = VanillaSwap::new(
            swap.swap_type(),
            swap.nominal()?,
            swap.fixed_schedule().clone(),
            swap.fixed_rate(),
            swap.fixed_day_count().clone(),
            swap.floating_schedule().clone(),
            shared(swap.ibor_index().clone_with(fwd_ts.handle())),
            swap.spread(),
            swap.floating_day_count().clone(),
            Some(swap.payment_convention()),
            settings,
        )?;

        Ok(FdmHullWhiteSwapInnerValue {
            dis_ts,
            fwd_ts,
            dis_affine: RefCell::new(None),
            fwd_affine: RefCell::new(None),
            dis_model,
            fwd_model,
            swap: rebuilt.into_fixed_vs_floating(),
            exercise_dates,
            mesher,
            direction,
        })
    }

    /// Hull–White `getState`: `dynamics()->shortRate(t, location)`.
    fn get_state(
        &self,
        model: &SharedMut<HullWhite>,
        t: Time,
        iter: &FdmLinearOpIterator,
    ) -> Array {
        let x = self.mesher.location(iter, self.direction);
        let r = model
            .borrow()
            .dynamics()
            .unwrap_or_else(|e| panic!("FdmHullWhiteSwapInnerValue dynamics: {e}"))
            .short_rate(t, x);
        Array::from([r])
    }

    fn exercise_date(&self, t: Time) -> Date {
        self.exercise_dates
            .iter()
            .find(|(time, _)| *time == t)
            .map(|(_, date)| *date)
            .unwrap_or_else(|| panic!("FdmHullWhiteSwapInnerValue: no exercise date for t={t}"))
    }

    fn link_or_set(
        model: &SharedMut<HullWhite>,
        rate: Array,
        exercise_date: Date,
        handle: &RelinkableHandle<dyn YieldTermStructure>,
        cached: &RefCell<Option<Shared<FdmAffineModelTermStructure>>>,
    ) -> QlResult<()> {
        let need_relink = handle.handle().is_empty()
            || handle.handle().current_link()?.reference_date()? != exercise_date;

        if need_relink {
            let discount = model.borrow().term_structure().current_link()?;
            let calendar = discount.calendar().unwrap_or_else(NullCalendar::new);
            let day_counter = discount.require_day_counter()?;
            let model_ref = discount.reference_date()?;
            let affine = shared(FdmAffineModelTermStructure::new(
                rate,
                calendar,
                day_counter,
                exercise_date,
                model_ref,
                SharedMut::clone(model) as SharedMut<dyn AffineModel>,
            ));
            *cached.borrow_mut() = Some(Shared::clone(&affine));
            handle.link_to(affine as Shared<dyn YieldTermStructure>);
        } else {
            cached
                .borrow()
                .as_ref()
                .expect("affine curve is cached whenever the handle is non-empty")
                .set_variable(rate);
        }
        Ok(())
    }

    fn npv_at(&self, exercise_date: Date) -> QlResult<Real> {
        let dis = self.dis_ts.handle().current_link()?;
        let mut npv = 0.0;
        for (j, leg) in [self.swap.fixed_leg(), self.swap.floating_leg()]
            .into_iter()
            .enumerate()
        {
            for flow in leg {
                let coupon = flow.as_coupon().expect("vanilla swap legs carry coupons");
                if coupon.accrual_start_date() >= exercise_date {
                    npv += flow.amount()? * dis.discount_date(flow.date(), false)?;
                }
            }
            if j == 0 {
                npv *= -1.0;
            }
        }
        if self.swap.swap_type() == SwapType::Receiver {
            npv *= -1.0;
        }
        Ok(npv.max(0.0))
    }
}

impl FdmInnerValueCalculator for FdmHullWhiteSwapInnerValue {
    fn inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        let exercise_date = self.exercise_date(t);
        let dis_rate = self.get_state(&self.dis_model, t, iter);
        let fwd_rate = self.get_state(&self.fwd_model, t, iter);

        Self::link_or_set(
            &self.dis_model,
            dis_rate,
            exercise_date,
            &self.dis_ts,
            &self.dis_affine,
        )
        .unwrap_or_else(|e| panic!("FdmHullWhiteSwapInnerValue discount curve: {e}"));
        Self::link_or_set(
            &self.fwd_model,
            fwd_rate,
            exercise_date,
            &self.fwd_ts,
            &self.fwd_affine,
        )
        .unwrap_or_else(|e| panic!("FdmHullWhiteSwapInnerValue forecast curve: {e}"));

        self.npv_at(exercise_date)
            .unwrap_or_else(|e| panic!("FdmHullWhiteSwapInnerValue NPV: {e}"))
    }

    fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        self.inner_value(iter, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::indexes::IborIndex;
    use crate::indexes::ibor::Euribor;
    use crate::interestrate::Compounding;
    use crate::methods::finitedifferences::meshers::{FdmMesher, UniformGridMesher};
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::settings::Settings;
    use crate::termstructures::yields::FlatForward;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::schedule::{MakeSchedule, Schedule};

    fn today() -> Date {
        Date::new(15, Month::January, 2026)
    }

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        s
    }

    fn flat_curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn schedule(from: Date, to: Date, frequency: Frequency) -> Schedule {
        MakeSchedule::new()
            .from(from)
            .to(to)
            .with_frequency(frequency)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .forwards()
            .end_of_month(false)
            .build()
    }

    fn fixture(fixed_rate: Real) -> (FdmAffineModelSwapInnerValue, Shared<dyn FdmMesher>) {
        let settings = settings();
        let curve = flat_curve();
        let model = G2::new(curve.clone(), 0.1, 0.01, 0.2, 0.008, -0.75).unwrap();
        let index: Shared<IborIndex> = shared(Euribor::six_months(curve, Shared::clone(&settings)));
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let swap = VanillaSwap::new(
            SwapType::Payer,
            100.0,
            schedule(start, end, Frequency::Annual),
            fixed_rate,
            Thirty360::with_convention(Convention::BondBasis),
            schedule(start, end, Frequency::Semiannual),
            index,
            0.0,
            Actual360::new(),
            None,
            Shared::clone(&settings),
        )
        .unwrap()
        .into_fixed_vs_floating();

        let layout = shared(FdmLinearOpLayout::new(vec![5, 5]));
        let mesher: Shared<dyn FdmMesher> = shared(
            UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05), (-0.05, 0.05)])
                .unwrap(),
        );
        let calculator = FdmAffineModelSwapInnerValue::new(
            SharedMut::clone(&model),
            model,
            &swap,
            vec![(0.0, today())],
            Shared::clone(&mesher),
            0,
        )
        .unwrap();
        (calculator, mesher)
    }

    fn origin(mesher: &Shared<dyn FdmMesher>) -> FdmLinearOpIterator {
        // Centre of a 5×5 layout.
        let mut iter = mesher.layout().begin();
        while iter.index() < mesher.layout().size() {
            if iter.coordinates() == [2, 2] {
                return iter;
            }
            iter.advance();
        }
        panic!("centre node missing");
    }

    #[test]
    fn atm_payer_is_near_zero_at_origin() {
        let (calc, mesher) = fixture(0.05);
        let v = calc.inner_value(&origin(&mesher), 0.0);
        assert!(v.is_finite(), "value={v}");
        // Fair swap on the fitting curve: exercise value should be ~0 (floor).
        assert!(v.abs() < 1.0, "expected ~0 ATM exercise value, got {v}");
    }

    #[test]
    fn deep_itm_payer_is_strictly_positive() {
        let (calc, mesher) = fixture(0.0);
        let v = calc.inner_value(&origin(&mesher), 0.0);
        assert!(v > 10.0, "expected material ITM value, got {v}");
    }

    #[test]
    fn avg_inner_value_matches_inner_value() {
        let (calc, mesher) = fixture(0.0);
        let iter = origin(&mesher);
        assert_eq!(
            calc.avg_inner_value(&iter, 0.0),
            calc.inner_value(&iter, 0.0)
        );
    }

    #[test]
    fn set_variable_path_reuses_curve_on_second_call() {
        let (calc, mesher) = fixture(0.0);
        let iter = origin(&mesher);
        let first = calc.inner_value(&iter, 0.0);
        let second = calc.inner_value(&iter, 0.0);
        assert!((first - second).abs() < 1e-12, "{first} vs {second}");
        assert!(!calc.dis_ts.handle().is_empty());
        assert!(!calc.fwd_ts.handle().is_empty());
    }

    fn hw_fixture(fixed_rate: Real) -> (FdmHullWhiteSwapInnerValue, Shared<dyn FdmMesher>) {
        let settings = settings();
        let curve = flat_curve();
        let model = HullWhite::new(curve.clone(), 0.1, 0.01).unwrap();
        let index: Shared<IborIndex> = shared(Euribor::six_months(curve, Shared::clone(&settings)));
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let swap = VanillaSwap::new(
            SwapType::Payer,
            100.0,
            schedule(start, end, Frequency::Annual),
            fixed_rate,
            Thirty360::with_convention(Convention::BondBasis),
            schedule(start, end, Frequency::Semiannual),
            index,
            0.0,
            Actual360::new(),
            None,
            Shared::clone(&settings),
        )
        .unwrap()
        .into_fixed_vs_floating();

        let layout = shared(FdmLinearOpLayout::new(vec![5]));
        let mesher: Shared<dyn FdmMesher> =
            shared(UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05)]).unwrap());
        let calculator = FdmHullWhiteSwapInnerValue::new(
            SharedMut::clone(&model),
            model,
            &swap,
            vec![(0.0, today())],
            Shared::clone(&mesher),
            0,
        )
        .unwrap();
        (calculator, mesher)
    }

    fn hw_origin(mesher: &Shared<dyn FdmMesher>) -> FdmLinearOpIterator {
        let mut iter = mesher.layout().begin();
        while iter.index() < mesher.layout().size() {
            if iter.coordinates() == [2] {
                return iter;
            }
            iter.advance();
        }
        panic!("centre node missing");
    }

    #[test]
    fn hw_atm_payer_is_near_zero_at_origin() {
        let (calc, mesher) = hw_fixture(0.05);
        let v = calc.inner_value(&hw_origin(&mesher), 0.0);
        assert!(v.is_finite(), "value={v}");
        assert!(v.abs() < 1.0, "expected ~0 ATM exercise value, got {v}");
    }

    #[test]
    fn hw_deep_itm_payer_is_strictly_positive() {
        let (calc, mesher) = hw_fixture(0.0);
        let v = calc.inner_value(&hw_origin(&mesher), 0.0);
        assert!(v > 10.0, "expected material ITM value, got {v}");
    }

    #[test]
    fn hw_get_state_is_the_short_rate_not_the_factor() {
        let (calc, mesher) = hw_fixture(0.0);
        let iter = hw_origin(&mesher);
        let x = mesher.location(&iter, 0);
        let expected = calc
            .dis_model
            .borrow()
            .dynamics()
            .unwrap()
            .short_rate(0.0, x);
        let got = calc.get_state(&calc.dis_model, 0.0, &iter);
        assert_eq!(got.size(), 1);
        assert!(
            (got[0] - expected).abs() < 1e-14,
            "state {} vs short rate {expected} (factor x={x})",
            got[0]
        );
        assert!(
            (got[0] - x).abs() > 1e-4,
            "state should include φ(t), not equal the raw factor"
        );
    }
}
