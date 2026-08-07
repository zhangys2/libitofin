//! Yield curve from an affine model at a fixed factor state.
//!
//! Port of `ql/methods/finitedifferences/utilities/fdmaffinemodeltermstructure.{hpp,cpp}`:
//! a [`YieldTermStructure`] whose discount from its reference date is the
//! model's `discountBond(t_, T + t_, r_)`, with `t_` the year-fraction from the
//! model's own reference date to this curve's. [`set_variable`](Self::set_variable)
//! updates the factor state mid-rollback (as
//! `FdmAffineModelSwapInnerValue` does) and notifies observers.
//!
//! ## Divergences from QuantLib
//!
//! - C++ `registerWith(model_)` is omitted: [`AffineModel`] is a pure trait here
//!   and does not expose an observable. Parameter changes still invalidate
//!   engines that observe the calibrated model directly (e.g. `FdmG2Solver`).
//! - The model is held as [`SharedMut`]`<dyn AffineModel>` so `G2` /
//!   `HullWhite` (which live in `RefCell`s) can be erased behind the trait.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::models::shortrate::AffineModel;
use crate::patterns::observable::{AsObservable, Observable};
use crate::shared::SharedMut;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::termstructures::{TermStructure, TermStructureBase};
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{DiscountFactor, Time};

/// Affine-model discount curve at a mutable factor state
/// (`fdmaffinemodeltermstructure.hpp:36`).
pub struct FdmAffineModelTermStructure {
    base: TermStructureBase,
    r: RefCell<Array>,
    t: Time,
    model: SharedMut<dyn AffineModel>,
}

impl FdmAffineModelTermStructure {
    /// `FdmAffineModelTermStructure(r, cal, dayCounter, referenceDate,
    /// modelReferenceDate, model)` (`fdmaffinemodeltermstructure.cpp:33-41`).
    pub fn new(
        r: Array,
        calendar: Calendar,
        day_counter: DayCounter,
        reference_date: Date,
        model_reference_date: Date,
        model: SharedMut<dyn AffineModel>,
    ) -> FdmAffineModelTermStructure {
        let t = day_counter.year_fraction(model_reference_date, reference_date);
        let base = TermStructureBase::with_reference_date(
            reference_date,
            Some(calendar),
            Some(day_counter),
        );
        FdmAffineModelTermStructure {
            base,
            r: RefCell::new(r),
            t,
            model,
        }
    }

    /// Replaces the factor state and notifies observers (`cpp:48-51`).
    pub fn set_variable(&self, r: Array) {
        *self.r.borrow_mut() = r;
        self.base.observable().notify_observers();
    }

    /// Year-fraction from the model reference date to this curve's reference
    /// date (`t_` in C++).
    pub fn model_time(&self) -> Time {
        self.t
    }
}

impl AsObservable for FdmAffineModelTermStructure {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for FdmAffineModelTermStructure {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn max_date(&self) -> Date {
        Date::max_date()
    }
}

impl YieldTermStructure for FdmAffineModelTermStructure {
    /// `discountImpl(T) = model.discountBond(t_, T + t_, r_)` (`cpp:53-55`).
    fn discount_impl(&self, t: Time) -> QlResult<DiscountFactor> {
        Ok(self
            .model
            .borrow()
            .discount_bond_factors(self.t, t + self.t, &self.r.borrow()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::models::shortrate::G2;
    use crate::shared::{Shared, shared};
    use crate::termstructures::yields::FlatForward;
    use crate::test_support::{Flag, as_observer};
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::Real;

    const A: Real = 0.1;
    const SIGMA: Real = 0.01;
    const B: Real = 0.2;
    const ETA: Real = 0.008;
    const RHO: Real = -0.75;

    fn today() -> Date {
        Date::new(19, Month::May, 2026)
    }

    fn model() -> SharedMut<G2> {
        let curve = Handle::new(shared(FlatForward::with_rate(
            today(),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        G2::new(curve, A, SIGMA, B, ETA, RHO).unwrap()
    }

    fn ts(
        g2: &SharedMut<G2>,
        r: Array,
        reference_date: Date,
        model_reference_date: Date,
    ) -> FdmAffineModelTermStructure {
        FdmAffineModelTermStructure::new(
            r,
            NullCalendar::new(),
            Actual365Fixed::new(),
            reference_date,
            model_reference_date,
            SharedMut::clone(g2) as SharedMut<dyn AffineModel>,
        )
    }

    #[test]
    fn origin_factors_at_model_ref_reprices_the_curve() {
        let g2 = model();
        let curve = g2.borrow().term_structure().current_link().unwrap();
        let affine = ts(&g2, Array::from([0.0, 0.0]), today(), today());
        assert_eq!(affine.model_time(), 0.0);
        assert_eq!(affine.max_date(), Date::max_date());

        for &t in &[0.5, 1.0, 2.0, 5.0] {
            let got = affine.discount(t, true).unwrap();
            let expected = curve.discount(t, false).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "t={t}: affine {got} vs curve {expected}"
            );
        }
    }

    #[test]
    fn nonzero_factors_match_discount_bond() {
        let g2 = model();
        let affine = ts(&g2, Array::from([0.02, -0.01]), today(), today());
        for &maturity in &[1.0, 3.0, 7.0] {
            let got = affine.discount(maturity, true).unwrap();
            let expected = g2
                .borrow()
                .discount_bond(0.0, maturity, 0.02, -0.01)
                .unwrap();
            assert!(
                (got - expected).abs() < 1e-14,
                "T={maturity}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn shifted_reference_uses_model_time_offset() {
        let g2 = model();
        let exercise = today() + 365;
        let affine = ts(&g2, Array::from([0.01, 0.005]), exercise, today());
        let t = affine.model_time();
        assert!(t > 0.0);

        let maturity_from_exercise = 2.0;
        let got = affine.discount(maturity_from_exercise, true).unwrap();
        let expected = g2
            .borrow()
            .discount_bond(t, maturity_from_exercise + t, 0.01, 0.005)
            .unwrap();
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn set_variable_changes_discounts_and_notifies() {
        let g2 = model();
        let affine = ts(&g2, Array::from([0.0, 0.0]), today(), today());
        let before = affine.discount(2.0, true).unwrap();

        let flag = Flag::new();
        affine.observable().register_observer(&as_observer(&flag));
        Flag::lower(&flag);

        affine.set_variable(Array::from([0.03, -0.02]));
        assert!(Flag::is_up(&flag));

        let after = affine.discount(2.0, true).unwrap();
        let expected = g2.borrow().discount_bond(0.0, 2.0, 0.03, -0.02).unwrap();
        assert!((after - expected).abs() < 1e-14);
        assert!((after - before).abs() > 1e-6);
    }
}
