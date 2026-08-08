//! Swaption priced on a lattice.
//!
//! Port of `ql/pricingengines/swaption/discretizedswaption.{hpp,cpp}`:
//! [`DiscretizedSwaption`] is a [`DiscretizedOption`] whose underlying is a
//! [`DiscretizedSwap`](super::DiscretizedSwap). On each exercise node the option
//! condition takes `max(continuation, exercise)`, where the exercise value is the
//! underlying swap rolled to that node.
//!
//! # Composition, not inheritance (the single load-bearing decision)
//! C++'s `DiscretizedSwaption` derives from `DiscretizedOption`, overriding only
//! [`reset`](DiscretizedSwaption::reset). Rust has no subclassing, so the type
//! EMBEDS a [`DiscretizedOption`] and forwards [`base`](DiscretizedAsset::base) /
//! [`base_mut`](DiscretizedAsset::base_mut) to it - the swaption owns NO
//! [`DiscretizedAssetBase`] of its own. This is essential: the lattice mutates
//! state (time, values, method) through the swaption trait object, and the
//! delegated [`post_adjust_values_impl`](DiscretizedAsset::post_adjust_values_impl)
//! reads that same state off `self.option`. Two separate base storages would make
//! the exercise pass read zeros and silently misprice (the
//! `rust-composition-loses-virtual-dispatch` trap). Every other adjustment method
//! forwards to the embedded option unchanged, mirroring C++'s single virtual
//! subobject.
//!
//! # Date snapping
//! The ctor calls [`prepare_swaption_with_snapped_dates`]
//! (`discretizedswaption.cpp:39,82`): coupon schedule dates within a week of an
//! exercise date collapse onto that exercise (and flip to the post-adjustment
//! pass when the unadjusted date sits in the previous week). Without this, the
//! annual/semi Bermudan geometry misprices floating resets that fall a few days
//! off the exercise nodes.

use crate::discretizedasset::{
    CouponAdjustment, DiscretizedAsset, DiscretizedAssetBase, DiscretizedOption,
};
use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{Swaption, SwaptionArguments, VanillaSwap};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::types::{Size, Time};

use super::DiscretizedSwap;

/// A swaption discretized on a [`Lattice`](crate::methods::lattices::lattice::Lattice)
/// (`discretizedswaption.hpp:34`).
///
/// Built from a [`SwaptionArguments`] with a reference date and day counter; the
/// exercise dates become year-fraction times and the underlying swap is a
/// [`DiscretizedSwap`](super::DiscretizedSwap).
pub struct DiscretizedSwaption {
    option: DiscretizedOption,
    last_payment: Time,
}

impl DiscretizedSwaption {
    /// `DiscretizedSwaption(args, referenceDate, dayCounter)`
    /// (`discretizedswaption.cpp:36`), with coupon dates snapped to nearby
    /// exercise dates (`prepareSwaptionWithSnappedDates`).
    ///
    /// # Errors
    /// Fails if the arguments carry no exercise, no swap, no fixed coupons or no
    /// floating coupons, if the snapped swap cannot be rebuilt, or if
    /// [`DiscretizedSwap::with_adjustments`](super::DiscretizedSwap::with_adjustments)
    /// fails.
    pub fn new(
        args: &SwaptionArguments,
        reference_date: Date,
        day_counter: &DayCounter,
        settings: &Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let (snapped_args, fixed_adj, floating_adj) =
            prepare_swaption_with_snapped_dates(args, Shared::clone(settings))?;

        let Some(exercise) = snapped_args.exercise.as_ref() else {
            fail!("exercise not set");
        };
        let swap_args = &snapped_args.swap_arguments;

        let exercise_times: Vec<Time> = exercise
            .dates()
            .iter()
            .map(|&date| day_counter.year_fraction(reference_date, date))
            .collect();
        let exercise_type = exercise.exercise_type();

        let Some(&last_fixed_date) = swap_args.fixed_pay_dates.last() else {
            fail!("swap has no fixed coupons");
        };
        let Some(&last_floating_date) = swap_args.floating_pay_dates.last() else {
            fail!("swap has no floating coupons");
        };
        let last_fixed = day_counter.year_fraction(reference_date, last_fixed_date);
        let last_floating = day_counter.year_fraction(reference_date, last_floating_date);
        let last_payment = last_fixed.max(last_floating);

        let swap = DiscretizedSwap::with_adjustments(
            swap_args,
            reference_date,
            day_counter,
            fixed_adj,
            floating_adj,
            settings.as_ref(),
        )?;
        let underlying: SharedMut<dyn DiscretizedAsset> = shared_mut(swap);
        let option = DiscretizedOption::new(underlying, exercise_type, exercise_times);

        Ok(DiscretizedSwaption {
            option,
            last_payment,
        })
    }
}

impl DiscretizedAsset for DiscretizedSwaption {
    fn base(&self) -> &DiscretizedAssetBase {
        self.option.base()
    }

    fn base_mut(&mut self) -> &mut DiscretizedAssetBase {
        self.option.base_mut()
    }

    fn as_asset_mut(&mut self) -> &mut dyn DiscretizedAsset {
        self
    }

    /// `reset(size)` (`discretizedswaption.cpp:73`): initialize the underlying swap
    /// at `last_payment` FIRST, then run the [`DiscretizedOption`] reset (which
    /// checks option and underlying share a method, zeros the values and adjusts).
    fn reset(&mut self, size: Size) -> QlResult<()> {
        let method = self.require_method()?;
        let underlying = SharedMut::clone(self.option.underlying());
        underlying
            .borrow_mut()
            .initialize(method, self.last_payment)?;
        self.option.reset(size)
    }

    /// The embedded option's times (its underlying's plus the exercise times).
    fn mandatory_times(&self) -> Vec<Time> {
        self.option.mandatory_times()
    }

    /// Forwarded to the embedded option (the C++ non-overridden virtual).
    fn pre_adjust_values_impl(&mut self) -> QlResult<()> {
        self.option.pre_adjust_values_impl()
    }

    /// The exercise machinery, forwarded to the embedded option. It reads the base
    /// state the lattice mutated through this swaption (single-storage base).
    fn post_adjust_values_impl(&mut self) -> QlResult<()> {
        self.option.post_adjust_values_impl()
    }
}

fn within_previous_week(d1: Date, d2: Date) -> bool {
    d2 >= d1 - 7 && d2 <= d1
}

fn within_next_week(d1: Date, d2: Date) -> bool {
    d2 >= d1 && d2 <= d1 + 7
}

fn within_one_week(d1: Date, d2: Date) -> bool {
    within_previous_week(d1, d2) || within_next_week(d1, d2)
}

/// `prepareSwaptionWithSnappedDates` (`discretizedswaption.cpp:82`): collapse
/// nearby coupon schedule dates onto exercise dates and tag previous-week snaps
/// as [`CouponAdjustment::Post`].
fn prepare_swaption_with_snapped_dates(
    args: &SwaptionArguments,
    settings: Shared<Settings<Date>>,
) -> QlResult<(
    SwaptionArguments,
    Vec<CouponAdjustment>,
    Vec<CouponAdjustment>,
)> {
    let Some(swap_rc) = args.swap.as_ref() else {
        fail!("swap not set");
    };
    let Some(exercise) = args.exercise.as_ref() else {
        fail!("exercise not set");
    };

    let swap = swap_rc.borrow();
    let mut fixed_dates = swap.fixed_schedule().dates().to_vec();
    let mut float_dates = swap.floating_schedule().dates().to_vec();

    let mut fixed_coupon_adjustments = vec![CouponAdjustment::Pre; swap.fixed_leg().len()];
    let mut floating_coupon_adjustments = vec![CouponAdjustment::Pre; swap.floating_leg().len()];

    require!(
        fixed_coupon_adjustments.len() + 1 == fixed_dates.len(),
        "fixed schedule date count must be one more than the fixed coupon count"
    );
    require!(
        floating_coupon_adjustments.len() + 1 == float_dates.len(),
        "floating schedule date count must be one more than the floating coupon count"
    );

    for &exercise_date in exercise.dates() {
        for j in 0..fixed_dates.len() - 1 {
            let unadjusted = fixed_dates[j];
            if exercise_date != unadjusted && within_one_week(exercise_date, unadjusted) {
                fixed_dates[j] = exercise_date;
                if within_previous_week(exercise_date, unadjusted) {
                    fixed_coupon_adjustments[j] = CouponAdjustment::Post;
                }
            }
        }
        for j in 0..float_dates.len() - 1 {
            let unadjusted = float_dates[j];
            if exercise_date != unadjusted && within_one_week(exercise_date, unadjusted) {
                float_dates[j] = exercise_date;
                if within_previous_week(exercise_date, unadjusted) {
                    floating_coupon_adjustments[j] = CouponAdjustment::Post;
                }
            }
        }
    }

    let snapped_swap = shared_mut(
        VanillaSwap::new(
            swap.swap_type(),
            swap.nominal()?,
            Schedule::from_dates(fixed_dates),
            swap.fixed_rate(),
            swap.fixed_day_count().clone(),
            Schedule::from_dates(float_dates),
            Shared::clone(swap.ibor_index()),
            swap.spread(),
            swap.floating_day_count().clone(),
            Some(swap.payment_convention()),
            Shared::clone(&settings),
        )?
        .into_fixed_vs_floating(),
    );
    drop(swap);

    let snapped_swaption = Swaption::new(
        snapped_swap,
        Shared::clone(exercise),
        args.settlement_type,
        args.settlement_method,
        settings,
    );
    let mut snapped_args = SwaptionArguments::default();
    snapped_swaption.setup_arguments(&mut snapped_args)?;
    Ok((
        snapped_args,
        fixed_coupon_adjustments,
        floating_coupon_adjustments,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::date::Month;

    #[test]
    fn within_week_helpers_match_quantlib() {
        let d1 = Date::new(19, Month::September, 2017);
        assert!(within_previous_week(d1, d1 - 3));
        assert!(within_previous_week(d1, d1));
        assert!(!within_previous_week(d1, d1 + 1));
        assert!(within_next_week(d1, d1 + 3));
        assert!(!within_next_week(d1, d1 - 1));
        assert!(within_one_week(d1, d1 - 7));
        assert!(within_one_week(d1, d1 + 7));
        assert!(!within_one_week(d1, d1 + 8));
    }
}
