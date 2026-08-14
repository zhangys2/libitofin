//! Floating-rate bond.
//!
//! Port of `ql/instruments/bonds/floatingratebond.{hpp,cpp}` (in-arrears path
//! deferred with the IborLeg gaps).

use super::super::bond::Bond;
use crate::cashflows::IborLeg;
use crate::errors::QlResult;
use crate::indexes::IborIndex;
use crate::indexes::index::Index;
use crate::instrument::Instrument;
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::period::Period;
use crate::time::schedule::Schedule;
use crate::types::{Natural, Rate, Real, Spread};

/// A bond paying floating Ibor coupons plus a single redemption.
pub struct FloatingRateBond {
    bond: Bond,
}

impl FloatingRateBond {
    /// Builds a floating-rate bond from an [`IborLeg`] schedule.
    ///
    /// `fixing_convention` defaults to [`Preceding`](BusinessDayConvention::Preceding)
    /// when `None`, matching C++ `FloatingRateBond`'s default and
    /// [`IborLeg`](crate::cashflows::IborLeg)'s default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        settlement_days: Natural,
        face_amount: Real,
        schedule: Schedule,
        ibor_index: Shared<IborIndex>,
        payment_day_counter: DayCounter,
        payment_convention: BusinessDayConvention,
        fixing_days: Option<Natural>,
        gearings: Vec<Real>,
        spreads: Vec<Spread>,
        caps: Vec<Rate>,
        floors: Vec<Rate>,
        redemption: Real,
        issue_date: Option<Date>,
        ex_coupon_period: Option<Period>,
        ex_coupon_calendar: Calendar,
        ex_coupon_convention: BusinessDayConvention,
        ex_coupon_end_of_month: bool,
        fixing_convention: Option<BusinessDayConvention>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let calendar = schedule.calendar().clone();
        let maturity = schedule.end_date();
        let mut leg = IborLeg::new(schedule, Shared::clone(&ibor_index))
            .with_notional(face_amount)
            .with_payment_day_counter(payment_day_counter)
            .with_payment_adjustment(payment_convention)
            .with_gearings(gearings)
            .with_spreads(spreads)
            .with_caps(caps)
            .with_floors(floors);
        if let Some(days) = fixing_days {
            leg = leg.with_fixing_days(days);
        }
        if let Some(convention) = fixing_convention {
            leg = leg.with_fixing_convention(convention);
        }
        if let Some(period) = ex_coupon_period {
            leg = leg.with_ex_coupon_period(
                period,
                ex_coupon_calendar,
                ex_coupon_convention,
                ex_coupon_end_of_month,
            );
        }
        let cashflows = leg.build()?;
        // Match C++ `FloatingRateBond`: construct with an empty leg so the base
        // issue-date check is skipped (seasoned bonds pay coupons before issue),
        // then assign cashflows and append the redemption.
        let mut bond = Bond::new(settlement_days, calendar, issue_date, Vec::new(), settings)?;
        bond.set_cashflows(cashflows);
        bond.add_redemptions_to_cashflows(&[redemption])?;
        bond.set_maturity_date(maturity);
        require!(!bond.cashflows().is_empty(), "bond with no cashflows!");
        require!(
            bond.redemptions().len() == 1,
            "multiple redemptions created"
        );
        bond.base().register_with(ibor_index.observable());
        Ok(Self { bond })
    }

    /// The underlying [`Bond`] base.
    pub fn bond(&self) -> &Bond {
        &self.bond
    }

    /// Mutable access to the underlying [`Bond`] base.
    pub fn bond_mut(&mut self) -> &mut Bond {
        &mut self.bond
    }

    /// Consumes the wrapper and yields its [`Bond`] base.
    pub fn into_bond(self) -> Bond {
        self.bond
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::indexes::AUDLibor;
    use crate::interestrate::Compounding;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::australia::{self, Australia};
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::Month;
    use crate::time::dategenerationrule::DateGeneration;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::timeunit::TimeUnit;

    /// `bonds.cpp` testFixingConvention (`:1827`): with `fixingDays=0` and an
    /// unadjusted quarterly schedule, a Saturday accrual start (22 Jun 2024)
    /// fixes on Friday under Preceding and Monday under Following.
    #[test]
    fn floating_bond_fixing_convention_moves_weekend_accrual_starts() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(Date::new(1, Month::January, 2024));
        let calendar = Australia::new(australia::Market::Settlement);
        let schedule = Schedule::new(
            Date::new(22, Month::March, 2024),
            Date::new(22, Month::December, 2024),
            Period::new(3, TimeUnit::Months),
            calendar,
            BusinessDayConvention::Unadjusted,
            BusinessDayConvention::Unadjusted,
            DateGeneration::Forward,
            false,
            Date::null(),
            Date::null(),
        );
        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            Date::new(1, Month::January, 2024),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let index = shared(AUDLibor::three_months(curve, Shared::clone(&settings)));

        let make_bond = |fixing_convention: Option<BusinessDayConvention>| {
            FloatingRateBond::new(
                0,
                100.0,
                schedule.clone(),
                Shared::clone(&index),
                Actual365Fixed::new(),
                BusinessDayConvention::Following,
                Some(0),
                vec![1.0],
                vec![0.0],
                Vec::new(),
                Vec::new(),
                100.0,
                None,
                None,
                NullCalendar::new(),
                BusinessDayConvention::Unadjusted,
                false,
                fixing_convention,
                Shared::clone(&settings),
            )
            .unwrap()
        };

        let preceding = make_bond(None);
        let following = make_bond(Some(BusinessDayConvention::Following));
        let june22 = Date::new(22, Month::June, 2024);

        let find_fixing = |bond: &FloatingRateBond| -> Date {
            for cf in bond.bond().cashflows() {
                if let Some(coupon) = cf.as_coupon()
                    && coupon.accrual_start_date() == june22
                {
                    return coupon
                        .fixing_date()
                        .expect("floating coupon must expose a fixing date");
                }
            }
            panic!("no coupon starting on {june22}");
        };

        assert_eq!(
            find_fixing(&preceding),
            Date::new(21, Month::June, 2024),
            "Preceding should land on Friday"
        );
        assert_eq!(
            find_fixing(&following),
            Date::new(24, Month::June, 2024),
            "Following should land on Monday"
        );
    }
}
