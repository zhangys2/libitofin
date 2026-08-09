//! Floating-rate bond.
//!
//! Port of `ql/instruments/bonds/floatingratebond.{hpp,cpp}` (in-arrears and
//! exotic fixing-convention paths deferred with the IborLeg gaps).

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
}
