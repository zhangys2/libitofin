//! ICE Libor indexes (non-EUR).
//!
//! Port of `ql/indexes/ibor/libor.{hpp,cpp}`. [`Libor`] configures an
//! [`IborIndex`] with the London Exchange fixing calendar and a joint
//! (UK Exchange ∪ financial-centre) calendar for value and maturity dates.
//! Daily tenors and EUR are rejected (dedicated constructors, not ported).

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::indexes::iborindex::IborIndex;
use crate::indexes::interestrateindex::InterestRateIndex;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::calendars::jointcalendar::{JointCalendar, JointCalendarRule};
use crate::time::calendars::unitedkingdom::{Market as UkMarket, UnitedKingdom};
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::period::Period;
use crate::time::timeunit::TimeUnit;
use crate::types::Natural;
use crate::{fail, require};

/// ICE Libor family constructors (`ql/indexes/ibor/libor.hpp`).
pub struct Libor;

impl Libor {
    /// Builds a Libor index of the given `tenor` over `forwarding`
    /// (`libor.cpp:59-84`).
    ///
    /// # Errors
    /// Daily tenors and EUR are rejected with the QuantLib messages.
    #[allow(clippy::new_ret_no_self, clippy::too_many_arguments)]
    pub fn new(
        family_name: String,
        tenor: Period,
        settlement_days: Natural,
        currency: Currency,
        financial_center_calendar: Calendar,
        day_counter: DayCounter,
        forwarding: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<IborIndex> {
        require!(
            currency != Currency::eur(),
            "for EUR Libor dedicated EurLibor constructor must be used"
        );
        let uk_exchange = UnitedKingdom::new(UkMarket::Exchange);
        let joint = JointCalendar::of_two(
            uk_exchange.clone(),
            financial_center_calendar.clone(),
            JointCalendarRule::JoinHolidays,
        );
        let index = IborIndex::new_with_joint_calendars(
            family_name,
            tenor,
            settlement_days,
            currency,
            uk_exchange,
            financial_center_calendar,
            joint,
            libor_convention(tenor)?,
            libor_eom(tenor)?,
            day_counter,
            forwarding,
            settings,
        );
        require!(
            index.tenor().units() != TimeUnit::Days,
            "for daily tenors ({}) dedicated DailyTenor constructor must be used",
            index.tenor()
        );
        Ok(index)
    }
}

fn libor_convention(tenor: Period) -> QlResult<BusinessDayConvention> {
    match tenor.units() {
        TimeUnit::Days | TimeUnit::Weeks => Ok(BusinessDayConvention::Following),
        TimeUnit::Months | TimeUnit::Years => Ok(BusinessDayConvention::ModifiedFollowing),
        _ => fail!("invalid time units"),
    }
}

fn libor_eom(tenor: Period) -> QlResult<bool> {
    match tenor.units() {
        TimeUnit::Days | TimeUnit::Weeks => Ok(false),
        TimeUnit::Months | TimeUnit::Years => Ok(true),
        _ => fail!("invalid time units"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexes::index::Index;
    use crate::shared::shared;
    use crate::time::calendars::unitedstates::{Market as UsMarket, UnitedStates};
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;

    #[test]
    fn daily_tenor_is_rejected() {
        let settings = shared(Settings::<Date>::new());
        match Libor::new(
            "USDLibor".into(),
            Period::new(1, TimeUnit::Days),
            2,
            Currency::usd(),
            UnitedStates::new(UsMarket::LiborImpact),
            Actual360::new(),
            Handle::empty(),
            settings,
        ) {
            Ok(_) => panic!("daily tenor must fail"),
            Err(err) => assert!(err.message().contains("dedicated DailyTenor constructor")),
        }
    }

    #[test]
    fn eur_is_rejected() {
        let settings = shared(Settings::<Date>::new());
        match Libor::new(
            "EURLibor".into(),
            Period::new(6, TimeUnit::Months),
            2,
            Currency::eur(),
            UnitedKingdom::new(UkMarket::Exchange),
            Actual360::new(),
            Handle::empty(),
            settings,
        ) {
            Ok(_) => panic!("EUR must fail"),
            Err(err) => assert!(err.message().contains("EurLibor")),
        }
    }

    #[test]
    fn value_date_joint_adjusts_when_us_is_closed() {
        // 2004-07-05 is a Monday; US Independence Day observed 5 Jul 2004.
        // London is open. Libor value date after a Fri 2-Jul fixing should
        // skip the US holiday via the joint calendar.
        let settings = shared(Settings::<Date>::new());
        settings.set_evaluation_date(Date::new(1, Month::July, 2004));
        let index = Libor::new(
            "USDLibor".into(),
            Period::new(6, TimeUnit::Months),
            2,
            Currency::usd(),
            UnitedStates::new(UsMarket::LiborImpact),
            Actual360::new(),
            Handle::empty(),
            settings,
        )
        .unwrap();
        let fixing = Date::new(2, Month::July, 2004); // Friday
        assert!(index.is_valid_fixing_date(fixing));
        let value = index.value_date(fixing).unwrap();
        // London advance by 2 BD → Tue 6 Jul; joint adjust is still 6 Jul
        // (5 Jul is US holiday but advance already landed on 6 Jul).
        // Stronger pin: a fixing whose London+2 lands on a US holiday.
        let fixing2 = Date::new(1, Month::July, 2004); // Thursday
        let london_plus_2 = UnitedKingdom::new(UkMarket::Exchange).advance(
            fixing2,
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        assert_eq!(london_plus_2, Date::new(5, Month::July, 2004));
        let value2 = index.value_date(fixing2).unwrap();
        assert_eq!(
            value2,
            Date::new(6, Month::July, 2004),
            "joint calendar must skip US Independence Day observed 5 Jul"
        );
        assert_eq!(value, Date::new(6, Month::July, 2004));
    }
}
