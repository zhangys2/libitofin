//! AUD Libor.
//!
//! Port of `ql/indexes/ibor/audlibor.hpp`. [`AUDLibor`] is ICE AUD Libor over
//! the Australia settlement calendar with two settlement days and Actual/360.
//! Australian Dollar LIBOR was discontinued as of 2013; the index remains for
//! historical/oracle use (`bonds.cpp` `testFixingConvention`).

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::indexes::ibor::libor::Libor;
use crate::indexes::iborindex::IborIndex;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::calendars::australia::{Australia, Market};
use crate::time::date::Date;
use crate::time::daycounters::actual360::Actual360;
use crate::time::period::Period;
use crate::time::timeunit::TimeUnit;

/// AUD Libor constructors (`ql/indexes/ibor/audlibor.hpp`).
pub struct AUDLibor;

impl AUDLibor {
    /// Builds an AUD Libor index of the given `tenor` over `forwarding`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        tenor: Period,
        forwarding: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<IborIndex> {
        Libor::new(
            "AUDLibor".into(),
            tenor,
            2,
            Currency::aud(),
            Australia::new(Market::Settlement),
            Actual360::new(),
            forwarding,
            settings,
        )
    }

    /// The 3-month AUD Libor index (`AUDLibor(3M)`).
    pub fn three_months(
        forwarding: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> IborIndex {
        Self::new(Period::new(3, TimeUnit::Months), forwarding, settings)
            .expect("a 3-month AUD Libor tenor is always valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexes::index::Index;
    use crate::indexes::interestrateindex::InterestRateIndex;
    use crate::shared::shared;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::unitedkingdom::{Market as UkMarket, UnitedKingdom};

    #[test]
    fn audlibor3m_matches_the_quantlib_construction_table() {
        let settings = shared(Settings::<Date>::new());
        let index = AUDLibor::three_months(Handle::empty(), settings);

        assert_eq!(index.name(), "AUDLibor3M Actual/360");
        assert_eq!(index.fixing_days(), 2);
        assert_eq!(*index.currency(), Currency::aud());
        assert_eq!(
            index.fixing_calendar().name(),
            UnitedKingdom::new(UkMarket::Exchange).name()
        );
        assert_eq!(index.day_counter().name(), "Actual/360");
        assert_eq!(
            index.business_day_convention(),
            BusinessDayConvention::ModifiedFollowing
        );
        assert!(index.end_of_month());
        assert!(index.joint_calendar().is_some());
        assert_eq!(
            index.financial_center_calendar().map(|c| c.name()),
            Some(Australia::new(Market::Settlement).name())
        );
    }
}
