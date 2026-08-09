//! USD Libor.
//!
//! Port of `ql/indexes/ibor/usdlibor.hpp`. [`USDLibor`] is ICE USD Libor over
//! the US Libor-impact calendar with two settlement days and Actual/360.

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::indexes::ibor::libor::Libor;
use crate::indexes::iborindex::IborIndex;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::calendars::unitedstates::{Market, UnitedStates};
use crate::time::date::Date;
use crate::time::daycounters::actual360::Actual360;
use crate::time::period::Period;
use crate::time::timeunit::TimeUnit;

/// USD Libor constructors (`ql/indexes/ibor/usdlibor.hpp`).
pub struct USDLibor;

impl USDLibor {
    /// Builds a USD Libor index of the given `tenor` over `forwarding`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        tenor: Period,
        forwarding: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<IborIndex> {
        Libor::new(
            "USDLibor".into(),
            tenor,
            2,
            Currency::usd(),
            UnitedStates::new(Market::LiborImpact),
            Actual360::new(),
            forwarding,
            settings,
        )
    }

    /// The 6-month USD Libor index (`USDLibor(6M)`).
    pub fn six_months(
        forwarding: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> IborIndex {
        Self::new(Period::new(6, TimeUnit::Months), forwarding, settings)
            .expect("a 6-month USD Libor tenor is always valid")
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
    fn usdlibor6m_matches_the_quantlib_construction_table() {
        let settings = shared(Settings::<Date>::new());
        let index = USDLibor::six_months(Handle::empty(), settings);

        assert_eq!(index.name(), "USDLibor6M Actual/360");
        assert_eq!(index.fixing_days(), 2);
        assert_eq!(*index.currency(), Currency::usd());
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
            Some(UnitedStates::new(Market::LiborImpact).name())
        );
    }
}
