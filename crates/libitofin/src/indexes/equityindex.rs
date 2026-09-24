//! Base equity index.
//!
//! Port of `ql/indexes/equityindex.{hpp,cpp}`. [`EquityIndex`] models an equity
//! asset or equity index whose fixings can be retrieved from historical records
//! or forecasted using interest-rate and dividend discount curves.
//!
//! When forecasting a fixing at date $T$, the index uses the spot value $S$ and
//! discount factors:
//!
//! $$I(t, T) = S \frac{P_D(t, T)}{P_R(t, T)}$$
//!
//! where $P_D$ is the dividend curve discount factor and $P_R$ is the risk-free
//! discount factor. If no dividend curve is provided, $P_D(t, T) = 1$.
//!
//! To obtain the spot value $S$, the index reads from the spot quote handle if
//! non-empty. If the spot handle is empty, it falls back to the past fixing on
//! the preceding business day.

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::indexes::index::Index;
use crate::patterns::observable::{Observable, Observer, ResetThenNotify};
use crate::quotes::Quote;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::types::Rate;
use crate::{fail, require};

/// Base equity index (`EquityIndex`).
pub struct EquityIndex {
    name: String,
    fixing_calendar: Calendar,
    currency: Currency,
    interest: Handle<dyn YieldTermStructure>,
    dividend: Handle<dyn YieldTermStructure>,
    spot: Handle<dyn Quote>,
    settings: Shared<Settings<Date>>,
    observable: Shared<Observable>,
    _forwarder: SharedMut<ResetThenNotify>,
}

impl EquityIndex {
    /// Builds an equity index with curves and spot quote.
    pub fn new(
        name: impl Into<String>,
        fixing_calendar: Calendar,
        currency: Currency,
        interest: Handle<dyn YieldTermStructure>,
        dividend: Handle<dyn YieldTermStructure>,
        spot: Handle<dyn Quote>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let name = name.into();
        let (observable, forwarder) = ResetThenNotify::forwarder();

        interest.register_observer(&(forwarder.clone() as SharedMut<dyn Observer>));
        dividend.register_observer(&(forwarder.clone() as SharedMut<dyn Observer>));
        spot.register_observer(&(forwarder.clone() as SharedMut<dyn Observer>));
        settings.register_eval_date_observer(&(forwarder.clone() as SharedMut<dyn Observer>));
        settings.register_fixing_observer(&name, &(forwarder.clone() as SharedMut<dyn Observer>));

        EquityIndex {
            name,
            fixing_calendar,
            currency,
            interest,
            dividend,
            spot,
            settings,
            observable,
            _forwarder: forwarder,
        }
    }

    /// The index name.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// The calendar defining valid fixing dates.
    pub fn fixing_calendar(&self) -> Calendar {
        self.fixing_calendar.clone()
    }

    /// Whether `fixing_date` is a business day for this index.
    pub fn is_valid_fixing_date(&self, fixing_date: Date) -> bool {
        self.fixing_calendar.is_business_day(fixing_date)
    }

    /// The index currency.
    pub fn currency(&self) -> &Currency {
        &self.currency
    }

    /// The rate curve used to forecast fixings.
    pub fn equity_interest_rate_curve(&self) -> &Handle<dyn YieldTermStructure> {
        &self.interest
    }

    /// The dividend curve used to forecast fixings.
    pub fn equity_dividend_curve(&self) -> &Handle<dyn YieldTermStructure> {
        &self.dividend
    }

    /// The index spot value quote handle.
    pub fn spot(&self) -> &Handle<dyn Quote> {
        &self.spot
    }

    /// The settings holding evaluation date and fixing history.
    pub fn settings(&self) -> &Settings<Date> {
        &self.settings
    }

    /// Returns a copy of itself linked to different interest, dividend curves or spot quote.
    pub fn clone_with(
        &self,
        interest: Handle<dyn YieldTermStructure>,
        dividend: Handle<dyn YieldTermStructure>,
        spot: Handle<dyn Quote>,
    ) -> Self {
        Self::new(
            self.name.clone(),
            self.fixing_calendar.clone(),
            self.currency.clone(),
            interest,
            dividend,
            spot,
            Shared::clone(&self.settings),
        )
    }

    /// Forecasts the fixing on `fixing_date`.
    pub fn forecast_fixing(&self, fixing_date: Date) -> QlResult<Rate> {
        require!(
            !self.interest.is_empty(),
            "null interest rate term structure set to this instance of {}",
            self.name()
        );

        let today = match self.settings.evaluation_date() {
            Some(d) => d,
            None => fail!("evaluation date not set"),
        };
        let last_fixing_date = self
            .fixing_calendar
            .adjust(today, BusinessDayConvention::Preceding);

        let spot = if !self.spot.is_empty() {
            self.spot.current_link()?.value()?
        } else if let Some(fixing) = self.past_fixing(last_fixing_date)? {
            fixing
        } else {
            fail!("Cannot forecast equity index, missing both spot and historical index");
        };

        let interest_ts = self.interest.current_link()?;
        let interest_discount = interest_ts.discount_date(fixing_date, false)?;

        let forward = if !self.dividend.is_empty() {
            let dividend_ts = self.dividend.current_link()?;
            let dividend_discount = dividend_ts.discount_date(fixing_date, false)?;
            spot * dividend_discount / interest_discount
        } else {
            spot / interest_discount
        };

        Ok(forward)
    }

    /// The fixing at `fixing_date`.
    pub fn fixing(&self, fixing_date: Date, forecast_todays_fixing: bool) -> QlResult<Rate> {
        require!(
            self.is_valid_fixing_date(fixing_date),
            "Fixing date {:?} is not valid",
            fixing_date
        );

        let today = match self.settings.evaluation_date() {
            Some(d) => d,
            None => fail!("evaluation date not set"),
        };

        if fixing_date > today || (fixing_date == today && forecast_todays_fixing) {
            return self.forecast_fixing(fixing_date);
        }

        if let Some(fixing) = self.past_fixing(fixing_date)? {
            return Ok(fixing);
        }

        if fixing_date == today && !self.spot.is_empty() {
            return self.spot.current_link()?.value();
        }

        fail!("Missing {} fixing for {:?}", self.name(), fixing_date);
    }

    /// The observable this index broadcasts its changes through.
    pub fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl Index for EquityIndex {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn fixing_calendar(&self) -> Calendar {
        self.fixing_calendar.clone()
    }

    fn is_valid_fixing_date(&self, fixing_date: Date) -> bool {
        self.is_valid_fixing_date(fixing_date)
    }

    fn fixing(&self, fixing_date: Date, forecast_todays_fixing: bool) -> QlResult<Rate> {
        self.fixing(fixing_date, forecast_todays_fixing)
    }

    fn settings(&self) -> &Settings<Date> {
        &self.settings
    }

    fn observable(&self) -> &Observable {
        &self.observable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::RelinkableHandle;
    use crate::quotes::SimpleQuote;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    struct TestObserver {
        updated: bool,
    }

    impl Observer for TestObserver {
        fn update(&mut self) {
            self.updated = true;
        }
    }

    #[test]
    fn test_equity_index_past_and_today_fixing() {
        let settings = shared(Settings::<Date>::new());
        let today = Date::new(27, Month::January, 2023);
        settings.set_evaluation_date(today);

        let calendar = Target::new();
        let currency = Currency::eur();
        let index = EquityIndex::new(
            "SX5E",
            calendar,
            currency,
            Handle::empty(),
            Handle::empty(),
            Handle::empty(),
            Shared::clone(&settings),
        );

        let past_date = Date::new(5, Month::January, 2023);
        index.add_fixing(past_date, 4000.0).unwrap();

        assert_eq!(index.fixing(past_date, false).unwrap(), 4000.0);

        // Missing today's fixing when spot is empty fails
        assert!(index.fixing(today, false).is_err());
    }

    #[test]
    fn test_equity_index_forecast_with_spot_and_curves() {
        let settings = shared(Settings::<Date>::new());
        let today = Date::new(27, Month::January, 2023);
        settings.set_evaluation_date(today);

        let calendar = NullCalendar::new();
        let currency = Currency::usd();
        let day_counter = Actual365Fixed::new();

        let r_ts = shared(FlatForward::with_rate(
            today,
            0.05,
            day_counter.clone(),
            crate::interestrate::Compounding::Continuous,
            crate::time::frequency::Frequency::Annual,
        ));
        let q_ts = shared(FlatForward::with_rate(
            today,
            0.02,
            day_counter.clone(),
            crate::interestrate::Compounding::Continuous,
            crate::time::frequency::Frequency::Annual,
        ));

        let spot_quote = shared(SimpleQuote::new(100.0));
        let spot_handle = Handle::new(spot_quote as Shared<dyn Quote>);

        let index = EquityIndex::new(
            "SPX",
            calendar,
            currency,
            Handle::new(r_ts as Shared<dyn YieldTermStructure>),
            Handle::new(q_ts as Shared<dyn YieldTermStructure>),
            spot_handle,
            Shared::clone(&settings),
        );

        let future_date = Date::new(27, Month::January, 2024);
        let forecast = index.fixing(future_date, false).unwrap();

        // Analytical forward = 100 * exp((0.05 - 0.02) * 1.0) = 100 * exp(0.03) = 103.045453395...
        let expected = 100.0 * (0.03f64).exp();
        assert!((forecast - expected).abs() < 1e-6);
    }

    #[test]
    fn test_equity_index_observation() {
        let settings = shared(Settings::<Date>::new());
        let today = Date::new(27, Month::January, 2023);
        settings.set_evaluation_date(today);

        let spot_quote = shared(SimpleQuote::new(100.0));
        let spot_relinkable =
            RelinkableHandle::<dyn Quote>::new(spot_quote.clone() as Shared<dyn Quote>);

        let index = EquityIndex::new(
            "TEST",
            NullCalendar::new(),
            Currency::usd(),
            Handle::empty(),
            Handle::empty(),
            spot_relinkable.handle(),
            Shared::clone(&settings),
        );

        let obs = shared_mut(TestObserver { updated: false });
        index
            .observable()
            .register_observer(&(obs.clone() as SharedMut<dyn Observer>));

        spot_quote.set_value(105.0);
        assert!(obs.borrow().updated);
    }
}
