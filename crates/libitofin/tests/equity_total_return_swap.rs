//! Tests for EquityTotalReturnSwap.
//!
//! Direct port of QuantLib's `test-suite/equitytotalreturnswap.cpp`.
//!
//! Test cases:
//! - `test_fair_margin`: fair margin replicates zero NPV across Libor and SOFR,
//!   receiver and payer, zero/positive/negative margins, gearing = 0, and payment delay.
//! - `test_error_when_negative_nominal`: verifies negative nominal errors with "Nominal cannot be negative".
//! - `test_error_when_no_payment_calendar`: verifies empty calendar errors with "Calendar in schedule cannot be empty".
//! - `test_equity_leg_npv`: equity leg NPV replicates analytical formula:
//!   $(I(T) / I(0) - 1) \cdot N \cdot P(0, T)$.
//! - `test_trs_npv`: checks that summing legs NPV equals instrument NPV and
//!   each leg NPV replicates discounted cash flows.

use libitofin::cashflow::Leg;
use libitofin::currency::Currency;
use libitofin::errors::QlResult;
use libitofin::handle::{Handle, RelinkableHandle};
use libitofin::indexes::EquityIndex;
use libitofin::indexes::ibor::{Sofr, UsdLibor};
use libitofin::indexes::iborindex::{IborIndex, OvernightIndex};
use libitofin::indexes::index::Index;
use libitofin::instruments::{EquityTotalReturnSwap, SwapType};
use libitofin::interestrate::Compounding;
use libitofin::pricingengine::PricingEngine;
use libitofin::pricingengines::DiscountingSwapEngine;
use libitofin::quotes::{Quote, SimpleQuote};
use libitofin::settings::Settings;
use libitofin::shared::{Shared, SharedMut, shared, shared_mut};
use libitofin::termstructures::yields::FlatForward;
use libitofin::termstructures::yieldtermstructure::YieldTermStructure;
use libitofin::time::businessdayconvention::BusinessDayConvention;
use libitofin::time::calendar::Calendar;
use libitofin::time::calendars::unitedstates::{Market, UnitedStates};
use libitofin::time::date::{Date, Month};
use libitofin::time::dategenerationrule::DateGeneration;
use libitofin::time::daycounter::DayCounter;
use libitofin::time::daycounters::actual365fixed::Actual365Fixed;
use libitofin::time::frequency::Frequency;
use libitofin::time::period::Period;
use libitofin::time::schedule::{MakeSchedule, Schedule};
use libitofin::time::timeunit::TimeUnit;
use libitofin::types::{Natural, Rate, Real};

#[allow(dead_code)]
struct CommonVars {
    today: Date,
    calendar: Calendar,
    day_count: DayCounter,
    settings: Shared<Settings<Date>>,
    equity_index: Shared<EquityIndex>,
    usd_libor: Shared<IborIndex>,
    sofr: Shared<OvernightIndex>,
    interest_handle: RelinkableHandle<dyn YieldTermStructure>,
    dividend_handle: RelinkableHandle<dyn YieldTermStructure>,
    spot: Shared<SimpleQuote>,
    spot_handle: RelinkableHandle<dyn Quote>,
    discount_engine: SharedMut<DiscountingSwapEngine>,
}

impl CommonVars {
    fn new() -> Self {
        let calendar = UnitedStates::new(Market::GovernmentBond);
        let day_count: DayCounter = Actual365Fixed::new();

        let today = calendar.adjust(
            Date::new(27, Month::January, 2023),
            BusinessDayConvention::Following,
        );
        let settings = shared(Settings::<Date>::new());
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(8700.0));
        let spot_handle = RelinkableHandle::new(spot.clone() as Shared<dyn Quote>);

        let interest_curve = shared(FlatForward::with_rate(
            today,
            0.0375,
            day_count.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        ));
        let interest_handle =
            RelinkableHandle::new(interest_curve as Shared<dyn YieldTermStructure>);

        let dividend_curve = shared(FlatForward::with_rate(
            today,
            0.005,
            day_count.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        ));
        let dividend_handle =
            RelinkableHandle::new(dividend_curve as Shared<dyn YieldTermStructure>);

        let equity_index = shared(EquityIndex::new(
            "eqIndex",
            calendar.clone(),
            Currency::usd(),
            interest_handle.handle(),
            dividend_handle.handle(),
            spot_handle.handle(),
            Shared::clone(&settings),
        ));
        equity_index
            .add_fixing(Date::new(5, Month::January, 2023), 9010.0)
            .unwrap();
        equity_index.add_fixing(today, 8690.0).unwrap();

        let sofr = shared(Sofr::new(
            interest_handle.handle(),
            Shared::clone(&settings),
        ));
        sofr.add_fixing(Date::new(3, Month::January, 2023), 0.03)
            .unwrap();
        sofr.add_fixing(Date::new(4, Month::January, 2023), 0.031)
            .unwrap();
        sofr.add_fixing(Date::new(5, Month::January, 2023), 0.031)
            .unwrap();
        sofr.add_fixing(Date::new(6, Month::January, 2023), 0.031)
            .unwrap();
        sofr.add_fixing(Date::new(9, Month::January, 2023), 0.032)
            .unwrap();
        sofr.add_fixing(Date::new(10, Month::January, 2023), 0.033)
            .unwrap();
        sofr.add_fixing(Date::new(11, Month::January, 2023), 0.033)
            .unwrap();
        sofr.add_fixing(Date::new(12, Month::January, 2023), 0.033)
            .unwrap();
        sofr.add_fixing(Date::new(13, Month::January, 2023), 0.033)
            .unwrap();
        sofr.add_fixing(Date::new(17, Month::January, 2023), 0.033)
            .unwrap();
        sofr.add_fixing(Date::new(18, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(19, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(20, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(23, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(24, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(25, Month::January, 2023), 0.034)
            .unwrap();
        sofr.add_fixing(Date::new(26, Month::January, 2023), 0.034)
            .unwrap();

        let usd_libor = shared(
            UsdLibor::new(
                Period::new(3, TimeUnit::Months),
                interest_handle.handle(),
                Shared::clone(&settings),
            )
            .unwrap(),
        );
        usd_libor
            .add_fixing(Date::new(3, Month::January, 2023), 0.035)
            .unwrap();

        let discount_engine = shared_mut(DiscountingSwapEngine::new(
            interest_handle.handle(),
            None,
            None,
            None,
            Shared::clone(&settings),
        ));

        CommonVars {
            today,
            calendar,
            day_count,
            settings,
            equity_index,
            usd_libor,
            sofr,
            interest_handle,
            dividend_handle,
            spot,
            spot_handle,
            discount_engine,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_trs(
        &self,
        swap_type: SwapType,
        schedule: Schedule,
        use_overnight_index: bool,
        margin: Rate,
        nominal: Real,
        gearing: Real,
        payment_delay: Natural,
    ) -> QlResult<EquityTotalReturnSwap> {
        let mut swap = if use_overnight_index {
            EquityTotalReturnSwap::new(
                swap_type,
                nominal,
                schedule.clone(),
                Shared::clone(&self.equity_index),
                Shared::clone(&self.sofr),
                self.day_count.clone(),
                margin,
                gearing,
                Some(schedule.calendar().clone()),
                BusinessDayConvention::Following,
                payment_delay,
                Shared::clone(&self.settings),
            )?
        } else {
            EquityTotalReturnSwap::new(
                swap_type,
                nominal,
                schedule.clone(),
                Shared::clone(&self.equity_index),
                Shared::clone(&self.usd_libor),
                self.day_count.clone(),
                margin,
                gearing,
                Some(schedule.calendar().clone()),
                BusinessDayConvention::Following,
                payment_delay,
                Shared::clone(&self.settings),
            )?
        };
        swap.set_pricing_engine(self.discount_engine.clone() as SharedMut<dyn PricingEngine>);
        Ok(swap)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_trs_dates(
        &self,
        swap_type: SwapType,
        start: Date,
        end: Date,
        use_overnight_index: bool,
        margin: Rate,
        nominal: Real,
        gearing: Real,
        payment_delay: Natural,
    ) -> QlResult<EquityTotalReturnSwap> {
        let schedule = MakeSchedule::new()
            .from(start)
            .to(end)
            .with_tenor(Period::new(3, TimeUnit::Months))
            .with_calendar(self.calendar.clone())
            .with_convention(BusinessDayConvention::Following)
            .backwards()
            .build();
        self.create_trs(
            swap_type,
            schedule,
            use_overnight_index,
            margin,
            nominal,
            gearing,
            payment_delay,
        )
    }
}

fn check_fair_margin_calculation(
    swap_type: SwapType,
    start: Date,
    end: Date,
    use_overnight_index: bool,
    margin: Rate,
    gearing: Real,
    payment_delay: Natural,
) {
    let vars = CommonVars::new();
    let tolerance = 1.0e-8;
    let nominal = 1.0e7;

    let mut trs = vars
        .create_trs_dates(
            swap_type,
            start,
            end,
            use_overnight_index,
            margin,
            nominal,
            gearing,
            payment_delay,
        )
        .unwrap();

    let fair_margin = trs.fair_margin().unwrap();
    let mut par_trs = vars
        .create_trs_dates(
            swap_type,
            start,
            end,
            use_overnight_index,
            fair_margin,
            nominal,
            gearing,
            payment_delay,
        )
        .unwrap();

    let par_npv = par_trs.npv().unwrap();
    assert!(
        par_npv.abs() <= tolerance,
        "unable to imply a fair margin: actual NPV = {}, expected = 0.0, fair margin = {}, IR index = {}",
        par_npv,
        fair_margin,
        trs.interest_rate_index().name()
    );
}

fn leg_npv(leg: &Leg, ts: &Handle<dyn YieldTermStructure>) -> QlResult<Real> {
    let curve = ts.current_link()?;
    let mut npv = 0.0;
    for cf in leg {
        npv += cf.amount()? * curve.discount_date(cf.date(), false)?;
    }
    Ok(npv)
}

fn check_npv_calculation(
    swap_type: SwapType,
    start: Date,
    end: Date,
    use_overnight_index: bool,
    margin: Rate,
    gearing: Real,
    payment_delay: Natural,
) {
    let vars = CommonVars::new();
    let tolerance = 1.0e-2;
    let nominal = 1.0e7;

    let mut trs = vars
        .create_trs_dates(
            swap_type,
            start,
            end,
            use_overnight_index,
            margin,
            nominal,
            gearing,
            payment_delay,
        )
        .unwrap();

    let npv = trs.npv().unwrap();
    let scaling = if swap_type == SwapType::Receiver {
        1.0
    } else {
        -1.0
    };

    let equity_leg_npv = trs.equity_leg_npv().unwrap();
    let replicated_equity_leg_npv =
        scaling * leg_npv(trs.equity_leg(), &vars.interest_handle.handle()).unwrap();
    assert!(
        (equity_leg_npv - replicated_equity_leg_npv).abs() <= tolerance,
        "incorrect NPV of the equity leg: actual = {}, expected = {}",
        equity_leg_npv,
        replicated_equity_leg_npv
    );

    let interest_leg_npv = trs.interest_rate_leg_npv().unwrap();
    let replicated_interest_leg_npv =
        -scaling * leg_npv(trs.interest_rate_leg(), &vars.interest_handle.handle()).unwrap();
    assert!(
        (interest_leg_npv - replicated_interest_leg_npv).abs() <= tolerance,
        "incorrect NPV of the interest leg: actual = {}, expected = {}",
        interest_leg_npv,
        replicated_interest_leg_npv
    );

    assert!(
        (npv - (equity_leg_npv + interest_leg_npv)).abs() <= tolerance,
        "summing legs NPV does not replicate instrument NPV: actual = {}, sum = {}",
        npv,
        equity_leg_npv + interest_leg_npv
    );
}

#[test]
fn test_fair_margin() {
    // Check TRS vs Libor-type index
    check_fair_margin_calculation(
        SwapType::Receiver,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.0,
        1.0,
        0,
    );
    check_fair_margin_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.01,
        1.0,
        0,
    );
    check_fair_margin_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.0,
        0.0,
        0,
    );
    check_fair_margin_calculation(
        SwapType::Receiver,
        Date::new(31, Month::January, 2023),
        Date::new(30, Month::April, 2023),
        false,
        -0.005,
        1.0,
        2,
    );

    // Check TRS vs overnight index
    check_fair_margin_calculation(
        SwapType::Receiver,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        true,
        0.0,
        1.0,
        0,
    );
    check_fair_margin_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        true,
        0.01,
        1.0,
        0,
    );
    check_fair_margin_calculation(
        SwapType::Receiver,
        Date::new(31, Month::January, 2023),
        Date::new(30, Month::April, 2023),
        true,
        -0.005,
        1.0,
        2,
    );
}

#[test]
fn test_error_when_negative_nominal() {
    let vars = CommonVars::new();
    let result = vars.create_trs_dates(
        SwapType::Receiver,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.0,
        -1.0e7,
        1.0,
        0,
    );

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.message().contains("Nominal cannot be negative"),
        "expected 'Nominal cannot be negative', got: {}",
        err.message()
    );
}

#[test]
fn test_error_when_no_payment_calendar() {
    let vars = CommonVars::new();
    let schedule = Schedule::new(
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        Period::new(3, TimeUnit::Months),
        Calendar::empty(),
        BusinessDayConvention::Unadjusted,
        BusinessDayConvention::Unadjusted,
        DateGeneration::Backward,
        false,
        Date::null(),
        Date::null(),
    );

    let result = vars.create_trs(SwapType::Receiver, schedule, false, 0.0, 1.0e7, 1.0, 0);

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.message()
            .contains("Calendar in schedule cannot be empty"),
        "expected 'Calendar in schedule cannot be empty', got: {}",
        err.message()
    );
}

#[test]
fn test_equity_leg_npv() {
    let vars = CommonVars::new();
    let tolerance = 1.0e-8;

    let start = Date::new(5, Month::January, 2023);
    let end = Date::new(5, Month::April, 2023);

    let mut trs = vars
        .create_trs_dates(SwapType::Receiver, start, end, false, 0.0, 1.0e7, 1.0, 0)
        .unwrap();

    let actual_equity_leg_npv = trs.equity_leg_npv().unwrap();

    let eq_idx = trs.equity_index();
    let curve = vars.interest_handle.handle().current_link().unwrap();
    let discount = curve.discount_date(end, false).unwrap();
    let expected_equity_leg_npv =
        (eq_idx.fixing(end, false).unwrap() / eq_idx.fixing(start, false).unwrap() - 1.0)
            * trs.nominal()
            * discount;

    assert!(
        (actual_equity_leg_npv - expected_equity_leg_npv).abs() <= tolerance,
        "unable to replicate equity leg NPV: actual = {}, expected = {}",
        actual_equity_leg_npv,
        expected_equity_leg_npv
    );
}

#[test]
fn test_trs_npv() {
    // Check TRS vs Libor-type index
    check_npv_calculation(
        SwapType::Receiver,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.0,
        1.0,
        0,
    );
    check_npv_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.01,
        1.0,
        0,
    );
    check_npv_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        false,
        0.0,
        0.0,
        0,
    );
    check_npv_calculation(
        SwapType::Receiver,
        Date::new(31, Month::January, 2023),
        Date::new(30, Month::April, 2023),
        false,
        -0.005,
        1.0,
        2,
    );

    // Check TRS vs overnight index
    check_npv_calculation(
        SwapType::Receiver,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        true,
        0.0,
        1.0,
        0,
    );
    check_npv_calculation(
        SwapType::Payer,
        Date::new(5, Month::January, 2023),
        Date::new(5, Month::April, 2023),
        true,
        0.01,
        1.0,
        0,
    );
    check_npv_calculation(
        SwapType::Receiver,
        Date::new(31, Month::January, 2023),
        Date::new(30, Month::April, 2023),
        true,
        -0.005,
        1.0,
        2,
    );
}
