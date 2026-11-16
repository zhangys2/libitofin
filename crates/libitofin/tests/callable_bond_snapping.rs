//! QuantLib v1.44 `callablebonds.cpp:571`
//! `testSnappingExerciseDate2ClosestCouponDate`, plus v1.44 pins of the
//! spread-inclusive snapped-exercise discount ratio from
//! `tests/fixtures/callable_v144_pins/snap_oas.py`.

use libitofin::handle::Handle;
use libitofin::instrument::Instrument;
use libitofin::instruments::{
    Bond, BondPrice, Callability, CallabilityType, CallableFixedRateBond, FixedRateBond,
};
use libitofin::interestrate::Compounding;
use libitofin::models::shortrate::HullWhite;
use libitofin::pricingengines::bond::{DiscountingBondEngine, TreeCallableFixedRateBondEngine};
use libitofin::settings::Settings;
use libitofin::shared::{Shared, shared, shared_mut};
use libitofin::termstructures::yields::FlatForward;
use libitofin::termstructures::yieldtermstructure::YieldTermStructure;
use libitofin::time::businessdayconvention::BusinessDayConvention;
use libitofin::time::calendar::Calendar;
use libitofin::time::calendars::unitedstates::{Market, UnitedStates};
use libitofin::time::date::{Date, Month};
use libitofin::time::daycounters::actual365fixed::Actual365Fixed;
use libitofin::time::daycounters::thirty360::{Convention, Thirty360};
use libitofin::time::frequency::Frequency;
use libitofin::time::schedule::{MakeSchedule, Schedule};

struct Setup {
    settings: Shared<Settings<Date>>,
    calendar: Calendar,
    curve: Shared<dyn YieldTermStructure>,
    schedule: Schedule,
    issue: Date,
}

fn setup() -> Setup {
    let today = Date::new(18, Month::May, 2021);
    let settings = shared(Settings::new());
    settings.set_evaluation_date(today);
    let calendar = UnitedStates::new(Market::FederalReserve);
    let curve: Shared<dyn YieldTermStructure> = shared(FlatForward::with_rate(
        today,
        0.02,
        Actual365Fixed::new(),
        Compounding::Continuous,
        Frequency::Annual,
    ));
    let issue = Date::new(20, Month::May, 2021) - 2 * 366;
    let schedule = MakeSchedule::new()
        .from(issue)
        .to(Date::new(14, Month::February, 2026))
        .with_frequency(Frequency::Semiannual)
        .with_calendar(calendar.clone())
        .with_convention(BusinessDayConvention::Unadjusted)
        .with_termination_date_convention(BusinessDayConvention::Unadjusted)
        .backwards()
        .end_of_month(false)
        .build();
    Setup {
        settings,
        calendar,
        curve,
        schedule,
        issue,
    }
}

fn accrual() -> libitofin::time::daycounter::DayCounter {
    Thirty360::with_convention(Convention::USA)
}

fn engine(s: &Setup) -> TreeCallableFixedRateBondEngine {
    let model = HullWhite::new(Handle::new(Shared::clone(&s.curve)), 1e-12, 0.003).unwrap();
    TreeCallableFixedRateBondEngine::new(model, 40, Shared::clone(&s.settings)).unwrap()
}

fn callable(s: &Setup, call_date: Date) -> CallableFixedRateBond {
    let calls =
        vec![Callability::new(BondPrice::Clean(100.0), CallabilityType::Call, call_date).unwrap()];
    let mut bond = CallableFixedRateBond::new(
        2,
        100.0,
        s.schedule.clone(),
        vec![0.05; s.schedule.len() - 1],
        accrual(),
        BusinessDayConvention::Following,
        100.0,
        Some(s.issue),
        calls,
        Shared::clone(&s.settings),
    )
    .unwrap();
    bond.base_mut().set_pricing_engine(shared_mut(engine(s)));
    bond
}

fn fixed(s: &Setup, call_date: Date) -> Bond {
    let mut bond = FixedRateBond::new(
        2,
        100.0,
        s.schedule.until(call_date),
        vec![0.05; s.schedule.len() - 1],
        accrual(),
        BusinessDayConvention::Following,
        100.0,
        Some(s.issue),
        None,
        None,
        s.calendar.clone(),
        BusinessDayConvention::Unadjusted,
        false,
        None,
        Shared::clone(&s.settings),
    )
    .unwrap();
    bond.bond_mut()
        .base_mut()
        .set_pricing_engine(shared_mut(DiscountingBondEngine::new(
            Handle::new(Shared::clone(&s.curve)),
            None,
            Shared::clone(&s.settings),
        )));
    bond.into_bond()
}

/// QuantLib v1.44 `OAS(cleanPrice() - 2, ...)` per business-day call date,
/// 4 to 24 Feb 2022, from `callable_v144_pins/snap_oas.py`.
const QUANTLIB_OAS: [f64; 14] = [
    0.026519876585038214,
    0.02631693405203969,
    0.026256140005112217,
    0.026195585622630736,
    0.026135269456360893,
    0.0260751900698953,
    0.025896358405991787,
    0.025837686317721358,
    0.025782570077713434,
    0.02570540628403276,
    0.025637298227712112,
    0.025372991638946834,
    0.02529078365643318,
    0.025206465957462083,
];

#[test]
fn snapping_exercise_date_to_closest_coupon_date() {
    let s = setup();
    let initial_call_date = Date::new(14, Month::February, 2022);
    let tolerance = 1e-10;
    let mut prev_oas = 0.0266;
    let expected_oas_step = 0.00005;
    let mut checked = 0;
    for i in -10..11 {
        let call_date = initial_call_date + i;
        if !s.calendar.is_business_day(call_date) {
            continue;
        }
        let npv_fixed = fixed(&s, call_date).npv().unwrap();
        let mut bond = callable(&s, call_date);
        let npv_callable = bond.npv().unwrap();
        assert!(
            (npv_callable - npv_fixed).abs() <= tolerance,
            "{call_date}: callable {npv_callable:.17}, fixed {npv_fixed:.17}"
        );
        let clean_price = bond.clean_price().unwrap() - 2.0;
        let settlement = bond.bond().settlement_date(None).unwrap();
        let accrued = bond.bond().accrued_amount(Some(settlement)).unwrap();
        let discount = s.curve.discount_date(settlement, false).unwrap();
        let oas = bond
            .oas(
                (clean_price + accrued) / discount - accrued,
                &engine(&s),
                accrual(),
                Compounding::Continuous,
                Frequency::Semiannual,
                None,
                1e-10,
                100,
                0.0,
            )
            .unwrap();
        assert!(
            (oas - QUANTLIB_OAS[checked]).abs() <= 1e-8,
            "{call_date}: oas {oas:.17}, QuantLib {:.17}",
            QUANTLIB_OAS[checked]
        );
        assert!(
            prev_oas - oas >= expected_oas_step,
            "{call_date}: oas {oas:.10}, previous {prev_oas:.10}"
        );
        prev_oas = oas;
        checked += 1;
    }
    assert_eq!(checked, QUANTLIB_OAS.len());
}

#[test]
fn snapped_exercise_discount_ratio_includes_the_oas_spread() {
    let s = setup();
    for (day, spread, expected) in [
        (4, 0.0, 102.06779542016746),
        (4, 0.01, 101.33989888788211),
        (7, 0.0, 102.09227883568438),
        (7, 0.01, 101.35597062085685),
        (11, 0.0, 102.12490662893346),
        (11, 0.01, 101.37737779825251),
    ] {
        let bond = callable(&s, Date::new(day, Month::February, 2022));
        let price = bond
            .clean_price_oas(
                spread,
                &engine(&s),
                accrual(),
                Compounding::Continuous,
                Frequency::Semiannual,
                None,
            )
            .unwrap();
        assert!(
            (price - expected).abs() <= 1e-10,
            "{day} Feb 2022, spread {spread}: actual {price:.17}, expected {expected:.17}"
        );
    }
}
