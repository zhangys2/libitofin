//! Native QuantLib 1.44 dirty-normalized OAS sensitivity fixtures.

#[path = "fixtures/medium6_callable_sensitivities/support.rs"]
mod support;

use libitofin::handle::Handle;
use libitofin::instrument::Instrument;
use libitofin::instruments::{BondPrice, Callability, CallabilityType, CallableFixedRateBond};
use libitofin::interestrate::Compounding;
use libitofin::models::shortrate::HullWhite;
use libitofin::pricingengines::bond::TreeCallableFixedRateBondEngine;
use libitofin::settings::Settings;
use libitofin::shared::{Shared, shared};
use libitofin::termstructures::yields::FlatForward;
use libitofin::termstructures::yieldtermstructure::YieldTermStructure;
use libitofin::time::businessdayconvention::BusinessDayConvention;
use libitofin::time::calendars::unitedstates::{Market, UnitedStates};
use libitofin::time::date::{Date, Month};
use libitofin::time::dategenerationrule::DateGeneration;
use libitofin::time::daycounters::actual365fixed::Actual365Fixed;
use libitofin::time::daycounters::thirty360::{Convention, Thirty360};
use libitofin::time::frequency::Frequency;
use libitofin::time::period::Period;
use libitofin::time::schedule::Schedule;
use libitofin::time::timeunit::TimeUnit;
use support::{close, engine, fixed, rights, settings, zero};

#[test]
fn thirty_six_native_clean_price_bumps_and_dirty_normalized_sensitivities() {
    for row in include_str!("fixtures/medium6_callable_sensitivities/native.tsv")
        .lines()
        .skip(1)
    {
        let columns: Vec<_> = row.split('\t').collect();
        let numbers: Vec<f64> = columns[3..].iter().map(|s| s.parse().unwrap()).collect();
        let (comp, freq) = match columns[2] {
            "continuous" => (Compounding::Continuous, Frequency::Annual),
            "compounded" => (Compounding::Compounded, Frequency::Semiannual),
            "simple" => (Compounding::Simple, Frequency::Annual),
            _ => unreachable!(),
        };
        let settings = settings();
        let engine = engine(settings.clone());
        let spread = 0.0075;
        let bump = numbers[0];
        macro_rules! check {
            ($bond:expr) => {{
                let bond = $bond;
                let accrued = bond.bond().accrued_amount(None).unwrap();
                close(accrued, numbers[1], 1e-12);
                let mut prices = vec![];
                for (value, expected) in [spread, spread + bump, spread - bump]
                    .into_iter()
                    .zip(&numbers[2..5])
                {
                    let price = bond
                        .clean_price_oas(value, &engine, Actual365Fixed::new(), comp, freq, None)
                        .unwrap();
                    close(price, *expected, 2e-11);
                    prices.push(price);
                }
                let duration = bond
                    .effective_duration(
                        spread,
                        &engine,
                        Actual365Fixed::new(),
                        comp,
                        freq,
                        Some(bump),
                    )
                    .unwrap();
                let convexity = bond
                    .effective_convexity(
                        spread,
                        &engine,
                        Actual365Fixed::new(),
                        comp,
                        freq,
                        Some(bump),
                    )
                    .unwrap();
                close(duration, numbers[5], 2e-9);
                close(convexity, numbers[6], 2e-6);
                let dirty = prices[0] + accrued;
                close(
                    duration,
                    (prices[2] - prices[1]) / (2.0 * dirty * bump),
                    1e-12,
                );
                close(
                    convexity,
                    (prices[1] + prices[2] - 2.0 * prices[0]) / (bump * bump * dirty),
                    1e-12,
                );
                if accrued > 0.0 {
                    let clean_duration = (prices[2] - prices[1]) / (2.0 * prices[0] * bump);
                    assert!((duration - clean_duration).abs() > 0.01);
                }
                if bump == 2e-4 {
                    assert_eq!(
                        duration,
                        bond.effective_duration(
                            spread,
                            &engine,
                            Actual365Fixed::new(),
                            comp,
                            freq,
                            None
                        )
                        .unwrap()
                    );
                    assert_eq!(
                        convexity,
                        bond.effective_convexity(
                            spread,
                            &engine,
                            Actual365Fixed::new(),
                            comp,
                            freq,
                            None
                        )
                        .unwrap()
                    );
                }
                assert!(bond.base().pricing_engine().is_none());
                assert!(!bond.base().is_calculated());
            }};
        }
        if columns[0] == "fixed" {
            check!(fixed(settings, rights(columns[1])));
        } else {
            check!(zero(settings, rights(columns[1]), 100.0));
        }
    }
}

/// Ports QuantLib v1.44 `callablebonds.cpp` `testEffectiveDurationAndConvexity`
/// (:1113) at its `BOOST_CHECK_CLOSE` tolerance of 1e-4 percent (1e-6
/// relative), and pins the OAS, duration and convexity from QuantLib-Python
/// 1.44 at the fixture tolerances above, which cover the OAS solver's 1e-10
/// accuracy (about 4e-10 on duration). Its explicit `settlementDate` (the evaluation date) precedes the
/// moving curve's reference date, which the Rust engine rejects
/// (`treecallablebondengine.rs:99`), and Rust's explicit-settlement quotes are
/// normalized differently (`oas.rs`, `clean_price_oas`). In v1.44 that date
/// reaches only `accruedAmount` and `notional`, while the engine discounts at
/// the bond's own settlement (`callablebond.cpp:266,292-293,333`,
/// `treecallablebondengine.cpp:85`). So the explicit accrued is kept, and the
/// prices use the default settlement with the same dirty target: the OAS solves
/// for `70.926 + accrued(settlementDate) - accrued(default)`.
#[test]
fn effective_duration_and_convexity_use_the_dirty_price() {
    let settlement_date = Date::new(30, Month::November, 2023);
    let settings = shared(Settings::new());
    settings.set_evaluation_date(settlement_date);
    let effective_date = Date::new(20, Month::May, 2021);
    let maturity_date = Date::new(1, Month::June, 2029);
    let first_coupon_date = Date::new(1, Month::December, 2021);
    let calendar = UnitedStates::new(Market::GovernmentBond);
    let day_count = Thirty360::with_convention(Convention::ISDA);
    let unadjusted = BusinessDayConvention::Unadjusted;
    let schedule = Schedule::new(
        effective_date,
        maturity_date,
        Period::new(6, TimeUnit::Months),
        calendar.clone(),
        unadjusted,
        unadjusted,
        DateGeneration::Backward,
        true,
        first_coupon_date,
        Date::null(),
    );
    let call_schedule = [
        (102.438, Date::new(1, Month::June, 2024)),
        (101.219, Date::new(1, Month::June, 2025)),
        (100.0, Date::new(1, Month::June, 2026)),
        (100.0, Date::new(1, Month::June, 2029)),
    ]
    .into_iter()
    .map(|(price, date)| {
        Callability::new(BondPrice::Clean(price), CallabilityType::Call, date).unwrap()
    })
    .collect();
    let settlement_days = 2;
    let callable_bond = CallableFixedRateBond::new(
        settlement_days,
        100.0,
        schedule,
        vec![0.04875],
        day_count.clone(),
        unadjusted,
        100.0,
        Some(effective_date),
        call_schedule,
        settings.clone(),
    )
    .unwrap();
    let flat_rate: Shared<dyn YieldTermStructure> = shared(FlatForward::moving_with_rate(
        settlement_days,
        calendar,
        0.05,
        day_count.clone(),
        Compounding::Continuous,
        Frequency::Annual,
        settings.clone(),
    ));
    let hw = HullWhite::new(Handle::new(flat_rate), 0.03, 0.012).unwrap();
    let grid_steps = ((maturity_date - settlement_date) / 30) as usize;
    let engine = TreeCallableFixedRateBondEngine::new(hw, grid_steps, settings).unwrap();
    let (comp, freq) = (Compounding::Compounded, Frequency::Semiannual);
    assert_eq!(
        callable_bond.bond().settlement_date(None).unwrap(),
        Date::new(4, Month::December, 2023)
    );
    let accrued = callable_bond
        .bond()
        .accrued_amount(Some(settlement_date))
        .unwrap();
    let default_accrued = callable_bond.bond().accrued_amount(None).unwrap();
    let oas = callable_bond
        .oas(
            70.926 + accrued - default_accrued,
            &engine,
            day_count.clone(),
            comp,
            freq,
            None,
            1e-10,
            100,
            0.0,
        )
        .unwrap();
    let shift = 0.001;
    let eff_dur = callable_bond
        .effective_duration(oas, &engine, day_count.clone(), comp, freq, Some(shift))
        .unwrap();
    let eff_conv = callable_bond
        .effective_convexity(oas, &engine, day_count.clone(), comp, freq, Some(shift))
        .unwrap();
    let dirty = |spread: f64| {
        callable_bond
            .clean_price_oas(spread, &engine, day_count.clone(), comp, freq, None)
            .unwrap()
            + default_accrued
    };
    let (p0, p_up, p_down) = (dirty(oas), dirty(oas + shift), dirty(oas - shift));
    let expected_eff_dur = (p_down - p_up) / (2.0 * p0 * shift);
    let expected_eff_conv = (p_down + p_up - 2.0 * p0) / (p0 * shift * shift);
    let incorrect_eff_dur = (p_down - p_up) / (2.0 * (p0 - accrued) * shift);
    let relative = |a: f64, b: f64, tolerance: f64| {
        assert!(
            (a - b).abs() <= tolerance * a.abs().max(b.abs()),
            "{a} != {b}"
        )
    };
    relative(eff_dur, expected_eff_dur, 1e-6);
    relative(eff_conv, expected_eff_conv, 1e-6);
    assert!((eff_dur - incorrect_eff_dur).abs() > 0.01);
    assert!((oas - 0.06514444314526413).abs() <= 1e-8);
    close(eff_dur, 4.494176075178809, 2e-9);
    close(eff_conv, 24.248354648489038, 2e-6);
}
