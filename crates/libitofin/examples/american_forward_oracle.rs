//! Synthetic American option quotes for validating `ImpliedForward`'s model-based path.
//!
//! Prices calls and puts on a 17-strike chain (80 to 120) with the finite-difference engine on an
//! 800 x 1600 Douglas grid, which is far finer than anything the estimator itself uses, and prints
//! them as a tab-separated file together with the exact forward each chain was generated from.
//! `tests/american_forward.rs` bakes the output in as `tests/fixtures/american_forward/scenarios.tsv`.
//!
//! ```sh
//! cargo run --release -p libitofin --example american_forward_oracle \
//!   > crates/libitofin/tests/fixtures/american_forward/scenarios.tsv
//! ```
//!
//! Format: a `S` line per scenario, `S name spot rate carry days true_forward dividends`, followed
//! by one `Q` line per strike, `Q strike call_bid call_ask put_bid put_ask`. Dividends are
//! `day:amount` pairs separated by `;`, or `-`. Times are `days / 360` (Act/360).

use libitofin::cashflows::{Dividend, FixedDividend};
use libitofin::exercise::{AmericanExercise, Exercise};
use libitofin::handle::Handle;
use libitofin::instrument::Instrument;
use libitofin::instruments::{OneAssetOption, PlainVanillaPayoff};
use libitofin::interestrate::Compounding;
use libitofin::methods::finitedifferences::solvers::FdmSchemeDesc;
use libitofin::option::OptionType;
use libitofin::pricingengine::PricingEngine;
use libitofin::pricingengines::vanilla::FdBlackScholesVanillaEngine;
use libitofin::processes::BlackScholesMertonProcess;
use libitofin::quotes::{Quote, SimpleQuote};
use libitofin::settings::Settings;
use libitofin::shared::{Shared, SharedMut, shared, shared_mut};
use libitofin::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use libitofin::termstructures::yields::FlatForward;
use libitofin::termstructures::yieldtermstructure::YieldTermStructure;
use libitofin::time::date::{Date, Month};
use libitofin::time::daycounters::actual360::Actual360;
use libitofin::time::frequency::Frequency;
use libitofin::types::Real;

const SPOT: Real = 100.0;
const T_GRID: usize = 800;
const X_GRID: usize = 1600;

fn today() -> Date {
    Date::new(15, Month::June, 2026)
}

fn handle(quote: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
    Handle::new(Shared::clone(quote) as Shared<dyn Quote>)
}

struct Market {
    settings: Shared<Settings<Date>>,
    spot: Shared<SimpleQuote>,
    carry: Shared<SimpleQuote>,
    rate: Shared<SimpleQuote>,
    vol: Shared<SimpleQuote>,
    process: Shared<BlackScholesMertonProcess>,
}

fn market() -> Market {
    let settings = shared(Settings::new());
    settings.set_evaluation_date(today());
    let spot = shared(SimpleQuote::new(SPOT));
    let carry = shared(SimpleQuote::new(0.0));
    let rate = shared(SimpleQuote::new(0.0));
    let vol = shared(SimpleQuote::new(0.25));
    let flat = |quote: &Shared<SimpleQuote>| -> Shared<dyn YieldTermStructure> {
        shared(FlatForward::new(
            today(),
            handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>
    };
    let vol_curve = shared(BlackConstantVol::with_quote(
        today(),
        None,
        handle(&vol),
        Actual360::new(),
    )) as Shared<dyn BlackVolTermStructure>;
    let process = shared(BlackScholesMertonProcess::new(
        handle(&spot),
        Handle::new(flat(&carry)),
        Handle::new(flat(&rate)),
        Handle::new(vol_curve),
    ));
    Market {
        settings,
        spot,
        carry,
        rate,
        vol,
        process,
    }
}

/// Black volatility of a strike: flat, or a skewed smile `0.25 - 0.5 x + 0.2 x^2` in x = ln(K/S).
fn vol_at(strike: Real, smile: bool) -> Real {
    if smile {
        let x = (strike / SPOT).ln();
        0.25 - 0.5 * x + 0.2 * x * x
    } else {
        0.25
    }
}

struct Scenario {
    name: &'static str,
    rate: Real,
    carry: Real,
    days: i32,
    dividends: Vec<(i32, Real)>,
    smile: bool,
}

fn price(
    m: &Market,
    s: &Scenario,
    option_type: OptionType,
    strike: Real,
    engine: &SharedMut<dyn PricingEngine>,
) -> Real {
    m.spot.set_value(SPOT);
    m.rate.set_value(s.rate);
    m.carry.set_value(s.carry);
    m.vol.set_value(vol_at(strike, s.smile));
    let exercise = AmericanExercise::over(today(), today() + s.days).expect("window");
    let mut option = OneAssetOption::new(
        shared(PlainVanillaPayoff::new(option_type, strike)),
        shared(exercise) as Shared<dyn Exercise>,
        Shared::clone(&m.settings),
    );
    option
        .base_mut()
        .set_pricing_engine(SharedMut::clone(engine));
    option.npv().expect("the grid prices it")
}

fn main() {
    let scenarios = vec![
        Scenario {
            name: "r5_q0_180d",
            rate: 0.05,
            carry: 0.0,
            days: 180,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r5_q0_720d",
            rate: 0.05,
            carry: 0.0,
            days: 720,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r4_q1.5_360d",
            rate: 0.04,
            carry: 0.015,
            days: 360,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r1_q3_360d",
            rate: 0.01,
            carry: 0.03,
            days: 360,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r3_q6_720d",
            rate: 0.03,
            carry: 0.06,
            days: 720,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r5_q0_36d",
            rate: 0.05,
            carry: 0.0,
            days: 36,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r0_q0_360d",
            rate: 0.0,
            carry: 0.0,
            days: 360,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r8_q2_360d",
            rate: 0.08,
            carry: 0.02,
            days: 360,
            dividends: vec![],
            smile: false,
        },
        Scenario {
            name: "r5_q0_360d_smile",
            rate: 0.05,
            carry: 0.0,
            days: 360,
            dividends: vec![],
            smile: true,
        },
        Scenario {
            name: "r4_div_quarterly_0.5_360d",
            rate: 0.04,
            carry: 0.0,
            days: 360,
            dividends: vec![(45, 0.5), (135, 0.5), (225, 0.5), (315, 0.5)],
            smile: false,
        },
        Scenario {
            name: "r4_div_quarterly_0.5_720d",
            rate: 0.04,
            carry: 0.0,
            days: 720,
            dividends: (0..8).map(|i| (45 + 90 * i, 0.5)).collect(),
            smile: false,
        },
        Scenario {
            name: "r4_div_quarterly_1.0_360d",
            rate: 0.04,
            carry: 0.0,
            days: 360,
            dividends: vec![(45, 1.0), (135, 1.0), (225, 1.0), (315, 1.0)],
            smile: false,
        },
        Scenario {
            name: "r4_div_special_3.0_360d",
            rate: 0.04,
            carry: 0.0,
            days: 360,
            dividends: vec![(100, 3.0)],
            smile: false,
        },
        Scenario {
            name: "r4_div_2x1.0_180d",
            rate: 0.04,
            carry: 0.0,
            days: 180,
            dividends: vec![(45, 1.0), (135, 1.0)],
            smile: false,
        },
        Scenario {
            name: "r4_div_quarterly_0.5_360d_smile",
            rate: 0.04,
            carry: 0.0,
            days: 360,
            dividends: vec![(45, 0.5), (135, 0.5), (225, 0.5), (315, 0.5)],
            smile: true,
        },
    ];
    let strikes: Vec<Real> = (0..17).map(|i| 80.0 + 2.5 * i as Real).collect();
    let m = market();
    for s in &scenarios {
        let dividends: Vec<Shared<dyn Dividend>> = s
            .dividends
            .iter()
            .map(|&(day, amount)| {
                shared(FixedDividend::new(amount, today() + day)) as Shared<dyn Dividend>
            })
            .collect();
        let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&m.process) as Shared<_>,
            dividends,
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        )) as SharedMut<dyn PricingEngine>;
        let t = Real::from(s.days) / 360.0;
        let pv: Real = s
            .dividends
            .iter()
            .map(|&(day, amount)| amount * (-s.rate * Real::from(day) / 360.0).exp())
            .sum();
        let true_forward = (SPOT - pv) * ((s.rate - s.carry) * t).exp();
        let dividends = if s.dividends.is_empty() {
            "-".to_string()
        } else {
            s.dividends
                .iter()
                .map(|&(day, amount)| format!("{day}:{amount}"))
                .collect::<Vec<_>>()
                .join(";")
        };
        println!(
            "S\t{}\t{SPOT}\t{}\t{}\t{}\t{true_forward:.10}\t{dividends}",
            s.name, s.rate, s.carry, s.days
        );
        for &k in &strikes {
            let call = price(&m, s, OptionType::Call, k, &engine);
            let put = price(&m, s, OptionType::Put, k, &engine);
            // One cent plus 0.2% of the mid, either side.
            let (hc, hp) = (0.01 + 0.002 * call, 0.01 + 0.002 * put);
            println!(
                "Q\t{k}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
                call - hc,
                call + hc,
                put - hp,
                put + hp
            );
        }
    }
}
