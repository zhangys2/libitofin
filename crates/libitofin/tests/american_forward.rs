//! Validation of the model-based American implied forward on synthetic quotes.
//!
//! `fixtures/american_forward/scenarios.tsv` holds 17-strike chains priced by an 800 x 1600
//! finite-difference grid (see `examples/american_forward_oracle.rs`) together with the exact
//! forward each chain was generated from. The estimator only has closed forms, a 400-step tree
//! and a 100 x 200 grid to work with, so the recovered forward is an independent check.

use libitofin::termstructures::forward::{
    AmericanLevel, AmericanModelConfig, ForwardFlags, ForwardStatus, ForwardTier, ImpliedForward,
    ImpliedForwardConfig, ImpliedForwardResult, MarketConvention, OptionQuotePair,
};

const SCENARIOS: &str = include_str!("fixtures/american_forward/scenarios.tsv");
const SPOT: f64 = 100.0;

struct Scenario {
    name: String,
    rate: f64,
    expiry: f64,
    true_forward: f64,
    dividends: Vec<(f64, f64)>,
    quotes: Vec<OptionQuotePair>,
}

fn load() -> Vec<Scenario> {
    let mut scenarios: Vec<Scenario> = Vec::new();
    for line in SCENARIOS.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "S" => {
                let days: f64 = f[5].parse().unwrap();
                let dividends = if f[7] == "-" {
                    Vec::new()
                } else {
                    f[7].split(';')
                        .map(|d| {
                            let (day, amount) = d.split_once(':').unwrap();
                            (day.parse::<f64>().unwrap() / 360.0, amount.parse().unwrap())
                        })
                        .collect()
                };
                scenarios.push(Scenario {
                    name: f[1].to_string(),
                    rate: f[3].parse().unwrap(),
                    expiry: days / 360.0,
                    true_forward: f[6].parse().unwrap(),
                    dividends,
                    quotes: Vec::new(),
                });
            }
            "Q" => {
                let v: Vec<f64> = f[1..].iter().map(|x| x.parse().unwrap()).collect();
                scenarios
                    .last_mut()
                    .unwrap()
                    .quotes
                    .push(OptionQuotePair::new(v[0], v[1], v[2], v[3], v[4]));
            }
            other => panic!("unknown record {other}"),
        }
    }
    scenarios
}

fn config(s: &Scenario, level: AmericanLevel, with_schedule: bool) -> ImpliedForwardConfig {
    ImpliedForwardConfig {
        discount_factor: Some((-s.rate * s.expiry).exp()),
        // Forwards of long expiries sit more than 10% from spot.
        max_spot_deviation: None,
        american: AmericanModelConfig {
            level,
            dividends: if with_schedule {
                s.dividends.clone()
            } else {
                Vec::new()
            },
            ..AmericanModelConfig::default()
        },
        ..Default::default()
    }
}

fn run(s: &Scenario, config: &ImpliedForwardConfig) -> ImpliedForwardResult {
    let convention = MarketConvention::AmericanEquity {
        spot: SPOT,
        atm_vol: None,
    };
    ImpliedForward::calculate(&s.quotes, convention, config, s.expiry).unwrap()
}

fn error_bp(result: &ImpliedForwardResult, s: &Scenario) -> f64 {
    (result.forward / s.true_forward - 1.0) * 1e4
}

fn plain_parity(s: &Scenario) -> ImpliedForwardResult {
    ImpliedForward::calculate(
        &s.quotes,
        MarketConvention::EuropeanVanilla,
        &config(s, AmericanLevel::Auto, false),
        s.expiry,
    )
    .unwrap()
}

#[test]
fn fixtures_cover_the_expected_scenarios() {
    let scenarios = load();
    assert_eq!(scenarios.len(), 15);
    assert!(scenarios.iter().all(|s| s.quotes.len() == 17));
    assert_eq!(
        scenarios.iter().filter(|s| !s.dividends.is_empty()).count(),
        6
    );
}

#[test]
fn fast_level_recovers_the_forward_without_dividends() {
    // Observed worst case 6.0 bp (r = 3%, q = 6%, two years); the closed-form error budget is 8.
    for s in load().iter().filter(|s| s.dividends.is_empty()) {
        let result = run(s, &config(s, AmericanLevel::Fast, false));
        assert_eq!(result.tier, Some(ForwardTier::ModelFast), "{}", s.name);
        assert!(
            result.flags.is_empty(),
            "{}: {:?}",
            s.name,
            result.flags.names()
        );
        assert_eq!(
            result.diagnostics.status,
            ForwardStatus::Valid,
            "{}",
            s.name
        );
        assert!(
            error_bp(&result, s).abs() < 8.0,
            "{}: {} bp",
            s.name,
            error_bp(&result, s)
        );
    }
}

#[test]
fn accurate_level_recovers_the_forward_everywhere() {
    // Observed worst case 0.82 bp (two years of quarterly dividends); the target is under 2 bp.
    for s in load() {
        let result = run(&s, &config(&s, AmericanLevel::Accurate, true));
        assert_eq!(result.tier, Some(ForwardTier::ModelAccurate), "{}", s.name);
        assert!(
            result.flags.is_empty(),
            "{}: {:?}",
            s.name,
            result.flags.names()
        );
        assert_eq!(
            result.diagnostics.status,
            ForwardStatus::Valid,
            "{}",
            s.name
        );
        assert!(
            error_bp(&result, &s).abs() < 1.5,
            "{}: {} bp",
            s.name,
            error_bp(&result, &s)
        );
        // The synthetic bid/ask interval, built from quotes corrected at the mid, contains the truth.
        assert!(
            result.forward_bid_strict <= s.true_forward
                && s.true_forward <= result.forward_ask_strict,
            "{}: [{}, {}] vs {}",
            s.name,
            result.forward_bid_strict,
            result.forward_ask_strict,
            s.true_forward
        );
        let work = result.american.unwrap();
        assert!(work.accurate_prices_used > 0);
        assert!(
            work.call_put_iv_gap.abs() < 1e-6,
            "{}: {}",
            s.name,
            work.call_put_iv_gap
        );
    }
}

#[test]
fn model_path_improves_on_plain_parity_where_parity_is_biased() {
    for s in load() {
        let plain = error_bp(&plain_parity(&s), &s).abs();
        let accurate = error_bp(&run(&s, &config(&s, AmericanLevel::Accurate, true)), &s).abs();
        if plain > 3.0 {
            assert!(
                accurate * 3.0 < plain,
                "{}: plain {plain} vs accurate {accurate}",
                s.name
            );
        }
        // With no early-exercise premium (zero rates) the model must not make things worse.
        assert!(accurate < 1.5 || accurate < plain, "{}", s.name);
    }
}

#[test]
fn default_work_budget_pays_for_a_seventeen_strike_chain_with_a_schedule() {
    // A full chain is 17 strikes x 2 prices x 2 passes = 68 prices, inside the default budget of 100.
    for s in load().iter().filter(|s| !s.dividends.is_empty()) {
        let result = run(s, &AmericanConfigOnly::default_for(s));
        assert_eq!(result.tier, Some(ForwardTier::ModelAccurate), "{}", s.name);
        assert!(
            !result.flags.contains(ForwardFlags::LATENCY_FALLBACK),
            "{}",
            s.name
        );
        assert_eq!(
            result.american.unwrap().accurate_prices_used,
            4 * result.diagnostics.pairs_used,
            "{}",
            s.name
        );
    }
}

#[test]
fn fast_level_ignores_the_schedule_and_pays_for_it_on_a_lumpy_dividend() {
    let scenarios = load();
    let special = scenarios
        .iter()
        .find(|s| s.name == "r4_div_special_3.0_360d")
        .unwrap();
    let fast = run(special, &config(special, AmericanLevel::Fast, true));
    assert_eq!(fast.tier, Some(ForwardTier::ModelFast));
    assert!(fast.flags.contains(ForwardFlags::SCHEDULE_IGNORED));
    // A single net carry cannot represent early exercise just before a 3.00 ex-date.
    assert!(
        error_bp(&fast, special).abs() > 20.0,
        "{} bp",
        error_bp(&fast, special)
    );
}

#[test]
fn results_are_deterministic() {
    let scenarios = load();
    let s = &scenarios[9];
    let cfg = config(s, AmericanLevel::Auto, true);
    assert_eq!(run(s, &cfg), run(s, &cfg));
}

/// Config with only the discount factor and the schedule: every other American setting default.
struct AmericanConfigOnly;

impl AmericanConfigOnly {
    fn default_for(s: &Scenario) -> ImpliedForwardConfig {
        ImpliedForwardConfig {
            discount_factor: Some((-s.rate * s.expiry).exp()),
            max_spot_deviation: None,
            american: AmericanModelConfig {
                dividends: s.dividends.clone(),
                ..AmericanModelConfig::default()
            },
            ..Default::default()
        }
    }
}
