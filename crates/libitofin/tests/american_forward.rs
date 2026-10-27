//! Validation of the model-based American implied forward on synthetic quotes.
//!
//! `fixtures/american_forward/scenarios.tsv` holds 17-strike chains priced by an 800 x 1600
//! finite-difference grid (see `examples/american_forward_oracle.rs`) together with the exact
//! forward each chain was generated from. The estimator only has closed forms, a 400-step tree
//! and a 100 x 200 grid to work with, so the recovered forward is an independent check.

use libitofin::termstructures::forward::{
    AmericanLevel, AmericanModelConfig, ForwardFlags, ForwardStatus, ForwardTier, ImpliedForward,
    ImpliedForwardConfig, ImpliedForwardResult, MarketConvention, ModelFailure, OptionQuotePair,
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

/// Deterministic uniform noise in `[-1, 1)` (SplitMix64), so the noisy chains are reproducible.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Every call and put moved, bid and ask together, by up to `relative` of its own price.
fn noisy(quotes: &[OptionQuotePair], relative: f64, seed: u64) -> Vec<OptionQuotePair> {
    let mut noise = Noise(seed);
    let mut jitter = |bid: f64, ask: f64| {
        let scale = 1.0 + relative * noise.next();
        (bid * scale, ask * scale)
    };
    quotes
        .iter()
        .map(|q| {
            let (cb, ca) = jitter(q.call_bid, q.call_ask);
            let (pb, pa) = jitter(q.put_bid, q.put_ask);
            OptionQuotePair::new(q.strike, cb, ca, pb, pa)
        })
        .collect()
}

fn run_quotes(
    s: &Scenario,
    quotes: &[OptionQuotePair],
    config: &ImpliedForwardConfig,
) -> ImpliedForwardResult {
    let convention = MarketConvention::AmericanEquity {
        spot: SPOT,
        atm_vol: None,
    };
    ImpliedForward::calculate(quotes, convention, config, s.expiry).unwrap()
}

fn bp_between(a: &ImpliedForwardResult, b: &ImpliedForwardResult) -> f64 {
    (a.forward / b.forward - 1.0).abs() * 1e4
}

#[test]
fn noisy_quotes_move_the_forward_by_about_the_noise_and_nothing_else() {
    // 0.3% of every price is a few cents on a 10-dollar option: tens of basis points of noise
    // in the call/put gap, which parity over 17 strikes averages down to a few basis points.
    let mut worst: f64 = 0.0;
    for s in load().iter().filter(|s| s.dividends.is_empty()) {
        for level in [AmericanLevel::Fast, AmericanLevel::Accurate] {
            let cfg = config(s, level, false);
            let clean = run(s, &cfg);
            for seed in 1..=4 {
                let res = run_quotes(s, &noisy(&s.quotes, 0.003, seed), &cfg);
                let moved = bp_between(&res, &clean);
                worst = worst.max(moved);
                let label = format!("{} {level:?} seed {seed}", s.name);
                assert!(res.forward.is_finite(), "{label}");
                assert!(moved < 8.0, "{label}: moved {moved} bp");
                assert_eq!(res.tier, clean.tier, "{label}");
                assert_eq!(res.flags, ForwardFlags::NONE, "{label}");
                assert_eq!(res.model_failure, None, "{label}");
                // Noise wider than the spread can cross the synthetic bounds; nothing else.
                assert!(
                    matches!(
                        res.diagnostics.status,
                        ForwardStatus::Valid | ForwardStatus::WarningCrossedSyntheticQuotes
                    ),
                    "{label}: {:?}",
                    res.diagnostics.status
                );
            }
        }
    }
    assert!(worst > 0.1, "the noise had no effect ({worst} bp)");
}

/// Scenario 0 (no dividends, 180 days) with the quotes of `strikes` replaced by `bad`.
fn with_replaced(
    s: &Scenario,
    strikes: &[f64],
    bad: impl Fn(&OptionQuotePair) -> OptionQuotePair,
) -> Vec<OptionQuotePair> {
    s.quotes
        .iter()
        .map(|q| {
            if strikes.contains(&q.strike) {
                bad(q)
            } else {
                *q
            }
        })
        .collect()
}

#[test]
fn quotes_that_cannot_be_inverted_are_dropped_and_counted() {
    let scenarios = load();
    let s = &scenarios[0];
    let cfg = config(s, AmericanLevel::Fast, false);
    let clean = run(s, &cfg);
    // A call above the spot has no implied volatility; so does a put above the strike.
    let junk =
        |q: &OptionQuotePair| OptionQuotePair::new(q.strike, 150.0, 151.0, q.put_bid, q.put_ask);

    let two = run_quotes(s, &with_replaced(s, &[97.5, 102.5], junk), &cfg);
    assert_eq!(two.tier, Some(ForwardTier::ModelFast));
    assert_eq!(
        two.diagnostics.pairs_pruned,
        clean.diagnostics.pairs_pruned + 2
    );
    assert!(!two.flags.contains(ForwardFlags::PAIRS_DROPPED));
    assert!(
        bp_between(&two, &clean) < 1.0,
        "{} bp",
        bp_between(&two, &clean)
    );

    // Most of the chain unusable: still a forward, but the result says so.
    let strikes: Vec<f64> = s.quotes.iter().map(|q| q.strike).step_by(2).collect();
    let many = run_quotes(s, &with_replaced(s, &strikes, junk), &cfg);
    assert!(many.flags.contains(ForwardFlags::PAIRS_DROPPED));
    assert!(many.forward.is_finite());
    assert!(
        bp_between(&many, &clean) < 5.0,
        "{} bp",
        bp_between(&many, &clean)
    );
}

#[test]
fn crossed_quotes_never_reach_the_model() {
    let scenarios = load();
    let s = &scenarios[0];
    let cfg = config(s, AmericanLevel::Fast, false);
    let clean = run(s, &cfg);
    // A bid above its ask is not a quote; the pair is dropped before the strike window.
    let crossed = s
        .quotes
        .iter()
        .map(|q| {
            if q.strike == 100.0 || q.strike == 105.0 {
                OptionQuotePair {
                    call_bid: q.call_ask + 0.5,
                    ..*q
                }
            } else {
                *q
            }
        })
        .collect::<Vec<_>>();
    let res = run_quotes(s, &crossed, &cfg);
    assert_eq!(res.tier, Some(ForwardTier::ModelFast));
    assert_eq!(res.flags, ForwardFlags::NONE);
    assert_eq!(res.diagnostics.total_pairs_received, s.quotes.len());
    assert_eq!(
        res.diagnostics.pairs_pruned,
        clean.diagnostics.pairs_pruned + 2
    );
    assert!(bp_between(&res, &clean) < 1.0);
}

#[test]
fn an_unusable_chain_falls_back_with_the_reason() {
    let scenarios = load();
    let s = &scenarios[0];
    let cfg = config(s, AmericanLevel::Fast, false);
    let junk = |q: &OptionQuotePair| OptionQuotePair::new(q.strike, 150.0, 151.0, 150.0, 151.0);
    let strikes: Vec<f64> = s.quotes.iter().map(|q| q.strike).collect();
    let res = run_quotes(s, &with_replaced(s, &strikes, junk), &cfg);
    assert_eq!(res.tier, Some(ForwardTier::Corridor));
    assert!(res.flags.contains(ForwardFlags::NON_CONVERGED));
    // The junk quotes cross the corridor's synthetic bounds, which takes precedence in status.
    assert_ne!(res.diagnostics.status, ForwardStatus::Valid);
    assert_eq!(res.model_failure, Some(ModelFailure::NoImpliedVolatility));
}
