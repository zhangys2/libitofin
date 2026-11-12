//! Acceptance tests transliterated from the in-scope `ql-essvi` C++ tests.
//! CSV helper checks, `runtime_seconds`, and scenario millisecond budgets are omitted.

use libitofin::termstructures::volatility::essvi::{
    CalibrationPath, Quote, Slice, Surface, WarmOptions, butterfly_safe,
    butterfly_safe_with_safety, calendar_safe, calibrate, calibrate_warm, detail, mm_psi2_bound,
    slice_total_variance, validate,
};

struct Noise {
    state: u64,
}

impl Noise {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn centered(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        2.0 * (self.state >> 11) as f64 * (1.0 / 9007199254740992.0) - 1.0
    }
}

fn synthetic_chain(expiry_count: usize, quotes_per_expiry: usize) -> Vec<Quote> {
    synthetic_chain_seed(expiry_count, quotes_per_expiry, 0x8f3f73b5cf1c9ad1)
}

fn synthetic_chain_seed(expiry_count: usize, quotes_per_expiry: usize, seed: u64) -> Vec<Quote> {
    let mut noise = Noise::new(seed);
    let mut quotes = Vec::with_capacity(expiry_count * quotes_per_expiry);
    for i in 0..expiry_count {
        let expiry = 0.05 + 0.05 * i as f64;
        let truth = Slice {
            expiry,
            theta: 0.02 + 0.03 * expiry,
            psi: 0.05 + 0.02 * expiry,
            rho: -0.35,
        };
        for j in 0..quotes_per_expiry {
            let k = if quotes_per_expiry == 1 {
                0.0
            } else {
                -0.5 + j as f64 / (quotes_per_expiry - 1) as f64
            };
            let strike = 100.0 * k.exp();
            let iv = (slice_total_variance(&truth, k) / expiry).sqrt();
            let mid = iv + noise.centered() * 0.00001;
            quotes.push(Quote::new(
                format!("{i}-{j}"),
                expiry,
                100.0,
                1.0,
                strike,
                (iv - 0.00005).max(0.001),
                iv + 0.00005,
                mid,
            ));
        }
    }
    quotes
}

fn per_slice_surface() -> Surface {
    Surface {
        slices: vec![
            Slice {
                expiry: 0.25,
                theta: 0.010,
                psi: 0.04,
                rho: -0.50,
            },
            Slice {
                expiry: 0.50,
                theta: 0.020,
                psi: 0.075,
                rho: -0.10,
            },
            Slice {
                expiry: 1.00,
                theta: 0.040,
                psi: 0.14,
                rho: -0.20,
            },
        ],
    }
}

fn quotes_on(surface: &Surface, bid_off: f64, ask_off: f64) -> Vec<Quote> {
    let mut quotes = Vec::new();
    for slice in &surface.slices {
        for i in -6..=6 {
            let k = 0.05 * i as f64;
            let strike = 100.0 * k.exp();
            let iv = surface.implied_volatility(slice.expiry, k);
            quotes.push(Quote::new(
                format!("q{}", quotes.len()),
                slice.expiry,
                100.0,
                1.0,
                strike,
                iv + bid_off,
                iv + ask_off,
                iv + 0.5 * (bid_off + ask_off),
            ));
        }
    }
    quotes
}

fn fitted_inside_band(fitted: &Surface, quote: &Quote, tol: f64) -> bool {
    let iv = fitted.implied_volatility(quote.expiry, (quote.strike / quote.forward).ln());
    iv >= quote.bid_iv - tol && iv <= quote.ask_iv + tol
}

#[test]
#[allow(clippy::excessive_precision)]
fn mm_bound_matches_cpp_probe() {
    let theta = 0.0215;
    let rho = 0.35;
    let safe = mm_psi2_bound(theta, rho);
    let hint = detail::mm_psi2_bound_search_hint(theta, rho);
    let reference = detail::mm_psi2_bound_reference(theta, rho);
    assert!(
        (safe - 0.25908305065150045).abs() <= 1e-15 * safe,
        "safe {safe:.17}"
    );
    assert!(
        (hint - 0.25908997796424799).abs() <= 1e-15 * hint,
        "hint {hint:.17}"
    );
    assert!(
        (reference - 0.25908997796424799).abs() <= 1e-15 * reference,
        "ref {reference:.17}"
    );
}

#[test]
fn evaluator_matches_cpp_surface_checks() {
    let slice = Slice {
        expiry: 1.0,
        theta: 0.04,
        psi: 0.20,
        rho: -0.4,
    };
    assert!((slice_total_variance(&slice, 0.0) - slice.theta).abs() < 1e-12);
    assert!(butterfly_safe(&slice));
    let gj2 = slice.psi * slice.psi * (1.0 + slice.rho.abs());
    assert!(gj2 < 4.0 * slice.theta);
    let mm_only = Slice {
        expiry: 1.0,
        theta: 0.04,
        psi: 0.40,
        rho: -0.4,
    };
    assert!(mm_only.psi * mm_only.psi * (1.0 + mm_only.rho.abs()) >= 4.0 * mm_only.theta);
    assert!(mm_only.psi * (1.0 + mm_only.rho.abs()) < 4.0);
    let short_dated = Slice {
        expiry: 0.019178,
        theta: 0.000213385774644,
        psi: 0.02,
        rho: 0.0,
    };
    assert!(mm_psi2_bound(short_dated.theta, 0.0) > short_dated.psi * short_dated.psi);
    assert!(butterfly_safe_with_safety(&short_dated, 1e-8));

    let surface = Surface {
        slices: vec![
            Slice {
                expiry: 0.5,
                theta: 0.04,
                psi: 0.20,
                rho: -0.4,
            },
            Slice {
                expiry: 1.0,
                theta: 0.07,
                psi: 0.30,
                rho: -0.35,
            },
        ],
    };
    assert!(validate(&surface).is_empty());
    assert!((surface.slice_at(0.5).theta - 0.04).abs() < 1e-12);
    assert!(surface.slice_at(0.75).theta > 0.04);
    assert!(surface.total_variance(0.75, 0.0) > 0.04);
    let rising_phi = Slice {
        expiry: 1.0,
        theta: 0.05,
        psi: 0.30,
        rho: -0.4,
    };
    assert!(!calendar_safe(
        &Slice {
            expiry: 0.5,
            theta: 0.04,
            psi: 0.20,
            rho: -0.4
        },
        &rising_phi
    ));
}

#[test]
fn calibration_recovers_free_rho_per_slice() {
    let truth = per_slice_surface();
    assert!(validate(&truth).is_empty());
    let quotes = quotes_on(&truth, -0.0005, 0.0005);
    let result = calibrate(&quotes);
    assert!(result.converged, "{}", result.error);
    assert_eq!(result.surface.slices.len(), 3);
    assert!(validate(&result.surface).is_empty());
    for quote in &quotes {
        assert!(fitted_inside_band(&result.surface, quote, 1e-5));
    }
    let slices = &result.surface.slices;
    assert!(slices[1].rho > slices[0].rho);
    assert!(slices[1].rho > slices[2].rho);
    assert!(slices[0].rho < -0.25);
    assert!(slices[1].rho > -0.25);
}

#[test]
fn calibration_keeps_fit_inside_every_bid_ask_band() {
    let truth = per_slice_surface();
    let mut quotes = Vec::new();
    for slice in &truth.slices {
        for i in -6..=6 {
            let k = 0.05 * i as f64;
            let strike = 100.0 * k.exp();
            let iv = truth.implied_volatility(slice.expiry, k);
            let add = |bid: f64, ask: f64, quotes: &mut Vec<Quote>| {
                quotes.push(Quote::new(
                    format!("q{}", quotes.len()),
                    slice.expiry,
                    100.0,
                    1.0,
                    strike,
                    bid,
                    ask,
                    0.5 * (bid + ask),
                ));
            };
            add(iv - 0.0005, iv + 0.0005, &mut quotes);
            add(iv - 0.0005, iv + 0.05, &mut quotes);
        }
    }
    let result = calibrate(&quotes);
    assert!(result.converged, "{}", result.error);
    for quote in &quotes {
        assert!(fitted_inside_band(&result.surface, quote, 1e-5));
    }
}

#[test]
fn slice_at_nan_expiry_does_not_panic() {
    let surface = Surface {
        slices: vec![
            Slice {
                expiry: 0.5,
                theta: 0.04,
                psi: 0.20,
                rho: -0.4,
            },
            Slice {
                expiry: 1.0,
                theta: 0.07,
                psi: 0.30,
                rho: -0.35,
            },
        ],
    };
    let got = surface.slice_at(f64::NAN);
    assert!(got.expiry.is_nan() && got.theta.is_nan() && got.psi.is_nan() && got.rho.is_nan());
    assert!(surface.total_variance(f64::NAN, 0.0).is_nan());
}

#[test]
fn crossed_quote_does_not_abort_calibration() {
    let quotes = [
        Quote::new("atm", 0.5, 100.0, 1.0, 100.0, 0.19, 0.21, 0.20),
        Quote::new("crossed", 0.5, 100.0, 1.0, 110.0, 0.30, 0.20, 0.25),
    ];
    let result = calibrate(&quotes);
    assert!(!result.error.is_empty() || result.converged);
}

#[test]
fn predecessor_theta_above_five_does_not_abort() {
    let mut first = Vec::new();
    let mut second = Vec::new();
    for i in 0..11 {
        let k = -0.5 + i as f64 * 0.1;
        let strike = 100.0 * k.exp();
        first.push(Quote::new(
            format!("a{i}"),
            1.0,
            100.0,
            1.0,
            strike,
            2.25,
            2.35,
            2.30,
        ));
        second.push(Quote::new(
            format!("b{i}"),
            1.5,
            100.0,
            1.0,
            strike,
            0.19,
            0.21,
            0.20,
        ));
    }
    first.append(&mut second);
    let result = calibrate(&first);
    let _ = result.converged;
}

#[test]
fn calibration_reports_error_when_no_included_quotes() {
    let mut quote = Quote::new("q0", 0.5, 100.0, 1.0, 100.0, 0.2, 0.21, 0.205);
    quote.included = false;
    let result = calibrate(&[quote]);
    assert!(!result.converged);
    assert!(!result.error.is_empty());
}

#[test]
fn cold_baseline_is_deterministic_and_matches_table() {
    let quotes = synthetic_chain(40, 100);
    let result = calibrate(&quotes);
    let repeated = calibrate(&quotes);
    assert!(result.converged && repeated.converged, "{}", result.error);
    assert_eq!(result.surface.slices.len(), 40);
    assert_eq!(repeated.objective, result.objective);
    assert_eq!(repeated.iterations, result.iterations);
    assert_eq!(repeated.objective_evaluations, result.objective_evaluations);
    assert_eq!(repeated.mm_bound_calls, result.mm_bound_calls);
    for (first, second) in result.surface.slices.iter().zip(&repeated.surface.slices) {
        assert_eq!(first.expiry, second.expiry);
        assert_eq!(first.theta, second.theta);
        assert_eq!(first.psi, second.psi);
        assert_eq!(first.rho, second.rho);
    }

    let baseline = include_str!("fixtures/essvi/baseline_essvi_40x100.csv");
    let mut rows = baseline.lines();
    rows.next();
    for slice in &result.surface.slices {
        let row = rows.next().expect("baseline row");
        let columns: Vec<f64> = row
            .split(',')
            .take(4)
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(slice.expiry, columns[0]);
        let theta_gap = (slice.theta - columns[1]).abs();
        let psi_gap = (slice.psi - columns[2]).abs();
        let rho_gap = (slice.rho - columns[3]).abs();
        assert!(
            theta_gap <= 1e-9 * columns[1].abs()
                && psi_gap <= 1e-9 * columns[2].abs()
                && rho_gap <= 1e-9 * columns[3].abs(),
            "expiry {} theta {} vs {} (gap {}) psi {} vs {} rho {} vs {}",
            slice.expiry,
            slice.theta,
            columns[1],
            theta_gap,
            slice.psi,
            columns[2],
            slice.rho,
            columns[3]
        );
    }
    assert!(rows.next().is_none());
}

#[test]
fn invalid_included_weight_fails_closed_and_warm() {
    let valid = synthetic_chain(1, 40);
    let previous = calibrate(&valid);
    assert!(previous.converged, "{}", previous.error);
    for invalid_weight in [0.0, -1.0, f64::NAN] {
        let mut invalid = valid.clone();
        invalid[0].weight = invalid_weight;
        let cold = calibrate(&invalid);
        assert!(!cold.converged);
        assert_eq!(
            cold.error,
            "included quote weight must be finite and positive"
        );
        let warm = calibrate_warm(&invalid, Some(&previous), &valid, WarmOptions::default());
        assert!(!warm.converged);
        assert_eq!(
            warm.error,
            "included quote weight must be finite and positive"
        );
    }
}

#[test]
fn adaptive_quote_cap_boundary() {
    assert!(detail::passes_adaptive_weighted_quote_cap(
        0.055, 0.05, 0.5, 0.01
    ));
    assert!(!detail::passes_adaptive_weighted_quote_cap(
        0.055001, 0.05, 0.5, 0.01
    ));
    assert!(detail::passes_adaptive_weighted_quote_cap(
        0.22, 0.2, 2.0, 0.01
    ));
    assert!(!detail::passes_adaptive_weighted_quote_cap(
        0.220001, 0.2, 2.0, 0.01
    ));
}

#[test]
fn weight_scales_cold_objective_by_four() {
    let mut quotes = synthetic_chain(3, 100);
    let legacy = calibrate(&quotes);
    for quote in &mut quotes {
        quote.weight = 2.0;
    }
    let weighted = calibrate(&quotes);
    assert!(legacy.converged && weighted.converged);
    assert!(
        (weighted.objective - 4.0 * legacy.objective).abs()
            <= 1e-12 * 1.0_f64.max(weighted.objective.abs())
    );
}

#[test]
fn null_previous_warm_matches_cold() {
    let quotes = synthetic_chain(3, 100);
    let cold = calibrate(&quotes);
    let warm = calibrate_warm(&quotes, None, &quotes, WarmOptions::default());
    assert!(warm.converged, "{}", warm.error);
    assert_eq!(warm.path, CalibrationPath::Cold);
    assert_eq!(warm.slices_refit, cold.surface.slices.len());
    for age in &warm.reuse_ages {
        assert_eq!(*age, 0);
    }
    for (left, right) in warm.surface.slices.iter().zip(&cold.surface.slices) {
        assert_eq!(left.theta, right.theta);
        assert_eq!(left.psi, right.psi);
        assert_eq!(left.rho, right.rho);
    }
}

fn fine_bound(theta: f64, abs_rho: f64) -> f64 {
    let ratio = |l: f64| {
        let s = (1.0 - abs_rho * abs_rho).sqrt();
        let root = (l * l + 1.0).sqrt();
        let n = s + abs_rho * l + root;
        let np = abs_rho + l / root;
        let npp = 1.0 / ((l * l + 1.0) * root);
        let g = np / 4.0;
        let h = 1.0 - (l - abs_rho / s) * np / (2.0 * n);
        let g2 = npp - np * np / (2.0 * n);
        let den = theta * s * g * g - g2;
        if den > 0.0 {
            4.0 * theta * s * h * h / den
        } else {
            f64::INFINITY
        }
    };
    let l2 = 1.0 / ((-abs_rho).acos() / 3.0).tan();
    let lo = l2 + 1e-8;
    let hi = 50.0_f64.max(lo + 1.0);
    const POINTS: i32 = 4096;
    let mut best = f64::INFINITY;
    let mut best_index = 0;
    for i in 0..=POINTS {
        let u = i as f64 / POINTS as f64;
        let l = lo * (hi / lo).powf(u);
        let value = ratio(l);
        if value < best {
            best = value;
            best_index = i;
        }
    }
    let left = lo * (hi / lo).powf((best_index - 1).max(0) as f64 / POINTS as f64);
    let right = lo * (hi / lo).powf((best_index + 1).min(POINTS) as f64 / POINTS as f64);
    const GOLDEN: f64 = 0.381966011250105;
    let mut a = left.ln();
    let mut b = right.ln();
    let mut x1 = a + GOLDEN * (b - a);
    let mut x2 = b - GOLDEN * (b - a);
    let mut f1 = ratio(x1.exp());
    let mut f2 = ratio(x2.exp());
    for _ in 0..48 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + GOLDEN * (b - a);
            f1 = ratio(x1.exp());
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - GOLDEN * (b - a);
            f2 = ratio(x2.exp());
        }
    }
    best.min(f1.min(f2))
}

fn check_conservative(theta: f64, rho: f64, truth: f64) {
    let fast = mm_psi2_bound(theta, rho);
    let tol = 1e-12 * truth.abs();
    assert!(fast.is_finite() && fast > 0.0);
    assert!(fast <= truth + tol, "fast {fast} truth {truth}");
    assert!(fast >= truth - 1e-9 * truth.abs());
}

#[test]
fn mm_bound_is_conservative_against_its_reference() {
    check_conservative(
        1.4902654986887248e-5,
        0.2993688263167309,
        fine_bound(1.4902654986887248e-5, 0.2993688263167309),
    );
    let mut random = Noise::new(0x8f3f73b5cf1c9ad1);
    let unit = |noise: &mut Noise| (noise.centered() + 1.0) * 0.5;
    for _ in 0..1_000_000 {
        let log_hi = 5_f64.log10();
        let sample_theta = 10_f64.powf(-6.0 + (log_hi + 6.0) * unit(&mut random));
        let sample_rho = 0.99 * unit(&mut random);
        let fast = mm_psi2_bound(sample_theta, sample_rho);
        let reference = detail::mm_psi2_bound_reference(sample_theta, sample_rho);
        let hint = detail::mm_psi2_bound_search_hint(sample_theta, sample_rho);
        let tolerance = 1e-12 * reference.abs();
        assert!(fast.is_finite() && fast > 0.0);
        assert!(fast <= reference + tolerance);
        assert!((hint - reference).abs() <= 1e-14 * reference.abs());
    }
    for _ in 0..4096 {
        let log_hi = 5_f64.log10();
        let sample_theta = 10_f64.powf(-6.0 + (log_hi + 6.0) * unit(&mut random));
        let sample_rho = 0.99 * unit(&mut random);
        check_conservative(
            sample_theta,
            sample_rho,
            fine_bound(sample_theta, sample_rho),
        );
    }
}

#[allow(clippy::collapsible_if)]
fn load_quotes(csv: &str) -> Vec<Quote> {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let column = |name: &str| header.iter().position(|item| *item == name).unwrap();
    let id = column("quote_id");
    let expiry = column("time_to_expiry");
    let forward = column("forward");
    let discount = column("discount_factor");
    let strike = column("strike");
    let bid = column("bid_iv");
    let ask = column("ask_iv");
    let mid = column("mid_iv");
    let included = column("included");
    let weight = header.iter().position(|item| *item == "weight");
    lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let mut quote = Quote::new(
                fields[id],
                fields[expiry].parse().unwrap(),
                fields[forward].parse().unwrap(),
                fields[discount].parse().unwrap(),
                fields[strike].parse().unwrap(),
                fields[bid].parse().unwrap(),
                fields[ask].parse().unwrap(),
                fields[mid].parse().unwrap(),
            );
            quote.included = fields[included] == "true";
            if let Some(weight) = weight {
                if !fields[weight].is_empty() {
                    quote.weight = fields[weight].parse().unwrap_or(f64::NAN);
                }
            }
            quote
        })
        .collect()
}

fn assert_dense_grid(surface: &Surface) {
    assert!(validate(surface).is_empty());
    let mut previous_variance = [0.0; 101];
    let mut has_previous = false;
    for slice in &surface.slices {
        assert!(butterfly_safe(slice));
        let mut variance = [0.0; 101];
        let mut call = [0.0; 101];
        let mut strike = [0.0; 101];
        for j in 0..=100 {
            let k = -0.5 + 0.01 * j as f64;
            strike[j] = 100.0 * k.exp();
            variance[j] = slice_total_variance(slice, k);
            assert!(variance[j] > 0.0);
            if has_previous {
                assert!(variance[j] >= previous_variance[j] - 1e-8);
            }
            call[j] = normalized_call(100.0, strike[j], variance[j]);
        }
        for j in 1..100 {
            let left_slope = (call[j] - call[j - 1]) / (strike[j] - strike[j - 1]);
            let right_slope = (call[j + 1] - call[j]) / (strike[j + 1] - strike[j]);
            assert!(right_slope >= left_slope - 1e-8);
        }
        previous_variance = variance;
        has_previous = true;
    }
}

fn normalized_call(forward: f64, strike: f64, variance: f64) -> f64 {
    if variance <= 0.0 {
        return (forward - strike).max(0.0);
    }
    let root = variance.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * variance) / root;
    let d2 = d1 - root;
    let cdf = |x: f64| 0.5 * (1.0 + libm_erf(x / 2.0_f64.sqrt()));
    forward * cdf(d1) - strike * cdf(d2)
}

#[link(name = "m")]
unsafe extern "C" {
    fn erf(x: f64) -> f64;
}

fn libm_erf(x: f64) -> f64 {
    unsafe { erf(x) }
}

#[test]
fn deribit_fixture_passes_dense_grid() {
    let quotes = load_quotes(include_str!(
        "fixtures/essvi/deribit_btc_20261018T1200_quotes.csv"
    ));
    let result = calibrate(&quotes);
    assert!(result.converged, "{}", result.error);
    assert_dense_grid(&result.surface);
}

#[test]
fn unchanged_quotes_are_reused_until_age_twelve() {
    let previous_quotes = synthetic_chain(40, 100);
    let mut previous = calibrate(&previous_quotes);
    assert!(previous.converged, "{}", previous.error);
    let options = WarmOptions {
        quote_tol: 1e-6,
        ..WarmOptions::default()
    };
    for _ in 1..=13 {
        let mut current = previous_quotes.clone();
        for quote in &mut current {
            quote.bid_iv += 1e-7;
            quote.ask_iv += 1e-7;
        }
        let reuse_allowed = previous.reuse_ages.is_empty() || previous.reuse_ages[0] < 12;
        let warm = calibrate_warm(&current, Some(&previous), &previous_quotes, options);
        assert!(warm.converged, "{}", warm.error);
        assert_eq!(warm.path, CalibrationPath::Warm);
        let expiries = warm.surface.slices.len();
        assert_eq!(
            warm.slices_reused,
            if reuse_allowed { expiries } else { 0 },
            "reused {} refit {} ages {:?} fallback {:?} cap {}",
            warm.slices_reused,
            warm.slices_refit,
            warm.reuse_ages,
            warm.fallback_reason_counts,
            options.quote_cap
        );
        assert_eq!(warm.slices_refit, if reuse_allowed { 0 } else { expiries });
        for age in &warm.reuse_ages {
            assert!(*age <= 12);
        }
        previous = warm;
    }
}
