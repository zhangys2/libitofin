//! #1463: `incomplete_beta` converges for large `a`, so the Student-t
//! quantile returns `Ok` and the infallible Student-t and Binomial cdfs do not
//! panic on inputs where the 100-iteration continued fraction failed. Oracle:
//! mpmath at 60 digits (`fixtures/incomplete_beta_large_a/generate.py`).

use libitofin::math::beta::incomplete_beta;
use libitofin::math::distributions::binomial::CumulativeBinomialDistribution;
use libitofin::math::distributions::studentt::StudentT;
use libitofin::math::distributions::{Cdf, Probability, Quantile};
use libitofin::types::Real;

const REL_TOL: Real = 1e-10;

/// The `log_gamma` prefactor cancelled to `2.1e-10` here; Boost's power
/// terms (#1468) bring the rows within `7e-14`.
const BINOMIAL_REL_TOL: Real = 1e-12;

fn cases(kind: &str) -> Vec<(Real, Real, Real)> {
    include_str!("fixtures/incomplete_beta_large_a/cases.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            (fields[0] == kind).then(|| {
                let parse = |i: usize| fields[i].parse::<Real>().unwrap();
                (parse(1), parse(2), parse(3))
            })
        })
        .collect()
}

fn assert_rel(got: Real, expected: Real, tol: Real, context: &str) {
    assert!(
        got.is_finite() && ((got - expected) / expected).abs() <= tol,
        "{context}: got {got}, expected {expected}"
    );
}

#[test]
fn student_quantile_sweep_converges_and_matches_mpmath() {
    let rows = cases("quantile");
    assert_eq!(rows.len(), 406);
    for (nu, p, expected) in rows {
        let dist = StudentT::new(nu).unwrap();
        let got = dist.quantile(Probability::try_from(p).unwrap()).unwrap();
        assert_rel(got, expected, REL_TOL, &format!("quantile nu={nu} p={p}"));
    }
}

#[test]
fn student_cdf_does_not_panic_and_matches_mpmath() {
    let rows = cases("cdf");
    assert_eq!(rows.len(), 9);
    for (nu, t, expected) in rows {
        let dist = StudentT::new(nu).unwrap();
        assert_rel(
            dist.cdf(t),
            expected,
            REL_TOL,
            &format!("cdf nu={nu} t={t}"),
        );
        assert_rel(
            dist.survival(-t),
            expected,
            REL_TOL,
            &format!("survival nu={nu} t={}", -t),
        );
    }
}

#[test]
fn binomial_cdf_does_not_panic_and_matches_mpmath() {
    let rows = cases("binomial");
    assert_eq!(rows.len(), 10);
    for (n, k, expected) in rows {
        let dist = CumulativeBinomialDistribution::new(0.5, n as u64).unwrap();
        assert_rel(
            dist.cdf(k as u64),
            expected,
            BINOMIAL_REL_TOL,
            &format!("binomial n={n} k={k}"),
        );
    }
}

#[test]
fn binomial_cdf_is_total_and_monotone_across_the_failing_band() {
    let dist = CumulativeBinomialDistribution::new(0.5, 100_000).unwrap();
    let mut previous = 0.0;
    for k in 45_000..55_000 {
        let p = dist.cdf(k);
        assert!(p >= previous && p <= 1.0, "k={k}: {p} after {previous}");
        previous = p;
    }
}

#[test]
fn incomplete_beta_errs_at_the_iteration_cap() {
    let err = incomplete_beta(1e25, 1e25, 0.5).unwrap_err().to_string();
    assert!(err.contains("did not converge"), "{err}");
}

#[test]
fn incomplete_beta_errs_on_a_nan_continued_fraction() {
    let err = incomplete_beta(1e300, 1e300, 0.5).unwrap_err().to_string();
    assert!(err.contains("produced NaN"), "{err}");
}
