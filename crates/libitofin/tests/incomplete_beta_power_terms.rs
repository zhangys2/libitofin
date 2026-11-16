//! #1468: the incomplete beta prefactor keeps its precision for very large
//! `a`, `b`, where the `log_gamma` form cancelled to garbage. Oracle: mpmath
//! at 60 digits (`fixtures/ibeta_power_terms/generate.py`).

use libitofin::math::distributions::binomial::CumulativeBinomialDistribution;
use libitofin::math::distributions::studentt::StudentT;
use libitofin::math::distributions::{Cdf, Probability, Quantile};
use libitofin::types::Real;

/// The forward form at `ν = 1e6` and the normal cdf's `erf` at `t = -5` set
/// this; the centre rows are within `1e-14`.
const STUDENT_REL_TOL: Real = 1e-10;

/// The quantile is one Halley step from the normal estimate at these `ν`.
const QUANTILE_REL_TOL: Real = 1e-12;

/// Per-`n` tolerance of the Binomial rows: the continued fraction cancels near
/// the saddle at these sizes (a documented deferral in `math/beta.rs`).
fn binomial_tol(n: u64) -> Real {
    match n {
        1_000_000_000_000 => 5e-10,
        100_000_000_000_000 => 1e-8,
        _ => 5e-7,
    }
}

fn rows(kind: &str) -> Vec<Vec<Real>> {
    include_str!("fixtures/ibeta_power_terms/cases.csv")
        .lines()
        .filter_map(|line| line.strip_prefix(kind))
        .map(|line| line.split(',').map(|f| f.parse().unwrap()).collect())
        .collect()
}

fn assert_rel(got: Real, expected: Real, tol: Real, context: &str) {
    let err = ((got - expected) / expected).abs();
    assert!(
        got.is_finite() && err <= tol,
        "{context}: got {got}, expected {expected}, rel err {err:e}"
    );
}

#[test]
fn binomial_cdf_at_huge_n_matches_mpmath() {
    let rows = rows("binomial,");
    assert_eq!(rows.len(), 20);
    for row in rows {
        let (n, p, k, expected) = (row[0] as u64, row[1], row[2] as u64, row[3]);
        let got = CumulativeBinomialDistribution::new(p, n).unwrap().cdf(k);
        let context = format!("binomial n={n} p={p} k={k}");
        assert_rel(got, expected, binomial_tol(n), &context);
    }
}

/// The forward form (`|t| > sqrt(3)` here) passes the exact
/// `y = t²/(ν + t²)`, where `1 - z` rounds away the tail, and beyond
/// `ν = 2^52` the cdf is the normal one.
#[test]
fn student_cdf_at_huge_nu_matches_mpmath() {
    let rows = rows("student,");
    assert_eq!(rows.len(), 32);
    for row in rows {
        let (nu, t, expected) = (row[0], row[1], row[2]);
        let dist = StudentT::new(nu).unwrap();
        let context = format!("cdf nu={nu} t={t}");
        assert_rel(dist.cdf(t), expected, STUDENT_REL_TOL, &context);
    }
}

#[test]
fn student_quantile_at_huge_nu_matches_mpmath() {
    let rows = rows("student_quantile,");
    assert_eq!(rows.len(), 2);
    for row in rows {
        let (nu, p, expected) = (row[0], row[1], row[2]);
        let dist = StudentT::new(nu).unwrap();
        let got = dist.quantile(Probability::try_from(p).unwrap()).unwrap();
        assert_rel(
            got,
            expected,
            QUANTILE_REL_TOL,
            &format!("quantile nu={nu} p={p}"),
        );
    }
}

#[test]
fn student_cdf_normal_limit_keeps_the_endpoints() {
    let dist = StudentT::new(1e16).unwrap();
    assert_eq!(dist.cdf(Real::NEG_INFINITY), 0.0);
    assert_eq!(dist.cdf(Real::INFINITY), 1.0);
    assert_eq!(dist.cdf(0.0), 0.5);
    assert!(dist.cdf(Real::NAN).is_nan());
}
