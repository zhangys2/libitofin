//! #1461: the Student-t cdf keeps full relative precision near `t = 0`, where
//! the forward `I_x(ν/2, 1/2)` with `x = ν/(ν+t²)` rounds to `1` and cancels,
//! and the tail it already got right stays right. Oracle: mpmath at 50 digits
//! (`fixtures/studentt_cdf_centre/generate.py`).

use libitofin::math::distributions::Cdf;
use libitofin::math::distributions::studentt::StudentT;
use libitofin::types::Real;

const CENTRE_REL_TOL: Real = 1e-14;
const TAIL_REL_TOL: Real = 1e-10;

fn cases(kind: &str) -> Vec<(Real, Real, Real)> {
    include_str!("fixtures/studentt_cdf_centre/cases.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split(',').collect::<Vec<&str>>())
        .filter(|fields| fields[0] == kind)
        .map(|fields| {
            let parse = |i: usize| fields[i].parse::<Real>().unwrap();
            (parse(1), parse(2), parse(3))
        })
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
fn cdf_near_zero_matches_mpmath() {
    let rows = cases("centre");
    assert_eq!(rows.len(), 18);
    for (nu, t, expected) in rows {
        let dist = StudentT::new(nu).unwrap();
        assert_rel(
            dist.cdf(t),
            expected,
            CENTRE_REL_TOL,
            &format!("cdf nu={nu} t={t}"),
        );
    }
}

/// One ulp either side of the centre/forward switch `y (nu + 5) = 3`, both
/// signs: a wrong threshold would put a cancelling form on one side.
#[test]
fn cdf_across_the_switch_matches_mpmath() {
    let rows = cases("switch");
    assert_eq!(rows.len(), 12);
    for (nu, t, expected) in rows {
        let dist = StudentT::new(nu).unwrap();
        assert_rel(
            dist.cdf(t),
            expected,
            TAIL_REL_TOL,
            &format!("cdf nu={nu} t={t}"),
        );
    }
}

#[test]
fn tail_guard_matches_mpmath() {
    let rows = cases("tail");
    assert_eq!(rows.len(), 3);
    for (nu, t, expected) in rows {
        let dist = StudentT::new(nu).unwrap();
        assert_rel(
            dist.cdf(t),
            expected,
            TAIL_REL_TOL,
            &format!("cdf nu={nu} t={t}"),
        );
    }
}

#[test]
fn cdf_at_infinity_takes_the_forward_form() {
    for nu in [3.0, 30.0, 1000.0] {
        let dist = StudentT::new(nu).unwrap();
        assert_eq!(dist.cdf(Real::NEG_INFINITY), 0.0);
        assert_eq!(dist.cdf(Real::INFINITY), 1.0);
        assert_eq!(dist.cdf(-1.0e200), 0.0);
        assert_eq!(dist.cdf(1.0e200), 1.0);
    }
}
