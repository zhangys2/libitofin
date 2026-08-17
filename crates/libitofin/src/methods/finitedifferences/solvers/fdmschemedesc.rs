//! The scheme choice a backward solver switches on.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp:35-59`.
//! `FdmBackwardSolver`, the rest of that header (`:61`), lands with #658.

use crate::types::Real;

/// The scheme families a descriptor can name
/// (`fdmbackwardsolver.hpp:36-40`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdmSchemeType {
    /// Hundsdorfer-Verwer splitting.
    Hundsdorfer,
    /// Douglas splitting, Crank-Nicolson in one dimension.
    Douglas,
    /// Craig-Sneyd splitting.
    CraigSneyd,
    /// Modified Craig-Sneyd splitting.
    ModifiedCraigSneyd,
    /// Fully implicit Euler.
    ImplicitEuler,
    /// Fully explicit Euler.
    ExplicitEuler,
    /// Method of lines.
    MethodOfLines,
    /// Trapezoidal rule with a second-order backward difference.
    TrBDF2,
    /// Crank-Nicolson.
    CrankNicolson,
}

/// A scheme type and the two parameters it is built from.
///
/// C++ holds all three fields `const` (`fdmbackwardsolver.hpp:44`); they are
/// plain public fields here.
///
/// The header declares ten factories over these nine types
/// (`fdmbackwardsolver.cpp:46-78`) - `ModifiedHundsdorfer` (`cpp:62`) is a
/// second `HundsdorferType` with a different `theta`. Ported factories:
/// [`douglas`](Self::douglas), [`implicit_euler`](Self::implicit_euler),
/// [`explicit_euler`](Self::explicit_euler),
/// [`crank_nicolson`](Self::crank_nicolson),
/// [`hundsdorfer`](Self::hundsdorfer),
/// [`modified_hundsdorfer`](Self::modified_hundsdorfer),
/// [`craig_sneyd`](Self::craig_sneyd),
/// [`modified_craig_sneyd`](Self::modified_craig_sneyd),
/// [`method_of_lines`](Self::method_of_lines). The remaining family
/// (`TrBDF2`) stays constructible through [`new`](Self::new) and is
/// rejected by the backward solver until its scheme lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FdmSchemeDesc {
    /// The scheme family to step with.
    pub scheme_type: FdmSchemeType,
    /// The implicitness weight of the splitting.
    pub theta: Real,
    /// The second parameter, whose meaning depends on the family.
    pub mu: Real,
}

impl FdmSchemeDesc {
    /// A descriptor with the given type and parameters
    /// (`fdmbackwardsolver.cpp:43-44`).
    pub fn new(scheme_type: FdmSchemeType, theta: Real, mu: Real) -> Self {
        FdmSchemeDesc {
            scheme_type,
            theta,
            mu,
        }
    }

    /// Douglas splitting, the same as Crank-Nicolson in one dimension
    /// (`fdmbackwardsolver.cpp:46`).
    pub fn douglas() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::Douglas, 0.5, 0.0)
    }

    /// Fully implicit Euler (`fdmbackwardsolver.cpp:70-72`).
    pub fn implicit_euler() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::ImplicitEuler, 0.0, 0.0)
    }

    /// Fully explicit Euler.
    pub fn explicit_euler() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::ExplicitEuler, 0.0, 0.0)
    }

    /// Crank-Nicolson, represented by Douglas with theta = 0.5 in 1-D.
    pub fn crank_nicolson() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::CrankNicolson, 0.5, 0.0)
    }

    /// Hundsdorfer–Verwer (`fdmbackwardsolver.cpp:59-61`):
    /// `θ = ½ + √3/6`, `μ = ½`.
    pub fn hundsdorfer() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::Hundsdorfer, 0.5 + 3.0_f64.sqrt() / 6.0, 0.5)
    }

    /// Modified Hundsdorfer (`fdmbackwardsolver.cpp:63-65`): same type with
    /// `θ = 1 − √2/2`, `μ = ½`.
    pub fn modified_hundsdorfer() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::Hundsdorfer, 1.0 - 2.0_f64.sqrt() / 2.0, 0.5)
    }

    /// Craig–Sneyd (`fdmbackwardsolver.cpp:52`): `θ = ½`, `μ = ½`.
    pub fn craig_sneyd() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::CraigSneyd, 0.5, 0.5)
    }

    /// Modified Craig–Sneyd (`fdmbackwardsolver.cpp:54-56`): `θ = μ = ⅓`.
    pub fn modified_craig_sneyd() -> Self {
        FdmSchemeDesc::new(FdmSchemeType::ModifiedCraigSneyd, 1.0 / 3.0, 1.0 / 3.0)
    }

    /// Method of lines (`fdmbackwardsolver.cpp:67-68`): `eps = 0.001`,
    /// `relInitStepSize = 0.01`, stored as `theta` and `mu`.
    pub fn method_of_lines() -> Self {
        Self::method_of_lines_with(0.001, 0.01)
    }

    /// Method of lines with prescribed RK tolerance and relative initial
    /// step (`fdmbackwardsolver.cpp:67`).
    pub fn method_of_lines_with(eps: Real, rel_init_step_size: Real) -> Self {
        FdmSchemeDesc::new(FdmSchemeType::MethodOfLines, eps, rel_init_step_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn douglas_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::douglas();

        assert_eq!(desc.scheme_type, FdmSchemeType::Douglas);
        assert_eq!(desc.theta, 0.5);
        assert_eq!(desc.mu, 0.0);
    }

    #[test]
    fn implicit_euler_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::implicit_euler();

        assert_eq!(desc.scheme_type, FdmSchemeType::ImplicitEuler);
        assert_eq!(desc.theta, 0.0);
        assert_eq!(desc.mu, 0.0);
    }

    /// The remaining unported factory (`TrBDF2`) is omitted, not its type:
    /// the rollback must be able to name it to reject it.
    #[test]
    fn every_scheme_type_is_constructible() {
        let types = [
            FdmSchemeType::Hundsdorfer,
            FdmSchemeType::Douglas,
            FdmSchemeType::CraigSneyd,
            FdmSchemeType::ModifiedCraigSneyd,
            FdmSchemeType::ImplicitEuler,
            FdmSchemeType::ExplicitEuler,
            FdmSchemeType::MethodOfLines,
            FdmSchemeType::TrBDF2,
            FdmSchemeType::CrankNicolson,
        ];

        for scheme_type in types {
            let desc = FdmSchemeDesc::new(scheme_type, 0.25, 0.75);
            assert_eq!(desc.scheme_type, scheme_type);
            assert_eq!(desc.theta, 0.25);
            assert_eq!(desc.mu, 0.75);
        }
    }

    #[test]
    fn the_two_ported_factories_differ() {
        assert_ne!(FdmSchemeDesc::douglas(), FdmSchemeDesc::implicit_euler());
    }

    #[test]
    fn hundsdorfer_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::hundsdorfer();
        assert_eq!(desc.scheme_type, FdmSchemeType::Hundsdorfer);
        assert!((desc.theta - (0.5 + 3.0_f64.sqrt() / 6.0)).abs() < 1e-15);
        assert_eq!(desc.mu, 0.5);
    }

    #[test]
    fn modified_hundsdorfer_shares_the_type_with_a_different_theta() {
        let desc = FdmSchemeDesc::modified_hundsdorfer();
        assert_eq!(desc.scheme_type, FdmSchemeType::Hundsdorfer);
        assert!((desc.theta - (1.0 - 2.0_f64.sqrt() / 2.0)).abs() < 1e-15);
        assert_eq!(desc.mu, 0.5);
        assert_ne!(desc.theta, FdmSchemeDesc::hundsdorfer().theta);
    }

    #[test]
    fn craig_sneyd_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::craig_sneyd();
        assert_eq!(desc.scheme_type, FdmSchemeType::CraigSneyd);
        assert_eq!(desc.theta, 0.5);
        assert_eq!(desc.mu, 0.5);
    }

    #[test]
    fn modified_craig_sneyd_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::modified_craig_sneyd();
        assert_eq!(desc.scheme_type, FdmSchemeType::ModifiedCraigSneyd);
        assert!((desc.theta - 1.0 / 3.0).abs() < 1e-15);
        assert!((desc.mu - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn method_of_lines_carries_the_cpp_parameters() {
        let desc = FdmSchemeDesc::method_of_lines();
        assert_eq!(desc.scheme_type, FdmSchemeType::MethodOfLines);
        assert_eq!(desc.theta, 0.001);
        assert_eq!(desc.mu, 0.01);

        let custom = FdmSchemeDesc::method_of_lines_with(1e-4, 0.05);
        assert_eq!(custom.scheme_type, FdmSchemeType::MethodOfLines);
        assert_eq!(custom.theta, 1e-4);
        assert_eq!(custom.mu, 0.05);
    }
}
