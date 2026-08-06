//! Stochastic processes for specific models.
//!
//! Port of `ql/processes/`: concrete implementations of the
//! [`StochasticProcess1D`](crate::stochasticprocess::StochasticProcess1D)
//! contract. The generalized Black-Scholes process (with its Merton
//! convenience) is the first resident; the sibling convenience names
//! (`BlackScholesProcess`, `BlackProcess`, `GarmanKohlagenProcess`) are thin
//! aliases to it for now, and the pluggable discretization objects follow as
//! noted on [`GeneralizedBlackScholesProcess`].

mod batesprocess;
mod blackscholesprocess;
mod g2process;
mod hestonprocess;
mod ornsteinuhlenbeckprocess;
mod stochasticprocessarray;

pub use batesprocess::BatesProcess;
pub use blackscholesprocess::{
    BlackProcess, BlackScholesMertonProcess, BlackScholesProcess, GarmanKohlagenProcess,
    GeneralizedBlackScholesProcess,
};
pub use g2process::G2Process;
pub use hestonprocess::HestonProcess;
pub use ornsteinuhlenbeckprocess::OrnsteinUhlenbeckProcess;
pub use stochasticprocessarray::StochasticProcessArray;
