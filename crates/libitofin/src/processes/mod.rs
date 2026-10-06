//! Stochastic processes for specific models.
//!
//! Port of `ql/processes/`: concrete implementations of the
//! [`StochasticProcess1D`](crate::stochasticprocess::StochasticProcess1D)
//! contract. The generalized Black-Scholes process (with its Merton
//! convenience) is the first resident; the sibling conveniences
//! (`BlackScholesProcess`, `BlackProcess`, `GarmanKohlagenProcess`) are thin
//! aliases to it for now, and the pluggable discretization objects follow as
//! noted on [`GeneralizedBlackScholesProcess`].

mod batesprocess;
mod blackscholesprocess;
pub mod discretization;
pub mod forwardmeasureprocess;
mod g2process;
mod geometricbrownianprocess;
mod gjrgarchprocess;
mod gsrprocess;
mod hestonprocess;
mod hestonslvprocess;
mod hullwhiteprocess;
mod merton76process;
mod mfstateprocess;
mod ornsteinuhlenbeckprocess;
mod stochasticprocessarray;

pub use batesprocess::BatesProcess;
pub use blackscholesprocess::{
    BlackProcess, BlackScholesMertonProcess, BlackScholesProcess, GarmanKohlagenProcess,
    GeneralizedBlackScholesProcess,
};
pub use discretization::{
    DiscretizedProcess, DiscretizedProcess1D, EulerDiscretization, ProcessDiscretization,
    ProcessDiscretization1D,
};
pub use forwardmeasureprocess::{ForwardMeasureProcess1D, ForwardMeasureTime};
pub use g2process::G2Process;
pub use geometricbrownianprocess::GeometricBrownianMotionProcess;
pub use gjrgarchprocess::{
    GjrGarchCoefficients, GjrGarchDiscretization, GjrGarchParameters, GjrGarchProcess,
};
pub use gsrprocess::GsrProcess;
pub use hestonprocess::{Discretization as HestonDiscretization, HestonProcess};
pub use hestonslvprocess::HestonSLVProcess;
pub use hullwhiteprocess::HullWhiteForwardProcess;
pub use merton76process::Merton76Process;
pub use mfstateprocess::MfStateProcess;
pub use ornsteinuhlenbeckprocess::OrnsteinUhlenbeckProcess;
pub use stochasticprocessarray::StochasticProcessArray;
