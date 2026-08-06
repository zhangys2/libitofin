//! Two-factor short-rate model scaffolding.
//!
//! Port of the dynamics surface from `ql/models/shortrate/twofactormodel.{hpp,cpp}`:
//! [`TwoFactorShortRateDynamics`] describes `r_t = f(t, x_t, y_t)` with two
//! correlated one-dimensional state processes. The two-factor lattice
//! (`TwoFactorModel::ShortRateTree` / `tree()`) is deferred.

use crate::errors::QlResult;
use crate::math::matrix::Matrix;
use crate::processes::StochasticProcessArray;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::{StochasticProcess, StochasticProcess1D};
use crate::types::{Rate, Real, Time};

/// Base description of a two-factor short-rate's dynamics
/// (`TwoFactorModel::ShortRateDynamics`, `twofactormodel.hpp:74`).
///
/// The short rate is a function of two Markov state variables `(x, y)` that
/// follow correlated one-dimensional risk-neutral processes. Trees and FD
/// engines read [`short_rate`](Self::short_rate) and discretize
/// [`process`](Self::process) (or the factor processes separately).
pub trait TwoFactorShortRateDynamics {
    /// `shortRate(Time t, Real x, Real y)` (`twofactormodel.hpp:87`).
    fn short_rate(&self, t: Time, x: Real, y: Real) -> Rate;

    /// Risk-neutral dynamics of the first state variable `x`.
    fn x_process(&self) -> Shared<dyn StochasticProcess1D>;

    /// Risk-neutral dynamics of the second state variable `y`.
    fn y_process(&self) -> Shared<dyn StochasticProcess1D>;

    /// Correlation `ρ` between the two Brownian motions.
    fn correlation(&self) -> Real;

    /// Joint process of the two variables (`twofactormodel.cpp:49-58`): a
    /// [`StochasticProcessArray`] of the factor processes under the
    /// instantaneous correlation matrix.
    ///
    /// # Errors
    ///
    /// Fails if the array cannot be assembled (should not happen for the
    /// fixed 2×2 correlation layout).
    fn process(&self) -> QlResult<Shared<dyn StochasticProcess>> {
        let correlation = Matrix::from([[1.0, self.correlation()], [self.correlation(), 1.0]]);
        let processes = vec![self.x_process(), self.y_process()];
        Ok(
            shared(StochasticProcessArray::new(processes, &correlation)?)
                as Shared<dyn StochasticProcess>,
        )
    }
}
