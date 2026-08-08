//! Two-factor short-rate model scaffolding.
//!
//! Port of `ql/models/shortrate/twofactormodel.{hpp,cpp}`:
//! [`TwoFactorShortRateDynamics`] describes `r_t = f(t, x_t, y_t)` with two
//! correlated one-dimensional state processes;
//! [`TwoFactorShortRateTree`] is the recombining product-tree discount surface
//! (`TwoFactorModel::ShortRateTree`). Concrete models (e.g. [`G2`](crate::models::shortrate::G2))
//! supply `dynamics()` and assemble the lattice via `tree()`.

use crate::errors::QlResult;
use crate::math::matrix::Matrix;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::treelattice::TreeLatticeImpl;
use crate::methods::lattices::treelattice2d::TwoFactorTree;
use crate::methods::lattices::trinomialtree::TrinomialTree;
use crate::processes::StochasticProcessArray;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::{StochasticProcess, StochasticProcess1D};
use crate::types::{Rate, Real, Size, Time};

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

/// Recombining two-dimensional tree discretizing a two-factor short-rate
/// (`TwoFactorModel::ShortRateTree`, `twofactormodel.hpp:128`).
///
/// C++'s `ShortRateTree` *is-a* `TreeLattice2D<ShortRateTree, TrinomialTree>`
/// (CRTP) and supplies the per-node discount off the model's dynamics. Here it
/// is the [`TreeLatticeImpl`] callback surface a
/// [`TreeLattice2D`](crate::methods::lattices::TreeLattice2D) induces over: it
/// embeds a [`TwoFactorTree`] (product navigation + Hull–White correlation
/// weights) and the [`TwoFactorShortRateDynamics`], and computes
/// [`discount`](TreeLatticeImpl::discount)`(i, index) =
/// exp(-short_rate(t_i, x, y) * dt_i)` after splitting the joint index
/// (`twofactormodel.hpp:134-143`).
///
/// It carries its own [`TimeGrid`] clone because the discount callback needs
/// `t_i`/`dt_i`, matching the one-factor [`ShortRateTree`](crate::models::shortrate::ShortRateTree)
/// pattern. Named `TwoFactorShortRateTree` so it does not collide with the
/// one-factor type.
pub struct TwoFactorShortRateTree {
    tree: TwoFactorTree,
    dynamics: Shared<dyn TwoFactorShortRateDynamics>,
    time_grid: TimeGrid,
}

impl TwoFactorShortRateTree {
    /// Plain build-up from factor trinomials and short-rate `dynamics` over
    /// `time_grid` (`twofactormodel.cpp:44-51`).
    ///
    /// # Errors
    ///
    /// Propagates [`TwoFactorTree::new`] (e.g. mismatched factor column counts).
    pub fn new(
        tree1: Shared<TrinomialTree>,
        tree2: Shared<TrinomialTree>,
        dynamics: Shared<dyn TwoFactorShortRateDynamics>,
        time_grid: TimeGrid,
    ) -> QlResult<Self> {
        let correlation = dynamics.correlation();
        let tree = TwoFactorTree::new(tree1, tree2, correlation)?;
        Ok(TwoFactorShortRateTree {
            tree,
            dynamics,
            time_grid,
        })
    }

    /// The product tree supplying joint `size`/`descendant`/`probability`.
    pub fn two_factor_tree(&self) -> &TwoFactorTree {
        &self.tree
    }

    /// Short-rate dynamics driving the per-node discount.
    pub fn dynamics(&self) -> &Shared<dyn TwoFactorShortRateDynamics> {
        &self.dynamics
    }
}

impl TreeLatticeImpl for TwoFactorShortRateTree {
    type Tree = TwoFactorTree;

    fn tree(&self) -> &TwoFactorTree {
        &self.tree
    }

    fn discount(&self, i: Size, index: Size) -> Real {
        let (x, y) = self.tree.state(i, index);
        let r = self.dynamics.short_rate(self.time_grid[i], x, y);
        (-r * self.time_grid.dt(i)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::comparison::close;
    use crate::methods::lattices::{Lattice, Tree, TreeLattice2D, TreeLatticeImpl};
    use crate::processes::OrnsteinUhlenbeckProcess;

    struct AffineDynamics {
        phi: Real,
        rho: Real,
        x: Shared<dyn StochasticProcess1D>,
        y: Shared<dyn StochasticProcess1D>,
    }

    impl TwoFactorShortRateDynamics for AffineDynamics {
        fn short_rate(&self, _t: Time, x: Real, y: Real) -> Rate {
            self.phi + x + y
        }
        fn x_process(&self) -> Shared<dyn StochasticProcess1D> {
            Shared::clone(&self.x)
        }
        fn y_process(&self) -> Shared<dyn StochasticProcess1D> {
            Shared::clone(&self.y)
        }
        fn correlation(&self) -> Real {
            self.rho
        }
    }

    fn ou() -> Shared<dyn StochasticProcess1D> {
        shared(OrnsteinUhlenbeckProcess::new(0.1, 0.01, 0.0, 0.0).unwrap())
            as Shared<dyn StochasticProcess1D>
    }

    fn lattice(
        phi: Real,
        rho: Real,
        steps: Size,
        end: Time,
    ) -> (TreeLattice2D<TwoFactorShortRateTree>, TimeGrid) {
        let grid = TimeGrid::new(end, steps).unwrap();
        let dynamics: Shared<dyn TwoFactorShortRateDynamics> = shared(AffineDynamics {
            phi,
            rho,
            x: ou(),
            y: ou(),
        });
        let t1 = shared(TrinomialTree::new(dynamics.x_process(), grid.clone(), false).unwrap());
        let t2 = shared(TrinomialTree::new(dynamics.y_process(), grid.clone(), false).unwrap());
        let impl_tree =
            TwoFactorShortRateTree::new(t1, t2, Shared::clone(&dynamics), grid.clone()).unwrap();
        (TreeLattice2D::new(impl_tree, grid.clone()).unwrap(), grid)
    }

    #[test]
    fn discount_is_exp_minus_short_rate_dt() {
        let phi = 0.04;
        let (lat, grid) = lattice(phi, -0.5, 5, 1.0);
        let tree = lat.implementation().two_factor_tree();
        for i in 0..grid.size() - 1 {
            for index in 0..tree.size(i) {
                let (x, y) = tree.state(i, index);
                let expected = (-(phi + x + y) * grid.dt(i)).exp();
                let got = lat.implementation().discount(i, index);
                assert!(
                    (got - expected).abs() < 1e-15,
                    "discount({i},{index}): {got} != {expected} (x={x}, y={y})"
                );
            }
        }
    }

    #[test]
    fn product_size_and_correlation_wire_through() {
        let rho = 0.3;
        let (lat, _) = lattice(0.05, rho, 4, 1.0);
        let tree = lat.implementation().two_factor_tree();
        assert_eq!(TwoFactorTree::BRANCHES, 9);
        assert!(close(tree.rho(), rho));
        for i in 0..tree.columns() {
            assert_eq!(tree.size(i), tree.tree1().size(i) * tree.tree2().size(i));
        }
    }

    #[test]
    fn root_discount_uses_phi_only() {
        // Slice 0 is the single (0,0) node; short_rate = phi.
        let phi = 0.05;
        let (lat, grid) = lattice(phi, 0.0, 3, 0.75);
        let expected = (-phi * grid.dt(0)).exp();
        assert!((lat.implementation().discount(0, 0) - expected).abs() < 1e-15);
        let (x, y) = lat.implementation().two_factor_tree().state(0, 0);
        assert!(close(x, 0.0) && close(y, 0.0));
    }

    #[test]
    fn grid_still_fails_on_two_factor_lattice() {
        let (lat, grid) = lattice(0.05, 0.0, 3, 0.5);
        let err = Lattice::grid(&lat, grid[0]).expect_err("2-D grid is unimplemented");
        assert_eq!(err.message(), "not implemented");
    }
}
