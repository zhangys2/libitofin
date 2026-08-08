//! Two-dimensional tree-based lattice.
//!
//! Port of `ql/methods/lattices/lattice2d.hpp`. A [`TwoFactorTree`] is the
//! recombining product of two [`TrinomialTree`]s (`BRANCHES = 9`) with the
//! Hull–White correlation correction on the transition weights. [`TreeLattice2D`]
//! newtype-wraps [`TreeLattice`] the same way [`TreeLattice1D`] does, but its
//! [`Lattice::grid`] fails with `"not implemented"` (`lattice2d.hpp:66`).
//!
//! # Index layout
//! Joint node `index` on slice `i` splits as
//! `index1 = index % size1(i)`, `index2 = index / size1(i)`; branches split the
//! same way over the factor trinomial's three legs. Descendants recombine with
//! stride `size1(i+1)` (`lattice2d.hpp:110-121`).
//!
//! # Correlation weights
//! `prob = p1*p2 + |ρ| * m[b1][b2] / 36`, where `m` is the fixed 3×3 Hull–White
//! matrix (sign pattern flipped when `ρ < 0`) (`lattice2d.hpp:80-106,123-134`).

use crate::discretizedasset::DiscretizedAsset;
use crate::errors::QlResult;
use crate::fail;
use crate::math::array::Array;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::lattice::Lattice;
use crate::methods::lattices::tree::Tree;
use crate::methods::lattices::treelattice::{TreeLattice, TreeLatticeImpl};
use crate::methods::lattices::trinomialtree::TrinomialTree;
use crate::require;
use crate::shared::Shared;
use crate::types::{Real, Size, Time};

/// Product of two trinomial trees with a Hull–White correlation adjustment
/// (`lattice2d.hpp` size/descendant/probability surface).
///
/// The joint state is not a scalar: [`Tree::underlying`] returns `0.0` and is
/// unused by lattice induction. Use [`state`](Self::state) / the factor trees
/// for `(x, y)`.
pub struct TwoFactorTree {
    tree1: Shared<TrinomialTree>,
    tree2: Shared<TrinomialTree>,
    rho: Real,
    m: [[Real; 3]; 3],
}

impl TwoFactorTree {
    /// Builds the product of `tree1` and `tree2` under `correlation`
    /// (`lattice2d.hpp:72-107`).
    ///
    /// # Errors
    /// Returns `Err` if the factor trees disagree on column count (they must
    /// share a time grid).
    pub fn new(
        tree1: Shared<TrinomialTree>,
        tree2: Shared<TrinomialTree>,
        correlation: Real,
    ) -> QlResult<Self> {
        require!(
            tree1.columns() == tree2.columns(),
            "two-factor tree factors must share the same number of columns \
             (got {} and {})",
            tree1.columns(),
            tree2.columns()
        );
        let m = if correlation < 0.0 {
            [[-1.0, -4.0, 5.0], [-4.0, 8.0, -4.0], [5.0, -4.0, -1.0]]
        } else {
            [[5.0, -4.0, -1.0], [-4.0, 8.0, -4.0], [-1.0, -4.0, 5.0]]
        };
        Ok(TwoFactorTree {
            tree1,
            tree2,
            rho: correlation.abs(),
            m,
        })
    }

    /// First-factor trinomial (`lattice2d.hpp` `tree1_`).
    pub fn tree1(&self) -> &TrinomialTree {
        &self.tree1
    }

    /// Second-factor trinomial (`lattice2d.hpp` `tree2_`).
    pub fn tree2(&self) -> &TrinomialTree {
        &self.tree2
    }

    /// Absolute correlation `|ρ|` used in the transition weights.
    pub fn rho(&self) -> Real {
        self.rho
    }

    /// Correlation matrix `m_[branch1][branch2]` (`lattice2d.hpp`).
    pub fn m(&self) -> &[[Real; 3]; 3] {
        &self.m
    }

    /// Splits a joint node index into factor indices on slice `i`
    /// (`lattice2d.hpp:112-114`).
    pub fn split_index(&self, i: Size, index: Size) -> (Size, Size) {
        let modulo = self.tree1.size(i);
        (index % modulo, index / modulo)
    }

    /// Factor state `(x, y)` at joint node `index` on slice `i`.
    pub fn state(&self, i: Size, index: Size) -> (Real, Real) {
        let (index1, index2) = self.split_index(i, index);
        (
            self.tree1.underlying(i, index1),
            self.tree2.underlying(i, index2),
        )
    }
}

impl Tree for TwoFactorTree {
    /// `T::branches * T::branches` with `T = TrinomialTree` (`lattice2d.hpp:75`).
    const BRANCHES: Size = 9;

    fn columns(&self) -> Size {
        self.tree1.columns()
    }

    fn size(&self, i: Size) -> Size {
        self.tree1.size(i) * self.tree2.size(i)
    }

    fn underlying(&self, _i: Size, _index: Size) -> Real {
        // Joint (x, y) is not a scalar; lattice induction never reads this.
        0.0
    }

    fn descendant(&self, i: Size, index: Size, branch: Size) -> Size {
        let modulo = self.tree1.size(i);
        let index1 = index % modulo;
        let index2 = index / modulo;
        let branch1 = branch % TrinomialTree::BRANCHES;
        let branch2 = branch / TrinomialTree::BRANCHES;
        let stride = self.tree1.size(i + 1);
        self.tree1.descendant(i, index1, branch1)
            + self.tree2.descendant(i, index2, branch2) * stride
    }

    fn probability(&self, i: Size, index: Size, branch: Size) -> Real {
        let modulo = self.tree1.size(i);
        let index1 = index % modulo;
        let index2 = index / modulo;
        let branch1 = branch % TrinomialTree::BRANCHES;
        let branch2 = branch / TrinomialTree::BRANCHES;
        let prob1 = self.tree1.probability(i, index1, branch1);
        let prob2 = self.tree2.probability(i, index2, branch2);
        prob1 * prob2 + self.rho * self.m[branch1][branch2] / 36.0
    }
}

/// Two-dimensional tree-based lattice (`lattice2d.hpp:48`): a [`TreeLattice`]
/// whose state grid is the product of two factor trees. Concrete engines hold
/// this as a `Shared<dyn Lattice>`; `TwoFactorModel::ShortRateTree` will supply
/// the [`TreeLatticeImpl`] discount surface in a follow-up.
pub struct TreeLattice2D<I: TreeLatticeImpl> {
    base: TreeLattice<I>,
}

impl<I: TreeLatticeImpl> TreeLattice2D<I> {
    /// Builds a 2-D lattice over `time_grid` driven by `implementation`
    /// (`lattice2d.hpp:72`).
    ///
    /// # Errors
    /// Propagates [`TreeLattice::new`].
    pub fn new(implementation: I, time_grid: TimeGrid) -> QlResult<Self> {
        Ok(TreeLattice2D {
            base: TreeLattice::new(implementation, time_grid)?,
        })
    }
}

impl<I: TreeLatticeImpl> std::ops::Deref for TreeLattice2D<I> {
    type Target = TreeLattice<I>;

    /// Exposes the base induction surface on the concrete 2-D type, modelling
    /// the C++ `TreeLattice2D : TreeLattice`.
    fn deref(&self) -> &TreeLattice<I> {
        &self.base
    }
}

impl<I: TreeLatticeImpl> Lattice for TreeLattice2D<I> {
    fn time_grid(&self) -> &TimeGrid {
        self.base.time_grid()
    }

    fn initialize(&self, asset: &mut dyn DiscretizedAsset, time: Time) -> QlResult<()> {
        self.base.initialize(asset, time)
    }

    fn rollback(&self, asset: &mut dyn DiscretizedAsset, to: Time) -> QlResult<()> {
        self.base.rollback(asset, to)
    }

    fn partial_rollback(&self, asset: &mut dyn DiscretizedAsset, to: Time) -> QlResult<()> {
        self.base.partial_rollback(asset, to)
    }

    fn present_value(&self, asset: &mut dyn DiscretizedAsset) -> QlResult<Real> {
        self.base.present_value(asset)
    }

    /// `lattice2d.hpp:66`: the joint state is not a 1-D grid.
    fn grid(&self, _time: Time) -> QlResult<Array> {
        fail!("not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretizedasset::DiscretizedDiscountBond;
    use crate::math::comparison::close;
    use crate::processes::OrnsteinUhlenbeckProcess;
    use crate::shared::shared;
    use crate::stochasticprocess::StochasticProcess1D;

    const SPEED: Real = 0.1;
    const VOL: Real = 0.01;
    const X0: Real = 0.10;
    const LEVEL: Real = 0.05;
    const R: Real = 0.05;

    fn process() -> Shared<dyn StochasticProcess1D> {
        shared(OrnsteinUhlenbeckProcess::new(SPEED, VOL, X0, LEVEL).unwrap())
    }

    fn factor_trees(
        steps: Size,
        end: Time,
    ) -> (Shared<TrinomialTree>, Shared<TrinomialTree>, TimeGrid) {
        let grid = TimeGrid::new(end, steps).unwrap();
        let t1 = shared(TrinomialTree::new(process(), grid.clone(), false).unwrap());
        let t2 = shared(TrinomialTree::new(process(), grid.clone(), false).unwrap());
        (t1, t2, grid)
    }

    fn product(correlation: Real) -> (TwoFactorTree, TimeGrid) {
        let (t1, t2, grid) = factor_trees(5, 1.0);
        (TwoFactorTree::new(t1, t2, correlation).unwrap(), grid)
    }

    struct FlatRate {
        tree: TwoFactorTree,
        grid: TimeGrid,
        rate: Real,
    }

    impl TreeLatticeImpl for FlatRate {
        type Tree = TwoFactorTree;
        fn tree(&self) -> &TwoFactorTree {
            &self.tree
        }
        fn discount(&self, i: Size, _index: Size) -> Real {
            (-self.rate * self.grid.dt(i)).exp()
        }
    }

    #[test]
    fn branches_is_nine() {
        assert_eq!(TwoFactorTree::BRANCHES, 9);
        assert_eq!(
            TwoFactorTree::BRANCHES,
            TrinomialTree::BRANCHES * TrinomialTree::BRANCHES
        );
    }

    #[test]
    fn size_is_product_of_factor_sizes() {
        let (tree, _) = product(0.0);
        for i in 0..tree.columns() {
            assert_eq!(
                tree.size(i),
                tree.tree1().size(i) * tree.tree2().size(i),
                "size mismatch at slice {i}"
            );
        }
    }

    #[test]
    fn mismatched_columns_are_rejected() {
        let grid_a = TimeGrid::new(1.0, 4).unwrap();
        let grid_b = TimeGrid::new(1.0, 5).unwrap();
        let t1 = shared(TrinomialTree::new(process(), grid_a, false).unwrap());
        let t2 = shared(TrinomialTree::new(process(), grid_b, false).unwrap());
        match TwoFactorTree::new(t1, t2, 0.0) {
            Ok(_) => panic!("mismatched columns must fail"),
            Err(err) => assert!(
                err.message().contains("same number of columns"),
                "unexpected message: {}",
                err.message()
            ),
        }
    }

    #[test]
    fn zero_correlation_is_independent_product() {
        let (tree, _) = product(0.0);
        let i = 2;
        for index in 0..tree.size(i) {
            let (index1, index2) = tree.split_index(i, index);
            for branch in 0..TwoFactorTree::BRANCHES {
                let b1 = branch % 3;
                let b2 = branch / 3;
                let expected = tree.tree1().probability(i, index1, b1)
                    * tree.tree2().probability(i, index2, b2);
                let got = tree.probability(i, index, branch);
                assert!(
                    (got - expected).abs() < 1e-15,
                    "ρ=0 prob mismatch at ({i},{index},{branch}): {got} != {expected}"
                );
            }
        }
    }

    #[test]
    fn nonzero_correlation_adds_hull_white_term() {
        let rho = 0.3;
        let (tree, _) = product(rho);
        assert!(close(tree.rho(), rho));
        let i = 1;
        let index = 0;
        let (index1, index2) = tree.split_index(i, index);
        for branch in 0..TwoFactorTree::BRANCHES {
            let b1 = branch % 3;
            let b2 = branch / 3;
            let expected = tree.tree1().probability(i, index1, b1)
                * tree.tree2().probability(i, index2, b2)
                + rho * tree.m()[b1][b2] / 36.0;
            let got = tree.probability(i, index, branch);
            assert!(
                (got - expected).abs() < 1e-15,
                "ρ≠0 prob mismatch at branch {branch}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn negative_correlation_flips_m_and_uses_abs_rho() {
        let (pos, _) = product(0.25);
        let (neg, _) = product(-0.25);
        assert!(close(pos.rho(), 0.25));
        assert!(close(neg.rho(), 0.25));
        for b1 in 0..3 {
            for b2 in 0..3 {
                // QL: negative ρ uses the sign-flipped corner pattern, which is
                // not a uniform negation of m (centre stays +8).
                assert_eq!(
                    neg.m()[b1][b2],
                    [[-1.0, -4.0, 5.0], [-4.0, 8.0, -4.0], [5.0, -4.0, -1.0],][b1][b2]
                );
                assert_eq!(
                    pos.m()[b1][b2],
                    [[5.0, -4.0, -1.0], [-4.0, 8.0, -4.0], [-1.0, -4.0, 5.0],][b1][b2]
                );
            }
        }
        // Probabilities at a fixed node must differ wherever m differs.
        let i = 1;
        let mut differed = false;
        for branch in 0..TwoFactorTree::BRANCHES {
            if (pos.probability(i, 0, branch) - neg.probability(i, 0, branch)).abs() > 1e-15 {
                differed = true;
                break;
            }
        }
        assert!(differed, "sign(ρ) must change at least one branch weight");
    }

    #[test]
    fn branch_probabilities_sum_to_one() {
        for correlation in [0.0, 0.5, -0.5] {
            let (tree, _) = product(correlation);
            for i in 0..tree.columns() - 1 {
                for index in 0..tree.size(i) {
                    let mut sum = 0.0;
                    for branch in 0..TwoFactorTree::BRANCHES {
                        sum += tree.probability(i, index, branch);
                    }
                    assert!(
                        (sum - 1.0).abs() < 1e-12,
                        "prob sum at ({i},{index}) ρ={correlation}: {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn descendant_recombines_with_size1_stride() {
        let (tree, _) = product(0.0);
        let i = 2;
        for index in 0..tree.size(i) {
            let (index1, index2) = tree.split_index(i, index);
            let stride = tree.tree1().size(i + 1);
            for branch in 0..TwoFactorTree::BRANCHES {
                let b1 = branch % 3;
                let b2 = branch / 3;
                let expected = tree.tree1().descendant(i, index1, b1)
                    + tree.tree2().descendant(i, index2, b2) * stride;
                assert_eq!(
                    tree.descendant(i, index, branch),
                    expected,
                    "descendant mismatch at ({i},{index},{branch})"
                );
            }
        }
    }

    #[test]
    fn state_splits_factor_underlyings() {
        let (tree, _) = product(0.0);
        let i = 3;
        for index in 0..tree.size(i) {
            let (x, y) = tree.state(i, index);
            let (i1, i2) = tree.split_index(i, index);
            assert!(close(x, tree.tree1().underlying(i, i1)));
            assert!(close(y, tree.tree2().underlying(i, i2)));
            assert!(close(tree.underlying(i, index), 0.0));
        }
    }

    #[test]
    fn grid_is_not_implemented() {
        let (tree, grid) = product(0.0);
        let lattice = TreeLattice2D::new(
            FlatRate {
                tree,
                grid: grid.clone(),
                rate: R,
            },
            grid.clone(),
        )
        .unwrap();
        let err = Lattice::grid(&lattice, grid[0]).expect_err("grid must fail");
        assert_eq!(err.message(), "not implemented");
    }

    #[test]
    fn flat_discount_rollback_matches_zero_bond() {
        // Smoke: unit payoff rolled back under flat r on the 9-branch product
        // recovers exp(-r T), exercising stepback with BRANCHES=9.
        let steps = 5;
        let end = 1.0;
        let (t1, t2, grid) = factor_trees(steps, end);
        let tree = TwoFactorTree::new(t1, t2, 0.25).unwrap();
        let lattice: Shared<dyn Lattice> = shared(
            TreeLattice2D::new(
                FlatRate {
                    tree,
                    grid: grid.clone(),
                    rate: R,
                },
                grid.clone(),
            )
            .unwrap(),
        );
        let mut bond = DiscretizedDiscountBond::new();
        bond.initialize(Shared::clone(&lattice), grid[steps])
            .unwrap();
        for v in bond.values_mut().iter_mut() {
            *v = 1.0;
        }
        bond.rollback(0.0).unwrap();
        let pv = bond.present_value().unwrap();
        let expected = (-R * end).exp();
        assert!(
            (pv - expected).abs() < 1e-12,
            "flat 2-D rollback pv={pv} != {expected}"
        );
    }
}
