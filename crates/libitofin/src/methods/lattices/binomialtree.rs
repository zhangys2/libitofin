//! Binomial equity trees.
//!
//! Port of the Cox-Ross-Rubinstein path of QuantLib's
//! `ql/methods/lattices/binomialtree.hpp` - a recombining, equal-jumps binomial
//! tree centered on the process's forward. It plugs into the existing
//! [`Tree`](crate::methods::lattices::tree::Tree) /
//! [`TreeLattice1D`](crate::methods::lattices::treelattice::TreeLattice1D)
//! machinery, so an equity option can be rolled back on it exactly as a
//! swaption is on the trinomial short-rate tree.

use crate::errors::QlResult;
use crate::methods::lattices::tree::Tree;
use crate::require;
use crate::types::{Real, Size, Time};

/// A Cox-Ross-Rubinstein (multiplicative, equal-jumps) binomial tree.
pub struct CoxRossRubinstein {
    x0: Real,
    dx: Real,
    pu: Real,
    pd: Real,
    columns: Size,
}

impl CoxRossRubinstein {
    /// Builds a CRR tree for a lognormal underlying with spot `x0`, continuous
    /// risk-free rate `r`, dividend yield `q` and volatility `sigma`, over
    /// `end` years in `steps` steps.
    ///
    /// The up jump is `dx = sigma·sqrt(dt)` and the up probability
    /// `0.5 + 0.5·drift/dx` with `drift = (r − q − sigma²/2)·dt`
    /// (`binomialtree.cpp` CRR ctor).
    ///
    /// # Errors
    ///
    /// Fails for non-positive `steps`, `end` or `sigma`, or if the risk-neutral
    /// probability leaves `[0, 1]` (too few steps for the drift).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(
        x0: Real,
        r: Real,
        q: Real,
        sigma: Real,
        end: Time,
        steps: Size,
    ) -> QlResult<CoxRossRubinstein> {
        require!(steps > 0, "at least one step required");
        require!(end > 0.0, "positive maturity required");
        require!(sigma > 0.0, "positive volatility required");
        let dt = end / steps as Real;
        let drift_per_step = (r - q - 0.5 * sigma * sigma) * dt;
        let dx = sigma * dt.sqrt();
        let pu = 0.5 + 0.5 * drift_per_step / dx;
        require!(
            (0.0..=1.0).contains(&pu),
            "negative probability: increase the number of steps"
        );
        Ok(CoxRossRubinstein {
            x0,
            dx,
            pu,
            pd: 1.0 - pu,
            columns: steps + 1,
        })
    }
}

impl Tree for CoxRossRubinstein {
    const BRANCHES: Size = 2;

    fn columns(&self) -> Size {
        self.columns
    }

    fn size(&self, i: Size) -> Size {
        i + 1
    }

    fn underlying(&self, i: Size, index: Size) -> Real {
        // Forward-centred: node j = 2·index − i jumps from the spot.
        let j = 2 * index as isize - i as isize;
        self.x0 * (j as Real * self.dx).exp()
    }

    fn descendant(&self, _i: Size, index: Size, branch: Size) -> Size {
        index + branch
    }

    fn probability(&self, _i: Size, _index: Size, branch: Size) -> Real {
        if branch == 1 { self.pu } else { self.pd }
    }
}
