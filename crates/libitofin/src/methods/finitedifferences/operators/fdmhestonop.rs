//! Finite-difference generator for the Heston stochastic-volatility model.
//!
//! Port of `ql/methods/finitedifferences/operators/fdmhestonop.{hpp,cpp}`:
//! [`FdmHestonOp`] is the two-direction [`FdmLinearOpComposite`] whose pieces
//! are the equity map in `ln(S)`, the CIR variance map, and the `ρ σ v` mixed
//! derivative.
//!
//! Quanto is [`with_quanto`](FdmHestonOp::with_quanto). The leverage-function
//! (local-vol) branch is still omitted; `mixingFactor` defaults to 1, so the
//! equity and variance maps are the plain Heston ones. `toMatrixDecomp` is
//! omitted with the sparse-matrix work (#636).

use crate::errors::QlResult;
use crate::fail;
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::utilities::FdmQuantoHelper;
use crate::processes::HestonProcess;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time};

use super::fdmlinearop::FdmLinearOp;
use super::fdmlinearopcomposite::FdmLinearOpComposite;
use super::firstderivativeop::first_derivative_op;
use super::ninepointlinearop::NinePointLinearOp;
use super::secondderivativeop::second_derivative_op;
use super::secondordermixedderivativeop::second_order_mixed_derivative_op;
use super::triplebandlinearop::TripleBandLinearOp;

/// Equity-direction map (`FdmHestonEquityPart`, `fdmhestonop.cpp:39-99`).
struct FdmHestonEquityPart {
    variance_values: Array,
    volatility_values: Array,
    dx_map: TripleBandLinearOp,
    dxx_map: TripleBandLinearOp,
    map_t: TripleBandLinearOp,
    r_ts: Shared<dyn YieldTermStructure>,
    q_ts: Shared<dyn YieldTermStructure>,
    leverage: Array,
    quanto: Option<Shared<FdmQuantoHelper>>,
}

impl FdmHestonEquityPart {
    fn new(
        mesher: Shared<dyn FdmMesher>,
        r_ts: Shared<dyn YieldTermStructure>,
        q_ts: Shared<dyn YieldTermStructure>,
        quanto: Option<Shared<FdmQuantoHelper>>,
    ) -> Self {
        let mut variance_values = 0.5 * &mesher.locations(1);
        let layout = mesher.layout();
        let last_x = layout.dim()[0] - 1;
        for iter in layout.iter() {
            let x = iter.coordinates()[0];
            if x == 0 || x == last_x {
                variance_values[iter.index()] = 0.0;
            }
        }
        let volatility_values = (2.0 * &variance_values).sqrt();
        let size = layout.size();
        FdmHestonEquityPart {
            variance_values,
            volatility_values,
            dx_map: first_derivative_op(0, Shared::clone(&mesher)),
            dxx_map: second_derivative_op(0, Shared::clone(&mesher))
                .mult(&(0.5 * &mesher.locations(1))),
            map_t: TripleBandLinearOp::new(0, mesher),
            r_ts,
            q_ts,
            leverage: Array::filled(size, 1.0),
            quanto,
        }
    }

    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        let r = self
            .r_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q = self
            .q_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let l_square = &self.leverage * &self.leverage;
        let mut drift = (r - q) - &(&self.variance_values * &l_square);
        if let Some(quanto) = &self.quanto {
            let adj = quanto.quanto_adjustment_array(
                &(&self.volatility_values * &self.leverage),
                t1,
                t2,
            )?;
            drift = &drift - &adj;
        }
        let dxx_scaled = self.dxx_map.mult(&l_square);
        self.map_t.axpyb(
            &drift,
            &self.dx_map,
            &dxx_scaled,
            &Array::filled(1, -0.5 * r),
        );
        Ok(())
    }
}

/// Variance-direction map (`FdmHestonVariancePart`, `fdmhestonop.cpp:101-121`).
struct FdmHestonVariancePart {
    dy_map: TripleBandLinearOp,
    map_t: TripleBandLinearOp,
    r_ts: Shared<dyn YieldTermStructure>,
}

impl FdmHestonVariancePart {
    fn new(
        mesher: Shared<dyn FdmMesher>,
        r_ts: Shared<dyn YieldTermStructure>,
        mixed_sigma: Real,
        kappa: Real,
        theta: Real,
    ) -> Self {
        let v = mesher.locations(1);
        let dy_map = second_derivative_op(1, Shared::clone(&mesher))
            .mult(&(0.5 * mixed_sigma * mixed_sigma * &v))
            .add_op(&first_derivative_op(1, Shared::clone(&mesher)).mult(&(kappa * &(theta - &v))));
        FdmHestonVariancePart {
            dy_map,
            map_t: TripleBandLinearOp::new(1, mesher),
            r_ts,
        }
    }

    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        let r = self
            .r_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        self.map_t.axpyb(
            &Array::new(),
            &self.dy_map,
            &self.dy_map,
            &Array::filled(1, -0.5 * r),
        );
        Ok(())
    }
}

/// Heston finite-difference generator (`fdmhestonop.hpp:80`).
pub struct FdmHestonOp {
    correlation_map: NinePointLinearOp,
    dy_map: FdmHestonVariancePart,
    dx_map: FdmHestonEquityPart,
}

impl FdmHestonOp {
    /// `FdmHestonOp(mesher, hestonProcess)` without quanto / leverage
    /// (`fdmhestonop.cpp:123-142`). `mixing_factor` defaults to 1 in C++.
    ///
    /// # Errors
    ///
    /// Fails if a curve handle is empty or the mixed-derivative stencil cannot
    /// be built.
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        process: &HestonProcess,
        mixing_factor: Real,
    ) -> QlResult<Self> {
        Self::with_quanto(mesher, process, mixing_factor, None)
    }

    /// As [`new`](Self::new), with the C++ `quantoHelper`.
    pub fn with_quanto(
        mesher: Shared<dyn FdmMesher>,
        process: &HestonProcess,
        mixing_factor: Real,
        quanto: Option<Shared<FdmQuantoHelper>>,
    ) -> QlResult<Self> {
        let r_ts = process.risk_free_rate().current_link()?;
        let q_ts = process.dividend_yield().current_link()?;
        let mixed_sigma = process.sigma() * mixing_factor;
        let correlation_map = second_order_mixed_derivative_op(0, 1, Shared::clone(&mesher))?
            .mult(&(process.rho() * mixed_sigma * &mesher.locations(1)));
        Ok(FdmHestonOp {
            correlation_map,
            dy_map: FdmHestonVariancePart::new(
                Shared::clone(&mesher),
                Shared::clone(&r_ts),
                mixed_sigma,
                process.kappa(),
                process.theta(),
            ),
            dx_map: FdmHestonEquityPart::new(mesher, r_ts, q_ts, quanto),
        })
    }
}

impl FdmLinearOp for FdmHestonOp {
    /// `apply` (`fdmhestonop.cpp:152-155`).
    fn apply(&self, u: &Array) -> Array {
        &(&self.dy_map.map_t.apply(u) + &self.dx_map.map_t.apply(u)) + &self.apply_mixed(u)
    }
}

impl FdmLinearOpComposite for FdmHestonOp {
    fn size(&self) -> Size {
        2
    }

    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        self.dx_map.set_time(t1, t2)?;
        self.dy_map.set_time(t1, t2)
    }

    fn apply_mixed(&self, r: &Array) -> Array {
        &self.dx_map.leverage * &self.correlation_map.apply(r)
    }

    fn apply_direction(&self, direction: Size, r: &Array) -> Array {
        match direction {
            0 => self.dx_map.map_t.apply(r),
            1 => self.dy_map.map_t.apply(r),
            _ => Array::with_size(r.size()),
        }
    }

    fn solve_splitting(&self, direction: Size, r: &Array, s: Real) -> QlResult<Array> {
        match direction {
            0 => self.dx_map.map_t.solve_splitting(r, s, 1.0),
            1 => self.dy_map.map_t.solve_splitting(r, s, 1.0),
            _ => fail!("direction too large"),
        }
    }

    fn preconditioner(&self, r: &Array, dt: Real) -> QlResult<Array> {
        self.solve_splitting(1, &self.solve_splitting(0, r, dt)?, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::methods::finitedifferences::meshers::{FdmMesherComposite, uniform_1d_mesher};
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    fn process(rho: Real) -> HestonProcess {
        let dc = Actual365Fixed::new();
        let today = Date::new(15, Month::January, 2026);
        let curve = Handle::new(shared(FlatForward::with_rate(
            today,
            0.05,
            dc,
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        HestonProcess::new(
            curve.clone(),
            curve,
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            0.04,
            1.0,
            0.04,
            0.3,
            rho,
        )
    }

    fn mesher_2d() -> Shared<dyn FdmMesher> {
        shared(FdmMesherComposite::new(vec![
            uniform_1d_mesher((80.0_f64).ln(), (120.0_f64).ln(), 7).unwrap(),
            uniform_1d_mesher(0.01, 0.09, 5).unwrap(),
        ])) as Shared<dyn FdmMesher>
    }

    #[test]
    fn rho_zero_kills_the_mixed_map() {
        let mut op = FdmHestonOp::new(mesher_2d(), &process(0.0), 1.0).unwrap();
        op.set_time(0.0, 0.1).unwrap();
        let r = Array::incremental(op.apply_mixed(&Array::filled(35, 1.0)).size(), 1.0, 0.1);
        let mixed = op.apply_mixed(&r);
        for i in 0..mixed.size() {
            assert!(mixed[i].abs() < 1e-14, "mixed[{i}] = {}", mixed[i]);
        }
    }

    #[test]
    fn apply_equals_directions_plus_mixed() {
        let mut op = FdmHestonOp::new(mesher_2d(), &process(-0.5), 1.0).unwrap();
        op.set_time(0.0, 0.25).unwrap();
        let r = Array::incremental(7 * 5, 1.0, 0.05);
        let apply = op.apply(&r);
        let split =
            &(&op.apply_direction(0, &r) + &op.apply_direction(1, &r)) + &op.apply_mixed(&r);
        for i in 0..apply.size() {
            let diff = (apply[i] - split[i]).abs();
            assert!(
                diff < 1e-12,
                "i={i} apply={} split={} diff={diff}",
                apply[i],
                split[i]
            );
        }
    }

    #[test]
    fn splitting_inverts_each_direction() {
        let mut op = FdmHestonOp::new(mesher_2d(), &process(-0.25), 1.0).unwrap();
        op.set_time(0.0, 0.1).unwrap();
        let r = Array::incremental(7 * 5, 0.5, 0.02);
        let dt = 0.05;
        for dir in [0, 1] {
            let solved = op.solve_splitting(dir, &r, dt).unwrap();
            let reconstructed = &solved + &(dt * &op.apply_direction(dir, &solved));
            for i in 0..r.size() {
                let diff = (reconstructed[i] - r[i]).abs();
                assert!(
                    diff < 1e-10,
                    "dir={dir} i={i} recon={} rhs={} diff={diff}",
                    reconstructed[i],
                    r[i]
                );
            }
        }
    }
}
