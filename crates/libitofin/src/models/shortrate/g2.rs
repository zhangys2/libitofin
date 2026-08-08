//! Two-additive-factor Gaussian short-rate model (G2++).
//!
//! Port of `ql/models/shortrate/twofactormodels/g2.{hpp,cpp}`: the curve-fitted
//! two-factor Gaussian model
//!
//! ```text
//! r_t = φ(t) + x_t + y_t
//! dx  = -a x dt + σ dW¹
//! dy  = -b y dt + η dW²
//! dW¹ dW² = ρ dt
//! ```
//!
//! with analytical fitting parameter `φ(t)` chosen so the model reprices the
//! input [`YieldTermStructure`]. This slice ports the closed-form affine surface
//! (`A`, `B`, `V`, `discountBond`, `discountBondOption`), the `φ(t)` fitting
//! law, and [`G2Dynamics`] / [`G2::dynamics`].
//!
//! ## Deferred (omitted, not stubbed)
//!
//! - Tree Bermudan swaption engine path (`TreeSwaptionEngine` for G2) — tree
//!   half of `bermudanswaption.cpp` `testCachedG2Values`. The lattice itself
//!   ([`G2::tree`] / [`TwoFactorShortRateTree`]) is ported.
//!
//! The standalone Monte Carlo factor process lives at
//! [`crate::processes::G2Process`]. [`TwoFactorShortRateDynamics::process`]
//! follows QuantLib and returns a
//! [`StochasticProcessArray`](crate::processes::StochasticProcessArray) of the
//! two OU factors (instantaneous correlation), not `G2Process`.
//!
//! ## Divergences from QuantLib
//!
//! - C++ multiply-inherits `TwoFactorModel`, `AffineModel`, and
//!   `TermStructureConsistentModel`. Here `TwoFactorModel` is collapsed (it only
//!   forwarded an argument count); [`G2`] embeds a [`CalibratedModel`] and a
//!   [`TermStructureConsistentModel`], implements [`AffineModel`] directly
//!   (not [`OneFactorAffineModel`]), and registers with the term-structure
//!   handle like Hull-White. Dynamics use [`TwoFactorShortRateDynamics`];
//!   `tree()` returns [`TreeLattice2D`]`<`[`TwoFactorShortRateTree`]`>` rather
//!   than a CRTP `ShortRateTree` subclass.
//! - C++'s `FittingParameter` subclass becomes a [`ParameterValue`] wrapped by
//!   [`TermStructureFittingParameter`], matching the ECIR seam.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instruments::SwapType;
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::integrals::Integrator;
use crate::math::integrals::segment::SegmentIntegral;
use crate::math::optimization::constraint::{BoundaryConstraint, PositiveConstraint};
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::treelattice2d::TreeLattice2D;
use crate::methods::lattices::trinomialtree::TrinomialTree;
use crate::models::model::{
    CalibratedModel, CalibratedModelHolder, TermStructureConsistentModel,
    register_with_term_structure,
};
use crate::models::parameter::{
    ConstantParameter, NullParameter, Parameter, ParameterValue, TermStructureFittingParameter,
};
use crate::models::shortrate::onefactormodel::AffineModel;
use crate::models::shortrate::twofactormodel::{
    TwoFactorShortRateDynamics, TwoFactorShortRateTree,
};
use crate::option::OptionType;
use crate::patterns::observable::Observer;
use crate::pricingengines::blackformula::black_formula;
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Size, Time};

/// Analytical fitting law `φ(t)` (`g2.hpp:155-163`).
///
/// ```text
/// φ(t) = f(t) + ½ (σ(1−e^{−at})/a)² + ½ (η(1−e^{−bt})/b)²
///              + ρ (σ(1−e^{−at})/a) (η(1−e^{−bt})/b)
/// ```
struct FittingParameterValue {
    term_structure: Handle<dyn YieldTermStructure>,
    a: Real,
    sigma: Real,
    b: Real,
    eta: Real,
    rho: Real,
}

impl ParameterValue for FittingParameterValue {
    fn value(&self, _params: &Array, t: Time) -> Real {
        let curve = self
            .term_structure
            .current_link()
            .expect("the G2 fitting law requires a non-empty term-structure handle");
        let forward = curve
            .forward_rate(t, t, Compounding::Continuous, Frequency::NoFrequency, false)
            .expect("the G2 fitting law's forward rate is well-defined on its curve")
            .rate();
        let temp1 = self.sigma * (1.0 - (-self.a * t).exp()) / self.a;
        let temp2 = self.eta * (1.0 - (-self.b * t).exp()) / self.b;
        0.5 * temp1 * temp1 + 0.5 * temp2 * temp2 + self.rho * temp1 * temp2 + forward
    }
}

/// Two-additive-factor Gaussian model G2++ (`g2.hpp:54`).
pub struct G2 {
    model: CalibratedModel,
    ts_model: TermStructureConsistentModel,
    phi: Parameter,
    #[allow(dead_code)]
    ts_observer: Option<SharedMut<dyn Observer>>,
}

impl G2 {
    /// `G2(termStructure, a, sigma, b, eta, rho)` (`g2.cpp:31-46`).
    ///
    /// Defaults in C++: `a = 0.1`, `sigma = 0.01`, `b = 0.1`, `eta = 0.01`,
    /// `rho = -0.75`.
    ///
    /// Returns a [`SharedMut`] so the term-structure observer can be stashed
    /// after the model is shared.
    ///
    /// # Errors
    ///
    /// Fails if `a`/`sigma`/`b`/`eta` are not strictly positive, or if `rho` is
    /// outside `[-1, 1]`.
    pub fn new(
        term_structure: Handle<dyn YieldTermStructure>,
        a: Real,
        sigma: Real,
        b: Real,
        eta: Real,
        rho: Real,
    ) -> QlResult<SharedMut<G2>> {
        let mut model = CalibratedModel::new(5);
        model.arguments_mut()[0] = ConstantParameter::new(a, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[1] = ConstantParameter::new(sigma, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[2] = ConstantParameter::new(b, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[3] = ConstantParameter::new(eta, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[4] =
            ConstantParameter::new(rho, Rc::new(BoundaryConstraint::new(-1.0, 1.0)))?;

        let ts_model = TermStructureConsistentModel::new(term_structure.clone());
        let mut g2 = G2 {
            model,
            ts_model,
            phi: NullParameter::new(),
            ts_observer: None,
        };
        g2.generate_arguments();

        let shared = shared_mut(g2);
        let observer = register_with_term_structure(&shared, &term_structure);
        shared.borrow_mut().ts_observer = Some(observer);
        Ok(shared)
    }

    /// Mean-reversion speed of the first factor `a`.
    pub fn a(&self) -> Real {
        self.model.arguments()[0].value(0.0)
    }

    /// Volatility of the first factor `σ`.
    pub fn sigma(&self) -> Real {
        self.model.arguments()[1].value(0.0)
    }

    /// Mean-reversion speed of the second factor `b`.
    pub fn b(&self) -> Real {
        self.model.arguments()[2].value(0.0)
    }

    /// Volatility of the second factor `η`.
    pub fn eta(&self) -> Real {
        self.model.arguments()[3].value(0.0)
    }

    /// Factor correlation `ρ`.
    pub fn rho(&self) -> Real {
        self.model.arguments()[4].value(0.0)
    }

    /// Fitted term-structure handle.
    pub fn term_structure(&self) -> Handle<dyn YieldTermStructure> {
        self.ts_model.term_structure().clone()
    }

    /// `discount(Time t)` (`g2.hpp:85`): the curve discount.
    pub fn discount(&self, t: Time) -> QlResult<Real> {
        self.term_structure().current_link()?.discount(t, false)
    }

    /// Analytical fitting parameter `φ(t)` (`g2.hpp:132-180`).
    ///
    /// Exposed for identity pins (`shortRate(t,0,0) = φ(t)`).
    pub fn phi(&self, t: Time) -> Real {
        self.phi.value(t)
    }

    /// `dynamics()` (`g2.cpp:48-51`): short-rate dynamics `r = φ(t) + x + y`
    /// with OU factors `(a,σ)` and `(b,η)` correlated by `ρ`.
    ///
    /// # Errors
    ///
    /// Fails if either factor volatility is negative (OU constructor).
    pub fn dynamics(&self) -> QlResult<Shared<G2Dynamics>> {
        G2Dynamics::new(
            self.phi.clone(),
            self.a(),
            self.sigma(),
            self.b(),
            self.eta(),
            self.rho(),
        )
        .map(shared)
    }

    /// `TwoFactorModel::tree(const TimeGrid&)` (`twofactormodel.cpp:29-42`):
    /// product of two factor trinomials under G2 dynamics, wrapped as a
    /// [`TreeLattice2D`]. Unlike Hull–White, `φ(t)` is the analytic fitting
    /// law (no numerical state-price fit).
    ///
    /// # Errors
    ///
    /// Fails if dynamics/OU construction fails, if either trinomial rejects
    /// the process/grid, or if the product tree / lattice cannot be built.
    pub fn tree(&self, grid: TimeGrid) -> QlResult<TreeLattice2D<TwoFactorShortRateTree>> {
        let dynamics = self.dynamics()?;
        let tree1 = shared(TrinomialTree::new(
            dynamics.x_process(),
            grid.clone(),
            false,
        )?);
        let tree2 = shared(TrinomialTree::new(
            dynamics.y_process(),
            grid.clone(),
            false,
        )?);
        let dynamics: Shared<dyn TwoFactorShortRateDynamics> = dynamics;
        let short_rate_tree = TwoFactorShortRateTree::new(tree1, tree2, dynamics, grid.clone())?;
        TreeLattice2D::new(short_rate_tree, grid)
    }

    /// `B(x, t) = (1 − e^{−x t})/x` (`g2.cpp:107-109`).
    pub fn bond_b(x: Real, t: Time) -> Real {
        (1.0 - (-x * t).exp()) / x
    }

    /// `V(t)` (`g2.cpp:89-100`): integrated variance of the state.
    pub fn v(&self, t: Time) -> Real {
        let a = self.a();
        let b = self.b();
        let sigma = self.sigma();
        let eta = self.eta();
        let rho = self.rho();
        let expat = (-a * t).exp();
        let expbt = (-b * t).exp();
        let cx = sigma / a;
        let cy = eta / b;
        let valuex = cx * cx * (t + (2.0 * expat - 0.5 * expat * expat - 1.5) / a);
        let valuey = cy * cy * (t + (2.0 * expbt - 0.5 * expbt * expbt - 1.5) / b);
        let value = 2.0
            * rho
            * cx
            * cy
            * (t + (expat - 1.0) / a + (expbt - 1.0) / b - (expat * expbt - 1.0) / (a + b));
        valuex + valuey + value
    }

    /// `A(t, T)` (`g2.cpp:102-105`).
    pub fn bond_a(&self, t: Time, maturity: Time) -> QlResult<Real> {
        let pt = self.discount(t)?;
        let p_maturity = self.discount(maturity)?;
        Ok(p_maturity / pt * (0.5 * (self.v(maturity - t) - self.v(maturity) + self.v(t))).exp())
    }

    /// `discountBond(Time, Time, Rate, Rate)` (`g2.cpp:75-77`).
    pub fn discount_bond(&self, now: Time, maturity: Time, x: Rate, y: Rate) -> QlResult<Real> {
        Ok(self.bond_a(now, maturity)?
            * (-Self::bond_b(self.a(), maturity - now) * x
                - Self::bond_b(self.b(), maturity - now) * y)
                .exp())
    }

    /// `sigmaP(t, s)` (`g2.cpp:59-73`): Black volatility of a zero-coupon bond
    /// option with option maturity `t` and bond maturity `s`.
    pub fn sigma_p(&self, t: Time, s: Time) -> Real {
        let a = self.a();
        let b = self.b();
        let sigma = self.sigma();
        let eta = self.eta();
        let rho = self.rho();
        let temp = 1.0 - (-(a + b) * t).exp();
        let temp1 = 1.0 - (-a * (s - t)).exp();
        let temp2 = 1.0 - (-b * (s - t)).exp();
        let a3 = a * a * a;
        let b3 = b * b * b;
        let sigma2 = sigma * sigma;
        let eta2 = eta * eta;
        let value = 0.5 * sigma2 * temp1 * temp1 * (1.0 - (-2.0 * a * t).exp()) / a3
            + 0.5 * eta2 * temp2 * temp2 * (1.0 - (-2.0 * b * t).exp()) / b3
            + 2.0 * rho * sigma * eta / (a * b * (a + b)) * temp1 * temp2 * temp;
        value.sqrt()
    }

    /// `discountBondOption` (`g2.cpp:79-87`): European option on a zero-coupon
    /// bond via Black with vol [`sigma_p`](Self::sigma_p).
    pub fn discount_bond_option(
        &self,
        option_type: OptionType,
        strike: Real,
        maturity: Time,
        bond_maturity: Time,
    ) -> QlResult<Real> {
        let v = self.sigma_p(maturity, bond_maturity);
        let f = self.discount(bond_maturity)?;
        let k = self.discount(maturity)? * strike;
        black_formula(option_type, k, f, v, 1.0, 0.0)
    }

    /// European swaption NPV via the one-dimensional integral
    /// (`G2::swaption`, `g2.cpp:218-246`).
    ///
    /// `fixed_rate` should already include any floating-spread correction
    /// applied by [`G2SwaptionEngine`](crate::pricingengines::swaption::G2SwaptionEngine).
    ///
    /// # Errors
    ///
    /// Fails if the nominal is missing, the reset/pay date lists are empty, the
    /// curve discounts fail, or the segment integral fails.
    #[allow(clippy::too_many_arguments)]
    pub fn swaption(
        &self,
        nominal: Real,
        swap_type: SwapType,
        floating_reset_dates: &[Date],
        fixed_pay_dates: &[Date],
        fixed_rate: Rate,
        range: Real,
        intervals: Size,
    ) -> QlResult<Real> {
        require!(
            !floating_reset_dates.is_empty(),
            "swap has no floating resets"
        );
        require!(
            !fixed_pay_dates.is_empty(),
            "swap has no fixed payment dates"
        );

        let curve = self.term_structure().current_link()?;
        let settlement = curve.reference_date()?;
        let day_counter = curve.require_day_counter()?;
        let start = day_counter.year_fraction(settlement, floating_reset_dates[0]);
        let w = if swap_type == SwapType::Payer {
            1.0
        } else {
            -1.0
        };

        let fixed_pay_times: Vec<Time> = fixed_pay_dates
            .iter()
            .map(|&date| day_counter.year_fraction(settlement, date))
            .collect();

        let function = SwaptionPricingFunction::new(
            self.a(),
            self.sigma(),
            self.b(),
            self.eta(),
            self.rho(),
            w,
            start,
            fixed_pay_times,
            fixed_rate,
            self,
        )?;

        let upper = function.mux() + range * function.sigmax();
        let lower = function.mux() - range * function.sigmax();
        let integrator = SegmentIntegral::new(intervals)?;
        let integral = integrator.integrate(|x| function.evaluate(x), lower, upper)?;
        Ok(nominal * w * self.discount(start)? * integral)
    }
}

/// `G2::Dynamics` (`g2.hpp:118-130`): short rate `r_t = φ(t) + x_t + y_t`.
pub struct G2Dynamics {
    fitting: Parameter,
    x_process: Shared<dyn StochasticProcess1D>,
    y_process: Shared<dyn StochasticProcess1D>,
    correlation: Real,
}

impl G2Dynamics {
    /// `Dynamics(fitting, a, sigma, b, eta, rho)` (`g2.hpp:120-126`).
    ///
    /// # Errors
    ///
    /// Fails if `sigma` or `eta` is negative.
    pub fn new(
        fitting: Parameter,
        a: Real,
        sigma: Real,
        b: Real,
        eta: Real,
        rho: Real,
    ) -> QlResult<G2Dynamics> {
        Ok(G2Dynamics {
            fitting,
            x_process: shared(OrnsteinUhlenbeckProcess::new(a, sigma, 0.0, 0.0)?)
                as Shared<dyn StochasticProcess1D>,
            y_process: shared(OrnsteinUhlenbeckProcess::new(b, eta, 0.0, 0.0)?)
                as Shared<dyn StochasticProcess1D>,
            correlation: rho,
        })
    }

    /// Fitting drift `φ(t)`.
    pub fn fitting(&self, t: Time) -> Real {
        self.fitting.value(t)
    }
}

impl TwoFactorShortRateDynamics for G2Dynamics {
    fn short_rate(&self, t: Time, x: Real, y: Real) -> Rate {
        self.fitting.value(t) + x + y
    }

    fn x_process(&self) -> Shared<dyn StochasticProcess1D> {
        Shared::clone(&self.x_process)
    }

    fn y_process(&self) -> Shared<dyn StochasticProcess1D> {
        Shared::clone(&self.y_process)
    }

    fn correlation(&self) -> Real {
        self.correlation
    }
}

/// European swaption integrand (`G2::SwaptionPricingFunction`, `g2.cpp:111-216`).
struct SwaptionPricingFunction {
    w: Real,
    start: Time,
    t: Vec<Time>,
    rate: Rate,
    a_cache: Vec<Real>,
    ba: Vec<Real>,
    bb: Vec<Real>,
    mux: Real,
    muy: Real,
    sigmax: Real,
    sigmay: Real,
    rhoxy: Real,
}

impl SwaptionPricingFunction {
    #[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
    fn new(
        a: Real,
        sigma: Real,
        b: Real,
        eta: Real,
        rho: Real,
        w: Real,
        start: Time,
        pay_times: Vec<Time>,
        fixed_rate: Rate,
        model: &G2,
    ) -> QlResult<Self> {
        let size = pay_times.len();
        require!(
            size > 0,
            "swaption pricing function needs at least one pay time"
        );

        let sigmax = sigma * (0.5 * (1.0 - (-2.0 * a * start).exp()) / a).sqrt();
        let sigmay = eta * (0.5 * (1.0 - (-2.0 * b * start).exp()) / b).sqrt();
        require!(sigmax > 0.0, "G2 swaption sigmax must be positive");
        require!(sigmay > 0.0, "G2 swaption sigmay must be positive");

        let rhoxy =
            rho * eta * sigma * (1.0 - (-(a + b) * start).exp()) / ((a + b) * sigmax * sigmay);

        let mut temp = sigma * sigma / (a * a);
        let mux = -((temp + rho * sigma * eta / (a * b)) * (1.0 - (-a * start).exp())
            - 0.5 * temp * (1.0 - (-2.0 * a * start).exp())
            - rho * sigma * eta / (b * (a + b)) * (1.0 - (-(b + a) * start).exp()));

        temp = eta * eta / (b * b);
        let muy = -((temp + rho * sigma * eta / (a * b)) * (1.0 - (-b * start).exp())
            - 0.5 * temp * (1.0 - (-2.0 * b * start).exp())
            - rho * sigma * eta / (a * (a + b)) * (1.0 - (-(b + a) * start).exp()));

        let mut a_cache = Vec::with_capacity(size);
        let mut ba = Vec::with_capacity(size);
        let mut bb = Vec::with_capacity(size);
        for &pay in &pay_times {
            a_cache.push(model.bond_a(start, pay)?);
            ba.push(G2::bond_b(a, pay - start));
            bb.push(G2::bond_b(b, pay - start));
        }

        Ok(SwaptionPricingFunction {
            w,
            start,
            t: pay_times,
            rate: fixed_rate,
            a_cache,
            ba,
            bb,
            mux,
            muy,
            sigmax,
            sigmay,
            rhoxy,
        })
    }

    fn mux(&self) -> Real {
        self.mux
    }

    fn sigmax(&self) -> Real {
        self.sigmax
    }

    /// `operator()(x)` (`g2.cpp:154-189`).
    ///
    /// Panics if the inner Brent root for `y*` fails (C++ throws).
    fn evaluate(&self, x: Real) -> Real {
        let phi = CumulativeNormalDistribution::standard();
        let temp = (x - self.mux) / self.sigmax;
        let txy = (1.0 - self.rhoxy * self.rhoxy).sqrt();
        let size = self.t.len();

        let mut lambda = Vec::with_capacity(size);
        for i in 0..size {
            let tau = if i == 0 {
                self.t[0] - self.start
            } else {
                self.t[i] - self.t[i - 1]
            };
            let c = if i == size - 1 {
                1.0 + self.rate * tau
            } else {
                self.rate * tau
            };
            lambda.push(c * self.a_cache[i] * (-self.ba[i] * x).exp());
        }

        let bb = &self.bb;
        let solving = |y: Real| {
            let mut value = 1.0;
            for i in 0..lambda.len() {
                value -= lambda[i] * (-bb[i] * y).exp();
            }
            value
        };
        let search_bound = (10.0 * self.sigmay).max(1.0);
        let mut solver = Brent::new().with_max_evaluations(1000);
        let yb = solver
            .solve_bracketed(solving, 1e-6, 0.0, -search_bound, search_bound)
            .expect("G2 swaption y* root must exist in the search band");

        let h1 = (yb - self.muy) / (self.sigmay * txy)
            - self.rhoxy * (x - self.mux) / (self.sigmax * txy);
        let mut value = phi.value(-self.w * h1);

        for (lambda_i, bb_i) in lambda.iter().zip(self.bb.iter()) {
            let h2 = h1 + bb_i * self.sigmay * (1.0 - self.rhoxy * self.rhoxy).sqrt();
            let kappa = -bb_i
                * (self.muy - 0.5 * txy * txy * self.sigmay * self.sigmay * bb_i
                    + self.rhoxy * self.sigmay * (x - self.mux) / self.sigmax);
            value -= lambda_i * kappa.exp() * phi.value(-self.w * h2);
        }

        (-0.5 * temp * temp).exp() * value / (self.sigmax * (2.0 * std::f64::consts::PI).sqrt())
    }
}

impl CalibratedModelHolder for G2 {
    fn calibrated_model(&self) -> &CalibratedModel {
        &self.model
    }

    fn calibrated_model_mut(&mut self) -> &mut CalibratedModel {
        &mut self.model
    }

    /// `generateArguments` (`g2.cpp:53-57`): rebuilds `φ_`.
    fn generate_arguments(&mut self) {
        let law = FittingParameterValue {
            term_structure: self.ts_model.term_structure().clone(),
            a: self.a(),
            sigma: self.sigma(),
            b: self.b(),
            eta: self.eta(),
            rho: self.rho(),
        };
        self.phi = TermStructureFittingParameter::new(Rc::new(law));
    }
}

impl AffineModel for G2 {
    /// `discountBond(Time, Time, Array)` (`g2.hpp:67-71`).
    ///
    /// Panics (mirroring C++ `QL_REQUIRE`) if fewer than two factors are given.
    fn discount_bond_factors(&self, now: Time, maturity: Time, factors: &Array) -> Real {
        assert!(
            factors.size() > 1,
            "g2 model needs two factors to compute discount bond"
        );
        self.discount_bond(now, maturity, factors[0], factors[1])
            .expect("G2 discountBond discounts are well-defined on its curve")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{Handle, RelinkableHandle};
    use crate::math::array::Array;
    use crate::math::interpolations::linear::Linear;
    use crate::shared::{Shared, shared};
    use crate::termstructures::yields::{FlatForward, ZeroCurve};
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    const A: Real = 0.1;
    const SIGMA: Real = 0.01;
    const B: Real = 0.2;
    const ETA: Real = 0.008;
    const RHO: Real = -0.75;

    /// Same sloped ZeroCurve fixture as the Hull-White discountBond oracle.
    fn sloped_curve() -> Handle<dyn YieldTermStructure> {
        let dates = vec![
            Date::new(15, Month::January, 2026),
            Date::new(15, Month::January, 2027),
            Date::new(15, Month::January, 2028),
            Date::new(15, Month::January, 2029),
            Date::new(15, Month::January, 2031),
        ];
        let zeros = vec![0.02, 0.025, 0.03, 0.033, 0.04];
        let curve = ZeroCurve::new(dates, zeros, Actual365Fixed::new(), Linear).unwrap();
        Handle::new(shared(curve) as Shared<dyn YieldTermStructure>)
    }

    fn flat(rate: Rate) -> Shared<dyn YieldTermStructure> {
        shared(FlatForward::with_rate(
            Date::new(19, Month::May, 2026),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>
    }

    fn make_g2(handle: Handle<dyn YieldTermStructure>) -> SharedMut<G2> {
        G2::new(handle, A, SIGMA, B, ETA, RHO).unwrap()
    }

    /// Independent transliteration of `V(t)` (`g2.cpp:89-100`) — not the code
    /// under test.
    fn ref_v(t: Time, a: Real, sigma: Real, b: Real, eta: Real, rho: Real) -> Real {
        let expat = (-a * t).exp();
        let expbt = (-b * t).exp();
        let cx = sigma / a;
        let cy = eta / b;
        let valuex = cx * cx * (t + (2.0 * expat - 0.5 * expat * expat - 1.5) / a);
        let valuey = cy * cy * (t + (2.0 * expbt - 0.5 * expbt * expbt - 1.5) / b);
        let value = 2.0
            * rho
            * cx
            * cy
            * (t + (expat - 1.0) / a + (expbt - 1.0) / b - (expat * expbt - 1.0) / (a + b));
        valuex + valuey + value
    }

    fn ref_bond_b(x: Real, t: Time) -> Real {
        (1.0 - (-x * t).exp()) / x
    }

    #[allow(clippy::too_many_arguments)]
    fn ref_discount_bond(
        now: Time,
        maturity: Time,
        x: Rate,
        y: Rate,
        p_now: Real,
        p_maturity: Real,
        a: Real,
        sigma: Real,
        b: Real,
        eta: Real,
        rho: Real,
    ) -> Real {
        let a_tt = p_maturity / p_now
            * (0.5
                * (ref_v(maturity - now, a, sigma, b, eta, rho)
                    - ref_v(maturity, a, sigma, b, eta, rho)
                    + ref_v(now, a, sigma, b, eta, rho)))
            .exp();
        a_tt * (-ref_bond_b(a, maturity - now) * x - ref_bond_b(b, maturity - now) * y).exp()
    }

    fn ref_sigma_p(t: Time, s: Time, a: Real, sigma: Real, b: Real, eta: Real, rho: Real) -> Real {
        let temp = 1.0 - (-(a + b) * t).exp();
        let temp1 = 1.0 - (-a * (s - t)).exp();
        let temp2 = 1.0 - (-b * (s - t)).exp();
        let a3 = a * a * a;
        let b3 = b * b * b;
        let value = 0.5 * sigma * sigma * temp1 * temp1 * (1.0 - (-2.0 * a * t).exp()) / a3
            + 0.5 * eta * eta * temp2 * temp2 * (1.0 - (-2.0 * b * t).exp()) / b3
            + 2.0 * rho * sigma * eta / (a * b * (a + b)) * temp1 * temp2 * temp;
        value.sqrt()
    }

    #[test]
    fn ctor_round_trips_params() {
        let m = make_g2(sloped_curve());
        let g = m.borrow();
        assert_eq!(g.a(), A);
        assert_eq!(g.sigma(), SIGMA);
        assert_eq!(g.b(), B);
        assert_eq!(g.eta(), ETA);
        assert_eq!(g.rho(), RHO);
    }

    #[test]
    fn discount_bond_at_origin_reprices_the_curve() {
        let handle = sloped_curve();
        let curve = handle.current_link().unwrap();
        let m = make_g2(handle);
        let g = m.borrow();
        for &t in &[1.0, 2.0, 3.0, 5.0] {
            let model_p = g.discount_bond(0.0, t, 0.0, 0.0).unwrap();
            let curve_p = curve.discount(t, false).unwrap();
            assert!(
                (model_p - curve_p).abs() < 1e-14,
                "t={t}: model {model_p} vs curve {curve_p}"
            );
        }
    }

    #[test]
    fn discount_bond_matches_independent_reference() {
        let handle = sloped_curve();
        let curve = handle.current_link().unwrap();
        assert!((curve.discount(2.0, false).unwrap() - 0.941_764_533_584_248_7).abs() < 1e-14);
        assert!((curve.discount(3.0, false).unwrap() - 0.905_764_980_659_064).abs() < 1e-14);
        assert!((curve.discount(5.0, false).unwrap() - 0.818_770_008_233_211_2).abs() < 1e-14);

        let m = make_g2(Handle::clone(&handle));
        let g = m.borrow();

        for &(now, maturity, x, y) in &[
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 2.0, 0.03, -0.01),
            (1.0, 3.0, 0.02, 0.01),
            (2.0, 5.0, -0.01, 0.02),
        ] {
            let p_now = curve.discount(now, false).unwrap();
            let p_mat = curve.discount(maturity, false).unwrap();
            let expected =
                ref_discount_bond(now, maturity, x, y, p_now, p_mat, A, SIGMA, B, ETA, RHO);
            let got = g.discount_bond(now, maturity, x, y).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "({now},{maturity},{x},{y}): got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn discount_bond_option_matches_black_with_independent_sigma_p() {
        let handle = sloped_curve();
        let curve = handle.current_link().unwrap();
        let m = make_g2(handle);
        let g = m.borrow();

        for &(option_type, strike, maturity, bond_maturity) in &[
            (OptionType::Call, 0.9, 1.0, 3.0),
            (OptionType::Call, 0.95, 1.0, 3.0),
            (OptionType::Put, 0.9, 1.0, 3.0),
            (OptionType::Put, 0.95, 1.0, 3.0),
        ] {
            let v = ref_sigma_p(maturity, bond_maturity, A, SIGMA, B, ETA, RHO);
            let f = curve.discount(bond_maturity, false).unwrap();
            let k = curve.discount(maturity, false).unwrap() * strike;
            let expected = black_formula(option_type, k, f, v, 1.0, 0.0).unwrap();
            let got = g
                .discount_bond_option(option_type, strike, maturity, bond_maturity)
                .unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "{option_type:?} K={strike}: got {got} expected {expected}"
            );
            // Live sigma_p must match the independent transliteration.
            assert!((g.sigma_p(maturity, bond_maturity) - v).abs() < 1e-14);
        }
    }

    #[test]
    fn phi_matches_closed_form_on_flat_curve() {
        let handle = Handle::new(flat(0.03));
        let curve = handle.current_link().unwrap();
        let m = make_g2(Handle::clone(&handle));
        let g = m.borrow();
        for &t in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            // Read the live forward (FlatForward's t=0 short end is DT-shifted).
            let forward = curve
                .forward_rate(t, t, Compounding::Continuous, Frequency::NoFrequency, false)
                .unwrap()
                .rate();
            let temp1 = SIGMA * (1.0 - (-A * t).exp()) / A;
            let temp2 = ETA * (1.0 - (-B * t).exp()) / B;
            let expected =
                forward + 0.5 * temp1 * temp1 + 0.5 * temp2 * temp2 + RHO * temp1 * temp2;
            let got = g.phi(t);
            assert!(
                (got - expected).abs() < 1e-14,
                "t={t}: phi {got} vs {expected}"
            );
        }
    }

    #[test]
    fn dynamics_short_rate_is_phi_plus_factors() {
        use crate::models::shortrate::TwoFactorShortRateDynamics;
        let m = make_g2(Handle::new(flat(0.03)));
        let g = m.borrow();
        let dynamics = g.dynamics().unwrap();
        for &(t, x, y) in &[(0.0, 0.0, 0.0), (1.0, 0.02, -0.01), (2.5, -0.03, 0.015)] {
            let expected = g.phi(t) + x + y;
            let got = dynamics.short_rate(t, x, y);
            assert!(
                (got - expected).abs() < 1e-14,
                "t={t} x={x} y={y}: {got} vs {expected}"
            );
        }
        assert_eq!(dynamics.correlation(), RHO);
    }

    #[test]
    fn dynamics_factor_processes_are_zero_level_ou() {
        use crate::models::shortrate::TwoFactorShortRateDynamics;
        let m = make_g2(Handle::new(flat(0.03)));
        let dynamics = m.borrow().dynamics().unwrap();
        let x = dynamics.x_process();
        let y = dynamics.y_process();
        assert_eq!(x.x0().unwrap(), 0.0);
        assert_eq!(y.x0().unwrap(), 0.0);
        // Drift at state z is -speed * z for zero-level OU.
        assert!((x.drift(0.0, 0.05).unwrap() + A * 0.05).abs() < 1e-14);
        assert!((y.drift(0.0, -0.04).unwrap() + B * (-0.04)).abs() < 1e-14);
        assert!((x.diffusion(0.0, 0.0).unwrap() - SIGMA).abs() < 1e-15);
        assert!((y.diffusion(0.0, 0.0).unwrap() - ETA).abs() < 1e-15);
    }

    #[test]
    fn dynamics_joint_process_has_instantaneous_correlation() {
        use crate::models::shortrate::TwoFactorShortRateDynamics;
        let m = make_g2(Handle::new(flat(0.03)));
        let dynamics = m.borrow().dynamics().unwrap();
        let process = dynamics.process().unwrap();
        assert_eq!(process.size(), 2);
        let x0 = process.initial_values().unwrap();
        assert_eq!(x0[0], 0.0);
        assert_eq!(x0[1], 0.0);
        let d = process.diffusion(0.0, &x0).unwrap();
        // StochasticProcessArray uses spectral sqrt of corr; cov = d dᵀ must
        // recover the instantaneous factor covariance.
        let cov = &d * &d.transpose();
        assert!((cov[(0, 0)] - SIGMA * SIGMA).abs() < 1e-12);
        assert!((cov[(1, 1)] - ETA * ETA).abs() < 1e-12);
        assert!((cov[(0, 1)] - RHO * SIGMA * ETA).abs() < 1e-12);
    }

    #[test]
    fn set_params_rebuilds_phi() {
        let handle = Handle::new(flat(0.03));
        let curve = handle.current_link().unwrap();
        let m = make_g2(Handle::clone(&handle));
        let new_params = Array::from([0.15, 0.02, 0.25, 0.01, -0.5]);
        m.borrow_mut().set_params(&new_params).unwrap();
        let g = m.borrow();
        assert!((g.a() - 0.15).abs() < 1e-15);
        assert!((g.rho() - (-0.5)).abs() < 1e-15);
        let t: Time = 1.0;
        let forward = curve
            .forward_rate(t, t, Compounding::Continuous, Frequency::NoFrequency, false)
            .unwrap()
            .rate();
        let temp1 = 0.02 * (1.0 - (-0.15 * t).exp()) / 0.15;
        let temp2 = 0.01 * (1.0 - (-0.25 * t).exp()) / 0.25;
        let expected = forward + 0.5 * temp1 * temp1 + 0.5 * temp2 * temp2 + (-0.5) * temp1 * temp2;
        assert!((g.phi(t) - expected).abs() < 1e-14);
    }

    #[test]
    fn affine_model_reads_two_factors() {
        let m = make_g2(sloped_curve());
        let g = m.borrow();
        let factors = Array::from([0.02, 0.01]);
        let via_trait = AffineModel::discount_bond_factors(&*g, 1.0, 3.0, &factors);
        let via_direct = g.discount_bond(1.0, 3.0, 0.02, 0.01).unwrap();
        assert!((via_trait - via_direct).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "g2 model needs two factors")]
    fn affine_model_rejects_single_factor() {
        let m = make_g2(sloped_curve());
        let g = m.borrow();
        let factors = Array::from([0.02]);
        let _ = AffineModel::discount_bond_factors(&*g, 1.0, 3.0, &factors);
    }

    #[test]
    fn phi_updates_when_term_structure_relinks() {
        let rh: RelinkableHandle<dyn YieldTermStructure> = RelinkableHandle::new(flat(0.02));
        let model = G2::new(rh.handle(), A, SIGMA, B, ETA, RHO).unwrap();
        let before = model.borrow().phi(1.0);
        rh.link_to(flat(0.05));
        // generate_arguments rebuilds φ with the new forward.
        let after = model.borrow().phi(1.0);
        assert!(
            (after - before - 0.03).abs() < 1e-11,
            "relink should shift φ by the forward difference: before={before} after={after}"
        );
    }

    #[test]
    fn v_matches_independent_reference() {
        let m = make_g2(sloped_curve());
        let g = m.borrow();
        for &t in &[0.5, 1.0, 2.0, 5.0] {
            assert!((g.v(t) - ref_v(t, A, SIGMA, B, ETA, RHO)).abs() < 1e-15);
        }
    }

    // ------------------------------------------------------------------
    // TwoFactorShortRateTree / G2::tree
    // ------------------------------------------------------------------

    #[test]
    fn tree_builds_with_product_layout_and_abs_rho() {
        use crate::methods::lattices::{Tree, TwoFactorTree};
        use crate::models::shortrate::TwoFactorShortRateDynamics;

        let m = make_g2(Handle::new(flat(0.03)));
        let grid = TimeGrid::new(2.0, 8).unwrap();
        let lattice = m.borrow().tree(grid).unwrap();
        let impl_tree = lattice.implementation();
        let product = impl_tree.two_factor_tree();

        assert_eq!(TwoFactorTree::BRANCHES, 9);
        assert!((product.rho() - RHO.abs()).abs() < 1e-15);
        assert_eq!(
            impl_tree.dynamics().correlation(),
            m.borrow().dynamics().unwrap().correlation()
        );
        for i in 0..product.columns() {
            assert_eq!(
                product.size(i),
                product.tree1().size(i) * product.tree2().size(i)
            );
        }
    }

    #[test]
    fn tree_discount_matches_phi_plus_factors() {
        use crate::math::comparison::close;
        use crate::methods::lattices::{Tree, TreeLatticeImpl};

        let m = make_g2(Handle::new(flat(0.04)));
        let grid = TimeGrid::new(1.5, 6).unwrap();
        let lattice = m.borrow().tree(grid.clone()).unwrap();
        let g = m.borrow();
        let impl_tree = lattice.implementation();
        let product = impl_tree.two_factor_tree();

        for i in 0..grid.size() - 1 {
            for index in 0..product.size(i) {
                let (x, y) = product.state(i, index);
                let expected = (-(g.phi(grid[i]) + x + y) * grid.dt(i)).exp();
                let got = impl_tree.discount(i, index);
                assert!(
                    (got - expected).abs() < 1e-14,
                    "discount({i},{index}): {got} != {expected}"
                );
            }
        }

        // Root node is (0,0): discount depends on φ only.
        let root = (-g.phi(grid[0]) * grid.dt(0)).exp();
        assert!((impl_tree.discount(0, 0) - root).abs() < 1e-15);
        let (x0, y0) = product.state(0, 0);
        assert!(close(x0, 0.0) && close(y0, 0.0));
    }

    #[test]
    fn tree_state_price_mass_discounts() {
        // Analytic φ does not force a HW-style exact curve reprice on the
        // discrete tree (unlike numerically fitted Hull–White). Pin that the
        // Arrow-Debreu mass is finite/positive and declines with maturity.
        // Individual node weights may go slightly negative under large |ρ|
        // because the Hull–White correlation correction is not a probability
        // simplex — that matches QuantLib's lattice2d weights.
        let m = make_g2(Handle::new(flat(0.05)));
        let grid = TimeGrid::new(2.0, 10).unwrap();
        let lattice = m.borrow().tree(grid.clone()).unwrap();
        let mut prev_sum = f64::INFINITY;
        for i in 0..grid.size() {
            let sum: Real = lattice.state_prices(i).iter().sum();
            assert!(
                sum > 0.0 && sum.is_finite(),
                "bad state-price mass at {i}: {sum}"
            );
            if i > 0 {
                assert!(
                    sum < prev_sum,
                    "state-price mass should decline: slice {i} {sum} >= {prev_sum}"
                );
            }
            prev_sum = sum;
        }
    }
}
