//! QD+ American approximation (Li 2009), the exercise-boundary guess used by
//! QuantLib's fixed-point American engine.
//!
//! Port of `ql/pricingengines/vanilla/qdplusamericanengine.{hpp,cpp}`. A call
//! is priced as the put with spot and strike swapped and the rates exchanged
//! (Andersen–Lake put-call symmetry). The double-boundary case `q < r < 0`
//! is rejected, as in QuantLib 1.43.
//!
//! Divergence: a non-positive year fraction returns intrinsic. QuantLib 1.43
//! divides the zero log-discount by that year fraction; the option is already
//! expired in the instrument layer, so the suite never reaches the division.
//! The price is then floored at immediate exercise, matching QuantLib 1.44.
//! An exercise that pays at expiry is rejected, as on the other American
//! approximations in this crate.

use std::cell::Cell;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{OneAssetOptionEngine, OneAssetOptionResults, OptionArguments};
use crate::math::comparison::{close, close_enough};
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::math::integrals::Integrator;
use crate::math::interpolations::chebyshev::{ChebyshevInterpolation, ChebyshevKind};
use crate::math::solver1d::{DerivativeSolver, Solver1D, func1d};
use crate::math::solvers1d::brent::Brent;
use crate::math::solvers1d::newtonsafe::NewtonSafe;
use crate::math::solvers1d::ridder::Ridder;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Rate, Real, Size, Time, Volatility};

/// Root finder used for the QD+ exercise boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverType {
    /// Brent with a bisection fallback.
    Brent,
    /// Newton–Raphson, handing off to Newton-safe when a step leaves the bracket.
    Newton,
    /// Ridder's method.
    Ridder,
    /// Halley's method, with Brent as a fallback.
    Halley,
    /// Super-Halley, with the same Brent fallback.
    SuperHalley,
}

/// American engine based on the QD+ approximation to the exercise boundary.
///
/// Reference: Li, M. (2009), "Analytical Approximations for the Critical
/// Stock Prices of American Options: A Performance Comparison".
pub struct QdPlusAmericanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    interpolation_points: Size,
    solver_type: SolverType,
    eps: Real,
    max_iter: Size,
}

impl QdPlusAmericanEngine {
    /// Halley, 8 Chebyshev nodes, tolerance `1e-6`, and the QuantLib
    /// iteration cap for that solver (10).
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_solver(process, 8, SolverType::Halley, 1e-6, None)
    }

    /// Full constructor. `max_iter = None` selects QuantLib's default: 100 for
    /// Brent, Newton and Ridder, 10 for Halley and Super-Halley.
    pub fn with_solver(
        process: Shared<GeneralizedBlackScholesProcess>,
        interpolation_points: Size,
        solver_type: SolverType,
        eps: Real,
        max_iter: Option<Size>,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        let max_iter = max_iter.unwrap_or(match solver_type {
            SolverType::Newton | SolverType::Brent | SolverType::Ridder => 100,
            SolverType::Halley | SolverType::SuperHalley => 10,
        });
        Self {
            base,
            process,
            interpolation_points,
            solver_type,
            eps,
            max_iter,
        }
    }

    /// Upper exercise boundary `B(0)` from Andersen, Lake and Offengenden.
    ///
    /// # Errors
    ///
    /// Returns an error if `r` and `q` are not a covered sign case (a NaN).
    #[allow(clippy::if_same_then_else, clippy::float_cmp)]
    pub fn x_max(strike: Real, r: Rate, q: Rate) -> QlResult<Real> {
        // Table 2 of Andersen, Lake (2021), "Fast American Option Pricing:
        // The Double-Boundary Case". Exact `== 0.0` matches the C++ branches.
        // Equal results stay on separate branches so each sign case stays visible.
        let boundary = if r > 0.0 && q > 0.0 {
            strike * 1.0_f64.min(r / q)
        } else if r > 0.0 && q <= 0.0 {
            strike
        } else if r == 0.0 && q < 0.0 {
            strike
        } else if r == 0.0 && q >= 0.0 {
            0.0
        } else if r < 0.0 && q >= 0.0 {
            0.0
        } else if r < 0.0 && q < r {
            strike
        } else if r < 0.0 && r <= q && q < 0.0 {
            0.0
        } else {
            fail!("QD+ exercise boundary: unsupported rates r={r}, q={q}");
        };
        Ok(boundary)
    }

    /// Put exercise boundary at time-to-go `tau`, and how many boundary
    /// evaluations the solver used.
    ///
    /// # Errors
    ///
    /// Returns an error if the root finder cannot bracket or exhausts its
    /// evaluation budget.
    #[allow(clippy::too_many_arguments)]
    pub fn put_exercise_boundary_at_tau(
        &self,
        spot: Real,
        strike: Real,
        r: Rate,
        q: Rate,
        vol: Volatility,
        maturity: Time,
        tau: Time,
    ) -> QlResult<(Size, Real)> {
        // `maturity` is part of the QuantLib signature. The boundary at a
        // given `tau` does not depend on it; the Chebyshev grid uses it only
        // to choose the `tau` nodes.
        let _ = maturity;
        if tau < Real::EPSILON {
            return Ok((0, Self::x_max(strike, r, q)?));
        }
        let eval = QdPlusBoundaryEvaluator::new(spot, strike, r, q, vol, tau)?;
        let x = match self.solver_type {
            SolverType::Brent => {
                self.build_in_solver(&eval, BuiltIn::Brent, spot, self.max_iter, None)?
            }
            SolverType::Newton => {
                self.build_in_solver(&eval, BuiltIn::Newton, spot, self.max_iter, None)?
            }
            SolverType::Ridder => {
                self.build_in_solver(&eval, BuiltIn::Ridder, spot, self.max_iter, None)?
            }
            SolverType::Halley | SolverType::SuperHalley => self.halley_boundary(&eval, spot)?,
        };
        Ok((eval.evaluations(), x))
    }

    fn halley_boundary(&self, eval: &QdPlusBoundaryEvaluator, spot: Real) -> QlResult<Real> {
        let xmin = eval.xmin();
        let mut x = eval.xmax();
        let mut fx;
        let x = loop {
            let x_old = x;
            fx = eval.value(x);
            let f_prime = eval.derivative(x);
            let lf = fx * eval.fprime2(x) / (f_prime * f_prime);
            let step = if self.solver_type == SolverType::Halley {
                (1.0 / (1.0 - 0.5 * lf)) * fx / f_prime
            } else {
                (1.0 + 0.5 * lf / (1.0 - lf)) * fx / f_prime
            };
            x = xmin.max(x - step);
            let done = (x - x_old).abs() < 0.5 * self.eps;
            if done || eval.evaluations() >= self.max_iter {
                if !done && !close(fx.abs(), 0.0) {
                    x = self.build_in_solver(
                        eval,
                        BuiltIn::Brent,
                        spot,
                        10 * self.max_iter,
                        Some(x),
                    )?;
                }
                break x;
            }
        };
        Ok(x)
    }

    fn build_in_solver(
        &self,
        eval: &QdPlusBoundaryEvaluator,
        kind: BuiltIn,
        spot: Real,
        max_iter: Size,
        guess: Option<Real>,
    ) -> QlResult<Real> {
        let xmin = eval.xmin();
        let fxmin = eval.value(xmin);
        let mut xmax = (0.5 * (eval.xmax() + spot)).max(eval.xmax());
        while eval.value(xmax) * fxmin > 0.0 && eval.evaluations() < self.max_iter {
            xmax *= 2.0;
        }
        let mut guess = guess.unwrap_or(0.5 * (xmax + spot));
        if guess >= xmax {
            guess = xmax.next_down();
        } else if guess <= xmin {
            guess = xmin.next_up();
        }
        let accuracy = self.eps.max(Real::EPSILON);
        match kind {
            BuiltIn::Brent => {
                let mut solver = Brent::new();
                solver.set_max_evaluations(max_iter);
                solver.set_lower_bound(xmin);
                solver.solve_bracketed(|x| eval.value(x), accuracy, guess, xmin, xmax)
            }
            BuiltIn::Ridder => {
                let mut solver = Ridder::new();
                solver.set_max_evaluations(max_iter);
                solver.set_lower_bound(xmin);
                solver.solve_bracketed(|x| eval.value(x), accuracy, guess, xmin, xmax)
            }
            BuiltIn::Newton => newton_solve(eval, accuracy, guess, xmin, xmax, max_iter),
        }
    }

    fn calculate_put(
        &self,
        spot: Real,
        strike: Real,
        r: Rate,
        q: Rate,
        vol: Volatility,
        maturity: Time,
    ) -> QlResult<Real> {
        if r < 0.0 && q < r {
            fail!("double-boundary case q<r<0 for a put option is given");
        }
        let xmax = Self::x_max(strike, r, q)?;
        require!(xmax > 0.0, "QD+ put exercise boundary is empty");
        let boundary = ChebyshevInterpolation::from_function(
            self.interpolation_points,
            ChebyshevKind::Second,
            |z| {
                let tau = 0.25 * maturity * (1.0 + z) * (1.0 + z);
                let (_, exercise) =
                    self.put_exercise_boundary_at_tau(spot, strike, r, q, vol, maturity, tau)?;
                Ok((exercise / xmax).ln().powi(2))
            },
        )?;
        // QuantLib prefers Boost tanh-sinh and otherwise uses
        // `GaussLobattoIntegral(100000, QL_MAX_REAL, 0.1 * eps)`. The port's
        // tanh-sinh does not reach `eps = 1e-10` on this integrand; Lobatto
        // is the path that reproduces the suite.
        let add_on =
            crate::math::integrals::lobatto::GaussLobattoIntegral::new(100_000, Real::MAX)?
                .with_relative_accuracy(0.1 * self.eps)?
                .integrate(
                    |z| add_on_integrand(z, maturity, spot, strike, r, q, vol, xmax, &boundary),
                    0.0,
                    maturity.sqrt(),
                )?;
        require!(
            add_on > -10.0 * self.eps,
            "negative early exercise value {add_on}"
        );
        let european = european_put(spot, strike, r, q, vol, maturity)?;
        Ok(european + add_on.max(0.0))
    }

    fn calculate_put_with_edge_cases(
        &self,
        spot: Real,
        strike: Real,
        r: Rate,
        q: Rate,
        vol: Volatility,
        maturity: Time,
    ) -> QlResult<Real> {
        if close(strike, 0.0) {
            return Ok(0.0);
        }
        if close(spot, 0.0) {
            return Ok(strike.max(strike * (-r * maturity).exp()));
        }
        if r <= 0.0 && r <= q {
            return european_put(spot, strike, r, q, vol, maturity);
        }
        if close(vol, 0.0) {
            let intrinsic = |t: Time| (strike * (-r * t).exp() - spot * (-q * t).exp()).max(0.0);
            let npv0 = intrinsic(0.0);
            let npv_t = intrinsic(maturity);
            let extrem_t = if close_enough(r, q) {
                Real::MAX
            } else {
                (r * strike / (q * spot)).ln() / (r - q)
            };
            if extrem_t > 0.0 && extrem_t < maturity {
                return Ok(npv0.max(npv_t).max(intrinsic(extrem_t)));
            }
            return Ok(npv0.max(npv_t));
        }
        self.calculate_put(spot, strike, r, q, vol, maturity)
    }
}

fn european_put(
    spot: Real,
    strike: Real,
    r: Rate,
    q: Rate,
    vol: Volatility,
    maturity: Time,
) -> QlResult<Real> {
    let black = BlackCalculator::new(
        OptionType::Put,
        strike,
        spot * ((r - q) * maturity).exp(),
        vol * maturity.sqrt(),
        (-r * maturity).exp(),
    )?;
    Ok(black.value().max(0.0))
}

#[allow(clippy::too_many_arguments)]
fn add_on_integrand(
    z: Real,
    maturity: Time,
    spot: Real,
    strike: Real,
    r: Rate,
    q: Rate,
    vol: Volatility,
    xmax: Real,
    boundary: &ChebyshevInterpolation,
) -> Real {
    let t = z * z;
    let q_node = boundary.value(2.0 * ((maturity - t).max(0.0) / maturity).sqrt() - 1.0);
    let b_t = xmax * (-q_node.max(0.0).sqrt()).exp();
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    let v = vol * t.sqrt();
    let phi = CumulativeNormalDistribution::standard();
    if v >= Real::EPSILON {
        if b_t > Real::EPSILON {
            let dp = (spot * df_q / (b_t * df_r)).ln() / v + 0.5 * v;
            2.0 * z * (r * strike * df_r * phi.value(-dp + v) - q * spot * df_q * phi.value(-dp))
        } else {
            0.0
        }
    } else if close_enough(spot * df_q, b_t * df_r) {
        z * (r * strike * df_r - q * spot * df_q)
    } else if b_t * df_r > spot * df_q {
        2.0 * z * (r * strike * df_r - q * spot * df_q)
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
enum BuiltIn {
    Brent,
    Newton,
    Ridder,
}

/// QuantLib's `Newton`, including the hand-off to Newton-safe when a step
/// leaves the bracket (`newton.hpp`). The shared [`Newton`](crate::math::solvers1d::newton::Newton)
/// reports that step as an error instead.
fn newton_solve(
    eval: &QdPlusBoundaryEvaluator,
    accuracy: Real,
    mut root: Real,
    xmin: Real,
    xmax: Real,
    max_evaluations: Size,
) -> QlResult<Real> {
    let mut froot = eval.value(root);
    let mut dfroot = eval.derivative(root);
    let mut evaluations = 1usize;
    while evaluations <= max_evaluations {
        let dx = froot / dfroot;
        root -= dx;
        if (xmin - root) * (root - xmax) < 0.0 {
            let safe = NewtonSafe::new()
                .with_max_evaluations(max_evaluations.saturating_sub(evaluations).max(1));
            return safe.solve_bracketed(
                func1d(|x| eval.value(x), |x| eval.derivative(x)),
                accuracy,
                root + dx,
                xmin,
                xmax,
            );
        }
        if dx.abs() < accuracy {
            let _ = eval.value(root);
            return Ok(root);
        }
        froot = eval.value(root);
        dfroot = eval.derivative(root);
        evaluations += 1;
    }
    fail!("maximum number of function evaluations ({max_evaluations}) exceeded")
}

#[derive(Clone, Copy)]
struct BoundaryCache {
    sc: Real,
    dp: Real,
    dm: Real,
    phi_minus_dp: Real,
    phi_dp: Real,
    npv: Real,
    theta: Real,
    charm: Real,
    evaluations: usize,
}

struct QdPlusBoundaryEvaluator {
    tau: Time,
    strike: Real,
    sigma2: Real,
    v: Real,
    r: Rate,
    q: Rate,
    df_r: Real,
    df_q: Real,
    lambda: Real,
    alpha: Real,
    beta: Real,
    x_max: Real,
    x_min: Real,
    phi: NormalDistribution,
    big_phi: CumulativeNormalDistribution,
    cache: Cell<BoundaryCache>,
}

impl QdPlusBoundaryEvaluator {
    fn new(
        spot: Real,
        strike: Real,
        r: Rate,
        q: Rate,
        vol: Volatility,
        tau: Time,
    ) -> QlResult<Self> {
        let sigma2 = vol * vol;
        let v = vol * tau.sqrt();
        let df_r = (-r * tau).exp();
        let df_q = (-q * tau).exp();
        let ddr = if (r * tau).abs() > 1e-5 {
            r / (1.0 - df_r)
        } else {
            1.0 / (tau * (1.0 - 0.5 * r * tau * (1.0 - r * tau / 3.0)))
        };
        let omega = 2.0 * (r - q) / sigma2;
        let disc = (omega - 1.0) * (omega - 1.0) + 8.0 * ddr / sigma2;
        let root = disc.sqrt();
        let lambda = 0.5 * (-(omega - 1.0) - root);
        let lambda_prime = 2.0 * ddr * ddr / (sigma2 * root);
        let alpha = 2.0 * df_r / (sigma2 * (2.0 * lambda + omega - 1.0));
        let beta = alpha * (ddr + lambda_prime / (2.0 * lambda + omega - 1.0)) - lambda;
        let x_max = QdPlusAmericanEngine::x_max(strike, r, q)?;
        let x_min = Real::EPSILON * 1e4 * (0.5 * (strike + spot)).min(x_max);
        Ok(Self {
            tau,
            strike,
            sigma2,
            v,
            r,
            q,
            df_r,
            df_q,
            lambda,
            alpha,
            beta,
            x_max,
            x_min,
            phi: NormalDistribution::standard(),
            big_phi: CumulativeNormalDistribution::standard(),
            cache: Cell::new(BoundaryCache {
                sc: Real::NAN,
                dp: 0.0,
                dm: 0.0,
                phi_minus_dp: 0.0,
                phi_dp: 0.0,
                npv: 0.0,
                theta: 0.0,
                charm: 0.0,
                evaluations: 0,
            }),
        })
    }

    fn xmin(&self) -> Real {
        self.x_min
    }

    fn xmax(&self) -> Real {
        self.x_max
    }

    fn evaluations(&self) -> Size {
        self.cache.get().evaluations
    }

    fn prepare(&self, spot: Real) {
        let mut cache = self.cache.get();
        #[allow(clippy::float_cmp)]
        if spot == cache.sc {
            return;
        }
        let spot_c = spot.max(Real::EPSILON);
        let dp = (spot_c * self.df_q / (self.strike * self.df_r)).ln() / self.v + 0.5 * self.v;
        let dm = dp - self.v;
        let phi_minus_dp = self.big_phi.value(-dp);
        let phi_minus_dm = self.big_phi.value(-dm);
        let phi_dp = self.phi.value(dp);
        let npv = self.df_r * self.strike * phi_minus_dm - spot_c * self.df_q * phi_minus_dp;
        let theta = self.r * self.strike * self.df_r * phi_minus_dm
            - self.q * spot_c * self.df_q * phi_minus_dp
            - self.sigma2 * spot_c / (2.0 * self.v) * self.df_q * phi_dp;
        let charm = -self.df_q
            * (phi_dp * ((self.r - self.q) / self.v - dm / (2.0 * self.tau))
                + self.q * phi_minus_dp);
        cache.sc = spot_c;
        cache.dp = dp;
        cache.dm = dm;
        cache.phi_minus_dp = phi_minus_dp;
        cache.phi_dp = phi_dp;
        cache.npv = npv;
        cache.theta = theta;
        cache.charm = charm;
        self.cache.set(cache);
    }

    fn value(&self, spot: Real) -> Real {
        let mut cache = self.cache.get();
        cache.evaluations += 1;
        self.cache.set(cache);
        self.prepare(spot);
        let cache = self.cache.get();
        let alive = 1.0 - self.df_q * cache.phi_minus_dp;
        if close_enough(self.strike - spot, cache.npv) {
            alive * spot + self.alpha * cache.theta / self.df_r
        } else {
            let c0 = -self.beta - self.lambda
                + self.alpha * cache.theta / (self.df_r * (self.strike - spot - cache.npv));
            alive * spot + (self.lambda + c0) * (self.strike - spot - cache.npv)
        }
    }

    fn derivative(&self, spot: Real) -> Real {
        self.prepare(spot);
        let cache = self.cache.get();
        1.0 - self.df_q * cache.phi_minus_dp
            + self.df_q / self.v * cache.phi_dp
            + self.beta * (1.0 - self.df_q * cache.phi_minus_dp)
            + self.alpha / self.df_r * cache.charm
    }

    fn fprime2(&self, spot: Real) -> Real {
        self.prepare(spot);
        let cache = self.cache.get();
        let gamma = cache.phi_dp * self.df_q / (self.v * spot);
        let colour = gamma
            * (self.q
                + (self.r - self.q) * cache.dp / self.v
                + (1.0 - cache.dp * cache.dm) / (2.0 * self.tau));
        self.df_q
            * (cache.phi_dp / (spot * self.v) - cache.phi_dp * cache.dp / (spot * self.v * self.v))
            + self.beta * gamma
            + self.alpha / self.df_r * colour
    }
}

impl AsObservable for QdPlusAmericanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QdPlusAmericanEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("no exercise given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::American,
            "not an American option"
        );
        require!(!exercise.payoff_at_expiry(), "payoff at expiry not handled");
        let Some(payoff) = args.payoff.as_ref() else {
            fail!("non-striked payoff given");
        };
        let maturity = exercise.last_date();
        let t = self.process.time(&maturity)?;
        let spot = self.process.x0()?;
        require!(spot >= 0.0, "negative underlying given");
        let strike = payoff.strike();
        require!(strike >= 0.0, "zero or positive strike is required");

        let value = if t <= 0.0 {
            match payoff.option_type() {
                OptionType::Put => (strike - spot).max(0.0),
                OptionType::Call => (spot - strike).max(0.0),
            }
        } else {
            let vol = self
                .process
                .black_volatility()
                .current_link()?
                .black_vol(t, strike, false)?;
            require!(vol >= 0.0, "zero or positive volatility is required");
            let r = -self
                .process
                .risk_free_rate()
                .current_link()?
                .discount_date(maturity, false)?
                .ln()
                / t;
            let q = -self
                .process
                .dividend_yield()
                .current_link()?
                .discount_date(maturity, false)?
                .ln()
                / t;
            match payoff.option_type() {
                OptionType::Put => {
                    self.calculate_put_with_edge_cases(spot, strike, r, q, vol, t)?
                }
                OptionType::Call => {
                    self.calculate_put_with_edge_cases(strike, spot, q, r, vol, t)?
                }
            }
        };
        let value = value.max(payoff.value(spot));
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::AmericanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(1, Month::June, 2022)
    }

    struct Market {
        settings: Shared<Settings<Date>>,
        spot: Shared<SimpleQuote>,
        q_rate: Shared<SimpleQuote>,
        r_rate: Shared<SimpleQuote>,
        vol: Shared<SimpleQuote>,
        process: Shared<GeneralizedBlackScholesProcess>,
    }

    impl Market {
        fn new() -> Self {
            let settings = shared(Settings::new());
            settings.set_evaluation_date(today());
            let spot = shared(SimpleQuote::new(1.0));
            let q_rate = shared(SimpleQuote::new(0.0));
            let r_rate = shared(SimpleQuote::new(0.0));
            let vol = shared(SimpleQuote::new(0.0));
            let quote =
                |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
            let flat = |q: &Shared<SimpleQuote>| {
                Handle::new(shared(FlatForward::new(
                    today(),
                    quote(q),
                    Actual365Fixed::new(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>)
            };
            let process = shared(BlackScholesMertonProcess::new(
                quote(&spot),
                flat(&q_rate),
                flat(&r_rate),
                Handle::new(shared(BlackConstantVol::with_quote(
                    today(),
                    None,
                    quote(&vol),
                    Actual365Fixed::new(),
                )) as Shared<dyn BlackVolTermStructure>),
            ));
            Self {
                settings,
                spot,
                q_rate,
                r_rate,
                vol,
                process,
            }
        }

        fn set(&self, spot: Real, q: Rate, r: Rate, vol: Volatility) {
            self.spot.set_value(spot);
            self.q_rate.set_value(q);
            self.r_rate.set_value(r);
            self.vol.set_value(vol);
        }
    }

    fn engine(
        market: &Market,
        points: Size,
        solver: SolverType,
        eps: Real,
    ) -> QdPlusAmericanEngine {
        QdPlusAmericanEngine::with_solver(Shared::clone(&market.process), points, solver, eps, None)
    }

    fn price(
        market: &Market,
        kind: OptionType,
        strike: Real,
        days: i32,
        points: Size,
        eps: Real,
    ) -> Real {
        let exercise = shared(AmericanExercise::over(today(), today() + days).unwrap());
        let payoff = shared(PlainVanillaPayoff::new(kind, strike));
        let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&market.settings));
        option
            .base_mut()
            .set_pricing_engine(shared_mut(QdPlusAmericanEngine::with_solver(
                Shared::clone(&market.process),
                points,
                SolverType::Halley,
                eps,
                None,
            )) as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    #[test]
    fn qd_plus_boundary_values() {
        let market = Market::new();
        let pricer = engine(&market, 10, SolverType::Halley, 1e-6);
        let cases = [
            (4.9, 87.769_609_499_653_87),
            (4.0, 88.390_530_036_146_12),
            (2.5, 90.143_273_157_622_56),
            (1.0, 94.497_938_030_959_84),
            (0.1, 106.258_896_444_233_8),
        ];
        for (tau, expected) in cases {
            let (evals, boundary) = pricer
                .put_exercise_boundary_at_tau(100.0, 120.0, 0.1, 0.03, 0.25, 5.0, tau)
                .unwrap();
            assert!(
                (boundary - expected).abs() <= 1e-12,
                "tau={tau}: {boundary} vs {expected}"
            );
            assert!(evals <= 10, "tau={tau}: {evals} evaluations");
        }
    }

    #[test]
    fn qd_plus_boundary_convergence() {
        let market = Market::new();
        let cases = [
            (0.10, 0.03, 120.0, 2000usize),
            (0.0001, 0.03, 120.0, 2000),
            (0.0001, 0.000002, 120.0, 2000),
            (0.01, 0.75, 120.0, 2000),
            (0.03, 0.0, 30.0, 2000),
            (0.03, 0.0, 1e7, 2500),
            (0.075, 0.0, 1e-8, 2000),
        ];
        let solvers = [
            SolverType::Brent,
            SolverType::Newton,
            SolverType::Ridder,
            SolverType::Halley,
            SolverType::SuperHalley,
        ];
        for (r, q, strike, max_evals) in cases {
            for solver in solvers {
                let pricer = engine(&market, 8, solver, 1e-8);
                let mut evaluations = 0usize;
                let mut tau = 0.0;
                while tau < 10.0 {
                    let (used, _) = pricer
                        .put_exercise_boundary_at_tau(100.0, strike, r, q, 0.25, 10.0, tau)
                        .unwrap();
                    evaluations += used;
                    tau += 0.1;
                }
                let cap = match solver {
                    SolverType::Halley | SolverType::SuperHalley => 750,
                    _ => max_evals,
                };
                assert!(
                    evaluations <= cap,
                    "{solver:?} r={r} q={q} K={strike}: {evaluations} > {cap}"
                );
            }
        }
    }

    #[test]
    fn qd_plus_edge_values() {
        let market = Market::new();
        // `americanoption.cpp` `testQdAmericanEngines` high-precision edges.
        #[allow(clippy::excessive_precision)]
        let cases = [
            (
                Put,
                100.0,
                120.0,
                3650,
                0.25,
                0.10,
                0.03,
                22.97383256003585,
                1e-8,
            ),
            (
                Call,
                120.0,
                100.0,
                3650,
                0.25,
                0.03,
                0.10,
                22.97383256003585,
                1e-8,
            ),
            (Put, 100.0, 0.0, 365, 0.25, 0.02, 0.02, 0.0, 1e-14),
            (Put, 100.0, 1e-8, 365, 0.25, 0.02, 0.02, 0.0, 1e-14),
            (Call, 100.0, 0.0, 365, 0.25, 0.05, 0.01, 100.0, 1e-11),
            (Call, 100.0, 1e-7, 365, 0.25, 0.05, 0.01, 100.0 - 1e-7, 1e-9),
            (
                Call,
                100.0,
                50.0,
                365,
                0.0,
                0.05,
                0.01,
                51.4435121498811085,
                1e-10,
            ),
            (
                Call,
                100.0,
                50.0,
                365,
                1e-8,
                0.05,
                0.01,
                51.4435121498811156,
                1e-8,
            ),
            (
                Put,
                100.0,
                120.0,
                4 * 3650,
                1e-6,
                0.01,
                0.50,
                108.980920365700442,
                1e-4,
            ),
            (
                Put,
                100.0,
                120.0,
                4 * 3650,
                0.0,
                0.01,
                0.50,
                108.980904561184602,
                1e-10,
            ),
            (Put, 100.0, 120.0, 365, 1e-7, 0.05, 0.01, 20.0, 1e-9),
            (Put, 100.0, 120.0, 365, 0.0, 0.05, 0.01, 20.0, 1e-12),
            (
                Put,
                100.0,
                120.0,
                365,
                1e-7,
                0.00,
                0.05,
                24.8770575499286082,
                1e-8,
            ),
            (
                Put,
                100.0,
                120.0,
                365,
                0.0,
                0.00,
                0.05,
                24.8770575499286082,
                1e-10,
            ),
            (
                Put,
                1e-6,
                120.0,
                365,
                0.25,
                -0.075,
                0.05,
                129.346097154926355,
                1e-9,
            ),
            (
                Put,
                0.0,
                120.0,
                365,
                0.25,
                -0.075,
                0.05,
                129.346098106155779,
                1e-10,
            ),
            (Call, 1e-6, 120.0, 365, 0.25, 0.075, 0.05, 0.0, 1e-14),
            (Call, 0.0, 120.0, 365, 0.25, 0.075, 0.05, 0.0, 1e-14),
            (Put, 100.0, 120.0, 1, 0.25, 0.05, 0.0, 20.0, 1e-10),
            (Put, 100.0, 120.0, 0, 0.25, 0.05, 0.0, 0.0, 1e-14),
            (Put, 0.0, 0.0, 365, 0.0, 0.0, 0.0, 0.0, 1e-14),
            (
                Call,
                100.0,
                1e-7,
                365,
                1e-8,
                0.05,
                0.025,
                100.0 - 1e-7,
                1e-8,
            ),
            (Call, 100.0, 0.0, 365, 1e-8, 0.05, 0.025, 100.0, 1e-8),
            (Call, 100.0, 1e-7, 365, 0.0, 0.05, 0.025, 100.0 - 1e-7, 1e-8),
            (Call, 100.0, 0.0, 365, 0.0, 0.05, 0.025, 100.0, 1e-8),
            (Call, 1e-8, 100.0, 365, 1e-8, 0.05, 0.025, 0.0, 1e-10),
            (Call, 0.0, 100.0, 365, 0.0, 0.05, 0.025, 0.0, 1e-14),
            (
                Call,
                100.0,
                100.0,
                365,
                0.25,
                0.0,
                0.025,
                8.871505915120776,
                1e-8,
            ),
            (
                Call,
                100.0,
                100.0,
                365,
                0.25,
                0.05,
                0.0,
                12.3359989303687243,
                1e-8,
            ),
            (Call, 1e10, 100.0, 365, 0.25, 0.01, 0.05, 1e10 - 100.0, -1.0),
            (Call, 100.0, 1e10, 365, 0.25, 0.01, 0.05, 0.0, 1e-14),
            (
                Call,
                100.0,
                100.0,
                365,
                100.0,
                0.01,
                0.05,
                99.9874942266127,
                1e-8,
            ),
            (
                Call,
                100.0,
                100.0,
                365,
                0.25,
                0.10,
                10.0,
                0.1159627202107989,
                1e-8,
            ),
            (
                Call,
                100.0,
                100.0,
                170 * 365,
                0.25,
                0.01,
                0.002,
                80.37468392429741,
                1e-8,
            ),
        ];
        for (kind, spot, strike, days, vol, r, q, expected, precision) in cases {
            market.set(spot, q, r, vol);
            let calculated = price(&market, kind, strike, days, 8, 1e-10);
            let err = (calculated - expected).abs();
            let ok = if precision > 0.0 {
                err <= precision
            } else if expected > 0.1 {
                err / expected <= 0.005
            } else {
                err <= 5e-4
            };
            assert!(
                ok,
                "{kind:?} S={spot} K={strike} T={days} r={r} q={q} vol={vol}: {calculated} vs {expected} (err={err})"
            );
        }
    }

    /// Cached PDE prices from `testQdAmericanEngines`, drawn with
    /// `InverseCumulativeRng<MersenneTwisterUniformRng, InverseCumulativeNormal>`
    /// seeded at 12345.
    #[rustfmt::skip]
    const PDE_VALUES: &[Real] = &[
        581.46895,113.78442,581.44547,1408.579,49.19448,1060.27367,
        834.83366,176.48305,120.38008,307.11264,602.7006,233.80171,
        204.74596,0.30987,0.0,0.0,5.36215,0.01711,0.0,84.51193,0.67131,
        0.06414,152.67188,54.75257,90.31861,168.50289,18.38926,0.0,
        282.4995,0.0,0.08428,12.30929,42.26359,139.87748,0.28724,0.00421,
        0.0,0.00206,0.0,658.60427,140.51139,23.17387,0.35612,0.0,909.14828,
        0.0,0.11549,5.46749,144.25428,2576.6754,562.16484,0.0,122.725,
        383.48463,278.7447,3.52566,82.34348,81.06139,0.0,10.42824,
        4.95917,25.28602,31.38869,3.53697,0.0,0.012,0.0,0.4263,162.16184,
        0.4618,97.714,283.03442,0.38176,70.25367,134.94142,2.19293,
        226.4746,76.74309,46.03123,15.76214,0.01666,1806.26208,0.0,
        103.93726,6.82956,337.81301,0.64236,677.63248,25.01763,
        443.79052,1793.78327,118.6293,185.79849,11.59313,679.01736,
        17.99005,403.57554,1.67418,0.0,0.03795,3326.09089,71.1996,
        0.0,485.10353,0.0,1681.25166,0.0,43.15432,0.75825,0.05895,34.71493,
        0.00015,5.58671,115.98793,37.7713,399.24494,0.00766,445.42207,
        152.65397,0.0,47.05874,0.96921,14.21875,257.84754,109.62533,
        2553.99295,138.46663,192.33614,81.41877,18.21403,113.926,
        27.28409,174.77093,42.70527,0.90326,0.0,967.9901,616.0143,
        253.56442,0.00397,2493.82098,9.29406,11.00023,0.0,0.0,234.12481,
        0.0,72.46356,0.0,9.00932,48.67934,29.42756,13.4271,0.0,0.0,0.0,0.0,20.71417,
        48.57474,2.26452,0.0,109.0243,0.0,21.26801,1.21164,0.0,86.25232,
        36.00437,4.53844,7.40503,313.53602,379.76105,165.84347,77.19665,
        9.02466,0.10634,214.84982,6.13387,133.44645,303.25953,0.0,
        134.26724,246.89804,0.0,123.32975,32.83429,9.56819,7.42582,0.0,
        73.82832,196.84831,0.00001,72.70391,2173.8649,123.00513,
        153.83539,21.63003,209.84752,30.12425,0.0,197.6502,0.0,164.02863,
        7.65143,56.57631,2392.70018,0.0,0.0,34.23457,171.08459,0.49387,
        31.13395,237.68801,0.01262,0.0,0.0,0.0,0.0,41.56635,0.0,8.41535,55.01775,
        310.50094,0.0,14.85456,174.34018,7.19772,0.00001,0.0,91.70874,
        0.00001,17.51724,0.00587,0.0,532.24902,2.05553,36.80843,0.0,
        33.39288,0.00006,0.04439,1.3434,0.0,0.41816,926.37642,0.0,247.61559,
        151.98965,0.35243,4.33198,23294.47744,0.00791,12.51996,53.47727,
        167.95572,0.0062,6.8482,0.0,347.83408,852.85742,558.21422,0.0,
        53.89293,78.61011,187.3978,9.18927,0.00553,113.48101,1467.30556,
        74.82251,94.84476,0.0,101.3649,59.27007,0.0,773.81251,0.0,542.7889,0.0,
        68.96209,96.0435,0.00004,0.10738,0.00187,324.97758,245.68455,
        30.52818,129.84472,0.0,46.86288,368.41675,139.29763,4.4393,16.29594,
        25.7554,64.02621,89.41363,0.62751,219.65237,0.26039,0.0,12.02172,
        101.97733,69.37456,45.81122,1263.33603,164.31607,15.88788,0.0,48.77797,
        0.13233,147.16808,10.31217,7.50634,7.48611,177.95409,225.77562,3.56947,
        0.02531,4.88869,8.76632,0.0,0.0,0.02214,305.08468,44.52185,182.17332,
        538.31458,0.0,46.97229,0.0,31.94202,410.43038,0.0,70.35432,15.58346,74.14177,
        953.67663,11.79128,59.83061,0.0,37.86557,1184.22731,2411.37823,0.0,0.0,0.0,0.0,
        49.3179,236.38654,21.36225,218.048,517.57006,0.0,0.0,12.52933,256.71967,
        0.00025,1.47981,158.19166,0.0,1923.70709,4.94441,1199.81196,45.92353,
        85.73255,14.91338,88.81459,21.42459,3456.9466,31.97838,233.26863,
        49.34801,2684.07758,0.0,0.0,32.24149,0.0,111.79552,0.00506,8.77602,0.0,
        406.54213,0.32974,365.53998,1.49714,19.65603,37.33877,205.06928,
        0.01805,589.23478,9.58273,0.02946,286.48706,463.34512,528.21392,0.0,
        47.71294,21.0864,114.81771,80.489,21.30905,41.95873,19.03598,156.09295,
        0.0,73.6509,0.0,0.0,168.17576,0.0,32.71243,36.75044,177.64583,0.05618,
        156.38616,1370.4754,24.5976,59.83173,0.0,354.93074,34.96889,0.00532,
        16.95287,1259.72993,241.05777,18.9778,0.57635,43.98093,25.2678,
        369.39896,0.31549,0.0,31.95512,101.60559,11.22079,970.16273,0.0,0.0,
        1.55445,0.0,18.6067,0.0,1124.20117,52.67762,10.38273,0.0,10.22588,251.27813,
        0.0,431.82244,0.0,1.31252,0.0,84.72154,100.98411,160.95557,129.51372,
        0.00026,103.81663,421.64767,0.00031,0.0,104.48529,162.59225,0.0,
        1504.0869,88.11253,4.14052,0.07195,203.78754,0.00002,42.5395,0.0,
        17.05087,26.89157,64.64923,0.0,390.87453,124.55406,0.01018,94.23963,
    ];

    #[test]
    fn qd_plus_pde_cache() {
        use crate::math::distributions::normal::InverseCumulativeNormal;
        use crate::math::randomnumbers::UniformRng;
        use crate::math::randomnumbers::mt19937uniformrng::MersenneTwisterUniformRng;

        let market = Market::new();
        let mut rng = MersenneTwisterUniformRng::new(12_345);
        let gaussian = InverseCumulativeNormal::standard();
        let mut next = || gaussian.value(rng.next_real()).unwrap();
        for &expected in PDE_VALUES {
            let kind = if next() > 0.0 { Call } else { Put };
            let spot = 100.0 * (1.5 * next()).exp();
            let strike = 100.0 * (1.5 * next()).exp();
            let days = (1.0 + 365.0 * (2.0 * next()).exp()) as i32;
            let vol = 0.5 * next().exp();
            let r = 0.10 * next().exp();
            let q = 0.10 * next().exp();
            market.set(spot, q, r, vol);
            let calculated = price(&market, kind, strike, days, 8, 1e-10);
            let err = (calculated - expected).abs();
            let ok = if expected > 0.1 {
                err / expected <= 0.005
            } else {
                err <= 5e-4
            };
            assert!(
                ok,
                "{kind:?} S={spot} K={strike} T={days} r={r} q={q} vol={vol}: {calculated} vs {expected} (err={err})"
            );
        }
    }

    #[test]
    fn qd_plus_negative_dividend_yield() {
        use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
        use crate::pricingengines::FdBlackScholesVanillaEngine;
        use crate::time::period::Period;
        use crate::time::timeunit::TimeUnit;

        let today = Date::new(5, Month::December, 2022);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let quote = |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let flat = |q: &Shared<SimpleQuote>| {
            Handle::new(shared(FlatForward::new(
                today,
                quote(q),
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let process = shared(BlackScholesMertonProcess::new(
            quote(&spot),
            flat(&q_rate),
            flat(&r_rate),
            Handle::new(shared(BlackConstantVol::with_quote(
                today,
                None,
                Handle::new(shared(SimpleQuote::new(0.4)) as Shared<dyn Quote>),
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let maturity = today + Period::new(18, TimeUnit::Months);
        let exercise = shared(AmericanExercise::over(today, maturity).unwrap());
        let payoff = shared(PlainVanillaPayoff::new(Put, 90.0));
        let mut option = VanillaOption::new(payoff, exercise, settings);
        let qd = shared_mut(QdPlusAmericanEngine::new(Shared::clone(&process)));
        let fd = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&process),
            Vec::new(),
            800,
            200,
            0,
            FdmSchemeDesc::douglas(),
        ));
        for r in [0.025, 0.5, 0.0, 1e-6] {
            r_rate.set_value(r);
            for q in [-0.05, -0.1, -0.5, 0.0] {
                q_rate.set_value(q);
                option
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&qd) as SharedMut<dyn PricingEngine>);
                let qd_npv = option.npv().unwrap();
                option
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&fd) as SharedMut<dyn PricingEngine>);
                let fd_npv = option.npv().unwrap();
                let diff = (fd_npv - qd_npv).abs();
                assert!(
                    diff <= 7.5e-2,
                    "r={r} q={q}: QD+ {qd_npv} vs FD {fd_npv} (diff={diff})"
                );
            }
        }
    }

    #[test]
    fn qd_plus_is_not_below_exercise() {
        use crate::time::period::Period;
        use crate::time::timeunit::TimeUnit;

        let today = Date::new(15, Month::August, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.2));
        let quote = |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let flat = |q: &Shared<SimpleQuote>| {
            Handle::new(shared(FlatForward::new(
                today,
                quote(q),
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let process = shared(BlackScholesMertonProcess::new(
            quote(&spot),
            flat(&q_rate),
            flat(&r_rate),
            Handle::new(shared(BlackConstantVol::with_quote(
                today,
                None,
                quote(&vol),
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let maturity = today + Period::new(10, TimeUnit::Years);
        let cases = [
            (Call, 60.0, -0.10, 0.06, 0.40, 40.0),
            (Call, 80.0, -0.05, 0.06, 0.20, 20.0),
            (Put, 140.0, 0.06, 0.0, 0.20, 40.0),
        ];
        for (kind, strike, r, q, sigma, intrinsic) in cases {
            spot.set_value(100.0);
            r_rate.set_value(r);
            q_rate.set_value(q);
            vol.set_value(sigma);
            let exercise = shared(AmericanExercise::over(today, maturity).unwrap());
            let payoff = shared(PlainVanillaPayoff::new(kind, strike));
            let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&settings));
            option
                .base_mut()
                .set_pricing_engine(
                    shared_mut(QdPlusAmericanEngine::new(Shared::clone(&process)))
                        as SharedMut<dyn PricingEngine>,
                );
            let calculated = option.npv().unwrap();
            assert!(
                calculated + 1e-12 >= intrinsic,
                "{kind:?} K={strike}: {calculated} below exercise {intrinsic}"
            );
        }
    }

    #[test]
    fn qd_plus_payoff_at_expiry_is_rejected() {
        let market = Market::new();
        market.set(100.0, 0.03, 0.10, 0.25);
        let exercise = shared(AmericanExercise::new(today(), today() + 365, true).unwrap());
        let payoff = shared(PlainVanillaPayoff::new(Put, 100.0));
        let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&market.settings));
        option
            .base_mut()
            .set_pricing_engine(shared_mut(QdPlusAmericanEngine::new(Shared::clone(
                &market.process,
            ))) as SharedMut<dyn PricingEngine>);
        let err = option.npv().unwrap_err();
        assert!(
            err.message().contains("payoff at expiry not handled"),
            "unexpected error: {}",
            err.message()
        );
    }
}
