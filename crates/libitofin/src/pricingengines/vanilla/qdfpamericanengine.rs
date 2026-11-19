//! Fixed-point QD American engine.
//!
//! Port of `ql/pricingengines/vanilla/qdfpamericanengine.{hpp,cpp}`. The
//! exercise boundary starts from [`QdPlusAmericanEngine`] and is refined by
//! Jacobi-Newton and naive fixed-point steps (Andersen, Lake and Offengenden,
//! 2015). A call is the put with spot and strike swapped. The double-boundary
//! case `q < r < 0` is rejected.
//!
//! The tanh-sinh integrators in QuantLib's accurate and high-precision
//! schemes are the Gauss-Lobatto fallback (`100000` evaluations, relative
//! `0.1 * eps`), the same substitution the QD+ engine uses. The price is
//! floored at immediate exercise, and an exercise that pays at expiry is
//! rejected, matching [`QdPlusAmericanEngine`].

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{OneAssetOptionEngine, OneAssetOptionResults, OptionArguments};
use crate::math::comparison::close_enough;
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::math::integrals::Integrator;
use crate::math::integrals::gaussianquadratures::{
    GaussianQuadrature, GaussianQuadratureIntegrator,
};
use crate::math::integrals::lobatto::GaussLobattoIntegral;
use crate::math::interpolations::chebyshev::ChebyshevInterpolation;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::pricingengines::vanilla::qdplusamericanengine::{
    QdPlusAmericanEngine, SolverType, add_on_integrand,
};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Rate, Real, Size, Time, Volatility};

/// Which fixed-point equation refines the exercise boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPointEquation {
    /// Equation A. Also selected by [`Auto`] when `|r - q| < 0.001`.
    A,
    /// Equation B.
    B,
    /// Equation A when `|r - q| < 0.001`, otherwise equation B.
    Auto,
}

/// How the fixed-point integrals and the final price integral are taken.
#[derive(Clone, Copy, Debug)]
pub struct QdFpScheme {
    nodes: Size,
    naive_steps: Size,
    fixed_point: IntegratorSpec,
    price: IntegratorSpec,
}

#[derive(Clone, Copy, Debug)]
enum IntegratorSpec {
    Legendre(Size),
    Lobatto(Real),
}

impl QdFpScheme {
    /// Gauss-Legendre `(l, m, n)-p` scheme.
    ///
    /// # Errors
    ///
    /// Returns an error unless `m > 0` and `n > 0`.
    pub fn legendre(l: Size, m: Size, n: Size, p: Size) -> QlResult<Self> {
        require!(m > 0, "at least one fixed point iteration step is needed");
        require!(n > 0, "at least one interpolation point is needed");
        Ok(Self {
            nodes: n,
            naive_steps: m - 1,
            fixed_point: IntegratorSpec::Legendre(l),
            price: IntegratorSpec::Legendre(p),
        })
    }

    /// Legendre fixed-point steps and a Lobatto price integral of accuracy `eps`.
    ///
    /// # Errors
    ///
    /// As [`legendre`](Self::legendre).
    pub fn legendre_tanh_sinh(l: Size, m: Size, n: Size, eps: Real) -> QlResult<Self> {
        let mut scheme = Self::legendre(l, m, n, 1)?;
        scheme.price = IntegratorSpec::Lobatto(eps);
        Ok(scheme)
    }

    /// Lobatto fixed-point and price integrals of accuracy `eps`.
    ///
    /// # Errors
    ///
    /// As [`legendre`](Self::legendre) for `m` and `n`.
    pub fn tanh_sinh(m: Size, n: Size, eps: Real) -> QlResult<Self> {
        require!(m > 0, "at least one fixed point iteration step is needed");
        require!(n > 0, "at least one interpolation point is needed");
        Ok(Self {
            nodes: n,
            naive_steps: m - 1,
            fixed_point: IntegratorSpec::Lobatto(eps),
            price: IntegratorSpec::Lobatto(eps),
        })
    }

    /// `(l, m, n, p) = (7, 2, 7, 27)`.
    pub fn fast() -> Self {
        Self {
            nodes: 7,
            naive_steps: 1,
            fixed_point: IntegratorSpec::Legendre(7),
            price: IntegratorSpec::Legendre(27),
        }
    }

    /// Legendre-Lobatto `(25, 5, 13)` at `1e-8`.
    pub fn accurate() -> Self {
        Self {
            nodes: 13,
            naive_steps: 4,
            fixed_point: IntegratorSpec::Legendre(25),
            price: IntegratorSpec::Lobatto(1e-8),
        }
    }

    /// Lobatto `(m, n, eps) = (10, 30, 1e-10)`, QuantLib's high-precision scheme.
    pub fn high_precision() -> Self {
        Self {
            nodes: 30,
            naive_steps: 9,
            fixed_point: IntegratorSpec::Lobatto(1e-10),
            price: IntegratorSpec::Lobatto(1e-10),
        }
    }

    /// Chebyshev nodes the scheme asks for. The engine interpolates `nodes + 1`.
    pub fn chebyshev_nodes(&self) -> Size {
        self.nodes
    }

    /// Naive Richardson steps (`m - 1`).
    pub fn naive_steps(&self) -> Size {
        self.naive_steps
    }

    /// Partial Jacobi-Newton steps. QuantLib always takes one.
    pub fn jacobi_steps(&self) -> Size {
        1
    }

    fn integrate_fixed_point<F>(&self, f: F, a: Real, b: Real) -> QlResult<Real>
    where
        F: FnMut(Real) -> Real,
    {
        integrate(self.fixed_point, f, a, b)
    }

    fn integrate_price<F>(&self, f: F, a: Real, b: Real) -> QlResult<Real>
    where
        F: FnMut(Real) -> Real,
    {
        integrate(self.price, f, a, b)
    }
}

fn integrate<F>(spec: IntegratorSpec, f: F, a: Real, b: Real) -> QlResult<Real>
where
    F: FnMut(Real) -> Real,
{
    match spec {
        IntegratorSpec::Legendre(n) => {
            GaussianQuadratureIntegrator::legendre(n)?.integrate(f, a, b)
        }
        IntegratorSpec::Lobatto(eps) => GaussLobattoIntegral::new(100_000, Real::MAX)?
            .with_relative_accuracy(0.1 * eps)?
            .integrate(f, a, b),
    }
}

/// American engine based on fixed-point iteration for the exercise boundary.
pub struct QdFpAmericanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    scheme: QdFpScheme,
    equation: FixedPointEquation,
}

impl QdFpAmericanEngine {
    /// Accurate scheme and [`FixedPointEquation::Auto`].
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_scheme(process, QdFpScheme::accurate(), FixedPointEquation::Auto)
    }

    /// Full constructor.
    pub fn with_scheme(
        process: Shared<GeneralizedBlackScholesProcess>,
        scheme: QdFpScheme,
        equation: FixedPointEquation,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            scheme,
            equation,
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
        let xmax = QdPlusAmericanEngine::x_max(strike, r, q)?;
        require!(xmax > 0.0, "QD+ put exercise boundary is empty");
        let starter = QdPlusAmericanEngine::with_solver(
            Shared::clone(&self.process),
            self.scheme.nodes + 1,
            SolverType::Halley,
            1e-8,
            None,
        );
        let mut interp = starter.exercise_boundary(spot, strike, r, q, vol, maturity)?;
        let z = interp.interpolation_nodes().to_vec();
        let sqrt_t = maturity.sqrt();
        let x: Vec<Real> = z.iter().map(|node| 0.5 * sqrt_t * (1.0 + node)).collect();
        let use_a = self.equation == FixedPointEquation::A
            || (self.equation == FixedPointEquation::Auto && (r - q).abs() < 0.001);
        let fixed_point = match self.scheme.fixed_point {
            IntegratorSpec::Legendre(n) => Some(GaussianQuadrature::legendre(n)?),
            IntegratorSpec::Lobatto(_) => None,
        };

        let mut y = vec![0.0; x.len()];
        for _ in 0..self.scheme.jacobi_steps() {
            for i in 1..x.len() {
                let tau = x[i] * x[i];
                let b = boundary_at(&interp, xmax, maturity, tau);
                let (n_val, d_val, fv) = if use_a {
                    f_a(
                        strike,
                        r,
                        q,
                        vol,
                        tau,
                        b,
                        &interp,
                        xmax,
                        maturity,
                        fixed_point.as_ref(),
                        &self.scheme,
                    )?
                } else {
                    f_b(
                        strike,
                        r,
                        q,
                        vol,
                        tau,
                        b,
                        &interp,
                        xmax,
                        maturity,
                        fixed_point.as_ref(),
                        &self.scheme,
                    )?
                };
                y[i] = if tau < Real::EPSILON {
                    squared_log(fv, xmax)
                } else {
                    let (nd, dd) = if use_a {
                        ndd_a(strike, r, q, vol, tau, b)
                    } else {
                        ndd_b(strike, r, q, vol, tau, b)
                    };
                    let fd = strike
                        * (-(r - q) * tau).exp()
                        * (nd / d_val - dd * n_val / (d_val * d_val));
                    squared_log(b - (fv - b) / (fd - 1.0), xmax)
                };
            }
            interp.update_y(&y)?;
        }
        for _ in 0..self.scheme.naive_steps() {
            for i in 1..x.len() {
                let tau = x[i] * x[i];
                let b = boundary_at(&interp, xmax, maturity, tau);
                let (.., fv) = if use_a {
                    f_a(
                        strike,
                        r,
                        q,
                        vol,
                        tau,
                        b,
                        &interp,
                        xmax,
                        maturity,
                        fixed_point.as_ref(),
                        &self.scheme,
                    )?
                } else {
                    f_b(
                        strike,
                        r,
                        q,
                        vol,
                        tau,
                        b,
                        &interp,
                        xmax,
                        maturity,
                        fixed_point.as_ref(),
                        &self.scheme,
                    )?
                };
                y[i] = squared_log(fv, xmax);
            }
            interp.update_y(&y)?;
        }

        let add_on = self.scheme.integrate_price(
            |z| add_on_integrand(z, maturity, spot, strike, r, q, vol, xmax, &interp),
            0.0,
            sqrt_t,
        )?;
        let european = BlackCalculator::new(
            OptionType::Put,
            strike,
            spot * ((r - q) * maturity).exp(),
            vol * sqrt_t,
            (-r * maturity).exp(),
        )?
        .value();
        Ok(european.max(0.0) + add_on.max(0.0))
    }
}

fn squared_log(value: Real, xmax: Real) -> Real {
    (value / xmax).ln().powi(2)
}

fn boundary_at(interp: &ChebyshevInterpolation, xmax: Real, maturity: Time, tau: Time) -> Real {
    let z = 2.0 * (tau.abs() / maturity).sqrt() - 1.0;
    xmax * (-interp.value(z).max(0.0).sqrt()).exp()
}

fn moneyness(r: Rate, q: Rate, vol: Volatility, t: Time, z: Real) -> (Real, Real) {
    let v = vol * t.sqrt();
    let d1 = (z.ln() + (r - q) * t) / v + 0.5 * v;
    (d1, d1 - v)
}

#[allow(clippy::too_many_arguments)]
fn f_a(
    strike: Real,
    r: Rate,
    q: Rate,
    vol: Volatility,
    tau: Time,
    b: Real,
    interp: &ChebyshevInterpolation,
    xmax: Real,
    maturity: Time,
    legendre: Option<&GaussianQuadrature>,
    scheme: &QdFpScheme,
) -> QlResult<(Real, Real, Real)> {
    let pdf = NormalDistribution::standard();
    let cdf = CumulativeNormalDistribution::standard();
    let v = vol * tau.sqrt();
    let (n_val, d_val) = if tau < Real::EPSILON * Real::EPSILON {
        if close_enough(b, strike) {
            let n_val = 1.0 / (std::f64::consts::SQRT_2 * std::f64::consts::PI.sqrt() * v);
            (n_val, n_val + 0.5)
        } else if b > strike {
            (0.0, 1.0)
        } else {
            (0.0, 0.0)
        }
    } else {
        let stv = tau.sqrt() / vol;
        let (k12, k3) = if let Some(rule) = legendre {
            let mut k12 = 0.0;
            let mut k3 = 0.0;
            let xs = rule.abscissas();
            let ws = rule.weights();
            for i in (0..xs.len()).rev() {
                let y = xs[i];
                let m = 0.25 * tau * (1.0 + y) * (1.0 + y);
                let (dp, dm) = moneyness(
                    r,
                    q,
                    vol,
                    m,
                    b / boundary_at(interp, xmax, maturity, tau - m),
                );
                k12 += ws[i]
                    * (q * tau - q * m).exp()
                    * (0.5 * tau * (y + 1.0) * cdf.value(dp) + stv * pdf.value(dp));
                k3 += ws[i] * stv * (r * tau - r * m).exp() * pdf.value(dm);
            }
            (k12, k3)
        } else {
            let k12 = scheme.integrate_fixed_point(
                |y| {
                    let m = 0.25 * tau * (1.0 + y) * (1.0 + y);
                    let df = (q * tau - q * m).exp();
                    if y <= 5.0 * Real::EPSILON - 1.0 {
                        if close_enough(b, boundary_at(interp, xmax, maturity, tau - m)) {
                            df * stv / (std::f64::consts::SQRT_2 * std::f64::consts::PI.sqrt())
                        } else {
                            0.0
                        }
                    } else {
                        let dp = moneyness(
                            r,
                            q,
                            vol,
                            m,
                            b / boundary_at(interp, xmax, maturity, tau - m),
                        )
                        .0;
                        df * (0.5 * tau * (y + 1.0) * cdf.value(dp) + stv * pdf.value(dp))
                    }
                },
                -1.0,
                1.0,
            )?;
            let k3 = scheme.integrate_fixed_point(
                |y| {
                    let m = 0.25 * tau * (1.0 + y) * (1.0 + y);
                    let df = (r * tau - r * m).exp();
                    if y <= 5.0 * Real::EPSILON - 1.0 {
                        if close_enough(b, boundary_at(interp, xmax, maturity, tau - m)) {
                            df * stv / (std::f64::consts::SQRT_2 * std::f64::consts::PI.sqrt())
                        } else {
                            0.0
                        }
                    } else {
                        let dm = moneyness(
                            r,
                            q,
                            vol,
                            m,
                            b / boundary_at(interp, xmax, maturity, tau - m),
                        )
                        .1;
                        df * stv * pdf.value(dm)
                    }
                },
                -1.0,
                1.0,
            )?;
            (k12, k3)
        };
        let (dp, dm) = moneyness(r, q, vol, tau, b / strike);
        (
            pdf.value(dm) / v + r * k3,
            pdf.value(dp) / v + cdf.value(dp) + q * k12,
        )
    };
    Ok((
        n_val,
        d_val,
        fixed_point_value(strike, r, q, tau, b, n_val, d_val, true),
    ))
}

#[allow(clippy::too_many_arguments)]
fn fixed_point_value(
    strike: Real,
    r: Rate,
    q: Rate,
    tau: Time,
    b: Real,
    n_val: Real,
    d_val: Real,
    equation_a: bool,
) -> Real {
    let alpha = strike * (-(r - q) * tau).exp();
    if tau < Real::EPSILON * Real::EPSILON {
        let at_strike = close_enough(b, strike);
        if equation_a {
            if at_strike {
                alpha
            } else if b > strike {
                0.0
            } else if close_enough(q, 0.0) {
                alpha * r * (if q < 0.0 { -1.0 } else { 1.0 }) / Real::EPSILON
            } else {
                alpha * r / q
            }
        } else if at_strike || b > strike {
            alpha
        } else if close_enough(q, 0.0) {
            alpha * r * (if q < 0.0 { -1.0 } else { 1.0 }) / Real::EPSILON
        } else {
            alpha * r / q
        }
    } else {
        alpha * n_val / d_val
    }
}

fn ndd_a(strike: Real, r: Rate, q: Rate, vol: Volatility, tau: Time, b: Real) -> (Real, Real) {
    let pdf = NormalDistribution::standard();
    if tau < Real::EPSILON * Real::EPSILON {
        if close_enough(b, strike) {
            let sqrt_tau = tau.sqrt();
            let vol2 = vol * vol;
            let scale = std::f64::consts::FRAC_2_SQRT_PI * 0.5 * std::f64::consts::SQRT_2;
            let dd = scale
                * (-(0.5 * vol2 + r - q) / (b * vol * vol2 * sqrt_tau)
                    + 1.0 / (b * vol * sqrt_tau));
            let nd = scale * (-0.5 * vol2 + r - q) / (b * vol * vol2 * sqrt_tau);
            (nd, dd)
        } else {
            (0.0, 0.0)
        }
    } else {
        let (dp, dm) = moneyness(r, q, vol, tau, b / strike);
        let dd =
            -pdf.value(dp) * dp / (b * vol * vol * tau) + pdf.value(dp) / (b * vol * tau.sqrt());
        let nd = -pdf.value(dm) * dm / (b * vol * vol * tau);
        (nd, dd)
    }
}

#[allow(clippy::too_many_arguments)]
fn f_b(
    strike: Real,
    r: Rate,
    q: Rate,
    vol: Volatility,
    tau: Time,
    b: Real,
    interp: &ChebyshevInterpolation,
    xmax: Real,
    maturity: Time,
    legendre: Option<&GaussianQuadrature>,
    scheme: &QdFpScheme,
) -> QlResult<(Real, Real, Real)> {
    let cdf = CumulativeNormalDistribution::standard();
    let (n_val, d_val) = if tau < Real::EPSILON * Real::EPSILON {
        if close_enough(b, strike) {
            (0.5, 0.5)
        } else if b < strike {
            (0.0, 0.0)
        } else {
            (1.0, 1.0)
        }
    } else {
        let (ni, di) = if let Some(rule) = legendre {
            let c = 0.5 * tau;
            let mut ni = 0.0;
            let mut di = 0.0;
            let xs = rule.abscissas();
            let ws = rule.weights();
            for i in (0..xs.len()).rev() {
                let u = c * xs[i] + c;
                let (dp, dm) = moneyness(
                    r,
                    q,
                    vol,
                    tau - u,
                    b / boundary_at(interp, xmax, maturity, u),
                );
                ni += ws[i] * (r * u).exp() * cdf.value(dm);
                di += ws[i] * (q * u).exp() * cdf.value(dp);
            }
            (ni * c, di * c)
        } else {
            let ni = scheme.integrate_fixed_point(
                |u| {
                    let df = (r * u).exp();
                    if u >= tau * (1.0 - 5.0 * Real::EPSILON) {
                        if close_enough(b, boundary_at(interp, xmax, maturity, u)) {
                            0.5 * df
                        } else if b < boundary_at(interp, xmax, maturity, u) {
                            0.0
                        } else {
                            df
                        }
                    } else {
                        let dm = moneyness(
                            r,
                            q,
                            vol,
                            tau - u,
                            b / boundary_at(interp, xmax, maturity, u),
                        )
                        .1;
                        df * cdf.value(dm)
                    }
                },
                0.0,
                tau,
            )?;
            let di = scheme.integrate_fixed_point(
                |u| {
                    let df = (q * u).exp();
                    if u >= tau * (1.0 - 5.0 * Real::EPSILON) {
                        if close_enough(b, boundary_at(interp, xmax, maturity, u)) {
                            0.5 * df
                        } else if b < boundary_at(interp, xmax, maturity, u) {
                            0.0
                        } else {
                            df
                        }
                    } else {
                        let dp = moneyness(
                            r,
                            q,
                            vol,
                            tau - u,
                            b / boundary_at(interp, xmax, maturity, u),
                        )
                        .0;
                        df * cdf.value(dp)
                    }
                },
                0.0,
                tau,
            )?;
            (ni, di)
        };
        let (dp, dm) = moneyness(r, q, vol, tau, b / strike);
        (cdf.value(dm) + r * ni, cdf.value(dp) + q * di)
    };
    Ok((
        n_val,
        d_val,
        fixed_point_value(strike, r, q, tau, b, n_val, d_val, false),
    ))
}

fn ndd_b(strike: Real, r: Rate, q: Rate, vol: Volatility, tau: Time, b: Real) -> (Real, Real) {
    let pdf = NormalDistribution::standard();
    let (dp, dm) = moneyness(r, q, vol, tau, b / strike);
    let scale = b * vol * tau.sqrt();
    (pdf.value(dm) / scale, pdf.value(dp) / scale)
}

impl AsObservable for QdFpAmericanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QdFpAmericanEngine {
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
        let maturity_date = exercise.last_date();
        let maturity = self.process.time(&maturity_date)?;
        let spot = self.process.x0()?;
        require!(spot >= 0.0, "negative underlying given");
        let strike = payoff.strike();
        require!(strike >= 0.0, "zero or positive strike is required");
        let value = if maturity <= 0.0 {
            match payoff.option_type() {
                OptionType::Put => (strike - spot).max(0.0),
                OptionType::Call => (spot - strike).max(0.0),
            }
        } else {
            let vol = self
                .process
                .black_volatility()
                .current_link()?
                .black_vol(maturity, strike, false)?;
            require!(vol >= 0.0, "zero or positive volatility is required");
            let r = -self
                .process
                .risk_free_rate()
                .current_link()?
                .discount_date(maturity_date, false)?
                .ln()
                / maturity;
            let q = -self
                .process
                .dividend_yield()
                .current_link()?
                .discount_date(maturity_date, false)?
                .ln()
                / maturity;
            let (spot, strike, r, q) = match payoff.option_type() {
                OptionType::Put => (spot, strike, r, q),
                OptionType::Call => (strike, spot, q, r),
            };
            edge_put(spot, strike, r, q, vol, maturity, |s, k, rr, qq, v, t| {
                self.calculate_put(s, k, rr, qq, v, t)
            })?
        };
        let value = value.max(payoff.value(spot));
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

fn edge_put(
    spot: Real,
    strike: Real,
    r: Rate,
    q: Rate,
    vol: Volatility,
    maturity: Time,
    calculate_put: impl FnOnce(Real, Real, Rate, Rate, Volatility, Time) -> QlResult<Real>,
) -> QlResult<Real> {
    use crate::math::comparison::{close, close_enough};
    if close(strike, 0.0) {
        return Ok(0.0);
    }
    if close(spot, 0.0) {
        return Ok(strike.max(strike * (-r * maturity).exp()));
    }
    if r <= 0.0 && r <= q {
        return Ok(BlackCalculator::new(
            OptionType::Put,
            strike,
            spot * ((r - q) * maturity).exp(),
            vol * maturity.sqrt(),
            (-r * maturity).exp(),
        )?
        .value()
        .max(0.0));
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
    calculate_put(spot, strike, r, q, vol, maturity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{EuropeanOption, PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::math::distributions::normal::NormalDistribution;
    use crate::math::statistics::{IncrementalStatistics, MeanStdDev, Statistics};
    use crate::option::OptionType::Put;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
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
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    struct Market {
        settings: Shared<Settings<Date>>,
        spot: Shared<SimpleQuote>,
        q_rate: Shared<SimpleQuote>,
        r_rate: Shared<SimpleQuote>,
        vol: Shared<SimpleQuote>,
        process: Shared<GeneralizedBlackScholesProcess>,
        today: Date,
    }

    impl Market {
        fn new(today: Date) -> Self {
            let settings = shared(Settings::new());
            settings.set_evaluation_date(today);
            let spot = shared(SimpleQuote::new(100.0));
            let q_rate = shared(SimpleQuote::new(0.0));
            let r_rate = shared(SimpleQuote::new(0.0));
            let vol = shared(SimpleQuote::new(0.2));
            let quote =
                |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
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
            Self {
                settings,
                spot,
                q_rate,
                r_rate,
                vol,
                process,
                today,
            }
        }

        fn set(&self, spot: Real, q: Rate, r: Rate, vol: Volatility) {
            self.spot.set_value(spot);
            self.q_rate.set_value(q);
            self.r_rate.set_value(r);
            self.vol.set_value(vol);
        }

        fn american(&self, kind: OptionType, strike: Real, maturity: Date) -> VanillaOption {
            let exercise = shared(AmericanExercise::over(self.today, maturity).unwrap());
            let payoff = shared(PlainVanillaPayoff::new(kind, strike));
            VanillaOption::new(payoff, exercise, Shared::clone(&self.settings))
        }

        fn european_npv(&self, kind: OptionType, strike: Real, maturity: Date) -> Real {
            let exercise = shared(EuropeanExercise::new(maturity));
            let payoff = shared(PlainVanillaPayoff::new(kind, strike));
            let mut option = EuropeanOption::new(payoff, exercise, Shared::clone(&self.settings));
            option
                .base_mut()
                .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(Shared::clone(
                    &self.process,
                ))) as SharedMut<dyn PricingEngine>);
            option.npv().unwrap()
        }
    }

    fn price(option: &mut VanillaOption, engine: SharedMut<QdFpAmericanEngine>) -> Real {
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    #[test]
    fn qd_fp_iteration_scheme() {
        let tol = 1e-8;
        let l = 32;
        let m = 6;
        let n = 18;
        let p = 36;
        let schemes = [
            QdFpScheme::legendre(l, m, n, p).unwrap(),
            QdFpScheme::legendre_tanh_sinh(l, m, n, tol).unwrap(),
            QdFpScheme::tanh_sinh(m, n, tol).unwrap(),
        ];
        let density = NormalDistribution::standard();
        for scheme in schemes {
            assert_eq!(scheme.chebyshev_nodes(), n);
            assert_eq!(scheme.jacobi_steps(), 1);
            assert_eq!(scheme.naive_steps(), m - 1);
            let fp = scheme
                .integrate_fixed_point(|x| density.value(x), -10.0, 10.0)
                .unwrap();
            let price = scheme
                .integrate_price(|x| density.value(x), -10.0, 10.0)
                .unwrap();
            assert!((fp - 1.0).abs() <= tol, "fixed-point integral {fp}");
            assert!((price - 1.0).abs() <= tol, "price integral {price}");
        }
    }

    #[test]
    fn qd_fp_andersen_lake_high_precision() {
        let today = Date::new(25, Month::July, 2022);
        let market = Market::new(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        #[allow(clippy::excessive_precision)]
        let cases = [
            (
                24usize,
                3usize,
                9usize,
                0.05,
                0.1069528125898476,
                0.1069524359360852,
                1e-6,
            ),
            (5, 1, 4, 0.05, 0.1070237787625299, 0.1070042740171235, 1e-3),
            (11, 2, 5, 0.05, 0.106938750864602, 0.1069479057531648, 1e-4),
            (35, 8, 16, 0.05, 0.1069527032381714, 0.106952558361499, 1e-9),
            (
                65,
                8,
                32,
                0.05,
                0.1069527028247546,
                0.1069526779971959,
                1e-11,
            ),
            (5, 1, 4, 0.075, 0.3674420299196104, 0.3674766444325588, 1e-3),
            (
                11,
                2,
                5,
                0.075,
                0.3671056766787473,
                0.3671024005532715,
                1e-4,
            ),
            (
                35,
                8,
                16,
                0.075,
                0.3671116758420414,
                0.3671111055677869,
                1e-9,
            ),
            (
                65,
                8,
                32,
                0.075,
                0.3671112309062572,
                0.3671111267813689,
                1e-11,
            ),
        ];
        for (l, m, n, r, expected_a, expected_b, tol) in cases {
            market.set(100.0, 0.05, r, 0.25);
            let european = market.european_npv(Put, 100.0, maturity);
            for (equation, expected) in [
                (FixedPointEquation::A, expected_a),
                (FixedPointEquation::B, expected_b),
            ] {
                let mut option = market.american(Put, 100.0, maturity);
                let engine = shared_mut(QdFpAmericanEngine::with_scheme(
                    Shared::clone(&market.process),
                    QdFpScheme::legendre_tanh_sinh(l, m, n, tol).unwrap(),
                    equation,
                ));
                let premium = price(&mut option, engine) - european;
                let diff = (premium - expected).abs();
                assert!(
                    diff <= tol,
                    "{equation:?} r={r} ({l},{m},{n}): premium {premium} vs {expected} (diff={diff})"
                );
            }
        }
    }

    #[test]
    fn qd_fp_standard_example() {
        let today = Date::new(1, Month::June, 2022);
        let market = Market::new(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        market.set(100.0, 0.05, 0.075, 0.25);
        let european = market.european_npv(Put, 95.0, maturity);
        #[allow(clippy::excessive_precision)]
        let expected = [
            (FixedPointEquation::A, 0.2386475283369327),
            (FixedPointEquation::B, 0.2386596962737606),
        ];
        for (equation, expected) in expected {
            let mut option = market.american(Put, 95.0, maturity);
            let engine = shared_mut(QdFpAmericanEngine::with_scheme(
                Shared::clone(&market.process),
                QdFpScheme::legendre(32, 2, 15, 48).unwrap(),
                equation,
            ));
            let premium = price(&mut option, engine) - european;
            let diff = (premium - expected).abs();
            assert!(
                diff <= 7e-15,
                "{equation:?}: premium {premium} vs {expected} (diff={diff})"
            );
        }
    }

    #[test]
    fn qd_fp_high_precision_matches_lobatto() {
        let today = Date::new(5, Month::November, 2022);
        let market = Market::new(today);
        let maturity = today + Period::new(2, TimeUnit::Years);
        market.set(36.0, 0.01, 0.075, 0.40);
        for equation in [FixedPointEquation::A, FixedPointEquation::B] {
            let high = shared_mut(QdFpAmericanEngine::with_scheme(
                Shared::clone(&market.process),
                QdFpScheme::high_precision(),
                equation,
            ));
            let lobatto = shared_mut(QdFpAmericanEngine::with_scheme(
                Shared::clone(&market.process),
                QdFpScheme::tanh_sinh(10, 30, 1e-10).unwrap(),
                equation,
            ));
            for spot in [36.0, 40.0 - 1e-8, 40.0, 40.0 + 1e-8, 50.0] {
                market.spot.set_value(spot);
                let mut option = market.american(Put, 40.0, maturity);
                let high_npv = price(&mut option, SharedMut::clone(&high));
                let lobatto_npv = price(&mut option, SharedMut::clone(&lobatto));
                let diff = (high_npv - lobatto_npv).abs();
                assert!(
                    diff <= 1e-11 && lobatto_npv.is_finite(),
                    "{equation:?} S={spot}: {high_npv} vs {lobatto_npv} (diff={diff})"
                );
            }
        }
    }

    #[test]
    fn qd_fp_negative_dividend_matches_fd() {
        use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
        use crate::pricingengines::FdBlackScholesVanillaEngine;

        let today = Date::new(5, Month::December, 2022);
        let market = Market::new(today);
        let maturity = today + Period::new(18, TimeUnit::Months);
        market.vol.set_value(0.4);
        let mut option = market.american(Put, 90.0, maturity);
        let fp = shared_mut(QdFpAmericanEngine::new(Shared::clone(&market.process)));
        let fd = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&market.process),
            Vec::new(),
            800,
            200,
            0,
            FdmSchemeDesc::douglas(),
        ));
        for r in [0.025, 0.5, 0.0, 1e-6] {
            market.r_rate.set_value(r);
            for q in [-0.05, -0.1, -0.5, 0.0] {
                market.q_rate.set_value(q);
                let fp_npv = price(&mut option, SharedMut::clone(&fp));
                option
                    .base_mut()
                    .set_pricing_engine(SharedMut::clone(&fd) as SharedMut<dyn PricingEngine>);
                let fd_npv = option.npv().unwrap();
                let diff = (fd_npv - fp_npv).abs();
                assert!(
                    diff <= 1.5e-2 && fp_npv.is_finite(),
                    "r={r} q={q}: fixed-point {fp_npv} vs FD {fd_npv} (diff={diff})"
                );
            }
        }
    }

    #[test]
    fn qd_fp_bulk_fast_matches_accurate() {
        let today = Date::new(1, Month::June, 2022);
        let market = Market::new(today);
        let fast = shared_mut(QdFpAmericanEngine::with_scheme(
            Shared::clone(&market.process),
            QdFpScheme::fast(),
            FixedPointEquation::Auto,
        ));
        let accurate = shared_mut(QdFpAmericanEngine::with_scheme(
            Shared::clone(&market.process),
            QdFpScheme::accurate(),
            FixedPointEquation::Auto,
        ));
        let lobatto = shared_mut(QdFpAmericanEngine::with_scheme(
            Shared::clone(&market.process),
            QdFpScheme::tanh_sinh(3, 7, 1e-5).unwrap(),
            FixedPointEquation::Auto,
        ));
        let mut stats_accurate = IncrementalStatistics::new();
        let mut stats_lobatto = IncrementalStatistics::new();
        for days in [30, 182, 365] {
            let maturity = today + days;
            for r in [0.02, 0.04, 0.06, 0.1] {
                for q in [0.0, 0.04, 0.08, 0.12] {
                    for vol in [0.1, 0.25, 0.6] {
                        for spot in [25.0, 75.0, 100.0, 125.0, 200.0] {
                            market.set(spot, q, r, vol);
                            let mut option = market.american(Put, 100.0, maturity);
                            let fast_npv = price(&mut option, SharedMut::clone(&fast));
                            let accurate_npv = price(&mut option, SharedMut::clone(&accurate));
                            stats_accurate.add((fast_npv - accurate_npv).abs()).unwrap();
                            let lobatto_npv = price(&mut option, SharedMut::clone(&lobatto));
                            stats_lobatto
                                .add((accurate_npv - lobatto_npv).abs())
                                .unwrap();
                        }
                    }
                }
            }
        }
        let accurate_std = stats_accurate.standard_deviation().unwrap();
        let lobatto_std = stats_lobatto.standard_deviation().unwrap();
        assert!(accurate_std <= 1e-4, "fast RMSE {accurate_std}");
        assert!(lobatto_std <= 1e-4, "lobatto RMSE {lobatto_std}");
        let accurate_max = stats_accurate.max().unwrap();
        let lobatto_max = stats_lobatto.max().unwrap();
        assert!(accurate_max <= 2.5e-3, "fast max {accurate_max}");
        assert!(lobatto_max <= 2.5e-3, "lobatto max {lobatto_max}");
    }
}
