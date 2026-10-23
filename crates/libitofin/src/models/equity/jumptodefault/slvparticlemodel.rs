//! Guyon–Labordère particle calibration of stochastic local volatility
//! with jump to default.
//!
//! At each time step a Newton solve matches the Rao–Blackwellized survival
//! weights to the input credit curve, a kernel estimate of `E[v | S, τ > t]`
//! turns the Carr–Madan local volatility into a leverage slice, and an
//! Euler–Milstein step with moment-matched antithetic normals is rescaled so
//! the survival-weighted spot equals the equity forward.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::math::matrix::Matrix;
use crate::math::randomnumbers::{UniformRng, Xoshiro256StarStarUniformRng};
use crate::models::equity::jumptodefault::{
    BlackVolCallSurface, CallSurface, MomentGrid, PowerHazard, truncated_power_price,
};
use crate::option::OptionType;
use crate::processes::{HestonProcess, HestonSLVProcess, SlvJumpToDefaultProcess};
use crate::require;
use crate::shared::{Shared, shared};
use crate::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
use crate::termstructures::volatility::{
    BlackVolTermStructure, InterpolatedLocalVolSurface, LocalVolTermStructure,
};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Real, Size, Time, Volatility};

/// Knobs for the forward particle calibration.
#[derive(Clone, Debug)]
pub struct SlvJtdParticleConfig {
    /// Even particle count, at least 100.
    pub n_particles: Size,
    /// Time steps per year, at least 10.
    pub time_steps_per_year: Size,
    /// Strike knots of the leverage grid, at least 11.
    pub n_strike_knots: Size,
    /// Lowest log-moneyness `ln(K / S_0)` on the leverage grid.
    pub log_moneyness_low: Real,
    /// Highest log-moneyness `ln(K / S_0)` on the leverage grid.
    pub log_moneyness_high: Real,
    /// Multiplier on Silverman's bandwidth.
    pub kde_bandwidth_scale: Real,
    /// Absolute floor on the KDE bandwidth. Zero leaves Silverman's rule in charge.
    pub kde_bandwidth_floor: Real,
    /// Ridge weight on the affine variance prior.
    pub wing_regularization: Real,
    /// Floor applied to Carr–Madan local volatility, and to the effective
    /// diffusion `L sqrt(v)` when a path is evolved. Not a cap on the
    /// dimensionless leverage `L`.
    pub min_local_vol: Volatility,
    /// Cap applied to Carr–Madan local volatility and to `L sqrt(v)`.
    pub max_local_vol: Volatility,
    /// `S_min / S_0` inside the power intensity.
    pub spot_floor_ratio: Real,
    /// Quadrature for the truncated power price `M_p`.
    pub moment_grid: MomentGrid,
    /// Seed of the xoshiro256** generator. Zero is rejected: that seed draws
    /// entropy and would break common-random-number Greeks.
    pub seed: u64,
}

impl Default for SlvJtdParticleConfig {
    fn default() -> Self {
        Self {
            n_particles: 20_000,
            time_steps_per_year: 100,
            n_strike_knots: 61,
            log_moneyness_low: -1.10,
            log_moneyness_high: 0.90,
            kde_bandwidth_scale: 0.85,
            kde_bandwidth_floor: 0.0,
            wing_regularization: 1.0e-4,
            min_local_vol: 0.03,
            max_local_vol: 1.50,
            spot_floor_ratio: 0.05,
            moment_grid: MomentGrid::new(10.0, 64).expect("standard moment grid"),
            seed: 42,
        }
    }
}

impl SlvJtdParticleConfig {
    fn validate(&self) -> QlResult<()> {
        require!(
            self.n_particles >= 100 && self.n_particles.is_multiple_of(2),
            "particle count must be even and at least 100, got {}",
            self.n_particles
        );
        require!(
            self.time_steps_per_year >= 10,
            "time steps per year must be at least 10, got {}",
            self.time_steps_per_year
        );
        require!(
            self.n_strike_knots >= 11,
            "strike-knot count must be at least 11, got {}",
            self.n_strike_knots
        );
        require!(
            self.log_moneyness_low.is_finite()
                && self.log_moneyness_high.is_finite()
                && self.log_moneyness_low < self.log_moneyness_high,
            "log-moneyness window must be finite and increasing, got [{}, {}]",
            self.log_moneyness_low,
            self.log_moneyness_high
        );
        require!(
            self.kde_bandwidth_scale.is_finite() && self.kde_bandwidth_scale > 0.0,
            "KDE bandwidth scale must be positive, got {}",
            self.kde_bandwidth_scale
        );
        require!(
            self.kde_bandwidth_floor.is_finite() && self.kde_bandwidth_floor >= 0.0,
            "KDE bandwidth floor must be non-negative, got {}",
            self.kde_bandwidth_floor
        );
        require!(
            self.wing_regularization.is_finite() && self.wing_regularization >= 0.0,
            "wing regularization must be non-negative, got {}",
            self.wing_regularization
        );
        require!(
            self.min_local_vol.is_finite()
                && self.max_local_vol.is_finite()
                && self.min_local_vol > 0.0
                && self.min_local_vol < self.max_local_vol,
            "local-vol bounds must satisfy 0 < min < max, got [{}, {}]",
            self.min_local_vol,
            self.max_local_vol
        );
        require!(
            self.spot_floor_ratio.is_finite() && self.spot_floor_ratio > 0.0,
            "spot-floor ratio must be positive, got {}",
            self.spot_floor_ratio
        );
        require!(self.seed != 0, "particle seed must be non-zero");
        Ok(())
    }
}

/// One time slice of the particle ensemble.
pub struct SlvJtdParticleSnapshot {
    /// Time of the slice, measured from the calibration reference date.
    pub time: Time,
    /// Spot of each particle.
    pub spots: Vec<Real>,
    /// Variance of each particle.
    pub variances: Vec<Real>,
    /// Cumulative survival weight of each particle.
    pub survival_weights: Vec<Real>,
    /// Risk-free discount `D_r(t)` frozen at calibration.
    pub discount: Real,
    /// Dividend discount `D_q(t)` frozen at calibration.
    pub dividend_discount: Real,
}

/// Leverage surface, hazard, and the paths that produced them.
pub struct SlvJtdCalibrationResult {
    /// Leverage `L(t, K)`, piecewise constant in time and linear in strike.
    pub leverage_surface: Shared<InterpolatedLocalVolSurface>,
    /// Piecewise-constant scale `f(t)` matched to the survival curve.
    pub calibrated_hazard: Shared<PowerHazard>,
    /// Diffusion, hazard, and spot floor used to evolve one particle.
    pub process: Shared<SlvJumpToDefaultProcess>,
    /// Calibration time grid, including 0 and `max_maturity`.
    pub time_grid: Vec<Time>,
    /// Strike knots of every stored grid.
    pub strike_grid: Vec<Real>,
    /// Leverage, rows are times and columns are strikes.
    pub leverage_grid: Matrix,
    /// Clamped Carr–Madan local volatility on the same axes.
    pub dupire_jtd_grid: Matrix,
    /// Clamped Dupire local volatility with the jump term removed.
    pub dupire_no_default_grid: Matrix,
    /// Kernel estimate of `E[v | S = K, τ > t]`.
    pub cond_var_grid: Matrix,
    /// Ensemble survival `mean(w)` at each time.
    pub survival_probs_mc: Vec<Real>,
    /// Input survival curve at each time.
    pub survival_probs_target: Vec<Real>,
    /// Particle state at each time, including inception.
    pub snapshots: Vec<SlvJtdParticleSnapshot>,
    /// Generator seed, reused by common-random-number Greeks.
    pub seed: u64,
    /// Floor applied to the effective diffusion `L sqrt(v)`.
    pub min_local_vol: Volatility,
    /// Cap applied to the effective diffusion `L sqrt(v)`.
    pub max_local_vol: Volatility,
    /// Knots where the strike-space variance was not positive, filled from the nearest valid strike.
    pub unreliable_local_vol: Vec<(Time, Real)>,
}

/// Jointly calibrate the leverage surface and the power-hazard scale.
///
/// # Errors
///
/// Returns an error when the configuration, the exponent, the mixing factor,
/// or the maturity is unusable, or when a market input cannot be read.
pub fn calibrate_slv_jump_to_default(
    black_surface: Handle<dyn BlackVolTermStructure>,
    heston: Shared<HestonProcess>,
    survival_curve: Handle<dyn DefaultProbabilityTermStructure>,
    exponent: Real,
    mixing_factor: Real,
    max_maturity: Time,
    config: &SlvJtdParticleConfig,
) -> QlResult<SlvJtdCalibrationResult> {
    config.validate()?;
    require!(
        exponent.is_finite() && exponent <= 0.0,
        "jump-to-default exponent must be finite and <= 0, got {exponent}"
    );
    require!(
        mixing_factor.is_finite() && mixing_factor >= 0.0,
        "mixing factor must be finite and non-negative, got {mixing_factor}"
    );
    require!(
        max_maturity.is_finite() && max_maturity > 0.0 && max_maturity <= 100.0,
        "calibration maturity must lie in (0, 100], got {max_maturity}"
    );
    require!(
        heston.theta().is_finite() && heston.theta() > 0.0,
        "Heston long-run variance must be positive so the variance floor is positive, got {}",
        heston.theta()
    );
    require!(
        heston.v0().is_finite() && heston.v0() >= 0.0,
        "Heston initial variance must be finite and non-negative, got {}",
        heston.v0()
    );

    let risk_free = heston.risk_free_rate().current_link()?;
    let dividend = heston.dividend_yield().current_link()?;
    let spot0 = heston.s0().current_link()?.value()?;
    require!(
        spot0.is_finite() && spot0 > 0.0,
        "calibration spot must be positive, got {spot0}"
    );
    let reference_date = risk_free.reference_date()?;
    let day_counter = risk_free.require_day_counter()?;
    let calls = BlackVolCallSurface::new(
        black_surface,
        heston.risk_free_rate(),
        heston.dividend_yield(),
        heston.s0(),
    );
    let survival = survival_curve.current_link()?;

    // Check the rounded count before the cast. `f64::max` would turn a NaN
    // product into 1, and a non-finite float cast saturates instead of failing.
    let rounded_steps = (max_maturity * config.time_steps_per_year as Real).round();
    require!(
        rounded_steps.is_finite() && (1.0..=100_000.0).contains(&rounded_steps),
        "calibration step count must be a finite number in [1, 100000], got {rounded_steps}"
    );
    let n_steps = rounded_steps as usize;
    let time_grid: Vec<Time> = (0..=n_steps)
        .map(|i| max_maturity * i as Real / n_steps as Real)
        .collect();
    let strike_grid: Vec<Real> = (0..config.n_strike_knots)
        .map(|j| {
            let unit = j as Real / (config.n_strike_knots - 1) as Real;
            let log_k = config.log_moneyness_low
                + (config.log_moneyness_high - config.log_moneyness_low) * unit;
            spot0 * log_k.exp()
        })
        .collect();
    let log_strikes: Vec<Real> = strike_grid
        .iter()
        .map(|strike| (strike / spot0).ln())
        .collect();

    let n = config.n_particles;
    let mut spots = vec![spot0; n];
    let mut variances = vec![heston.v0(); n];
    let mut weights = vec![1.0; n];
    let mut rng = Xoshiro256StarStarUniformRng::new(config.seed);

    let mut leverage_grid = Matrix::with_size(n_steps + 1, strike_grid.len());
    let mut dupire_jtd_grid = Matrix::with_size(n_steps + 1, strike_grid.len());
    let mut dupire_no_default_grid = Matrix::with_size(n_steps + 1, strike_grid.len());
    let mut cond_var_grid = Matrix::with_size(n_steps + 1, strike_grid.len());
    let mut scales = Vec::with_capacity(n_steps);
    let mut survival_probs_mc = Vec::with_capacity(n_steps + 1);
    let mut survival_probs_target = Vec::with_capacity(n_steps + 1);
    let mut snapshots = Vec::with_capacity(n_steps + 1);
    let mut unreliable_local_vol = Vec::new();

    let v0 = heston.v0();
    let spot_floor = config.spot_floor_ratio * spot0;
    let rho_complement = (1.0 - heston.rho() * heston.rho()).max(0.0).sqrt();
    let xi = mixing_factor * heston.sigma();

    let dr0 = risk_free.discount(0.0, true)?;
    let dq0 = dividend.discount(0.0, true)?;
    survival_probs_target.push(survival.survival_probability(0.0, true)?);
    survival_probs_mc.push(1.0);
    snapshots.push(snapshot(0.0, &spots, &variances, &weights, dr0, dq0));

    for step in 0..n_steps {
        let t0 = time_grid[step];
        let t1 = time_grid[step + 1];
        let dt = t1 - t0;
        let t_eval = t0 + 0.5 * dt;
        let target = survival.survival_probability(t1, true)?;
        survival_probs_target.push(target);
        let g = state_factors(&spots, spot_floor, exponent);
        let scale = solve_step_scale(&weights, &g, dt, target)?;
        scales.push(scale);

        let conditional = if step == 0 {
            vec![v0; strike_grid.len()]
        } else {
            conditional_variance(
                &spots,
                &variances,
                &weights,
                &log_strikes,
                spot0,
                v0,
                config.kde_bandwidth_scale,
                config.kde_bandwidth_floor,
                config.wing_regularization,
            )?
        };
        let (vol_jtd, vol_plain, unreliable) = local_vol_slice(
            &calls,
            &strike_grid,
            exponent,
            scale,
            t_eval,
            &config.moment_grid,
            config.min_local_vol,
            config.max_local_vol,
        )?;
        for strike in unreliable {
            unreliable_local_vol.push((t_eval, strike));
        }
        let leverage = leverage_slice(&vol_jtd, &conditional)?;
        write_row(&mut leverage_grid, step, &leverage);
        write_row(&mut dupire_jtd_grid, step, &vol_jtd);
        write_row(&mut dupire_no_default_grid, step, &vol_plain);
        write_row(&mut cond_var_grid, step, &conditional);

        let (rate, div_yield, forward) = step_carry(&*risk_free, &*dividend, spot0, t0, t1)?;
        advance(
            &mut rng,
            &mut spots,
            &mut variances,
            &mut weights,
            &strike_grid,
            &leverage,
            &StepCoeffs {
                dt,
                rate,
                dividend: div_yield,
                forward,
                kappa: heston.kappa(),
                theta: heston.theta(),
                xi,
                rho: heston.rho(),
                rho_complement,
                exponent,
                spot_floor,
                hazard_scale: scale,
                intensity_shift: 0.0,
                leverage_shift: 0.0,
                min_local_vol: config.min_local_vol,
                max_local_vol: config.max_local_vol,
                moneyness_spot: spot0,
                scenario_spot: spot0,
            },
        )?;
        survival_probs_mc.push(mean(&weights));
        let dr1 = risk_free.discount(t1, true)?;
        let dq1 = dividend.discount(t1, true)?;
        snapshots.push(snapshot(t1, &spots, &variances, &weights, dr1, dq1));
    }

    let t_final = time_grid[n_steps];
    let scale_final = scales[n_steps - 1];
    let conditional = conditional_variance(
        &spots,
        &variances,
        &weights,
        &log_strikes,
        spot0,
        v0,
        config.kde_bandwidth_scale,
        config.kde_bandwidth_floor,
        config.wing_regularization,
    )?;
    let (vol_jtd, vol_plain, unreliable) = local_vol_slice(
        &calls,
        &strike_grid,
        exponent,
        scale_final,
        t_final,
        &config.moment_grid,
        config.min_local_vol,
        config.max_local_vol,
    )?;
    for strike in unreliable {
        unreliable_local_vol.push((t_final, strike));
    }
    let leverage = leverage_slice(&vol_jtd, &conditional)?;
    write_row(&mut leverage_grid, n_steps, &leverage);
    write_row(&mut dupire_jtd_grid, n_steps, &vol_jtd);
    write_row(&mut dupire_no_default_grid, n_steps, &vol_plain);
    write_row(&mut cond_var_grid, n_steps, &conditional);

    let hazard = shared(PowerHazard::new(time_grid[1..].to_vec(), scales, exponent)?);
    let surface = shared(InterpolatedLocalVolSurface::new(
        reference_date,
        day_counter,
        time_grid.clone(),
        strike_grid.clone(),
        leverage_grid.clone(),
    )?);
    let slv = shared(HestonSLVProcess::new(
        heston,
        Shared::clone(&surface) as Shared<dyn LocalVolTermStructure>,
        mixing_factor,
    ));
    let process = shared(SlvJumpToDefaultProcess::new(
        slv,
        Shared::clone(&hazard),
        spot_floor,
    )?);

    Ok(SlvJtdCalibrationResult {
        leverage_surface: surface,
        calibrated_hazard: hazard,
        process,
        time_grid,
        strike_grid,
        leverage_grid,
        dupire_jtd_grid,
        dupire_no_default_grid,
        cond_var_grid,
        survival_probs_mc,
        survival_probs_target,
        snapshots,
        seed: config.seed,
        min_local_vol: config.min_local_vol,
        max_local_vol: config.max_local_vol,
        unreliable_local_vol,
    })
}

/// Call, or put including the default claim `K` paid on `τ ≤ T`.
///
/// Returns `(npv, survival component, default-floor component)`.
///
/// # Errors
///
/// Returns an error when the strike, discount, or particle weights are not
/// usable.
pub(crate) fn defaultable_option_value(
    option_type: OptionType,
    strike: Real,
    discount: Real,
    spots: &[Real],
    weights: &[Real],
) -> QlResult<(Real, Real, Real)> {
    require!(
        strike.is_finite() && strike > 0.0,
        "defaultable option strike must be positive, got {strike}"
    );
    require!(
        discount.is_finite() && discount > 0.0,
        "defaultable option discount must be positive, got {discount}"
    );
    require!(
        spots.len() == weights.len() && !spots.is_empty(),
        "defaultable option needs one weight per spot"
    );
    let n = spots.len() as Real;
    let mut survival_payoff = 0.0;
    let mut mean_weight = 0.0;
    for (spot, weight) in spots.iter().zip(weights) {
        require!(
            spot.is_finite() && *spot > 0.0 && weight.is_finite() && *weight >= 0.0,
            "particle spot and survival weight must be finite and non-negative"
        );
        let intrinsic = match option_type {
            OptionType::Call => (*spot - strike).max(0.0),
            OptionType::Put => (strike - *spot).max(0.0),
        };
        survival_payoff += intrinsic * weight;
        mean_weight += weight;
    }
    mean_weight /= n;
    survival_payoff = discount * survival_payoff / n;
    let default_floor = match option_type {
        OptionType::Call => 0.0,
        OptionType::Put => discount * strike * (1.0 - mean_weight),
    };
    Ok((
        survival_payoff + default_floor,
        survival_payoff,
        default_floor,
    ))
}

/// Replay the calibrated leverage and hazard with common random numbers.
///
/// `spot0` replaces the inception spot. `intensity_shift` is added to
/// `λ(t, S)` and `leverage_shift` is added to `L`. The returned time is the
/// grid node closest to `expiry`.
///
/// # Errors
///
/// Returns an error when `expiry` is outside the calibrated grid or a market
/// input cannot be read.
pub(crate) fn simulate_fixed_leverage(
    result: &SlvJtdCalibrationResult,
    expiry: Time,
    spot0: Real,
    intensity_shift: Real,
    leverage_shift: Real,
) -> QlResult<(Time, Vec<Real>, Vec<Real>)> {
    require!(
        expiry.is_finite() && expiry >= 0.0,
        "resimulation expiry must be non-negative, got {expiry}"
    );
    require!(
        spot0.is_finite() && spot0 > 0.0,
        "resimulation spot must be positive, got {spot0}"
    );
    require!(
        intensity_shift.is_finite() && leverage_shift.is_finite(),
        "resimulation shifts must be finite"
    );
    let index = nearest_time(&result.time_grid, expiry);
    let step = if result.time_grid.len() > 1 {
        result.time_grid[1] - result.time_grid[0]
    } else {
        0.0
    };
    require!(
        (result.time_grid[index] - expiry).abs() <= 0.5 * step + 1.0e-10,
        "expiry {expiry} is off the calibration grid (nearest {})",
        result.time_grid[index]
    );
    let slv = result.process.slv();
    let heston_spot = slv.s0().current_link()?.value()?;
    let risk_free = slv.risk_free_rate().current_link()?;
    let dividend = slv.dividend_yield().current_link()?;
    let n = result.snapshots[0].spots.len();
    let mut spots = vec![spot0; n];
    let mut variances = vec![slv.v0(); n];
    let mut weights = vec![1.0; n];
    let mut rng = Xoshiro256StarStarUniformRng::new(result.seed);
    let exponent = result.process.hazard().exponent();
    let spot_floor = result.process.spot_floor();
    let rho = slv.rho();
    let rho_complement = (1.0 - rho * rho).max(0.0).sqrt();
    let xi = slv.mixing_factor() * slv.sigma();
    for step in 0..index {
        let t0 = result.time_grid[step];
        let t1 = result.time_grid[step + 1];
        let dt = t1 - t0;
        let (rate, div_yield, forward) = step_carry(&*risk_free, &*dividend, spot0, t0, t1)?;
        let leverage = result.leverage_grid.row(step);
        advance(
            &mut rng,
            &mut spots,
            &mut variances,
            &mut weights,
            &result.strike_grid,
            leverage,
            &StepCoeffs {
                dt,
                rate,
                dividend: div_yield,
                forward,
                kappa: slv.kappa(),
                theta: slv.theta(),
                xi,
                rho,
                rho_complement,
                exponent,
                spot_floor,
                hazard_scale: result.process.hazard().scale(t0 + 0.5 * dt),
                intensity_shift,
                leverage_shift,
                min_local_vol: result.min_local_vol,
                max_local_vol: result.max_local_vol,
                moneyness_spot: heston_spot,
                scenario_spot: spot0,
            },
        )?;
    }
    Ok((result.time_grid[index], spots, weights))
}

struct StepCoeffs {
    dt: Time,
    rate: Real,
    dividend: Real,
    forward: Real,
    kappa: Real,
    theta: Real,
    xi: Real,
    rho: Real,
    rho_complement: Real,
    exponent: Real,
    spot_floor: Real,
    hazard_scale: Real,
    intensity_shift: Real,
    leverage_shift: Real,
    min_local_vol: Real,
    max_local_vol: Real,
    moneyness_spot: Real,
    scenario_spot: Real,
}

fn advance(
    rng: &mut Xoshiro256StarStarUniformRng,
    spots: &mut [Real],
    variances: &mut [Real],
    weights: &mut [Real],
    strikes: &[Real],
    leverage: &[Real],
    coeffs: &StepCoeffs,
) -> QlResult<()> {
    require!(
        coeffs.theta.is_finite() && coeffs.theta > 0.0,
        "variance floor needs a positive long-run variance, got {}",
        coeffs.theta
    );
    require!(
        coeffs.min_local_vol.is_finite() && coeffs.min_local_vol > 0.0,
        "leverage floor must be positive, got {}",
        coeffs.min_local_vol
    );
    let n = spots.len();
    let sqrt_dt = coeffs.dt.sqrt();
    let (z1, z2) = moment_matched_normals(rng, n)?;
    let mut weighted_spot = 0.0;
    for i in 0..n {
        let g = state_factor(spots[i], coeffs.spot_floor, coeffs.exponent);
        let raw_lambda = coeffs.hazard_scale * g + coeffs.intensity_shift;
        require!(raw_lambda.is_finite(), "intensity is not finite");
        let lambda = raw_lambda.max(0.0);
        weights[i] *= (-lambda * coeffs.dt).exp();
        let query = spots[i] * (coeffs.moneyness_spot / coeffs.scenario_spot);
        let local = linear_interp(strikes, leverage, query) + coeffs.leverage_shift;
        require!(
            local.is_finite() && local > 0.0,
            "leverage at spot {} is not finite",
            spots[i]
        );
        let sigma = local * variances[i].max(0.0).sqrt();
        require!(
            sigma.is_finite() && sigma >= 0.0,
            "effective local volatility is not finite"
        );
        let sigma = sigma.min(coeffs.max_local_vol);
        let dw_spot = sqrt_dt * z1[i];
        let dw_var = sqrt_dt * (coeffs.rho * z1[i] + coeffs.rho_complement * z2[i]);
        let next = spots[i]
            * ((coeffs.rate - coeffs.dividend + lambda - 0.5 * sigma * sigma) * coeffs.dt
                + sigma * dw_spot)
                .exp();
        spots[i] = next;
        let sqrt_v = variances[i].max(0.0).sqrt();
        let next_v = variances[i]
            + coeffs.kappa * (coeffs.theta - variances[i]) * coeffs.dt
            + coeffs.xi * sqrt_v * dw_var
            + 0.25 * coeffs.xi * coeffs.xi * (dw_var * dw_var - coeffs.dt);
        require!(next_v.is_finite(), "variance step is not finite");
        variances[i] = next_v.max(1.0e-4 * coeffs.theta);
        weighted_spot += weights[i] * spots[i];
    }
    let mean_weighted = weighted_spot / n as Real;
    require!(
        mean_weighted.is_finite() && mean_weighted > 0.0,
        "survival-weighted spot collapsed during a particle step"
    );
    let scale = coeffs.forward / mean_weighted;
    for spot in spots.iter_mut() {
        *spot *= scale;
    }
    Ok(())
}

fn solve_step_scale(weights: &[Real], g: &[Real], dt: Time, target: Real) -> QlResult<Real> {
    require!(
        target.is_finite() && target > 0.0,
        "step survival target must be positive, got {target}"
    );
    let n = weights.len() as Real;
    let mean_w = mean(weights);
    if target >= mean_w {
        return Ok(0.0);
    }
    let mut sum_wg = 0.0;
    for (weight, factor) in weights.iter().zip(g) {
        sum_wg += weight * factor;
    }
    let mean_g = sum_wg / (mean_w * n);
    let mut scale = if mean_g > 0.0 {
        -(target / mean_w).ln() / (mean_g * dt)
    } else {
        0.0
    };
    scale = scale.max(0.0);
    for _ in 0..16 {
        let mut predicted = 0.0;
        let mut slope_sum = 0.0;
        for (weight, factor) in weights.iter().zip(g) {
            let decay = (-scale * factor * dt).exp();
            let weighted = weight * decay;
            predicted += weighted;
            slope_sum += weighted * factor;
        }
        predicted /= n;
        let gap = predicted - target;
        if gap.abs() < 1.0e-15 {
            break;
        }
        let slope = -(dt / n) * slope_sum;
        if slope.abs() < 1.0e-18 {
            break;
        }
        scale = (scale - gap / slope).max(0.0);
    }
    Ok(scale)
}

#[allow(clippy::too_many_arguments)]
fn conditional_variance(
    spots: &[Real],
    variances: &[Real],
    weights: &[Real],
    log_strikes: &[Real],
    spot0: Real,
    v0: Real,
    bandwidth_scale: Real,
    bandwidth_floor: Real,
    regularization: Real,
) -> QlResult<Vec<Real>> {
    let n = spots.len();
    let log_spots: Vec<Real> = spots
        .iter()
        .map(|spot| (spot.max(1.0e-8) / spot0).ln())
        .collect();
    let weight_sum: Real = weights.iter().sum();
    let bandwidth = kde_bandwidth(&log_spots, weights, bandwidth_scale, bandwidth_floor);
    if weight_sum <= 0.0 || bandwidth == 0.0 {
        let average = if weight_sum <= 0.0 {
            v0
        } else {
            weights
                .iter()
                .zip(variances)
                .map(|(weight, variance)| weight * variance)
                .sum::<Real>()
                / weight_sum
        };
        require!(
            average.is_finite() && average > 0.0,
            "conditional variance is not positive"
        );
        return Ok(vec![average; log_strikes.len()]);
    }

    let mut mean_x = 0.0;
    let mut mean_v = 0.0;
    for i in 0..n {
        let share = weights[i] / weight_sum;
        mean_x += share * log_spots[i];
        mean_v += share * variances[i];
    }
    let mut var_x = 0.0;
    let mut cov = 0.0;
    for i in 0..n {
        let share = weights[i] / weight_sum;
        let dx = log_spots[i] - mean_x;
        var_x += share * dx * dx;
        cov += share * dx * (variances[i] - mean_v);
    }
    let slope = cov / var_x.max(1.0e-8);
    let floor = 0.15 * v0;
    let cap = 3.50 * v0;
    let ridge = regularization * weight_sum;
    let mut conditional = Vec::with_capacity(log_strikes.len());
    for log_strike in log_strikes {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let prior = (mean_v + slope * (log_strike - mean_x)).clamp(floor, cap);
        for i in 0..n {
            let unit = (log_strike - log_spots[i]) / bandwidth;
            let kernel = (-0.5 * unit * unit).exp();
            let weighted = weights[i] * kernel;
            numerator += weighted * variances[i];
            denominator += weighted;
        }
        let value = (numerator + ridge * prior) / (denominator + ridge);
        require!(
            value.is_finite() && value > 0.0,
            "conditional variance is not positive"
        );
        conditional.push(value);
    }
    Ok(conditional)
}

/// Survival-weighted Silverman bandwidth.
///
/// `n_eff = (sum w)^2 / sum w^2`. The absolute `floor` binds only when it
/// exceeds that width.
fn kde_bandwidth(log_spots: &[Real], weights: &[Real], scale: Real, floor: Real) -> Real {
    let weight_sum: Real = weights.iter().copied().sum();
    let weight_square: Real = weights.iter().map(|weight| weight * weight).sum();
    if weight_sum <= 0.0 || weight_square <= 0.0 {
        return floor;
    }
    let n_eff = (weight_sum * weight_sum / weight_square).max(1.0);
    let mean = log_spots
        .iter()
        .zip(weights)
        .map(|(spot, weight)| weight * spot)
        .sum::<Real>()
        / weight_sum;
    let variance = log_spots
        .iter()
        .zip(weights)
        .map(|(spot, weight)| weight * (spot - mean) * (spot - mean))
        .sum::<Real>()
        / weight_sum;
    let silverman = scale * variance.sqrt() * n_eff.powf(-0.2);
    if !silverman.is_finite() {
        return floor;
    }
    silverman.max(floor)
}

#[allow(clippy::too_many_arguments)]
fn local_vol_slice(
    calls: &BlackVolCallSurface,
    strikes: &[Real],
    exponent: Real,
    scale: Real,
    t: Time,
    grid: &MomentGrid,
    min_vol: Volatility,
    max_vol: Volatility,
) -> QlResult<(Vec<Real>, Vec<Real>, Vec<Real>)> {
    let mut jumped_raw = vec![None; strikes.len()];
    let mut plain_raw = vec![None; strikes.len()];
    for (index, strike) in strikes.iter().enumerate() {
        let (var_jtd, var_plain) =
            match strike_space_variances(calls, exponent, scale, *strike, t, grid) {
                Ok(pair) => pair,
                Err(_) => continue,
            };
        if var_jtd.is_finite() && var_jtd > 0.0 && var_plain.is_finite() && var_plain > 0.0 {
            jumped_raw[index] = Some(var_jtd.sqrt().clamp(min_vol, max_vol));
            plain_raw[index] = Some(var_plain.sqrt().clamp(min_vol, max_vol));
        }
    }
    let unreliable = strikes
        .iter()
        .zip(&jumped_raw)
        .filter_map(|(strike, value)| value.is_none().then_some(*strike))
        .collect();
    Ok((
        fill_missing_local_vol(&jumped_raw, strikes)?,
        fill_missing_local_vol(&plain_raw, strikes)?,
        unreliable,
    ))
}

/// Copies each missing knot from the nearest strike that had a positive variance.
///
/// A slice with no positive variance is an error, the same outcome as
/// `CarrMadanLocalVolSurface` on a negative local variance. A far wing whose
/// call density underflows is filled from that neighbor and reported, rather
/// than published as the volatility floor.
fn fill_missing_local_vol(values: &[Option<Real>], strikes: &[Real]) -> QlResult<Vec<Real>> {
    require!(
        values.iter().any(Option::is_some),
        "negative Carr-Madan local vol^2 across the strike slice; the black vol surface is not smooth enough"
    );
    let mut filled = vec![0.0; values.len()];
    for (index, value) in values.iter().enumerate() {
        if let Some(vol) = value {
            filled[index] = *vol;
            continue;
        }
        let mut nearest = 0.0;
        let mut nearest_distance = Real::MAX;
        for (other, defined) in strikes.iter().zip(values) {
            if let Some(vol) = defined {
                let distance = (other - strikes[index]).abs();
                if distance < nearest_distance {
                    nearest_distance = distance;
                    nearest = *vol;
                }
            }
        }
        filled[index] = nearest;
    }
    Ok(filled)
}

fn strike_space_variances(
    calls: &BlackVolCallSurface,
    exponent: Real,
    scale: Real,
    strike: Real,
    t: Time,
    grid: &MomentGrid,
) -> QlResult<(Real, Real)> {
    let dt = (1.0e-4_f64).min(t / 2.0);
    let earlier = t - dt;
    let later = t + dt;
    let span = later - earlier;
    let call = calls.call(strike, t)?;
    let call_t = (calls.call(strike, later)? - calls.call(strike, earlier)?) / span;
    let call_k = calls.call_dk(strike, t)?;
    let call_kk = calls.call_dkk(strike, t)?;
    require!(
        call_kk.is_finite() && call_kk > 0.0,
        "call density is not positive at strike {strike} and time {t}; cannot recover Carr-Madan local variance"
    );
    let discount_up = calls.discount(later)?;
    let discount_dn = calls.discount(earlier)?;
    let rate = -(discount_up / discount_dn).ln() / span;
    let carry = (calls.forward(later)? / calls.forward(earlier)?).ln() / span;
    let dividend = rate - carry;
    let adjustment = if scale == 0.0 {
        0.0
    } else {
        strike * scale * truncated_power_price(calls, exponent, strike, t, grid)?
    };
    let base = call_t + dividend * call + carry * strike * call_k;
    let denom = strike * strike * call_kk;
    Ok((2.0 * (base - adjustment) / denom, 2.0 * base / denom))
}

fn leverage_slice(local_vol: &[Real], conditional: &[Real]) -> QlResult<Vec<Real>> {
    let mut leverage = Vec::with_capacity(local_vol.len());
    for (vol, variance) in local_vol.iter().zip(conditional) {
        require!(
            variance.is_finite() && *variance > 0.0,
            "conditional variance must be positive, got {variance}"
        );
        let ratio = vol / variance.sqrt();
        require!(
            ratio.is_finite() && ratio > 0.0,
            "leverage must be positive, got {ratio}"
        );
        leverage.push(ratio);
    }
    Ok(leverage)
}

#[cfg(test)]
fn bounded_local_vol(
    variance: Real,
    min_vol: Volatility,
    max_vol: Volatility,
    strike: Real,
    t: Time,
) -> QlResult<Real> {
    require!(
        variance.is_finite() && variance > 0.0,
        "negative Carr-Madan local vol^2 at strike {strike} and time {t}; the black vol surface is not smooth enough"
    );
    Ok(variance.sqrt().clamp(min_vol, max_vol))
}

fn moment_matched_normals(
    rng: &mut Xoshiro256StarStarUniformRng,
    n: usize,
) -> QlResult<(Vec<Real>, Vec<Real>)> {
    let half = n / 2;
    let mut z1 = vec![0.0; n];
    let mut z2 = vec![0.0; n];
    for k in 0..half {
        let g1 = InverseCumulativeNormal::standard_value(rng.next_real())?;
        let g2 = InverseCumulativeNormal::standard_value(rng.next_real())?;
        z1[k] = g1;
        z1[k + half] = -g1;
        z2[k] = g2;
        z2[k + half] = -g2;
    }
    normalize_and_orthogonalize(&mut z1, &mut z2)?;
    Ok((z1, z2))
}

fn normalize_and_orthogonalize(z1: &mut [Real], z2: &mut [Real]) -> QlResult<()> {
    let n = z1.len() as Real;
    let rms1 = (z1.iter().map(|z| z * z).sum::<Real>() / n).sqrt();
    require!(rms1 > 0.0, "antithetic spot draws collapsed");
    for z in z1.iter_mut() {
        *z /= rms1;
    }
    let covariance = z1.iter().zip(z2.iter()).map(|(a, b)| a * b).sum::<Real>() / n;
    for i in 0..z1.len() {
        z2[i] -= covariance * z1[i];
    }
    let rms2 = (z2.iter().map(|z| z * z).sum::<Real>() / n).sqrt();
    require!(rms2 > 0.0, "antithetic variance draws collapsed");
    for z in z2.iter_mut() {
        *z /= rms2;
    }
    Ok(())
}

fn step_carry(
    risk_free: &dyn YieldTermStructure,
    dividend: &dyn YieldTermStructure,
    spot: Real,
    t0: Time,
    t1: Time,
) -> QlResult<(Real, Real, Real)> {
    let dt = t1 - t0;
    let dr0 = risk_free.discount(t0, true)?;
    let dr1 = risk_free.discount(t1, true)?;
    let dq0 = dividend.discount(t0, true)?;
    let dq1 = dividend.discount(t1, true)?;
    require!(
        dr0 > 0.0 && dr1 > 0.0 && dq0 > 0.0 && dq1 > 0.0,
        "step discounts must be positive"
    );
    let rate = -(dr1 / dr0).ln() / dt;
    let dividend_rate = -(dq1 / dq0).ln() / dt;
    Ok((rate, dividend_rate, spot * dq1 / dr1))
}

fn state_factors(spots: &[Real], floor: Real, exponent: Real) -> Vec<Real> {
    spots
        .iter()
        .map(|spot| state_factor(*spot, floor, exponent))
        .collect()
}

fn state_factor(spot: Real, floor: Real, exponent: Real) -> Real {
    spot.max(floor).powf(exponent)
}

fn linear_interp(knots: &[Real], values: &[Real], x: Real) -> Real {
    if x <= knots[0] {
        return values[0];
    }
    let last = knots.len() - 1;
    if x >= knots[last] {
        return values[last];
    }
    let right = knots.partition_point(|knot| *knot < x);
    let left = right - 1;
    let weight = (x - knots[left]) / (knots[right] - knots[left]);
    values[left] * (1.0 - weight) + values[right] * weight
}

fn write_row(matrix: &mut Matrix, row: usize, values: &[Real]) {
    for (column, value) in values.iter().enumerate() {
        matrix[(row, column)] = *value;
    }
}

fn snapshot(
    time: Time,
    spots: &[Real],
    variances: &[Real],
    weights: &[Real],
    discount: Real,
    dividend_discount: Real,
) -> SlvJtdParticleSnapshot {
    SlvJtdParticleSnapshot {
        time,
        spots: spots.to_vec(),
        variances: variances.to_vec(),
        survival_weights: weights.to_vec(),
        discount,
        dividend_discount,
    }
}

fn mean(values: &[Real]) -> Real {
    values.iter().sum::<Real>() / values.len() as Real
}

fn nearest_time(times: &[Time], expiry: Time) -> usize {
    let mut best = 0;
    let mut best_gap = Real::MAX;
    for (index, time) in times.iter().enumerate() {
        let gap = (time - expiry).abs();
        if gap < best_gap {
            best = index;
            best_gap = gap;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::math::interpolations::flat::BackwardFlat;
    use crate::models::equity::jumptodefault::BlackVolCallSurface;
    use crate::quotes::make_quote_handle;
    use crate::shared::shared;
    use crate::termstructures::TermStructure;
    use crate::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
    use crate::termstructures::credit::flathazardrate::FlatHazardRate;
    use crate::termstructures::credit::interpolatedhazardratecurve::InterpolatedHazardRateCurve;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn yield_curve(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn black(vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::new(
            today(),
            None,
            vol,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    fn heston(v0: Real, kappa: Real, theta: Real, sigma: Real, rho: Real) -> Shared<HestonProcess> {
        shared(HestonProcess::new(
            yield_curve(0.03),
            yield_curve(0.01),
            make_quote_handle(100.0).handle(),
            v0,
            kappa,
            theta,
            sigma,
            rho,
        ))
    }

    fn config(particles: Size, steps: Size) -> SlvJtdParticleConfig {
        SlvJtdParticleConfig {
            n_particles: particles,
            time_steps_per_year: steps,
            n_strike_knots: 11,
            moment_grid: MomentGrid::new(8.0, 16).unwrap(),
            ..SlvJtdParticleConfig::default()
        }
    }

    fn flat_survival(hazard: Real) -> Handle<dyn DefaultProbabilityTermStructure> {
        Handle::new(shared(FlatHazardRate::with_rate(
            today(),
            hazard,
            Actual365Fixed::new(),
        )) as Shared<dyn DefaultProbabilityTermStructure>)
    }

    #[test]
    fn step_hazard_tracks_constant_and_piecewise_survival() {
        let cfg = config(100, 40);
        let constant = calibrate_slv_jump_to_default(
            black(0.25),
            heston(1.0, 1.5, 1.0, 0.30, -0.4),
            flat_survival(0.03),
            0.0,
            1.0,
            1.0,
            &cfg,
        )
        .unwrap();
        for (time, (mc, target)) in constant.time_grid.iter().zip(
            constant
                .survival_probs_mc
                .iter()
                .zip(&constant.survival_probs_target),
        ) {
            assert!(
                (mc - target).abs() <= 1e-13,
                "t={time} mc={mc} target={target}"
            );
        }
        for step in 0..constant.time_grid.len() - 1 {
            let mid = 0.5 * (constant.time_grid[step] + constant.time_grid[step + 1]);
            let scale = constant.calibrated_hazard.scale(mid);
            assert!((scale - 0.03).abs() <= 1e-10, "scale {scale} at {mid}");
        }

        let curve = InterpolatedHazardRateCurve::new(
            vec![today(), today() + 365, today() + 730],
            vec![0.01, 0.02, 0.04],
            Actual365Fixed::new(),
            BackwardFlat,
        )
        .unwrap();
        curve.enable_extrapolation();
        let stepped = calibrate_slv_jump_to_default(
            black(0.2),
            heston(1.0, 1.2, 1.0, 0.25, 0.0),
            Handle::new(shared(curve) as Shared<dyn DefaultProbabilityTermStructure>),
            0.0,
            1.0,
            1.0,
            &cfg,
        )
        .unwrap();
        for (mc, target) in stepped
            .survival_probs_mc
            .iter()
            .zip(&stepped.survival_probs_target)
        {
            assert!((mc - target).abs() <= 1e-13, "{mc} vs {target}");
        }
    }

    #[test]
    fn non_finite_maturity_and_non_positive_theta_are_rejected() {
        let cfg = config(100, 20);
        let market = black(0.2);
        let credit = flat_survival(0.01);
        assert!(
            calibrate_slv_jump_to_default(
                market.clone(),
                heston(1.0, 1.0, 1.0, 0.2, 0.0),
                credit.clone(),
                0.0,
                1.0,
                Time::NAN,
                &cfg,
            )
            .is_err()
        );
        assert!(
            calibrate_slv_jump_to_default(
                market,
                heston(1.0, 1.0, 0.0, 0.2, 0.0),
                credit,
                0.0,
                1.0,
                1.0,
                &cfg,
            )
            .is_err()
        );
    }

    #[test]
    fn a_non_finite_leverage_or_variance_update_is_rejected() {
        let mut spots = vec![100.0, 100.0];
        let mut variances = vec![1.0, 1.0];
        let mut weights = vec![1.0, 1.0];
        let strikes = [50.0, 150.0];
        let coeffs = StepCoeffs {
            dt: 0.1,
            rate: 0.0,
            dividend: 0.0,
            forward: 100.0,
            kappa: 1.0,
            theta: 1.0,
            xi: 0.0,
            rho: 0.0,
            rho_complement: 1.0,
            exponent: 0.0,
            spot_floor: 1.0,
            hazard_scale: 0.0,
            intensity_shift: 0.0,
            leverage_shift: 0.0,
            min_local_vol: 0.03,
            max_local_vol: 1.50,
            moneyness_spot: 100.0,
            scenario_spot: 100.0,
        };
        let mut rng = Xoshiro256StarStarUniformRng::new(7);
        assert!(
            advance(
                &mut rng,
                &mut spots,
                &mut variances,
                &mut weights,
                &strikes,
                &[Real::NAN, 0.2],
                &coeffs,
            )
            .is_err()
        );
        let nan_variance = StepCoeffs {
            xi: Real::NAN,
            ..coeffs
        };
        assert!(
            advance(
                &mut rng,
                &mut spots,
                &mut variances,
                &mut weights,
                &strikes,
                &[0.2, 0.2],
                &nan_variance,
            )
            .is_err()
        );
    }

    #[test]
    fn kde_bandwidth_follows_the_weighted_cloud_unless_the_floor_binds() {
        let logs = [0.0, 0.0, 1.0, 1.0];
        let weights = [1.0, 1.0, 1.0, 1.0];
        let open = kde_bandwidth(&logs, &weights, 1.0, 0.0);
        let expected = 0.5 * 4.0_f64.powf(-0.2);
        assert!((open - expected).abs() <= 1e-12, "{open} vs {expected}");
        assert!((kde_bandwidth(&logs, &weights, 1.0, 10.0) - 10.0).abs() <= 1e-12);
    }

    #[test]
    fn low_variance_leverage_is_not_capped_like_a_volatility() {
        let result = calibrate_slv_jump_to_default(
            black(0.45),
            heston(0.04, 1.5, 0.04, 0.01, 0.0),
            flat_survival(0.0),
            0.0,
            0.0,
            0.5,
            &config(100, 20),
        )
        .unwrap();
        let column = result
            .strike_grid
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (**left - 100.0)
                    .abs()
                    .partial_cmp(&(**right - 100.0).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        let leverage = result.leverage_grid[(0, column)];
        let expected = 0.45 / 0.04_f64.sqrt();
        assert!(
            (leverage - expected).abs() < 0.05,
            "leverage {leverage} vs {expected}"
        );
        assert!(leverage > result.max_local_vol);
    }

    #[test]
    fn a_non_positive_local_variance_is_an_error() {
        assert!(bounded_local_vol(-1.0, 0.03, 1.5, 100.0, 1.0).is_err());
        let calls = BlackVolCallSurface::new(
            black(0.2),
            yield_curve(0.01),
            yield_curve(0.0),
            make_quote_handle(100.0).handle(),
        );
        let grid = MomentGrid::new(8.0, 16).unwrap();
        assert!(
            local_vol_slice(
                &calls,
                &[90.0, 100.0, 110.0],
                0.0,
                2.0,
                1.0,
                &grid,
                0.03,
                1.5,
            )
            .is_err()
        );
    }

    #[test]
    fn discounted_survival_weighted_spot_is_the_spot() {
        let result = calibrate_slv_jump_to_default(
            black(0.22),
            heston(1.0, 1.8, 1.0, 0.40, -0.5),
            flat_survival(0.02),
            -1.25,
            1.0,
            2.0,
            &config(200, 20),
        )
        .unwrap();
        let risk_free = result
            .process
            .slv()
            .risk_free_rate()
            .current_link()
            .unwrap();
        let dividend = result
            .process
            .slv()
            .dividend_yield()
            .current_link()
            .unwrap();
        for maturity in [0.5, 1.0, 1.5, 2.0] {
            let index = nearest_time(&result.time_grid, maturity);
            let time = result.time_grid[index];
            let snap = &result.snapshots[index];
            let weighted = mean(
                &snap
                    .spots
                    .iter()
                    .zip(&snap.survival_weights)
                    .map(|(spot, weight)| spot * weight)
                    .collect::<Vec<_>>(),
            );
            let discounted = risk_free.discount(time, true).unwrap()
                / dividend.discount(time, true).unwrap()
                * weighted;
            assert!(
                (discounted - 100.0).abs() / 100.0 <= 1e-13,
                "T={time} discounted spot {discounted}"
            );
        }
    }

    #[test]
    fn cds_par_spreads_match_the_input_curve() {
        let hazard = 0.025;
        let recovery = 0.40;
        let rate = 0.03;
        let result = calibrate_slv_jump_to_default(
            black(0.2),
            heston(1.0, 1.4, 1.0, 0.3, -0.3),
            flat_survival(hazard),
            -0.5,
            1.0,
            2.0,
            &config(100, 100),
        )
        .unwrap();
        let market = (1.0 - recovery) * hazard;
        for maturity in [0.5, 1.0, 1.5, 2.0] {
            let quoted = par_spread(
                &result.time_grid,
                &result.survival_probs_mc,
                rate,
                recovery,
                maturity,
            );
            assert!(
                (quoted - market).abs() <= 5.0e-6,
                "T={maturity} spread {quoted} vs {market}"
            );
        }
    }

    #[test]
    fn zero_mixing_leverage_equals_carr_madan_local_vol() {
        let result = calibrate_slv_jump_to_default(
            black(0.25),
            heston(1.0, 1.5, 1.0, 0.35, -0.4),
            flat_survival(0.02),
            -1.0,
            0.0,
            0.5,
            &config(100, 20),
        )
        .unwrap();
        for row in 0..result.time_grid.len() {
            for column in 0..result.strike_grid.len() {
                let conditional = result.cond_var_grid[(row, column)];
                let local = result.dupire_jtd_grid[(row, column)];
                let leverage = result.leverage_grid[(row, column)];
                assert!(
                    (conditional - 1.0).abs() <= 1e-12,
                    "conditional variance {conditional}"
                );
                assert!((leverage - local).abs() <= 1e-12, "{leverage} vs {local}");
            }
        }
    }

    #[test]
    fn zero_hazard_leaves_dupire_and_unit_weights() {
        let result = calibrate_slv_jump_to_default(
            black(0.2),
            heston(1.0, 1.2, 1.0, 0.3, -0.2),
            flat_survival(0.0),
            -1.25,
            1.0,
            0.5,
            &config(100, 20),
        )
        .unwrap();
        for row in 0..result.time_grid.len() {
            for column in 0..result.strike_grid.len() {
                let jumped = result.dupire_jtd_grid[(row, column)];
                let plain = result.dupire_no_default_grid[(row, column)];
                assert!((jumped - plain).abs() <= 1e-14, "{jumped} vs {plain}");
            }
        }
        for snap in &result.snapshots {
            for weight in &snap.survival_weights {
                assert!((weight - 1.0).abs() <= 1e-14);
            }
        }
        let discount = result
            .process
            .slv()
            .risk_free_rate()
            .current_link()
            .unwrap()
            .discount(result.time_grid[result.time_grid.len() / 2], true)
            .unwrap();
        let index = result.time_grid.len() / 2;
        let (_, _, floor) = defaultable_option_value(
            OptionType::Put,
            100.0,
            discount,
            &result.snapshots[index].spots,
            &result.snapshots[index].survival_weights,
        )
        .unwrap();
        assert!(floor.abs() <= 1e-14, "default floor {floor}");
    }

    #[test]
    fn jump_to_default_cuts_downside_local_vol_more_than_upside() {
        let spot = 100.0;
        let result = calibrate_slv_jump_to_default(
            black(0.25),
            heston(1.0, 1.6, 1.0, 0.3, -0.45),
            flat_survival(0.04),
            -1.25,
            1.0,
            1.0,
            &SlvJtdParticleConfig {
                log_moneyness_low: -0.55,
                log_moneyness_high: 0.40,
                ..config(100, 20)
            },
        )
        .unwrap();
        let row = nearest_time(&result.time_grid, 1.0);
        let mut low_strike_gap = Real::NAN;
        let mut high_strike_gap = Real::NAN;
        for column in 0..result.strike_grid.len() {
            let plain = result.dupire_no_default_grid[(row, column)];
            let jumped = result.dupire_jtd_grid[(row, column)];
            assert!(
                jumped < plain,
                "K={} jumped {jumped} plain {plain}",
                result.strike_grid[column]
            );
            let strike = result.strike_grid[column];
            if (strike - 0.75 * spot).abs() < 3.0 {
                low_strike_gap = plain - jumped;
            }
            if (strike - 1.15 * spot).abs() < 3.0 {
                high_strike_gap = plain - jumped;
            }
        }
        let low = interp_row(&result, row, 0.75 * spot);
        let high = interp_row(&result, row, 1.15 * spot);
        assert!(
            low.0 - low.1 > high.0 - high.1,
            "downside reduction {} vs upside {}, nearest gaps {low_strike_gap} {high_strike_gap}",
            low.0 - low.1,
            high.0 - high.1
        );
    }

    fn interp_row(result: &SlvJtdCalibrationResult, row: usize, strike: Real) -> (Real, Real) {
        let plain = result.dupire_no_default_grid.row(row);
        let jumped = result.dupire_jtd_grid.row(row);
        (
            linear_interp(&result.strike_grid, plain, strike),
            linear_interp(&result.strike_grid, jumped, strike),
        )
    }

    fn par_spread(
        times: &[Time],
        survival: &[Real],
        rate: Real,
        recovery: Real,
        maturity: Time,
    ) -> Real {
        let lgd = 1.0 - recovery;
        let mut protection = 0.0;
        let mut annuity = 0.0;
        for i in 0..times.len() - 1 {
            if times[i] >= maturity - 1e-12 {
                break;
            }
            let t1 = times[i + 1].min(maturity);
            let dt = t1 - times[i];
            let df = (-rate * 0.5 * (times[i] + t1)).exp();
            let q_mid = 0.5 * (survival[i] + survival[i + 1]);
            protection += lgd * df * (survival[i] - survival[i + 1]);
            annuity += df * q_mid * dt;
        }
        protection / annuity
    }
}
