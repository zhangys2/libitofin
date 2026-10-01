//! C ABI for the one-expiry cubic mid-IV smile.
use crate::boundary::*;
use libitofin::shared::{Shared, shared};
use libitofin::termstructures::volatility::{
    ArbitrageFallbackPolicy, CubicSmileSection, DEFAULT_SMILE_SMOOTHING, DEFAULT_STD_DEV_POINTS,
    RogerLeeWingConfig, SmileSection, TotalVarianceCubicSmileSection,
};

/// # Safety
/// Follow the crate C caller contract; arrays must have their stated lengths.
/// A null `std_dev_points` pointer with `n_points == 0` selects the default grid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_cubic_smile_new(
    ctx: *mut Context,
    strikes: *const f64,
    n_strikes: usize,
    mid_ivs: *const f64,
    n_ivs: usize,
    forward: f64,
    exercise_time: f64,
    atm_vol: f64,
    std_dev_points: *const f64,
    n_points: usize,
    extrapolate: u8,
    out: *mut u64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        itofin_cubic_smile_new_with_smoothing(
            ctx,
            strikes,
            n_strikes,
            mid_ivs,
            n_ivs,
            forward,
            exercise_time,
            atm_vol,
            std_dev_points,
            n_points,
            extrapolate,
            DEFAULT_SMILE_SMOOTHING,
            out,
            error,
        )
    }
}

/// Fit fixed knots with an explicit nonnegative curvature penalty weight.
/// Zero smoothing requests the unregularized fit and requires full quote rank.
/// # Safety
/// Follow the crate C caller contract; arrays must have their stated lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_cubic_smile_new_with_smoothing(
    ctx: *mut Context,
    strikes: *const f64,
    n_strikes: usize,
    mid_ivs: *const f64,
    n_ivs: usize,
    forward: f64,
    exercise_time: f64,
    atm_vol: f64,
    std_dev_points: *const f64,
    n_points: usize,
    extrapolate: u8,
    smoothing: f64,
    out: *mut u64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out)?;
            let strikes = input_slice(strikes, n_strikes)?.to_vec();
            let mid_ivs = input_slice(mid_ivs, n_ivs)?.to_vec();
            let points = if n_points == 0 {
                DEFAULT_STD_DEV_POINTS.to_vec()
            } else {
                input_slice(std_dev_points, n_points)?.to_vec()
            };
            let mut smile = CubicSmileSection::with_smoothing(
                strikes,
                mid_ivs,
                forward,
                exercise_time,
                atm_vol,
                points,
                smoothing,
            )?;
            smile = smile.with_extrapolation(extrapolate != 0);
            output(out, c.insert(shared(smile))?)
        })
    }
}

/// Query: 0 volatility(strike), 1 volatility_at_std_dev, 2 strike_at_std_dev,
/// 3 forward, 4 atm vol, 5 exercise time, 6 min strike, 7 max strike, 8 variance,
/// 9 smoothing weight.
/// # Safety
/// Follow the crate C caller contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_cubic_smile_query(
    ctx: *mut Context,
    id: u64,
    kind: i32,
    x: f64,
    out: *mut f64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out)?;
            let smile = c.get::<Shared<CubicSmileSection>>(id)?;
            let value = match kind {
                0 => smile.volatility(x)?,
                1 => smile.volatility_at_std_dev(x)?,
                2 => smile.strike_at_std_dev(x)?,
                3 => smile.forward(),
                4 => smile.atm_vol(),
                5 => smile.exercise_time(),
                6 => smile.min_strike(),
                7 => smile.max_strike(),
                8 => smile.variance(x)?,
                9 => smile.smoothing(),
                _ => return Err(BindingError::invalid("unknown cubic smile query")),
            };
            output(out, value)
        })
    }
}

/// Series: 0 knot points, 1 knot points, 2 fitted knot IVs, 3 sampled knot IVs,
/// 4 fitted-minus-knot residuals, 5 segment coefficients flattened as (a, b, c)
/// triples; 6 observed strikes, 7 observed coordinates, 8 observed IVs,
/// 9 observation residuals. A null `out` reports the required length. For
/// series 3 and 9, `valid` receives 1 for entries inside the knot domain.
/// # Safety
/// Follow the crate C caller contract; `out` and `valid` must hold `capacity` slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_cubic_smile_series(
    ctx: *mut Context,
    id: u64,
    kind: i32,
    out: *mut f64,
    valid: *mut u8,
    capacity: usize,
    out_len: *mut usize,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out_len)?;
            let smile = c.get::<Shared<CubicSmileSection>>(id)?;
            let sampled = match kind {
                3 => Some(smile.sampled_mid_ivs()?),
                9 => Some(smile.observation_residuals()?),
                _ => None,
            };
            let residuals = if kind == 4 {
                Some(smile.node_residuals()?)
            } else {
                None
            };
            let coefficients = if kind == 5 {
                Some(smile.segment_coefficients())
            } else {
                None
            };
            let length = match kind {
                0 => smile.std_dev_points().len(),
                1 => smile.node_std_dev_points().len(),
                2 => smile.node_mid_ivs().len(),
                3 | 9 => sampled.as_ref().map(|v| v.len()).unwrap_or(0),
                4 => residuals.as_ref().map(|v| v.len()).unwrap_or(0),
                5 => coefficients.as_ref().map(|v| v.len() * 3).unwrap_or(0),
                6 => smile.observed_strikes().len(),
                7 => smile.observed_std_dev_points().len(),
                8 => smile.observed_mid_ivs().len(),
                _ => return Err(BindingError::invalid("unknown cubic smile series")),
            };
            *out_len = length;
            if out.is_null() {
                return Ok(());
            }
            if capacity < length {
                return Err(BindingError::invalid("cubic smile buffer is too small"));
            }
            let dest = std::slice::from_raw_parts_mut(out, length);
            match kind {
                0 => dest.copy_from_slice(smile.std_dev_points()),
                1 => dest.copy_from_slice(smile.node_std_dev_points()),
                2 => dest.copy_from_slice(smile.node_mid_ivs()),
                4 => dest.copy_from_slice(&residuals.unwrap()),
                6 => dest.copy_from_slice(smile.observed_strikes()),
                7 => dest.copy_from_slice(smile.observed_std_dev_points()),
                8 => dest.copy_from_slice(smile.observed_mid_ivs()),
                5 => {
                    for (i, [a, b, c]) in coefficients.unwrap().into_iter().enumerate() {
                        dest[3 * i] = a;
                        dest[3 * i + 1] = b;
                        dest[3 * i + 2] = c;
                    }
                }
                3 | 9 => {
                    check_ptr(valid)?;
                    let flags = std::slice::from_raw_parts_mut(valid, length);
                    for (i, value) in sampled.unwrap().into_iter().enumerate() {
                        match value {
                            Some(vol) => {
                                dest[i] = vol;
                                flags[i] = 1;
                            }
                            None => {
                                dest[i] = 0.0;
                                flags[i] = 0;
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
            Ok(())
        })
    }
}

/// Fit a total variance cubic smile with Roger Lee wing asymptotics.
/// # Safety
/// Follow the crate C caller contract; arrays must have their stated lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_total_variance_cubic_smile_new(
    ctx: *mut Context,
    strikes: *const f64,
    n_strikes: usize,
    mid_ivs: *const f64,
    n_ivs: usize,
    forward: f64,
    exercise_time: f64,
    atm_vol: f64,
    smoothing: f64,
    arbitrage_repair: u8,
    out: *mut u64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out)?;
            let strikes = input_slice(strikes, n_strikes)?.to_vec();
            let mid_ivs = input_slice(mid_ivs, n_ivs)?.to_vec();
            let smile = TotalVarianceCubicSmileSection::with_options(
                strikes,
                mid_ivs,
                forward,
                exercise_time,
                atm_vol,
                DEFAULT_STD_DEV_POINTS.to_vec(),
                smoothing,
                RogerLeeWingConfig::default(),
                arbitrage_repair != 0,
                ArbitrageFallbackPolicy::default(),
            )?;
            output(out, c.insert(shared(smile))?)
        })
    }
}

/// Query a total variance cubic smile:
/// 0 volatility(strike), 1 variance(strike), 2 total_variance(k),
/// 3 total_variance_derivative(k), 4 total_variance_second_derivative(k),
/// 5 durrleman_density(k), 6 forward, 7 atm_vol, 8 exercise_time,
/// 9 smoothing, 10 min_strike, 11 max_strike, 12 right_wing_slope, 13 left_wing_slope.
/// # Safety
/// Follow the crate C caller contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_total_variance_cubic_smile_query(
    ctx: *mut Context,
    id: u64,
    kind: i32,
    x: f64,
    out: *mut f64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out)?;
            let smile = c.get::<Shared<TotalVarianceCubicSmileSection>>(id)?;
            let value = match kind {
                0 => smile.volatility(x)?,
                1 => smile.variance(x)?,
                2 => smile.total_variance_at_log_moneyness(x)?,
                3 => smile.total_variance_derivative(x)?,
                4 => smile.total_variance_second_derivative(x)?,
                5 => smile.durrleman_density(x)?,
                6 => smile.forward(),
                7 => smile.atm_vol(),
                8 => smile.exercise_time(),
                9 => smile.smoothing(),
                10 => smile.min_strike(),
                11 => smile.max_strike(),
                12 => smile.right_wing().asymptotic_slope(),
                13 => smile.left_wing().asymptotic_slope(),
                14 => smile.butterfly_report().ramp_iterations as f64,
                15 => smile.requested_smoothing(),
                _ => {
                    return Err(BindingError::invalid(
                        "unknown total variance cubic smile query",
                    ));
                }
            };
            output(out, value)
        })
    }
}

/// Check butterfly arbitrage report on a total variance cubic smile.
/// # Safety
/// Follow the crate C caller contract; out pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_total_variance_cubic_smile_check_arbitrage(
    ctx: *mut Context,
    id: u64,
    out_min_density: *mut f64,
    out_argmin_k: *mut f64,
    out_has_arbitrage: *mut u8,
    out_final_smoothing: *mut f64,
    out_ramp_iterations: *mut u32,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        with_context(ctx, error, |c| {
            check_ptr(out_min_density)?;
            check_ptr(out_argmin_k)?;
            check_ptr(out_has_arbitrage)?;
            if !out_final_smoothing.is_null() {
                check_ptr(out_final_smoothing)?;
            }
            if !out_ramp_iterations.is_null() {
                check_ptr(out_ramp_iterations)?;
            }
            let smile = c.get::<Shared<TotalVarianceCubicSmileSection>>(id)?;
            let report = smile.butterfly_report();
            *out_min_density = report.min_density;
            *out_argmin_k = report.argmin_k;
            *out_has_arbitrage = if report.has_arbitrage { 1 } else { 0 };
            if !out_final_smoothing.is_null() {
                *out_final_smoothing = report.final_smoothing;
            }
            if !out_ramp_iterations.is_null() {
                *out_ramp_iterations = report.ramp_iterations as u32;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn sparse_default_fit_and_explicit_smoothing_cross_c_boundary() {
        let mut ctx = Context::new();
        let xs: [f64; 7] = [-1.4, -1.0, -0.6, 0.0, 0.6, 1.0, 1.4];
        let strikes: Vec<_> = xs.iter().map(|x| 100.0 * (x * 0.2).exp()).collect();
        let ivs: Vec<_> = xs.iter().map(|x| 0.3 + 0.02 * x * x).collect();
        let mut handle = 0;
        unsafe {
            assert_eq!(
                itofin_cubic_smile_new(
                    &mut ctx,
                    strikes.as_ptr(),
                    7,
                    ivs.as_ptr(),
                    7,
                    100.0,
                    1.0,
                    0.2,
                    null(),
                    0,
                    0,
                    &mut handle,
                    null_mut(),
                ),
                0
            );
            let mut lambda = -1.0;
            assert_eq!(
                itofin_cubic_smile_query(&mut ctx, handle, 9, 0.0, &mut lambda, null_mut()),
                0
            );
            assert_eq!(lambda, DEFAULT_SMILE_SMOOTHING);
            let mut len = 0;
            assert_eq!(
                itofin_cubic_smile_series(
                    &mut ctx,
                    handle,
                    1,
                    null_mut(),
                    null_mut(),
                    0,
                    &mut len,
                    null_mut()
                ),
                0
            );
            assert_eq!(len, 9);
            let mut output = 123;
            assert_ne!(
                itofin_cubic_smile_new_with_smoothing(
                    &mut ctx,
                    strikes.as_ptr(),
                    7,
                    ivs.as_ptr(),
                    7,
                    100.0,
                    1.0,
                    0.2,
                    null(),
                    0,
                    0,
                    0.0,
                    &mut output,
                    null_mut(),
                ),
                0
            );
            assert_eq!(output, 123);
            assert_eq!(
                itofin_cubic_smile_new_with_smoothing(
                    &mut ctx,
                    strikes.as_ptr(),
                    7,
                    ivs.as_ptr(),
                    7,
                    100.0,
                    1.0,
                    0.2,
                    null(),
                    0,
                    0,
                    0.05,
                    &mut output,
                    null_mut(),
                ),
                0
            );
            assert_eq!(
                itofin_cubic_smile_query(&mut ctx, output, 9, 0.0, &mut lambda, null_mut()),
                0
            );
            assert_eq!(lambda, 0.05);

            // Test total variance cubic smile C-FFI
            let mut tv_handle = 0;
            assert_eq!(
                itofin_total_variance_cubic_smile_new(
                    &mut ctx,
                    strikes.as_ptr(),
                    7,
                    ivs.as_ptr(),
                    7,
                    100.0,
                    1.0,
                    0.2,
                    0.01,
                    1,
                    &mut tv_handle,
                    null_mut(),
                ),
                0
            );

            let mut vol_atm = 0.0;
            assert_eq!(
                itofin_total_variance_cubic_smile_query(
                    &mut ctx,
                    tv_handle,
                    0,
                    100.0,
                    &mut vol_atm,
                    null_mut()
                ),
                0
            );
            assert!((vol_atm - 0.30).abs() < 0.01);

            let mut tv_atm = 0.0;
            assert_eq!(
                itofin_total_variance_cubic_smile_query(
                    &mut ctx,
                    tv_handle,
                    2,
                    0.0,
                    &mut tv_atm,
                    null_mut()
                ),
                0
            );
            assert!((tv_atm - vol_atm * vol_atm).abs() < 1e-6);

            let mut min_d = 0.0;
            let mut arg_k = 0.0;
            let mut has_arb = 1;
            let mut final_lambda = 0.0;
            let mut ramp_iters = 99;
            assert_eq!(
                itofin_total_variance_cubic_smile_check_arbitrage(
                    &mut ctx,
                    tv_handle,
                    &mut min_d,
                    &mut arg_k,
                    &mut has_arb,
                    &mut final_lambda,
                    &mut ramp_iters,
                    null_mut(),
                ),
                0
            );
            assert_eq!(has_arb, 0);
            assert!(min_d >= 0.0);
            assert_eq!(final_lambda, 0.01);
            assert_eq!(ramp_iters, 0);

            let mut query_lambda = 0.0;
            assert_eq!(
                itofin_total_variance_cubic_smile_query(
                    &mut ctx,
                    tv_handle,
                    9,
                    0.0,
                    &mut query_lambda,
                    null_mut(),
                ),
                0
            );
            assert_eq!(query_lambda, 0.01);
        }
    }
}
