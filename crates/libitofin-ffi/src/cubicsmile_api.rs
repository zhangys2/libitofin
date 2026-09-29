//! C ABI for the one-expiry cubic mid-IV smile.
use crate::boundary::*;
use libitofin::shared::{Shared, shared};
use libitofin::termstructures::volatility::{CubicSmileSection, SmileSection};

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
        with_context(ctx, error, |c| {
            check_ptr(out)?;
            let strikes = input_slice(strikes, n_strikes)?.to_vec();
            let mid_ivs = input_slice(mid_ivs, n_ivs)?.to_vec();
            let mut smile = if n_points == 0 {
                CubicSmileSection::new(strikes, mid_ivs, forward, exercise_time, atm_vol)?
            } else {
                let points = input_slice(std_dev_points, n_points)?.to_vec();
                CubicSmileSection::with_std_dev_points(
                    strikes,
                    mid_ivs,
                    forward,
                    exercise_time,
                    atm_vol,
                    points,
                )?
            };
            smile = smile.with_extrapolation(extrapolate != 0);
            output(out, c.insert(shared(smile))?)
        })
    }
}

/// Query: 0 volatility(strike), 1 volatility_at_std_dev, 2 strike_at_std_dev,
/// 3 forward, 4 atm vol, 5 exercise time, 6 min strike, 7 max strike, 8 variance.
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
                _ => return Err(BindingError::invalid("unknown cubic smile query")),
            };
            output(out, value)
        })
    }
}

/// Series: 0 sample points, 1 node points, 2 node IVs, 3 sampled IVs,
/// 4 node residuals, 5 segment coefficients flattened as (a, b, c) triples.
/// A null `out` reports the required length. For series 3, `valid` receives 1
/// when the sample is inside the domain.
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
            let sampled = if kind == 3 {
                Some(smile.sampled_mid_ivs()?)
            } else {
                None
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
                3 => sampled.as_ref().map(|v| v.len()).unwrap_or(0),
                4 => residuals.as_ref().map(|v| v.len()).unwrap_or(0),
                5 => coefficients.as_ref().map(|v| v.len() * 3).unwrap_or(0),
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
                5 => {
                    for (i, [a, b, c]) in coefficients.unwrap().into_iter().enumerate() {
                        dest[3 * i] = a;
                        dest[3 * i + 1] = b;
                        dest[3 * i + 2] = c;
                    }
                }
                3 => {
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
