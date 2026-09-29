//! C ABI for the Black implied-standard-deviation and implied-volatility solvers.
use crate::boundary::{
    BindingError, BindingResult, ItofinError, check_ptr, output, without_context,
};
use libitofin::option::OptionType;
use libitofin::pricingengines::black_formula_implied_std_dev;

fn option_type(code: i32) -> BindingResult<OptionType> {
    match code {
        1 => Ok(OptionType::Call),
        -1 => Ok(OptionType::Put),
        _ => Err(BindingError::invalid(
            "option type must be 1 (call) or -1 (put)",
        )),
    }
}

fn iterations(max_iterations: i64) -> BindingResult<u32> {
    u32::try_from(max_iterations).map_err(|_| {
        BindingError::invalid(
            "max_iterations must be positive and fit in a 32-bit unsigned integer",
        )
    })
}

#[unsafe(no_mangle)]
/// # Safety
/// Pointers must be aligned, live and valid for their stated lengths. Outputs
/// must not overlap inputs or other outputs. `guess` may be null. Any context
/// and its handles must belong to the calling thread; serialize calls including
/// destruction. See the crate-level C caller contract for lifetime requirements.
pub unsafe extern "C" fn itofin_black_formula_implied_std_dev(
    option_type_code: i32,
    strike: f64,
    forward: f64,
    black_price: f64,
    discount: f64,
    displacement: f64,
    guess: *const f64,
    accuracy: f64,
    max_iterations: i64,
    out: *mut f64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        without_context(error, || {
            let guess = if guess.is_null() {
                None
            } else {
                check_ptr(guess)?;
                Some(*guess)
            };
            let value = black_formula_implied_std_dev(
                option_type(option_type_code)?,
                strike,
                forward,
                black_price,
                discount,
                displacement,
                guess,
                accuracy,
                iterations(max_iterations)?,
            )?;
            output(out, value)
        })
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Pointers must be aligned, live and valid for their stated lengths. Outputs
/// must not overlap inputs or other outputs. Any context and its handles must
/// belong to the calling thread; serialize calls including destruction.
/// See the crate-level C caller contract for lifetime requirements.
pub unsafe extern "C" fn itofin_black_formula_implied_volatility(
    option_type_code: i32,
    strike: f64,
    forward: f64,
    expiry: f64,
    black_price: f64,
    discount: f64,
    displacement: f64,
    accuracy: f64,
    max_iterations: i64,
    out: *mut f64,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        without_context(error, || {
            if !expiry.is_finite() || expiry <= 0.0 {
                return Err(BindingError::invalid(
                    "expiry must be a positive finite year fraction",
                ));
            }
            let std_dev = black_formula_implied_std_dev(
                option_type(option_type_code)?,
                strike,
                forward,
                black_price,
                discount,
                displacement,
                None,
                accuracy,
                iterations(max_iterations)?,
            )?;
            output(out, std_dev / expiry.sqrt())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libitofin::pricingengines::black_formula;

    #[test]
    fn black_implied_std_dev_matches_core() {
        let price = black_formula(OptionType::Call, 100.0, 100.0, 0.2, 1.0, 0.0).unwrap();
        let mut out = 0.0;
        let mut error = ItofinError {
            code: 0,
            message: [0; 1024],
        };
        let status = unsafe {
            itofin_black_formula_implied_std_dev(
                1,
                100.0,
                100.0,
                price,
                1.0,
                0.0,
                std::ptr::null(),
                1e-12,
                100,
                &mut out,
                &mut error,
            )
        };
        assert_eq!(status, 0);
        assert!((out - 0.2).abs() < 1e-8);
    }

    #[test]
    fn black_implied_volatility_matches_core() {
        let price = black_formula(OptionType::Put, 80.0, 100.0, 0.15, 0.95, 0.01).unwrap();
        let mut out = 0.0;
        let mut error = ItofinError {
            code: 0,
            message: [0; 1024],
        };
        let expiry = 2.0;
        let status = unsafe {
            itofin_black_formula_implied_volatility(
                -1, 80.0, 100.0, expiry, price, 0.95, 0.01, 1e-12, 100, &mut out, &mut error,
            )
        };
        assert_eq!(status, 0);
        assert!((out - 0.15 / expiry.sqrt()).abs() < 1e-8);
    }
}
