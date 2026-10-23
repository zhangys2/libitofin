//! C ABI for implied forward and discount factor calculation from option quotes.

use crate::boundary::{BindingError, ItofinError, check_ptr, input_slice, output, without_context};
use libitofin::termstructures::forward::{
    ForwardStatus, ImpliedForward, ImpliedForwardConfig, MarketConvention, OptionQuotePair,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ItofinImpliedForwardResult {
    pub forward: f64,
    pub forward_bid_strict: f64,
    pub forward_ask_strict: f64,
    pub forward_bid_robust: f64,
    pub forward_ask_robust: f64,
    pub discount_factor: f64,
    pub implied_carry_rate: f64,
    pub status_code: i32,
    pub pairs_used: u32,
    pub pairs_pruned: u32,
    pub wls_rmse: f64,
}

#[unsafe(no_mangle)]
/// # Safety
/// Pointers must be live, valid, aligned, and non-overlapping.
pub unsafe extern "C" fn itofin_implied_forward_calculate(
    strikes: *const f64,
    call_bids: *const f64,
    call_asks: *const f64,
    put_bids: *const f64,
    put_asks: *const f64,
    count: usize,
    convention_code: i32,
    spot: *const f64,
    discount_factor: *const f64,
    expiry_years: f64,
    out: *mut ItofinImpliedForwardResult,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        without_context(error, || {
            let strikes = input_slice(strikes, count)?;
            let call_bids = input_slice(call_bids, count)?;
            let call_asks = input_slice(call_asks, count)?;
            let put_bids = input_slice(put_bids, count)?;
            let put_asks = input_slice(put_asks, count)?;

            let spot_val = if spot.is_null() {
                None
            } else {
                check_ptr(spot)?;
                Some(*spot)
            };

            let df_val = if discount_factor.is_null() {
                None
            } else {
                check_ptr(discount_factor)?;
                Some(*discount_factor)
            };

            let convention = match convention_code {
                0 => MarketConvention::EuropeanVanilla,
                1 => MarketConvention::CryptoInverse,
                2 => {
                    let s = spot_val.ok_or_else(|| {
                        BindingError::invalid("AmericanEquity convention requires spot pointer")
                    })?;
                    MarketConvention::AmericanEquity {
                        spot: s,
                        atm_vol: None,
                    }
                }
                _ => {
                    return Err(BindingError::invalid(
                        "unknown convention code: must be 0, 1, or 2",
                    ));
                }
            };

            let config = ImpliedForwardConfig {
                discount_factor: df_val,
                spot: spot_val,
                ..Default::default()
            };

            let mut quotes = Vec::with_capacity(count);
            for i in 0..count {
                quotes.push(OptionQuotePair::new(
                    strikes[i],
                    call_bids[i],
                    call_asks[i],
                    put_bids[i],
                    put_asks[i],
                ));
            }

            let res = ImpliedForward::calculate(&quotes, convention, &config, expiry_years)?;

            let status_code = match res.diagnostics.status {
                ForwardStatus::Valid => 0,
                ForwardStatus::WarningCrossedSyntheticQuotes => 1,
                ForwardStatus::WarningBoxSpreadArbitrage => 2,
                ForwardStatus::WarningImplausibleDiscountFactor => 3,
                ForwardStatus::WarningSpotDeviationExceeded => 4,
                ForwardStatus::DegradedAmericanBounds => 5,
            };

            output(
                out,
                ItofinImpliedForwardResult {
                    forward: res.forward,
                    forward_bid_strict: res.forward_bid_strict,
                    forward_ask_strict: res.forward_ask_strict,
                    forward_bid_robust: res.forward_bid_robust,
                    forward_ask_robust: res.forward_ask_robust,
                    discount_factor: res.discount_factor,
                    implied_carry_rate: res.implied_carry_rate.unwrap_or(f64::NAN),
                    status_code,
                    pairs_used: res.diagnostics.pairs_used as u32,
                    pairs_pruned: res.diagnostics.pairs_pruned as u32,
                    wls_rmse: res.diagnostics.wls_rmse,
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn test_c_ffi_european_forward() {
        let strikes = [4900.0, 5000.0, 5100.0];
        let cb = [150.0, 80.0, 30.0];
        let ca = [151.0, 81.0, 31.0];
        let pb = [30.0, 60.0, 110.0];
        let pa = [31.0, 61.0, 111.0];

        let mut out = ItofinImpliedForwardResult {
            forward: 0.0,
            forward_bid_strict: 0.0,
            forward_ask_strict: 0.0,
            forward_bid_robust: 0.0,
            forward_ask_robust: 0.0,
            discount_factor: 0.0,
            implied_carry_rate: 0.0,
            status_code: -1,
            pairs_used: 0,
            pairs_pruned: 0,
            wls_rmse: 0.0,
        };

        let rc = unsafe {
            itofin_implied_forward_calculate(
                strikes.as_ptr(),
                cb.as_ptr(),
                ca.as_ptr(),
                pb.as_ptr(),
                pa.as_ptr(),
                strikes.len(),
                0, // EuropeanVanilla
                null(),
                null(),
                0.25,
                &mut out,
                null_mut(),
            )
        };

        assert_eq!(rc, 0);
        assert!(out.forward > 4900.0 && out.forward < 5200.0);
        assert_eq!(out.status_code, 0);
    }
}
