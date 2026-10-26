//! C ABI for implied forward and discount factor calculation from option quotes.

use crate::boundary::{
    BindingError, BindingResult, ItofinError, check_ptr, input_slice, output, without_context,
};
use libitofin::termstructures::forward::{
    AmericanLevel, AmericanModelConfig, ForwardStatus, ForwardTier, ImpliedForward,
    ImpliedForwardConfig, ImpliedForwardResult, MarketConvention, OptionQuotePair,
};

/// Stable status codes of the C ABI; new variants take the next free value.
fn status_code(status: ForwardStatus) -> i32 {
    match status {
        ForwardStatus::Valid => 0,
        ForwardStatus::WarningCrossedSyntheticQuotes => 1,
        ForwardStatus::WarningBoxSpreadArbitrage => 2,
        ForwardStatus::WarningImplausibleDiscountFactor => 3,
        ForwardStatus::WarningSpotDeviationExceeded => 4,
        ForwardStatus::DegradedAmericanBounds => 5,
        ForwardStatus::WarningImplausibleCarry => 6,
    }
}

/// Tier codes of the extended result: -1 for conventions that have no tier.
fn tier_code(tier: Option<ForwardTier>) -> i32 {
    match tier {
        None => -1,
        Some(ForwardTier::Corridor) => 0,
        Some(ForwardTier::ModelFast) => 1,
        Some(ForwardTier::ModelAccurate) => 2,
    }
}

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

/// Settings of the model-based American path. `itofin_implied_forward_options_default` fills
/// the defaults; the dividend arrays are read during the call and not retained.
///
/// `level` is 0 auto, 1 fast, 2 accurate. Optional numbers use a `has_*` flag that must be 0 or 1.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ItofinImpliedForwardOptions {
    pub atm_vol: f64,
    pub has_atm_vol: i32,
    pub dividend_times: *const f64,
    pub dividend_amounts: *const f64,
    pub dividend_count: usize,
    pub level: i32,
    pub carry_search_low: f64,
    pub carry_search_high: f64,
    pub carry_flag_low: f64,
    pub carry_flag_high: f64,
    pub correction_passes: usize,
    pub has_correction_passes: i32,
    pub carry_tolerance: f64,
    pub corridor_kappa: f64,
    pub max_accurate_prices: usize,
}

/// The result of `itofin_implied_forward_calculate` plus the model-path outputs.
///
/// `tier_code` is -1 (no tier), 0 corridor, 1 model fast, 2 model accurate. `flags` bits:
/// 1 schedule ignored, 2 latency fallback, 4 not converged. `has_model` is 1 when the
/// `solved_carry`, `call_put_iv_gap`, `iterations` and `accurate_prices_used` fields are
/// meaningful (NaN or 0 otherwise).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ItofinImpliedForwardResultEx {
    pub core: ItofinImpliedForwardResult,
    pub tier_code: i32,
    pub flags: u32,
    pub has_model: i32,
    pub solved_carry: f64,
    pub call_put_iv_gap: f64,
    pub iterations: u32,
    pub accurate_prices_used: u32,
    pub total_pairs_received: u32,
    pub is_crossed: i32,
    pub box_arbitrage_violations: u32,
    pub min_spread: f64,
    pub max_spread: f64,
}

impl ItofinImpliedForwardOptions {
    fn defaults() -> Self {
        let defaults = AmericanModelConfig::default();
        Self {
            atm_vol: 0.0,
            has_atm_vol: 0,
            dividend_times: std::ptr::null(),
            dividend_amounts: std::ptr::null(),
            dividend_count: 0,
            level: 0,
            carry_search_low: defaults.carry_search_range.0,
            carry_search_high: defaults.carry_search_range.1,
            carry_flag_low: defaults.carry_flag_range.0,
            carry_flag_high: defaults.carry_flag_range.1,
            correction_passes: 0,
            has_correction_passes: 0,
            carry_tolerance: defaults.carry_tolerance,
            corridor_kappa: defaults.corridor_kappa,
            max_accurate_prices: defaults.max_accurate_prices,
        }
    }

    /// Copies the options into the core type, validating flags, codes and arrays.
    ///
    /// # Safety
    /// The dividend pointers must reference `dividend_count` readable values when it is nonzero.
    unsafe fn parse(&self) -> BindingResult<(AmericanModelConfig, Option<f64>)> {
        let flag = |value: i32, name: &str| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(BindingError::invalid(format!("{name} must be 0 or 1"))),
        };
        let level = match self.level {
            0 => AmericanLevel::Auto,
            1 => AmericanLevel::Fast,
            2 => AmericanLevel::Accurate,
            _ => return Err(BindingError::invalid("level must be 0, 1 or 2")),
        };
        let times = unsafe { input_slice(self.dividend_times, self.dividend_count)? };
        let amounts = unsafe { input_slice(self.dividend_amounts, self.dividend_count)? };
        let config = AmericanModelConfig {
            level,
            dividends: times.iter().copied().zip(amounts.iter().copied()).collect(),
            corridor_kappa: self.corridor_kappa,
            carry_search_range: (self.carry_search_low, self.carry_search_high),
            carry_flag_range: (self.carry_flag_low, self.carry_flag_high),
            correction_passes: flag(self.has_correction_passes, "has_correction_passes")?
                .then_some(self.correction_passes),
            carry_tolerance: self.carry_tolerance,
            max_accurate_prices: self.max_accurate_prices,
        };
        let atm_vol = flag(self.has_atm_vol, "has_atm_vol")?.then_some(self.atm_vol);
        Ok((config, atm_vol))
    }
}

fn core_result(res: &ImpliedForwardResult) -> ItofinImpliedForwardResult {
    ItofinImpliedForwardResult {
        forward: res.forward,
        forward_bid_strict: res.forward_bid_strict,
        forward_ask_strict: res.forward_ask_strict,
        forward_bid_robust: res.forward_bid_robust,
        forward_ask_robust: res.forward_ask_robust,
        discount_factor: res.discount_factor,
        implied_carry_rate: res.implied_carry_rate.unwrap_or(f64::NAN),
        status_code: status_code(res.diagnostics.status),
        pairs_used: res.diagnostics.pairs_used as u32,
        pairs_pruned: res.diagnostics.pairs_pruned as u32,
        wls_rmse: res.diagnostics.wls_rmse,
    }
}

fn extended_result(res: &ImpliedForwardResult) -> ItofinImpliedForwardResultEx {
    let model = res.american;
    ItofinImpliedForwardResultEx {
        core: core_result(res),
        tier_code: tier_code(res.tier),
        flags: u32::from(res.flags.bits()),
        has_model: i32::from(model.is_some()),
        solved_carry: model.map_or(f64::NAN, |m| m.solved_carry),
        call_put_iv_gap: model.map_or(f64::NAN, |m| m.call_put_iv_gap),
        iterations: model.map_or(0, |m| m.iterations as u32),
        accurate_prices_used: model.map_or(0, |m| m.accurate_prices_used as u32),
        total_pairs_received: res.diagnostics.total_pairs_received as u32,
        is_crossed: i32::from(res.diagnostics.is_crossed),
        box_arbitrage_violations: res.diagnostics.box_arbitrage_violations as u32,
        min_spread: res.diagnostics.min_spread,
        max_spread: res.diagnostics.max_spread,
    }
}

/// Reads the quotes and settings, runs the estimator and returns the core result.
///
/// # Safety
/// Pointers must be live, valid and aligned, and the quote arrays must hold `count` values.
#[allow(clippy::too_many_arguments)]
unsafe fn run(
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
    american: AmericanModelConfig,
    atm_vol: Option<f64>,
) -> BindingResult<ImpliedForwardResult> {
    let strikes = unsafe { input_slice(strikes, count)? };
    let call_bids = unsafe { input_slice(call_bids, count)? };
    let call_asks = unsafe { input_slice(call_asks, count)? };
    let put_bids = unsafe { input_slice(put_bids, count)? };
    let put_asks = unsafe { input_slice(put_asks, count)? };

    let spot_val = if spot.is_null() {
        None
    } else {
        check_ptr(spot)?;
        Some(unsafe { *spot })
    };

    let df_val = if discount_factor.is_null() {
        None
    } else {
        check_ptr(discount_factor)?;
        Some(unsafe { *discount_factor })
    };

    let convention = match convention_code {
        0 => MarketConvention::EuropeanVanilla,
        1 => MarketConvention::CryptoInverse,
        2 => {
            let s = spot_val.ok_or_else(|| {
                BindingError::invalid("AmericanEquity convention requires spot pointer")
            })?;
            MarketConvention::AmericanEquity { spot: s, atm_vol }
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
        american,
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

    Ok(ImpliedForward::calculate(
        &quotes,
        convention,
        &config,
        expiry_years,
    )?)
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
            let res = run(
                strikes,
                call_bids,
                call_asks,
                put_bids,
                put_asks,
                count,
                convention_code,
                spot,
                discount_factor,
                expiry_years,
                AmericanModelConfig::default(),
                None,
            )?;
            output(out, core_result(&res))
        })
    }
}

/// Write the default settings of the model-based American path.
/// # Safety
/// Output and error pointers must follow the crate C caller contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn itofin_implied_forward_options_default(
    out: *mut ItofinImpliedForwardOptions,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        without_context(error, || {
            output(out, ItofinImpliedForwardOptions::defaults())
        })
    }
}

/// Like `itofin_implied_forward_calculate`, with the model-path settings and extended result.
/// The options and their dividend arrays are copied and not retained; `out` is written only on
/// success.
/// # Safety
/// Pointers must be live, valid, aligned, and non-overlapping.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn itofin_implied_forward_calculate_with_options(
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
    options: *const ItofinImpliedForwardOptions,
    out: *mut ItofinImpliedForwardResultEx,
    error: *mut ItofinError,
) -> i32 {
    unsafe {
        without_context(error, || {
            check_ptr(options)?;
            let (american, atm_vol) = (*options).parse()?;
            let res = run(
                strikes,
                call_bids,
                call_asks,
                put_bids,
                put_asks,
                count,
                convention_code,
                spot,
                discount_factor,
                expiry_years,
                american,
                atm_vol,
            )?;
            output(out, extended_result(&res))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn status_codes_are_stable() {
        let codes = [
            (ForwardStatus::Valid, 0),
            (ForwardStatus::WarningCrossedSyntheticQuotes, 1),
            (ForwardStatus::WarningBoxSpreadArbitrage, 2),
            (ForwardStatus::WarningImplausibleDiscountFactor, 3),
            (ForwardStatus::WarningSpotDeviationExceeded, 4),
            (ForwardStatus::DegradedAmericanBounds, 5),
            (ForwardStatus::WarningImplausibleCarry, 6),
        ];
        for (status, code) in codes {
            assert_eq!(status_code(status), code, "{status:?}");
        }
    }

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

    // --- extended entry point -------------------------------------------------------------

    fn erf(x: f64) -> f64 {
        let mut sum = x;
        let mut term = x;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x * x / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// Strikes, call bids and asks, put bids and asks.
    type Quotes = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// Black-priced quotes with the given half spread: parity-consistent inputs for the model path.
    fn black_quotes(half: f64) -> Quotes {
        let (spot, rate, carry, t, vol) = (100.0_f64, 0.01_f64, 0.02_f64, 0.5_f64, 0.25_f64);
        let forward = spot * ((rate - carry) * t).exp();
        let df = (-rate * t).exp();
        let sd = vol * t.sqrt();
        let n = |x: f64| 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
        let strikes = vec![92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5];
        let (mut cb, mut ca, mut pb, mut pa) = (vec![], vec![], vec![], vec![]);
        for &k in &strikes {
            let d1 = ((forward / k).ln() + 0.5 * sd * sd) / sd;
            let d2 = d1 - sd;
            let call = df * (forward * n(d1) - k * n(d2));
            let put = df * (k * n(-d2) - forward * n(-d1));
            cb.push(call - half);
            ca.push(call + half);
            pb.push(put - half);
            pa.push(put + half);
        }
        (strikes, cb, ca, pb, pa)
    }

    fn empty_result() -> ItofinImpliedForwardResultEx {
        ItofinImpliedForwardResultEx {
            core: ItofinImpliedForwardResult {
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
            },
            tier_code: -2,
            flags: 0xffff,
            has_model: -1,
            solved_carry: 0.0,
            call_put_iv_gap: 0.0,
            iterations: 0,
            accurate_prices_used: 0,
            total_pairs_received: 0,
            is_crossed: -1,
            box_arbitrage_violations: 0,
            min_spread: 0.0,
            max_spread: 0.0,
        }
    }

    fn default_options() -> ItofinImpliedForwardOptions {
        let mut options = std::mem::MaybeUninit::<ItofinImpliedForwardOptions>::uninit();
        let rc =
            unsafe { itofin_implied_forward_options_default(options.as_mut_ptr(), null_mut()) };
        assert_eq!(rc, 0);
        unsafe { options.assume_init() }
    }

    #[allow(clippy::too_many_arguments)]
    fn american_call(
        quotes: &Quotes,
        convention: i32,
        discount_factor: Option<f64>,
        options: &ItofinImpliedForwardOptions,
        out: &mut ItofinImpliedForwardResultEx,
    ) -> i32 {
        let spot = 100.0_f64;
        let df = discount_factor.unwrap_or(1.0);
        unsafe {
            itofin_implied_forward_calculate_with_options(
                quotes.0.as_ptr(),
                quotes.1.as_ptr(),
                quotes.2.as_ptr(),
                quotes.3.as_ptr(),
                quotes.4.as_ptr(),
                quotes.0.len(),
                convention,
                &spot,
                if discount_factor.is_some() {
                    &df
                } else {
                    null()
                },
                0.5,
                options,
                out,
                null_mut(),
            )
        }
    }

    #[test]
    fn options_default_matches_the_core_defaults() {
        let o = default_options();
        assert_eq!(o.has_atm_vol, 0);
        assert_eq!(o.dividend_count, 0);
        assert_eq!(o.level, 0);
        assert_eq!((o.carry_search_low, o.carry_search_high), (-0.10, 0.25));
        assert_eq!((o.carry_flag_low, o.carry_flag_high), (-0.05, 0.15));
        assert_eq!(o.has_correction_passes, 0);
        assert_eq!(o.carry_tolerance, 1e-7);
        assert_eq!(o.corridor_kappa, 1.0);
        assert_eq!(o.max_accurate_prices, 64);
    }

    #[test]
    fn extended_call_reports_the_model_tier() {
        let quotes = black_quotes(0.01);
        let mut out = empty_result();
        let rc = american_call(
            &quotes,
            2,
            Some((-0.01_f64 * 0.5).exp()),
            &default_options(),
            &mut out,
        );
        assert_eq!(rc, 0);
        assert_eq!(out.tier_code, 1);
        assert_eq!(out.flags, 0);
        assert_eq!(out.has_model, 1);
        assert!(out.solved_carry.is_finite() && (out.solved_carry - 0.02).abs() < 0.02);
        assert!(out.call_put_iv_gap.abs() < 1e-6);
        assert!(out.iterations >= 1);
        assert_eq!(out.accurate_prices_used, 0);
        assert_eq!(out.total_pairs_received, 7);
        assert!(out.min_spread > 0.0 && out.max_spread >= out.min_spread);
        assert!(out.core.forward > 0.0);
        assert!(out.core.pairs_used >= 2);
    }

    #[test]
    fn extended_call_without_a_discount_factor_reports_the_corridor_tier() {
        let quotes = black_quotes(0.01);
        let mut out = empty_result();
        let rc = american_call(&quotes, 2, None, &default_options(), &mut out);
        assert_eq!(rc, 0);
        assert_eq!(out.tier_code, 0);
        assert_eq!(out.has_model, 0);
        assert!(out.solved_carry.is_nan());
    }

    #[test]
    fn extended_call_on_a_european_convention_reports_no_tier() {
        let quotes = black_quotes(0.5);
        let mut out = empty_result();
        let rc = american_call(&quotes, 0, Some(1.0), &default_options(), &mut out);
        assert_eq!(rc, 0);
        assert_eq!(out.tier_code, -1);
        assert_eq!(out.has_model, 0);
        assert_eq!(out.flags, 0);
    }

    #[test]
    fn extended_call_accurate_level_and_schedule_flags() {
        let quotes = black_quotes(0.01);
        let df = (-0.01_f64 * 0.5).exp();

        let mut accurate = default_options();
        accurate.level = 2;
        let mut out = empty_result();
        assert_eq!(american_call(&quotes, 2, Some(df), &accurate, &mut out), 0);
        assert_eq!(out.tier_code, 2);
        assert_eq!(out.accurate_prices_used as usize, 2 * quotes.0.len());

        // A schedule with the fast level is ignored and flagged (bit 0).
        let times = [0.2_f64];
        let amounts = [0.5_f64];
        let mut fast = default_options();
        fast.level = 1;
        fast.dividend_times = times.as_ptr();
        fast.dividend_amounts = amounts.as_ptr();
        fast.dividend_count = 1;
        let mut out = empty_result();
        assert_eq!(american_call(&quotes, 2, Some(df), &fast, &mut out), 0);
        assert_eq!(out.tier_code, 1);
        assert_eq!(out.flags, 1);

        // Auto with a schedule is accurate and unflagged.
        let mut auto = default_options();
        auto.dividend_times = times.as_ptr();
        auto.dividend_amounts = amounts.as_ptr();
        auto.dividend_count = 1;
        let mut out = empty_result();
        assert_eq!(american_call(&quotes, 2, Some(df), &auto, &mut out), 0);
        assert_eq!(out.tier_code, 2);
        assert_eq!(out.flags, 0);

        // A budget that cannot pay for the correction falls back to fast, bit 1.
        let mut tight = default_options();
        tight.level = 2;
        tight.max_accurate_prices = 1;
        let mut out = empty_result();
        assert_eq!(american_call(&quotes, 2, Some(df), &tight, &mut out), 0);
        assert_eq!(out.tier_code, 1);
        assert_eq!(out.flags, 2);
    }

    #[test]
    fn extended_call_rejects_invalid_options() {
        let quotes = black_quotes(0.01);
        let df = Some((-0.01_f64 * 0.5).exp());
        let bad = |edit: &dyn Fn(&mut ItofinImpliedForwardOptions)| {
            let mut options = default_options();
            edit(&mut options);
            let mut out = empty_result();
            let rc = american_call(&quotes, 2, df, &options, &mut out);
            (rc, out.tier_code)
        };
        // Invalid level, flag value, null dividend arrays with a count, inverted range.
        assert_ne!(bad(&|o| o.level = 7).0, 0);
        assert_ne!(bad(&|o| o.has_atm_vol = 2).0, 0);
        assert_ne!(bad(&|o| o.has_correction_passes = -1).0, 0);
        assert_ne!(bad(&|o| o.dividend_count = 2).0, 0);
        assert_ne!(
            bad(&|o| {
                o.carry_search_low = 0.25;
                o.carry_search_high = -0.10;
            })
            .0,
            0
        );
        // The output is untouched on failure.
        assert_eq!(bad(&|o| o.level = 7).1, -2);

        let mut out = empty_result();
        let rc = unsafe {
            itofin_implied_forward_calculate_with_options(
                quotes.0.as_ptr(),
                quotes.1.as_ptr(),
                quotes.2.as_ptr(),
                quotes.3.as_ptr(),
                quotes.4.as_ptr(),
                quotes.0.len(),
                2,
                null(),
                null(),
                0.5,
                null(),
                &mut out,
                null_mut(),
            )
        };
        assert_ne!(rc, 0, "a null options pointer must be rejected");
    }

    #[test]
    fn old_and_extended_entry_points_agree_on_the_shared_fields() {
        let quotes = black_quotes(0.5);
        let spot = 100.0_f64;
        let df = 1.0_f64;
        let mut old = empty_result().core;
        let rc = unsafe {
            itofin_implied_forward_calculate(
                quotes.0.as_ptr(),
                quotes.1.as_ptr(),
                quotes.2.as_ptr(),
                quotes.3.as_ptr(),
                quotes.4.as_ptr(),
                quotes.0.len(),
                0,
                &spot,
                &df,
                0.5,
                &mut old,
                null_mut(),
            )
        };
        assert_eq!(rc, 0);
        let mut ext = empty_result();
        assert_eq!(
            american_call(&quotes, 0, Some(1.0), &default_options(), &mut ext),
            0
        );
        assert_eq!(old.forward, ext.core.forward);
        assert_eq!(old.status_code, ext.core.status_code);
        assert_eq!(old.pairs_used, ext.core.pairs_used);
        assert_eq!(old.wls_rmse, ext.core.wls_rmse);
    }
}
