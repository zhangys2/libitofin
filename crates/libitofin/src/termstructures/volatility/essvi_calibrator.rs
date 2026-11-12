#![allow(
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
//! Sequential eSSVI calibration.
//!
//! Transliteration of `ql-essvi` `cpp/calibrator.cpp`. Rho candidates run on the
//! calling thread. Bound-call totals still match the C++ counter.

use super::essvi::detail::{
    mm_psi2_bound_search_hint_with_safety, passes_adaptive_weighted_quote_cap,
};
use super::essvi::{
    mm_psi2_bound_with_safety, slice_total_variance, validate, MmBoundCallScope, Slice, Surface,
};

const EPS: f64 = 1e-8;
const CALENDAR_HEADROOM: f64 = 0.005;
const FULL_COLD_SEARCH_BUDGET: usize = 8;
const DEGRADED_AGE_BEFORE_FORCED_COLD: usize = 3;
const FAST_SCENARIO_MEAN_ABSOLUTE_MOVE: f64 = 0.5 * 0.20680703819458962;
const COLD_STATE_VARIANCE: f64 = 0.25;
const STATE_PROCESS_VARIANCE: [f64; 3] = [1e-2, 5.0, 5.0];
const POSTERIOR_VARIANCE_FLOOR: f64 = 1e-8;
const RHO_OFFSETS: [f64; 3] = [0.0, -0.05, 0.05];
const ALPHAS: [f64; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
const THETA_MULTIPLIERS: [f64; 6] = [0.90, 0.95, 1.00, 1.05, 1.10, 1.20];

type StateMatrix = [[f64; 3]; 3];
type StateCovariance = [f64; 6];

fn cpp_dtoa(value: f64) -> String {
    format!("{value:.6}")
}

fn atanh(value: f64) -> f64 {
    0.5 * ((1.0 + value) / (1.0 - value)).ln()
}

fn same_value(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// How a calibration arrived at its slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationPath {
    /// Cold sequential fit, or a warm call with no usable prior.
    Cold,
    /// Every accepted slice came from the warm path.
    Warm,
    /// At least one slice took a fallback search.
    Fallback,
    /// The full-cold budget was exhausted and the surface still validated.
    Degraded,
}

/// One included or excluded option quote. `weight` defaults to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Quote {
    /// Stable quote identifier.
    pub id: String,
    /// Year fraction.
    pub expiry: f64,
    /// Forward price.
    pub forward: f64,
    /// Discount factor. Carried with the quote; the fit uses total variance.
    pub discount_factor: f64,
    /// Strike.
    pub strike: f64,
    /// Bid implied volatility.
    pub bid_iv: f64,
    /// Ask implied volatility.
    pub ask_iv: f64,
    /// Mid implied volatility.
    pub mid_iv: f64,
    /// False quotes are ignored by calibration.
    pub included: bool,
    /// Positive finite weight for included quotes.
    pub weight: f64,
}

impl Default for Quote {
    fn default() -> Self {
        Self {
            id: String::new(),
            expiry: 0.0,
            forward: 0.0,
            discount_factor: 0.0,
            strike: 0.0,
            bid_iv: 0.0,
            ask_iv: 0.0,
            mid_iv: 0.0,
            included: false,
            weight: 1.0,
        }
    }
}

impl Quote {
    /// Builds an included quote with weight 1.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        expiry: f64,
        forward: f64,
        discount_factor: f64,
        strike: f64,
        bid_iv: f64,
        ask_iv: f64,
        mid_iv: f64,
    ) -> Self {
        Self {
            id: id.into(),
            expiry,
            forward,
            discount_factor,
            strike,
            bid_iv,
            ask_iv,
            mid_iv,
            included: true,
            weight: 1.0,
        }
    }
}

/// Knobs for [`calibrate_warm`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmOptions {
    /// Maximum bid/ask move that still counts as an unchanged quote.
    pub quote_tol: f64,
    /// Echoed onto [`CalibrationResult::loss_tolerance`]. The gate itself uses `quote_cap`.
    pub loss_tol: f64,
    /// Cap on the weighted excess ratio.
    pub quote_cap: f64,
}

impl Default for WarmOptions {
    fn default() -> Self {
        Self {
            quote_tol: 1e-5,
            loss_tol: 0.05,
            quote_cap: 0.1,
        }
    }
}

/// Counts of warm-path rejections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FallbackReasonCounts {
    /// Local solver returned an infeasible slice.
    pub infeasible_local_result: u64,
    /// Quote-cap gate rejected a candidate.
    pub loss_above_gate: u64,
    /// Calendar repair did not restore the headroom margin.
    pub calendar_repair_failed: u64,
    /// A trial left the butterfly or psi bounds.
    pub bound_check_failed: u64,
    /// The full-cold search budget was spent.
    pub budget_exhausted: u64,
}

/// Cold or warm calibration outcome. A failed fit still returns this struct.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationResult {
    /// Fitted slices. A failed cold fit keeps the slices that succeeded.
    pub surface: Surface,
    /// Consecutive unchanged-parameter reuses, aligned with `surface.slices`.
    pub reuse_ages: Vec<usize>,
    /// Packed covariance `[00, 01, 02, 11, 12, 22]` in `(log theta, atanh rho, log psi)`.
    pub posterior_covariances: Vec<StateCovariance>,
    /// True when `error` is empty and the surface validated.
    pub converged: bool,
    /// Band-loss objective.
    pub objective: f64,
    /// Feasibility and loss evaluations charged to this fit.
    pub iterations: i32,
    /// Set equal to `iterations` when the fit returns.
    pub objective_evaluations: u64,
    /// Slices that entered the local solver.
    pub local_solver_slices: usize,
    /// Local-solver starting points.
    pub local_solver_starts: usize,
    /// Why warm candidates were rejected.
    pub fallback_reason_counts: FallbackReasonCounts,
    /// Restricted cold searches.
    pub restricted_cold_searches: usize,
    /// Full cold searches.
    pub full_cold_searches: usize,
    /// Full cold searches forced by degraded age.
    pub forced_full_cold_searches: usize,
    /// Slices whose degraded age is positive.
    pub degraded_slices: usize,
    /// Consecutive quality-gate failures, aligned with `surface.slices`.
    pub degraded_ages: Vec<usize>,
    /// Quotes that contributed an excess ratio.
    pub quality_quote_count: u64,
    /// Largest unweighted excess ratio across quotes.
    pub quality_maximum_excess_ratio: f64,
    /// Root-mean-square unweighted excess ratio.
    pub quality_rms_excess_ratio: f64,
    /// Largest per-slice maximum excess ratio.
    pub worst_slice_max_excess_ratio: f64,
    /// Largest per-slice root-mean-square excess ratio.
    pub worst_slice_rms_excess_ratio: f64,
    /// Objective evaluations inside local solves.
    pub local_objective_evaluations: u64,
    /// Martini–Mingone evaluations inside local solves.
    pub local_mm_bound_calls: u64,
    /// Martini–Mingone evaluations for the whole fit.
    pub mm_bound_calls: u64,
    /// Slices that were refit.
    pub slices_refit: usize,
    /// Slices reused from the previous surface.
    pub slices_reused: usize,
    /// Slices that took a fallback candidate.
    pub fallback_slices: usize,
    /// Quote-quality comparisons against a cold reference.
    pub quality_checks: usize,
    /// Band loss of restricted candidates that missed the cap.
    pub quality_warm_loss: f64,
    /// Band loss of the cold references used in those comparisons.
    pub quality_reference_loss: f64,
    /// Sum of band-loss scales for those comparisons.
    pub quality_loss_scale: f64,
    /// Copy of [`WarmOptions::loss_tol`] for warm fits.
    pub loss_tolerance: f64,
    /// Path taken by this fit.
    pub path: CalibrationPath,
    /// Empty when `converged` is true.
    pub error: String,
}

impl Default for CalibrationResult {
    fn default() -> Self {
        Self {
            surface: Surface::default(),
            reuse_ages: Vec::new(),
            posterior_covariances: Vec::new(),
            converged: false,
            objective: 0.0,
            iterations: 0,
            objective_evaluations: 0,
            local_solver_slices: 0,
            local_solver_starts: 0,
            fallback_reason_counts: FallbackReasonCounts::default(),
            restricted_cold_searches: 0,
            full_cold_searches: 0,
            forced_full_cold_searches: 0,
            degraded_slices: 0,
            degraded_ages: Vec::new(),
            quality_quote_count: 0,
            quality_maximum_excess_ratio: 0.0,
            quality_rms_excess_ratio: 0.0,
            worst_slice_max_excess_ratio: 0.0,
            worst_slice_rms_excess_ratio: 0.0,
            local_objective_evaluations: 0,
            local_mm_bound_calls: 0,
            mm_bound_calls: 0,
            slices_refit: 0,
            slices_reused: 0,
            fallback_slices: 0,
            quality_checks: 0,
            quality_warm_loss: 0.0,
            quality_reference_loss: 0.0,
            quality_loss_scale: 0.0,
            loss_tolerance: 0.0,
            path: CalibrationPath::Cold,
            error: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct PreparedQuote {
    k: f64,
    weight: f64,
    bid_variance: f64,
    ask_variance: f64,
    mid_iv: f64,
    has_band: bool,
    has_mid: bool,
}

struct QuoteGroups {
    expiries: Vec<f64>,
    slices: Vec<Vec<usize>>,
}

struct CalendarBounds {
    present: bool,
    expiry_increases: bool,
    previous_psi_valid: bool,
    theta_lower: f64,
    psi_lower: f64,
    ratio_lower: f64,
    previous_psi_over_theta: f64,
}

impl Default for CalendarBounds {
    fn default() -> Self {
        Self {
            present: false,
            expiry_increases: false,
            previous_psi_valid: false,
            theta_lower: 0.0,
            psi_lower: 0.0,
            ratio_lower: 0.0,
            previous_psi_over_theta: 0.0,
        }
    }
}

struct BoundMemo {
    search_hint: bool,
    theta: f64,
    abs_rho: f64,
    value: f64,
}

impl BoundMemo {
    fn new(search_hint: bool) -> Self {
        Self {
            search_hint,
            theta: f64::NAN,
            abs_rho: f64::NAN,
            value: 0.0,
        }
    }

    fn get(&mut self, theta: f64, abs_rho: f64) -> f64 {
        if theta == self.theta && abs_rho == self.abs_rho {
            return self.value;
        }
        self.value = if self.search_hint {
            mm_psi2_bound_search_hint_with_safety(theta, abs_rho, 1e-10)
        } else {
            mm_psi2_bound_with_safety(theta, abs_rho, EPS)
        };
        self.theta = theta;
        self.abs_rho = abs_rho;
        self.value
    }
}

struct Atm {
    theta_star: f64,
    k_star: f64,
    quote_index: Option<usize>,
}

struct FitOutput {
    result: CalibrationResult,
    calendar_headroom: f64,
}

fn finish_result(result: &mut CalibrationResult, bound_calls: u64) {
    result.objective_evaluations = result.iterations as u64;
    result.mm_bound_calls = bound_calls;
}

fn group_quotes(quotes: &[Quote]) -> QuoteGroups {
    let mut expiries = Vec::new();
    for quote in quotes {
        if quote.included && quote.expiry.is_finite() {
            expiries.push(quote.expiry);
        }
    }
    expiries.sort_by(f64::total_cmp);
    expiries.dedup();
    let mut slices = vec![Vec::new(); expiries.len()];
    for (index, quote) in quotes.iter().enumerate() {
        if !quote.included || !quote.expiry.is_finite() {
            continue;
        }
        if let Ok(position) = expiries.binary_search_by(|expiry| expiry.total_cmp(&quote.expiry)) {
            slices[position].push(index);
        }
    }
    QuoteGroups { expiries, slices }
}

fn quote_group_at(groups: &QuoteGroups, expiry: f64) -> Option<usize> {
    groups
        .expiries
        .binary_search_by(|item| item.total_cmp(&expiry))
        .ok()
}

fn surface_slice_at(surface: &Surface, expiry: f64) -> Option<Slice> {
    surface
        .slices
        .iter()
        .find(|slice| slice.expiry == expiry)
        .copied()
}

fn movement_below_tolerance(current: f64, previous: f64, tolerance: f64) -> bool {
    if same_value(current, previous) {
        return true;
    }
    current.is_finite() && previous.is_finite() && (current - previous).abs() < tolerance
}

fn quote_group_reusable(
    quotes: &[Quote],
    current: &[usize],
    previous_quotes: &[Quote],
    previous: &[usize],
    quote_tol: f64,
) -> bool {
    if current.len() != previous.len() {
        return false;
    }
    for (&current_index, &previous_index) in current.iter().zip(previous) {
        let a = &quotes[current_index];
        let b = &previous_quotes[previous_index];
        if a.id != b.id
            || !same_value(a.expiry, b.expiry)
            || !same_value(a.forward, b.forward)
            || !same_value(a.strike, b.strike)
            || !same_value(a.mid_iv, b.mid_iv)
            || !same_value(a.weight, b.weight)
            || !movement_below_tolerance(a.bid_iv, b.bid_iv, quote_tol)
            || !movement_below_tolerance(a.ask_iv, b.ask_iv, quote_tol)
        {
            return false;
        }
    }
    true
}

fn mean_absolute_mid_iv_move(groups: &QuoteGroups, quotes: &[Quote], previous: &Surface) -> f64 {
    let mut absolute_move_sum = 0.0;
    let mut matched_quotes = 0usize;
    for (group, &expiry) in groups.expiries.iter().enumerate() {
        let Some(prior) = surface_slice_at(previous, expiry) else {
            continue;
        };
        if !(expiry > 0.0) || !expiry.is_finite() {
            continue;
        }
        for &index in &groups.slices[group] {
            let now = &quotes[index];
            if !now.mid_iv.is_finite()
                || !(now.forward > 0.0)
                || !now.forward.is_finite()
                || !(now.strike > 0.0)
                || !now.strike.is_finite()
            {
                continue;
            }
            let k = (now.strike / now.forward).ln();
            let total_variance = slice_total_variance(&prior, k);
            if !k.is_finite() || !(total_variance > 0.0) || !total_variance.is_finite() {
                continue;
            }
            let prior_iv = (total_variance / expiry).sqrt();
            if !prior_iv.is_finite() {
                continue;
            }
            absolute_move_sum += (now.mid_iv - prior_iv).abs();
            matched_quotes += 1;
        }
    }
    if matched_quotes == 0 {
        0.0
    } else {
        absolute_move_sum / matched_quotes as f64
    }
}

fn prepare_quotes(quotes: &[Quote], indexes: &[usize]) -> Vec<PreparedQuote> {
    indexes
        .iter()
        .map(|&index| {
            let quote = &quotes[index];
            let has_band = quote.included && quote.bid_iv.is_finite() && quote.ask_iv.is_finite();
            let has_mid = quote.included && quote.mid_iv.is_finite();
            let k = if has_band || has_mid {
                (quote.strike / quote.forward).ln()
            } else {
                0.0
            };
            PreparedQuote {
                k,
                weight: quote.weight,
                bid_variance: quote.expiry * quote.bid_iv * quote.bid_iv,
                ask_variance: quote.expiry * quote.ask_iv * quote.ask_iv,
                mid_iv: quote.mid_iv,
                has_band,
                has_mid,
            }
        })
        .collect()
}

fn band_loss_scale(quotes: &[PreparedQuote]) -> f64 {
    let mut scale = 0.0;
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let width = (quote.ask_variance - quote.bid_variance).max(1e-6);
        scale += quote.weight * quote.weight * width * width;
    }
    scale
}

fn band_loss(slice: &Slice, quotes: &[PreparedQuote]) -> f64 {
    let mut loss = 0.0;
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let variance = slice_total_variance(slice, quote.k);
        let residual = variance - variance.clamp(quote.bid_variance, quote.ask_variance);
        loss += quote.weight * quote.weight * residual * residual;
    }
    loss
}

fn quote_excess_ratio(slice: &Slice, quote: &PreparedQuote) -> f64 {
    let variance = slice_total_variance(slice, quote.k);
    let excess = 0.0_f64.max((quote.bid_variance - variance).max(variance - quote.ask_variance));
    let width = (quote.ask_variance - quote.bid_variance).max(1e-6);
    excess / width
}

fn passes_quote_cap(slice: &Slice, quotes: &[PreparedQuote], cap: f64) -> bool {
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let ratio = quote.weight * quote_excess_ratio(slice, quote);
        if !ratio.is_finite() || ratio > cap {
            return false;
        }
    }
    true
}

fn passes_adaptive_quote_cap(
    candidate: &Slice,
    cold: &Slice,
    quotes: &[PreparedQuote],
    cap: f64,
) -> bool {
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let candidate_ratio = quote.weight * quote_excess_ratio(candidate, quote);
        let cold_ratio = quote.weight * quote_excess_ratio(cold, quote);
        if !passes_adaptive_weighted_quote_cap(candidate_ratio, cold_ratio, quote.weight, cap) {
            return false;
        }
    }
    true
}

fn calendar_bounds(
    previous: Option<&Slice>,
    expiry: f64,
    rho: f64,
    headroom: f64,
) -> CalendarBounds {
    let Some(previous) = previous else {
        return CalendarBounds::default();
    };
    let ratio = ((1.0 + previous.rho) / (1.0 + rho)).max((1.0 - previous.rho) / (1.0 - rho));
    CalendarBounds {
        present: true,
        expiry_increases: expiry > previous.expiry,
        previous_psi_valid: previous.psi > EPS,
        theta_lower: previous.theta + EPS,
        psi_lower: previous.psi + EPS,
        ratio_lower: previous.psi * ratio * (1.0 + headroom) + EPS,
        previous_psi_over_theta: previous.psi / previous.theta,
    }
}

fn calendar_bounds_safe(slice: &Slice, bounds: &CalendarBounds) -> bool {
    if !bounds.present {
        return true;
    }
    if !(bounds.expiry_increases
        && slice.theta > bounds.theta_lower
        && slice.psi > bounds.psi_lower
        && bounds.previous_psi_valid
        && slice.psi > EPS)
    {
        return false;
    }
    let upper = bounds.previous_psi_over_theta * slice.theta;
    slice.psi > bounds.ratio_lower && slice.psi < upper - EPS
}

fn bound_safe(slice: &Slice, bound: &mut BoundMemo) -> bool {
    let abs_rho = slice.rho.abs();
    if !(slice.theta > EPS
        && slice.psi > EPS
        && abs_rho < 1.0 - EPS
        && slice.psi * (1.0 + abs_rho) < 4.0 - EPS)
    {
        return false;
    }
    slice.psi * slice.psi <= bound.get(slice.theta, abs_rho) - EPS
}

fn pair_safe(slice: &Slice, bound: &mut BoundMemo, calendar: &CalendarBounds) -> bool {
    bound_safe(slice, bound) && calendar_bounds_safe(slice, calendar)
}

fn nearest_atm(quotes: &[PreparedQuote], expiry: f64) -> Atm {
    let mut atm = Atm {
        theta_star: f64::NAN,
        k_star: 0.0,
        quote_index: None,
    };
    let mut best = f64::INFINITY;
    for (index, quote) in quotes.iter().enumerate() {
        if !quote.has_mid {
            continue;
        }
        let abs_k = quote.k.abs();
        if abs_k < best {
            best = abs_k;
            atm.k_star = quote.k;
            atm.quote_index = Some(index);
            atm.theta_star = expiry * quote.mid_iv * quote.mid_iv;
        }
    }
    atm
}

fn anchored(
    expiry: f64,
    atm: &Atm,
    rho: f64,
    psi: f64,
    previous: Option<&Slice>,
    bound: &mut BoundMemo,
) -> Slice {
    let mut theta = atm.theta_star - rho * psi * atm.k_star;
    if let Some(previous) = previous {
        theta = theta.max(previous.theta + EPS);
        if previous.psi > EPS {
            theta = theta.max(psi * previous.theta / previous.psi + EPS);
        }
    }
    theta = theta.max(EPS);
    for _ in 0..48 {
        if psi * psi <= bound.get(theta, rho.abs()) - 2.0 * EPS {
            break;
        }
        theta *= 1.08;
    }
    Slice {
        expiry,
        theta,
        psi,
        rho,
    }
}

fn psi_upper(theta: f64, rho: f64, bound: &mut BoundMemo) -> f64 {
    let abs_rho = rho.abs();
    if !(abs_rho < 1.0 - EPS) || !(theta > EPS) {
        return 0.0;
    }
    let roger_lee = 4.0 / (1.0 + abs_rho) - EPS;
    let mm = bound.get(theta, abs_rho).max(0.0).sqrt() - EPS;
    roger_lee.min(mm)
}

fn psi_lower(previous: Option<&Slice>, rho: f64) -> f64 {
    let floor = 2.0 * EPS;
    let Some(previous) = previous else {
        return floor;
    };
    let ratio = ((1.0 + previous.rho) / (1.0 + rho)).max((1.0 - previous.rho) / (1.0 - rho));
    floor.max((previous.psi + EPS).max(previous.psi * ratio + EPS))
}

fn cold_covariance() -> StateCovariance {
    [
        COLD_STATE_VARIANCE,
        0.0,
        0.0,
        COLD_STATE_VARIANCE,
        0.0,
        COLD_STATE_VARIANCE,
    ]
}

fn unpack_covariance(packed: &StateCovariance) -> StateMatrix {
    [
        [packed[0], packed[1], packed[2]],
        [packed[1], packed[3], packed[4]],
        [packed[2], packed[4], packed[5]],
    ]
}

fn pack_covariance(matrix: &StateMatrix) -> StateCovariance {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][1],
        matrix[1][2],
        matrix[2][2],
    ]
}

fn cholesky(matrix: &StateMatrix) -> Option<StateMatrix> {
    let mut lower = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..=row {
            let mut value = matrix[row][column];
            if !value.is_finite() {
                return None;
            }
            for index in 0..column {
                value -= lower[row][index] * lower[column][index];
            }
            if row == column {
                if !(value > 0.0) || !value.is_finite() {
                    return None;
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
                if !lower[row][column].is_finite() {
                    return None;
                }
            }
        }
    }
    Some(lower)
}

fn valid_covariance(covariance: &StateCovariance) -> bool {
    cholesky(&unpack_covariance(covariance)).is_some()
}

fn invert_positive_definite(matrix: &StateMatrix) -> Option<StateMatrix> {
    let lower = cholesky(matrix)?;
    let mut inverse = [[0.0; 3]; 3];
    for column in 0..3 {
        let mut forward = [0.0; 3];
        let mut solution = [0.0; 3];
        for row in 0..3 {
            let mut value = if row == column { 1.0 } else { 0.0 };
            for index in 0..row {
                value -= lower[row][index] * forward[index];
            }
            forward[row] = value / lower[row][row];
        }
        for row in (0..3).rev() {
            let mut value = forward[row];
            for index in row + 1..3 {
                value -= lower[index][row] * solution[index];
            }
            solution[row] = value / lower[row][row];
        }
        for row in 0..3 {
            if !solution[row].is_finite() {
                return None;
            }
            inverse[row][column] = solution[row];
        }
    }
    for row in 0..3 {
        for column in row + 1..3 {
            let symmetric = 0.5 * (inverse[row][column] + inverse[column][row]);
            inverse[row][column] = symmetric;
            inverse[column][row] = symmetric;
        }
    }
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(inverse)
}

fn prior_precision_for(covariance: &StateCovariance) -> Option<(StateMatrix, StateCovariance)> {
    let mut predicted = unpack_covariance(covariance);
    for index in 0..3 {
        predicted[index][index] += STATE_PROCESS_VARIANCE[index];
    }
    let precision = invert_positive_definite(&predicted)?;
    let predicted_covariance = pack_covariance(&predicted);
    Some((precision, predicted_covariance))
}

fn state_coordinates(slice: &Slice) -> [f64; 3] {
    [slice.theta.ln(), atanh(slice.rho), slice.psi.ln()]
}

fn active_quote_information(
    slice: &Slice,
    quotes: &[PreparedQuote],
    loss_scale: f64,
) -> Option<StateMatrix> {
    if !(loss_scale > 0.0) || !loss_scale.is_finite() {
        return None;
    }
    let mut information = [[0.0; 3]; 3];
    let mut has_active_quote = false;
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let variance = slice_total_variance(slice, quote.k);
        if !variance.is_finite()
            || (variance >= quote.bid_variance && variance <= quote.ask_variance)
        {
            continue;
        }
        let theta = slice.theta;
        let rho = slice.rho;
        let psi = slice.psi;
        let a = psi * quote.k + rho * theta;
        let d = (a * a + theta * theta * (1.0 - rho * rho))
            .max(1e-30)
            .sqrt();
        let state_jacobian = [
            0.5 * (1.0 + (a * rho + theta * (1.0 - rho * rho)) / d) * theta,
            0.5 * (psi * quote.k + (a * theta - theta * theta * rho) / d) * (1.0 - rho * rho),
            0.5 * (rho * quote.k + a * quote.k / d) * psi,
        ];
        let scale = quote.weight * quote.weight / loss_scale;
        if !scale.is_finite() {
            return None;
        }
        for row in 0..3 {
            if !state_jacobian[row].is_finite() {
                return None;
            }
            for column in 0..3 {
                information[row][column] += scale * state_jacobian[row] * state_jacobian[column];
                if !information[row][column].is_finite() {
                    return None;
                }
            }
        }
        has_active_quote = true;
    }
    if has_active_quote {
        Some(information)
    } else {
        None
    }
}

fn refresh_covariance(
    predicted: &StateCovariance,
    fitted: &Slice,
    quotes: &[PreparedQuote],
    loss_scale: f64,
) -> StateCovariance {
    let Some(information) = active_quote_information(fitted, quotes, loss_scale) else {
        return *predicted;
    };
    let Some(mut precision) = invert_positive_definite(&unpack_covariance(predicted)) else {
        return *predicted;
    };
    for row in 0..3 {
        for column in 0..3 {
            precision[row][column] += information[row][column];
            if !precision[row][column].is_finite() {
                return *predicted;
            }
        }
    }
    let Some(mut covariance) = invert_positive_definite(&precision) else {
        return *predicted;
    };
    for coordinate in 0..3 {
        covariance[coordinate][coordinate] =
            covariance[coordinate][coordinate].max(POSTERIOR_VARIANCE_FLOOR);
    }
    let packed = pack_covariance(&covariance);
    if valid_covariance(&packed) {
        packed
    } else {
        *predicted
    }
}

fn posterior_covariance_at(result: &CalibrationResult, expiry: f64) -> Option<StateCovariance> {
    if !result.converged || result.posterior_covariances.len() != result.surface.slices.len() {
        return None;
    }
    for (slice, covariance) in result
        .surface
        .slices
        .iter()
        .zip(&result.posterior_covariances)
    {
        if slice.expiry != expiry {
            continue;
        }
        return if valid_covariance(covariance) {
            Some(*covariance)
        } else {
            None
        };
    }
    None
}

struct LossEvaluation {
    loss: f64,
    maximum_excess_ratio: f64,
    score: f64,
    normal: StateMatrix,
    gradient: [f64; 3],
}

fn loss_and_normal(
    slice: &Slice,
    quotes: &[PreparedQuote],
    prior_mean: Option<&[f64; 3]>,
    prior_precision: Option<&StateMatrix>,
    loss_scale: f64,
) -> LossEvaluation {
    let mut gradient = [0.0; 3];
    let mut normal = [[0.0; 3]; 3];
    let parameter_scale = [
        (slice.theta * 0.1).max(1e-3),
        0.1,
        (slice.psi * 0.1).max(1e-3),
    ];
    let evidence_scale = if prior_precision.is_some() && loss_scale > 0.0 {
        1.0 / loss_scale
    } else {
        1.0
    };
    let mut loss = 0.0;
    let mut score = 0.0;
    let mut maximum_excess_ratio: f64 = 0.0;
    for quote in quotes {
        if !quote.has_band {
            continue;
        }
        let weight_squared = quote.weight * quote.weight;
        let theta = slice.theta;
        let rho = slice.rho;
        let psi = slice.psi;
        let k = quote.k;
        let a = psi * k + rho * theta;
        let d = (a * a + theta * theta * (1.0 - rho * rho))
            .max(1e-30)
            .sqrt();
        let jacobian = [
            0.5 * (1.0 + (a * rho + theta * (1.0 - rho * rho)) / d),
            0.5 * (psi * k + (a * theta - theta * theta * rho) / d),
            0.5 * (rho * k + a * k / d),
        ];
        let scaled_jacobian = [
            jacobian[0] * parameter_scale[0],
            jacobian[1] * parameter_scale[1],
            jacobian[2] * parameter_scale[2],
        ];
        let variance = slice_total_variance(slice, k);
        let residual = variance - variance.clamp(quote.bid_variance, quote.ask_variance);
        loss += weight_squared * residual * residual;
        let score_weight = if prior_precision.is_some() {
            0.5 * evidence_scale
        } else {
            1.0
        };
        score += score_weight * weight_squared * residual * residual;
        let quote_width = (quote.ask_variance - quote.bid_variance).max(1e-6);
        let excess_ratio = quote.weight * residual.abs() / quote_width;
        maximum_excess_ratio = if excess_ratio.is_finite() {
            maximum_excess_ratio.max(excess_ratio)
        } else {
            f64::INFINITY
        };
        if residual == 0.0 {
            continue;
        }
        for row in 0..3 {
            gradient[row] += weight_squared * evidence_scale * scaled_jacobian[row] * residual;
            for column in 0..3 {
                normal[row][column] += weight_squared
                    * evidence_scale
                    * scaled_jacobian[row]
                    * scaled_jacobian[column];
            }
        }
    }
    if let (Some(prior_mean), Some(prior_precision)) = (prior_mean, prior_precision) {
        let delta = [
            slice.theta.ln() - prior_mean[0],
            atanh(slice.rho) - prior_mean[1],
            slice.psi.ln() - prior_mean[2],
        ];
        let state_scale = [
            parameter_scale[0] / slice.theta,
            parameter_scale[1] / (1.0 - slice.rho * slice.rho),
            parameter_scale[2] / slice.psi,
        ];
        for row in 0..3 {
            for column in 0..3 {
                score += 0.5 * delta[row] * prior_precision[row][column] * delta[column];
                gradient[row] += state_scale[row] * prior_precision[row][column] * delta[column];
                normal[row][column] +=
                    state_scale[row] * prior_precision[row][column] * state_scale[column];
            }
        }
    }
    LossEvaluation {
        loss,
        maximum_excess_ratio,
        score,
        normal,
        gradient,
    }
}

struct RhoCandidate {
    slice: Slice,
    loss: f64,
    evaluations: i32,
}

fn accept_rho_candidate(
    result: &mut RhoCandidate,
    slice: Slice,
    safe_bound: &mut BoundMemo,
    calendar: &CalendarBounds,
    quotes: &[PreparedQuote],
) {
    if !pair_safe(&slice, safe_bound, calendar) {
        return;
    }
    result.evaluations += 1;
    let loss = band_loss(&slice, quotes);
    if loss < result.loss {
        result.loss = loss;
        result.slice = slice;
    }
}

fn psi_search_loss(
    result: &mut RhoCandidate,
    psi: f64,
    rho: f64,
    quotes: &[PreparedQuote],
    expiry: f64,
    atm: &Atm,
    previous: Option<&Slice>,
    candidate_bound: &mut BoundMemo,
    safe_bound: &mut BoundMemo,
    calendar: &CalendarBounds,
) -> f64 {
    let slice = anchored(expiry, atm, rho, psi, previous, candidate_bound);
    if !pair_safe(&slice, safe_bound, calendar) {
        return 1e300;
    }
    result.evaluations += 1;
    band_loss(&slice, quotes)
}

struct LocalPick {
    best: Slice,
    best_loss: f64,
    best_score: f64,
    quote_cap_candidate: Slice,
    quote_cap_score: f64,
    quote_cap_loss: f64,
    have_quote_cap: bool,
}

impl LocalPick {
    fn fresh() -> Self {
        Self {
            best: Slice::default(),
            best_loss: f64::INFINITY,
            best_score: f64::INFINITY,
            quote_cap_candidate: Slice::default(),
            quote_cap_score: f64::INFINITY,
            quote_cap_loss: f64::INFINITY,
            have_quote_cap: false,
        }
    }

    fn consider(
        &mut self,
        candidate: Slice,
        loss: f64,
        score: f64,
        maximum_excess_ratio: f64,
        quote_cap: f64,
    ) {
        if score < self.best_score {
            self.best_score = score;
            self.best_loss = loss;
            self.best = candidate;
        }
        if maximum_excess_ratio.is_finite()
            && !(maximum_excess_ratio > quote_cap)
            && score < self.quote_cap_score
        {
            self.quote_cap_score = score;
            self.quote_cap_loss = loss;
            self.quote_cap_candidate = candidate;
            self.have_quote_cap = true;
        }
    }
}

fn evaluate_rho(
    rho: f64,
    quotes: &[PreparedQuote],
    expiry: f64,
    atm: &Atm,
    previous: Option<&Slice>,
    theta0: f64,
    calendar_headroom: f64,
    psi_steps: i32,
) -> RhoCandidate {
    let mut result = RhoCandidate {
        slice: Slice::default(),
        loss: f64::INFINITY,
        evaluations: 0,
    };
    let mut candidate_bound = BoundMemo::new(true);
    let mut safe_bound = BoundMemo::new(false);
    let calendar = calendar_bounds(previous, expiry, rho, calendar_headroom);
    let lo = psi_lower(previous, rho).max(calendar.ratio_lower);
    let hi = psi_upper(theta0, rho, &mut candidate_bound);
    if !(hi > lo) {
        return result;
    }
    let low_slice = anchored(expiry, atm, rho, lo, previous, &mut candidate_bound);
    accept_rho_candidate(&mut result, low_slice, &mut safe_bound, &calendar, quotes);
    let high_slice = anchored(expiry, atm, rho, hi, previous, &mut candidate_bound);
    accept_rho_candidate(&mut result, high_slice, &mut safe_bound, &calendar, quotes);

    let mut a = lo;
    let mut b = hi;
    const GOLDEN: f64 = 0.381966011250105;
    let mut x1 = a + GOLDEN * (b - a);
    let mut x2 = b - GOLDEN * (b - a);
    let mut f1 = psi_search_loss(
        &mut result,
        x1,
        rho,
        quotes,
        expiry,
        atm,
        previous,
        &mut candidate_bound,
        &mut safe_bound,
        &calendar,
    );
    let mut f2 = psi_search_loss(
        &mut result,
        x2,
        rho,
        quotes,
        expiry,
        atm,
        previous,
        &mut candidate_bound,
        &mut safe_bound,
        &calendar,
    );
    for _ in 0..psi_steps {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + GOLDEN * (b - a);
            f1 = psi_search_loss(
                &mut result,
                x1,
                rho,
                quotes,
                expiry,
                atm,
                previous,
                &mut candidate_bound,
                &mut safe_bound,
                &calendar,
            );
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - GOLDEN * (b - a);
            f2 = psi_search_loss(
                &mut result,
                x2,
                rho,
                quotes,
                expiry,
                atm,
                previous,
                &mut candidate_bound,
                &mut safe_bound,
                &calendar,
            );
        }
    }
    let chosen = if f1 < f2 { x1 } else { x2 };
    let chosen_slice = anchored(expiry, atm, rho, chosen, previous, &mut candidate_bound);
    accept_rho_candidate(
        &mut result,
        chosen_slice,
        &mut safe_bound,
        &calendar,
        quotes,
    );
    result
}

fn search_rho_window(
    rho_lo: f64,
    rho_hi: f64,
    rho_step: f64,
    psi_steps: i32,
    quotes: &[PreparedQuote],
    expiry: f64,
    atm: &Atm,
    previous: Option<&Slice>,
    theta0: f64,
    calendar_headroom: f64,
    best: &mut Slice,
    best_loss: &mut f64,
    evals: &mut i32,
) {
    let mut rho_values = Vec::new();
    let mut rho = rho_lo;
    while rho <= rho_hi + 1e-12 {
        if rho.abs() < 0.98 {
            rho_values.push(rho);
        }
        rho += rho_step;
    }
    if rho_values.is_empty() {
        return;
    }
    let mut results = Vec::with_capacity(rho_values.len());
    for rho in rho_values {
        results.push(evaluate_rho(
            rho,
            quotes,
            expiry,
            atm,
            previous,
            theta0,
            calendar_headroom,
            psi_steps,
        ));
    }
    for result in results {
        *evals += result.evaluations;
        if result.loss < *best_loss {
            *best_loss = result.loss;
            *best = result.slice;
        }
    }
}

fn solve_damped_normal(
    normal: &StateMatrix,
    gradient: &[f64; 3],
    damping: f64,
) -> Option<[f64; 3]> {
    let mut matrix = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..3 {
            matrix[row][column] = normal[row][column];
        }
        matrix[row][row] += damping * normal[row][row].max(1e-12);
        matrix[row][3] = -gradient[row];
    }
    for column in 0..3 {
        let mut pivot = column;
        for row in column + 1..3 {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() < 1e-24 {
            return None;
        }
        matrix.swap(column, pivot);
        let diagonal = matrix[column][column];
        for index in column..4 {
            matrix[column][index] /= diagonal;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for index in column..4 {
                matrix[row][index] -= factor * matrix[column][index];
            }
        }
    }
    Some([matrix[0][3], matrix[1][3], matrix[2][3]])
}

fn fit_slice_at_headroom(
    quotes: &[PreparedQuote],
    expiry: f64,
    previous: Option<&Slice>,
    warm_start: Option<&Slice>,
    restricted_center: Option<&Slice>,
    quote_cap: f64,
    calendar_headroom: f64,
    prior_mean: Option<&[f64; 3]>,
    prior_precision: Option<&StateMatrix>,
    loss_scale: f64,
) -> CalibrationResult {
    let mut out = CalibrationResult::default();
    let mut atm = nearest_atm(quotes, expiry);
    if let (Some(warm_start_unused), Some(index)) = (warm_start, atm.quote_index) {
        let _ = warm_start_unused;
        let quote = &quotes[index];
        if quote.has_band && expiry > 0.0 && quote.bid_variance <= quote.ask_variance {
            let anchor_iv =
                0.5 * ((quote.bid_variance / expiry).sqrt() + (quote.ask_variance / expiry).sqrt());
            if anchor_iv > 0.0 && anchor_iv.is_finite() {
                atm.theta_star = expiry * anchor_iv * anchor_iv;
            }
        }
    }
    if !atm.theta_star.is_finite() || atm.theta_star <= 0.0 {
        out.error = "no ATM quote".to_owned();
        return out;
    }

    let mut evals = 0i32;
    let mut best = Slice::default();
    let mut best_loss = f64::INFINITY;
    let theta0 = match previous {
        Some(previous) => atm.theta_star.max(previous.theta + EPS),
        None => atm.theta_star,
    };

    if let Some(warm_start) = warm_start {
        let mut safe_bound = BoundMemo::new(false);
        let mut pick = LocalPick::fresh();

        let mut incumbent = *warm_start;
        incumbent.expiry = expiry;
        let incumbent_calendar =
            calendar_bounds(previous, expiry, incumbent.rho, calendar_headroom);
        if pair_safe(&incumbent, &mut safe_bound, &incumbent_calendar) {
            if prior_precision.is_some() {
                let evaluation =
                    loss_and_normal(&incumbent, quotes, prior_mean, prior_precision, loss_scale);
                pick.best_loss = evaluation.loss;
                pick.best_score = evaluation.score;
            } else {
                pick.best_loss = band_loss(&incumbent, quotes);
                pick.best_score = pick.best_loss;
            }
            pick.best = incumbent;
            if passes_quote_cap(&incumbent, quotes, quote_cap) {
                pick.quote_cap_score = pick.best_score;
                pick.quote_cap_loss = pick.best_loss;
                pick.quote_cap_candidate = incumbent;
                pick.have_quote_cap = true;
            }
            evals += 1;
        }

        let local_evaluation_budget = if prior_precision.is_some() { 100 } else { 25 };
        let rho_lower = if prior_precision.is_some() {
            -0.979999
        } else {
            (-0.979999_f64).max(warm_start.rho - 0.15)
        };
        let rho_upper = if prior_precision.is_some() {
            0.979999
        } else {
            0.979999_f64.min(warm_start.rho + 0.15)
        };
        let theta_min = match previous {
            Some(previous) => previous.theta + EPS,
            None => EPS,
        };

        for start_index in 0..RHO_OFFSETS.len() {
            if evals >= local_evaluation_budget {
                break;
            }
            let rho = (warm_start.rho + RHO_OFFSETS[start_index]).clamp(rho_lower, rho_upper);
            let mut candidate_bound = BoundMemo::new(true);
            let calendar = calendar_bounds(previous, expiry, rho, calendar_headroom);
            let psi_min = psi_lower(previous, rho).max(calendar.ratio_lower);
            let theta_seed = match previous {
                Some(previous) => atm.theta_star.max(previous.theta + EPS),
                None => atm.theta_star,
            };
            let psi_max = psi_upper(theta_seed, rho, &mut candidate_bound);
            if !(psi_max > psi_min) {
                out.fallback_reason_counts.bound_check_failed += 1;
                continue;
            }
            let psi_fraction = if start_index == 1 {
                0.25
            } else if start_index == 2 {
                0.75
            } else {
                0.0
            };
            let psi_seed = if start_index == 0 {
                warm_start.psi
            } else {
                psi_min + psi_fraction * (psi_max - psi_min)
            };
            let psi = psi_seed.clamp(psi_min, psi_max);
            let mut current = anchored(expiry, &atm, rho, psi, previous, &mut candidate_bound);
            if !bound_safe(&current, &mut safe_bound) {
                out.fallback_reason_counts.bound_check_failed += 1;
                continue;
            }
            if !calendar_bounds_safe(&current, &calendar) {
                continue;
            }

            out.local_solver_starts += 1;
            let mut evaluation =
                loss_and_normal(&current, quotes, prior_mean, prior_precision, loss_scale);
            let mut current_loss = evaluation.loss;
            let mut current_score = evaluation.score;
            pick.consider(
                current,
                current_loss,
                current_score,
                evaluation.maximum_excess_ratio,
                quote_cap,
            );
            evals += 1;
            if current_score == 0.0 {
                break;
            }

            let starts_remaining = RHO_OFFSETS.len() - start_index;
            let start_evaluation_limit =
                evals + (local_evaluation_budget - evals) / starts_remaining as i32;
            let mut damping = 1e-6;
            let mut iteration = 0;
            while iteration < 12
                && evals < start_evaluation_limit
                && theta_min < 5.0
                && current_score > 0.0
            {
                iteration += 1;
                let Some(step) =
                    solve_damped_normal(&evaluation.normal, &evaluation.gradient, damping)
                else {
                    damping *= 10.0;
                    if damping > 1e12 {
                        break;
                    }
                    continue;
                };
                let parameter_scale = [
                    (current.theta * 0.1).max(1e-3),
                    0.1,
                    (current.psi * 0.1).max(1e-3),
                ];
                let mut selected = current;
                let mut selected_loss = current_loss;
                let mut selected_score = current_score;
                let mut selected_normal = evaluation.normal;
                let mut selected_gradient = evaluation.gradient;
                let mut found_improvement = false;
                for alpha in ALPHAS {
                    if evals >= start_evaluation_limit {
                        break;
                    }
                    let mut trial = current;
                    trial.theta = (current.theta + alpha * parameter_scale[0] * step[0])
                        .clamp(theta_min, 5.0);
                    trial.rho = (current.rho + alpha * parameter_scale[1] * step[1])
                        .clamp(rho_lower, rho_upper);
                    let trial_calendar =
                        calendar_bounds(previous, expiry, trial.rho, calendar_headroom);
                    let trial_psi_min =
                        psi_lower(previous, trial.rho).max(trial_calendar.ratio_lower);
                    let trial_psi_max = psi_upper(trial.theta, trial.rho, &mut candidate_bound);
                    if !(trial_psi_max > trial_psi_min) {
                        out.fallback_reason_counts.bound_check_failed += 1;
                        continue;
                    }
                    trial.psi = (current.psi + alpha * parameter_scale[2] * step[2])
                        .clamp(trial_psi_min, trial_psi_max);
                    if !bound_safe(&trial, &mut safe_bound) {
                        out.fallback_reason_counts.bound_check_failed += 1;
                        continue;
                    }
                    if !calendar_bounds_safe(&trial, &trial_calendar) {
                        continue;
                    }
                    let trial_evaluation =
                        loss_and_normal(&trial, quotes, prior_mean, prior_precision, loss_scale);
                    pick.consider(
                        trial,
                        trial_evaluation.loss,
                        trial_evaluation.score,
                        trial_evaluation.maximum_excess_ratio,
                        quote_cap,
                    );
                    evals += 1;
                    if trial_evaluation.score < current_score {
                        selected = trial;
                        selected_loss = trial_evaluation.loss;
                        selected_score = trial_evaluation.score;
                        selected_normal = trial_evaluation.normal;
                        selected_gradient = trial_evaluation.gradient;
                        found_improvement = true;
                        break;
                    }
                }
                if !found_improvement {
                    damping *= 10.0;
                    if damping > 1e12 {
                        break;
                    }
                    continue;
                }
                current = selected;
                current_loss = selected_loss;
                current_score = selected_score;
                evaluation.normal = selected_normal;
                evaluation.gradient = selected_gradient;
                damping = (damping * 0.3).max(1e-12);
            }
            if pick.best_score == 0.0 {
                break;
            }
        }
        best = pick.best;
        best_loss = pick.best_loss;
        if pick.have_quote_cap {
            best_loss = pick.quote_cap_loss;
            best = pick.quote_cap_candidate;
        }
    } else if let Some(restricted_center) = restricted_center {
        let rho_lower = (-0.95_f64).max(restricted_center.rho - 0.15);
        let rho_upper = 0.95_f64.min(restricted_center.rho + 0.15);
        search_rho_window(
            rho_lower,
            rho_upper,
            0.05,
            22,
            quotes,
            expiry,
            &atm,
            previous,
            theta0,
            calendar_headroom,
            &mut best,
            &mut best_loss,
            &mut evals,
        );
        if best_loss < f64::INFINITY {
            search_rho_window(
                rho_lower.max(best.rho - 0.06),
                rho_upper.min(best.rho + 0.06),
                0.01,
                18,
                quotes,
                expiry,
                &atm,
                previous,
                theta0,
                calendar_headroom,
                &mut best,
                &mut best_loss,
                &mut evals,
            );
        }
    } else {
        search_rho_window(
            -0.95,
            0.95,
            0.05,
            22,
            quotes,
            expiry,
            &atm,
            previous,
            theta0,
            calendar_headroom,
            &mut best,
            &mut best_loss,
            &mut evals,
        );
        if best_loss < f64::INFINITY {
            search_rho_window(
                (-0.95_f64).max(best.rho - 0.06),
                0.95_f64.min(best.rho + 0.06),
                0.01,
                18,
                quotes,
                expiry,
                &atm,
                previous,
                theta0,
                calendar_headroom,
                &mut best,
                &mut best_loss,
                &mut evals,
            );
        }
    }

    if warm_start.is_none() && best_loss < f64::INFINITY {
        let theta_floor = match previous {
            Some(previous) => previous.theta + EPS,
            None => EPS,
        };
        let mut safe_bound = BoundMemo::new(false);
        let calendar = calendar_bounds(previous, expiry, best.rho, calendar_headroom);
        for multiplier in THETA_MULTIPLIERS {
            let mut slice = best;
            slice.theta = (best.theta * multiplier).clamp(theta_floor, 5.0);
            if !pair_safe(&slice, &mut safe_bound, &calendar) {
                continue;
            }
            evals += 1;
            let loss = band_loss(&slice, quotes);
            if loss < best_loss {
                best_loss = loss;
                best = slice;
            }
        }
    }

    out.iterations = evals;
    out.objective_evaluations = evals as u64;
    out.objective = best_loss;
    if !(best_loss < f64::INFINITY) {
        out.error = format!("empty feasible set at t={}", cpp_dtoa(expiry));
        return out;
    }
    out.surface.slices.push(best);
    let mut check = Surface {
        slices: out.surface.slices.clone(),
    };
    if let Some(previous) = previous {
        check.slices.insert(0, *previous);
    }
    out.error = validate(&check);
    out.converged = out.error.is_empty();
    out
}

fn fit_slice(
    quotes: &[PreparedQuote],
    expiry: f64,
    previous: Option<&Slice>,
    warm_start: Option<&Slice>,
    restricted_center: Option<&Slice>,
    quote_cap: f64,
) -> FitOutput {
    let strict = fit_slice_at_headroom(
        quotes,
        expiry,
        previous,
        warm_start,
        restricted_center,
        quote_cap,
        0.0,
        None,
        None,
        0.0,
    );
    let Some(previous_slice) = previous else {
        return FitOutput {
            result: strict,
            calendar_headroom: 0.0,
        };
    };
    if !strict.converged {
        return FitOutput {
            result: strict,
            calendar_headroom: 0.0,
        };
    }
    let strict_candidate = strict.surface.slices[0];
    if calendar_bounds_safe(
        &strict_candidate,
        &calendar_bounds(
            Some(previous_slice),
            expiry,
            strict_candidate.rho,
            CALENDAR_HEADROOM,
        ),
    ) {
        return FitOutput {
            result: strict,
            calendar_headroom: CALENDAR_HEADROOM,
        };
    }
    let with_headroom = fit_slice_at_headroom(
        quotes,
        expiry,
        previous,
        warm_start,
        restricted_center,
        quote_cap,
        CALENDAR_HEADROOM,
        None,
        None,
        0.0,
    );
    if with_headroom.converged {
        FitOutput {
            result: with_headroom,
            calendar_headroom: CALENDAR_HEADROOM,
        }
    } else {
        FitOutput {
            result: strict,
            calendar_headroom: 0.0,
        }
    }
}

fn fit_slice_with_prior(
    quotes: &[PreparedQuote],
    expiry: f64,
    previous: Option<&Slice>,
    warm_start: &Slice,
    quote_cap: f64,
    calendar_headroom: f64,
    prior_mean: &[f64; 3],
    prior_precision: &StateMatrix,
    loss_scale: f64,
) -> CalibrationResult {
    fit_slice_at_headroom(
        quotes,
        expiry,
        previous,
        Some(warm_start),
        None,
        quote_cap,
        calendar_headroom,
        Some(prior_mean),
        Some(prior_precision),
        loss_scale,
    )
}

fn has_invalid_included_weight(quotes: &[Quote]) -> bool {
    quotes
        .iter()
        .any(|quote| quote.included && (!quote.weight.is_finite() || quote.weight <= 0.0))
}

/// Cold sequential calibration. A failed fit keeps successful earlier slices.
pub fn calibrate(quotes: &[Quote]) -> CalibrationResult {
    let mut out = CalibrationResult {
        path: CalibrationPath::Cold,
        ..CalibrationResult::default()
    };
    let mut bound_calls = 0u64;
    let _scope = MmBoundCallScope::new(&mut bound_calls);
    if has_invalid_included_weight(quotes) {
        out.error = "included quote weight must be finite and positive".to_owned();
        finish_result(&mut out, bound_calls);
        return out;
    }
    let groups = group_quotes(quotes);
    if groups.expiries.is_empty() {
        out.error = "no included quotes".to_owned();
        finish_result(&mut out, bound_calls);
        return out;
    }
    for (group_index, &expiry) in groups.expiries.iter().enumerate() {
        let prepared = prepare_quotes(quotes, &groups.slices[group_index]);
        let previous = out.surface.slices.last().copied();
        let part = fit_slice(
            &prepared,
            expiry,
            previous.as_ref(),
            None,
            None,
            f64::INFINITY,
        );
        out.iterations += part.result.iterations;
        if part.result.objective < f64::INFINITY {
            out.objective += part.result.objective;
        }
        if !part.result.converged {
            out.error = if part.result.error.is_empty() {
                "slice fit failed".to_owned()
            } else {
                part.result.error
            };
            finish_result(&mut out, bound_calls);
            return out;
        }
        let fitted = part.result.surface.slices[0];
        out.surface.slices.push(fitted);
        out.reuse_ages.push(0);
        out.posterior_covariances.push(refresh_covariance(
            &cold_covariance(),
            &fitted,
            &prepared,
            band_loss_scale(&prepared),
        ));
        out.slices_refit += 1;
    }
    out.error = validate(&out.surface);
    out.converged = out.error.is_empty();
    finish_result(&mut out, bound_calls);
    out
}

struct WarmPlan {
    prepared: Vec<PreparedQuote>,
    warm_start: Option<Slice>,
    old_previous: Option<Slice>,
    loss_scale: f64,
    candidate: Slice,
    candidate_loss: f64,
    candidate_headroom: f64,
    full_cold_headroom: f64,
    prior_precision: StateMatrix,
    predicted_covariance: StateCovariance,
    have_candidate: bool,
    candidate_reused: bool,
    candidate_from_warm: bool,
    local_attempted: bool,
    local: CalibrationResult,
    local_bound_calls: u64,
    full_cold_reference: CalibrationResult,
    full_cold_reference_attempted: bool,
    full_cold_budget_exhausted: bool,
    force_full_cold: bool,
}

impl Default for WarmPlan {
    fn default() -> Self {
        Self {
            prepared: Vec::new(),
            warm_start: None,
            old_previous: None,
            loss_scale: 0.0,
            candidate: Slice::default(),
            candidate_loss: f64::INFINITY,
            candidate_headroom: CALENDAR_HEADROOM,
            full_cold_headroom: CALENDAR_HEADROOM,
            prior_precision: [[0.0; 3]; 3],
            predicted_covariance: cold_covariance(),
            have_candidate: false,
            candidate_reused: false,
            candidate_from_warm: false,
            local_attempted: false,
            local: CalibrationResult::default(),
            local_bound_calls: 0,
            full_cold_reference: CalibrationResult::default(),
            full_cold_reference_attempted: false,
            full_cold_budget_exhausted: false,
            force_full_cold: false,
        }
    }
}

fn select_budget_fallback(
    prepared: &[PreparedQuote],
    previous_slice: Option<&Slice>,
    expiry: f64,
    warm_start: Option<Slice>,
    restricted: Option<(Slice, f64, f64)>,
    candidate: &mut Slice,
    candidate_loss: &mut f64,
    candidate_headroom: &mut f64,
    have_candidate: &mut bool,
    candidate_reused: &mut bool,
    candidate_from_warm: &mut bool,
    used_fallback: &mut bool,
    out: &mut CalibrationResult,
) -> bool {
    let mut choices = Vec::new();
    if let Some((slice, loss, headroom)) = restricted {
        choices.push((slice, loss, headroom));
    }
    if *have_candidate {
        choices.push((*candidate, *candidate_loss, *candidate_headroom));
    }
    if let Some(mut prior) = warm_start {
        prior.expiry = expiry;
        let loss = band_loss(&prior, prepared);
        choices.push((prior, loss, 0.0));
    }
    let mut best_slice = Slice::default();
    let mut best_loss = f64::INFINITY;
    let mut best_headroom = 0.0;
    let mut found = false;
    let mut safe_bound = BoundMemo::new(false);
    for (possible, loss, headroom) in choices {
        if !loss.is_finite() {
            continue;
        }
        let calendar = calendar_bounds(previous_slice, expiry, possible.rho, headroom);
        if !pair_safe(&possible, &mut safe_bound, &calendar) {
            continue;
        }
        if !found || loss < best_loss {
            best_slice = possible;
            best_loss = loss;
            best_headroom = headroom;
            found = true;
        }
    }
    if !found {
        return false;
    }
    *candidate = best_slice;
    *candidate_loss = best_loss;
    *candidate_headroom = best_headroom;
    *have_candidate = true;
    *candidate_reused = false;
    *candidate_from_warm = false;
    *used_fallback = true;
    out.fallback_slices += 1;
    true
}

/// Warm update from a previous calibration. `previous` of `None` requests a cold start.
pub fn calibrate_warm(
    current_quotes: &[Quote],
    previous: Option<&CalibrationResult>,
    previous_quotes: &[Quote],
    options: WarmOptions,
) -> CalibrationResult {
    let mut out = CalibrationResult {
        path: CalibrationPath::Warm,
        loss_tolerance: options.loss_tol,
        ..CalibrationResult::default()
    };
    let mut bound_calls = 0u64;
    let _scope = MmBoundCallScope::new(&mut bound_calls);
    if !options.quote_tol.is_finite()
        || options.quote_tol < 0.0
        || !options.quote_cap.is_finite()
        || options.quote_cap < 0.0
    {
        out.error = "invalid warm calibration options".to_owned();
        finish_result(&mut out, bound_calls);
        return out;
    }
    if has_invalid_included_weight(current_quotes) {
        out.error = "included quote weight must be finite and positive".to_owned();
        finish_result(&mut out, bound_calls);
        return out;
    }

    let has_previous =
        previous.is_some_and(|result| result.converged && !result.surface.slices.is_empty());
    let empty_previous = Surface::default();
    let previous_surface = if has_previous {
        &previous.expect("has_previous guarantees a result").surface
    } else {
        &empty_previous
    };
    let previous_result = previous.filter(|_| has_previous);
    let current_groups = group_quotes(current_quotes);
    let previous_groups = if has_previous {
        group_quotes(previous_quotes)
    } else {
        QuoteGroups {
            expiries: Vec::new(),
            slices: Vec::new(),
        }
    };
    if current_groups.expiries.is_empty() {
        out.error = "no included quotes".to_owned();
        finish_result(&mut out, bound_calls);
        return out;
    }

    let mut used_warm_path = false;
    let mut used_fallback = false;
    let mut quality_ratio_square_sum = 0.0;
    let mut plans = Vec::with_capacity(current_groups.expiries.len());
    for _ in 0..current_groups.expiries.len() {
        plans.push(WarmPlan::default());
    }

    let previous_reuse_age = |expiry: f64| -> usize {
        let Some(result) = previous_result else {
            return 0;
        };
        for (index, slice) in previous_surface.slices.iter().enumerate() {
            if slice.expiry != expiry {
                continue;
            }
            return result.reuse_ages.get(index).copied().unwrap_or(0);
        }
        0
    };
    let previous_degraded_age = |expiry: f64| -> usize {
        let Some(result) = previous_result else {
            return 0;
        };
        for (index, slice) in previous_surface.slices.iter().enumerate() {
            if slice.expiry != expiry {
                continue;
            }
            return result.degraded_ages.get(index).copied().unwrap_or(0);
        }
        0
    };
    let previous_slice_before = |expiry: f64| -> Option<Slice> {
        previous_surface
            .slices
            .iter()
            .filter(|slice| slice.expiry < expiry)
            .max_by(|left, right| left.expiry.total_cmp(&right.expiry))
            .copied()
    };

    let prior_fallback_majority = has_previous
        && previous_result.expect("previous result").fallback_slices * 2
            > previous_surface.slices.len();
    let mean_absolute_move = if has_previous {
        mean_absolute_mid_iv_move(&current_groups, current_quotes, previous_surface)
    } else {
        0.0
    };
    let skip_local_for_large_move =
        prior_fallback_majority && mean_absolute_move >= FAST_SCENARIO_MEAN_ABSOLUTE_MOVE;

    for (index, &expiry) in current_groups.expiries.iter().enumerate() {
        let plan = &mut plans[index];
        plan.prepared = prepare_quotes(current_quotes, &current_groups.slices[index]);
        plan.loss_scale = band_loss_scale(&plan.prepared);
        let matched_previous_slice = if has_previous {
            surface_slice_at(previous_surface, expiry)
        } else {
            None
        };
        let prior_covariance =
            previous_result.and_then(|result| posterior_covariance_at(result, expiry));
        if let Some(covariance) = prior_covariance {
            if let Some((precision, predicted)) = prior_precision_for(&covariance) {
                plan.prior_precision = precision;
                plan.predicted_covariance = predicted;
                plan.warm_start = matched_previous_slice;
            } else if matched_previous_slice.is_some() {
                plan.force_full_cold = true;
            }
        } else if matched_previous_slice.is_some() {
            plan.force_full_cold = true;
        }
        plan.old_previous = previous_slice_before(expiry);
        if plan.warm_start.is_none()
            || previous_degraded_age(expiry) >= DEGRADED_AGE_BEFORE_FORCED_COLD
        {
            continue;
        }
        let warm_start = plan.warm_start.expect("checked above");
        let old_quote_group = quote_group_at(&previous_groups, expiry);
        let quotes_unchanged = old_quote_group.is_some_and(|group| {
            quote_group_reusable(
                current_quotes,
                &current_groups.slices[index],
                previous_quotes,
                &previous_groups.slices[group],
                options.quote_tol,
            )
        });
        let reuse_age = previous_reuse_age(expiry);
        if quotes_unchanged && reuse_age < 12 {
            let mut safe_bound = BoundMemo::new(false);
            let calendar = calendar_bounds(plan.old_previous.as_ref(), expiry, warm_start.rho, 0.0);
            let mut reused = warm_start;
            reused.expiry = expiry;
            if pair_safe(&reused, &mut safe_bound, &calendar) {
                let loss = band_loss(&reused, &plan.prepared);
                if loss.is_finite() && passes_quote_cap(&reused, &plan.prepared, options.quote_cap)
                {
                    plan.candidate = reused;
                    plan.candidate_loss = loss;
                    plan.candidate_headroom = if calendar_bounds_safe(
                        &reused,
                        &calendar_bounds(
                            plan.old_previous.as_ref(),
                            expiry,
                            reused.rho,
                            CALENDAR_HEADROOM,
                        ),
                    ) {
                        CALENDAR_HEADROOM
                    } else {
                        0.0
                    };
                    plan.have_candidate = true;
                    plan.candidate_reused = true;
                    plan.candidate_from_warm = true;
                }
            }
        }
        if !plan.have_candidate && reuse_age < 12 {
            let mut safe_bound = BoundMemo::new(false);
            let mut prior = warm_start;
            prior.expiry = expiry;
            let calendar = calendar_bounds(plan.old_previous.as_ref(), expiry, prior.rho, 0.0);
            let prior_bound_safe = bound_safe(&prior, &mut safe_bound);
            if !prior_bound_safe {
                out.fallback_reason_counts.bound_check_failed += 1;
            } else if calendar_bounds_safe(&prior, &calendar) {
                if passes_quote_cap(&prior, &plan.prepared, options.quote_cap) {
                    plan.candidate = prior;
                    plan.candidate_loss = band_loss(&prior, &plan.prepared);
                    plan.have_candidate = plan.candidate_loss.is_finite();
                    plan.candidate_reused = plan.have_candidate;
                    plan.candidate_from_warm = plan.have_candidate;
                    if plan.have_candidate {
                        plan.candidate_headroom = if calendar_bounds_safe(
                            &prior,
                            &calendar_bounds(
                                plan.old_previous.as_ref(),
                                expiry,
                                prior.rho,
                                CALENDAR_HEADROOM,
                            ),
                        ) {
                            CALENDAR_HEADROOM
                        } else {
                            0.0
                        };
                    }
                } else {
                    out.fallback_reason_counts.loss_above_gate += 1;
                }
            }
        }
    }

    let mut local_indices = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        if plan.warm_start.is_some()
            && !plan.have_candidate
            && previous_degraded_age(current_groups.expiries[index])
                < DEGRADED_AGE_BEFORE_FORCED_COLD
            && !skip_local_for_large_move
        {
            local_indices.push(index);
        }
    }
    local_indices.sort_by(|&left, &right| {
        let age_left = previous_degraded_age(current_groups.expiries[left]);
        let age_right = previous_degraded_age(current_groups.expiries[right]);
        age_right.cmp(&age_left).then(left.cmp(&right))
    });
    for index in local_indices {
        let expiry = current_groups.expiries[index];
        let old_previous = plans[index].old_previous;
        let warm_start = plans[index]
            .warm_start
            .expect("local index has a warm start");
        plans[index].local_attempted = true;
        let local = {
            let mut local_bound_calls = 0u64;
            let _local_scope = MmBoundCallScope::new(&mut local_bound_calls);
            let fitted = fit_slice(
                &plans[index].prepared,
                expiry,
                old_previous.as_ref(),
                Some(&warm_start),
                None,
                options.quote_cap,
            );
            plans[index].local_bound_calls = local_bound_calls;
            plans[index].candidate_headroom = fitted.calendar_headroom;
            fitted.result
        };
        if local.converged {
            plans[index].candidate = local.surface.slices[0];
            plans[index].candidate_loss = local.objective;
            plans[index].have_candidate = true;
            plans[index].candidate_from_warm = true;
        }
        plans[index].local = local;
    }
    for plan in &plans {
        if !plan.local_attempted {
            continue;
        }
        out.local_solver_slices += 1;
        out.local_solver_starts += plan.local.local_solver_starts;
        out.local_objective_evaluations += plan.local.objective_evaluations;
        out.local_mm_bound_calls += plan.local_bound_calls;
        out.fallback_reason_counts.bound_check_failed +=
            plan.local.fallback_reason_counts.bound_check_failed;
        if !plan.local.converged {
            out.fallback_reason_counts.infeasible_local_result += 1;
        }
        bound_calls += plan.local_bound_calls;
        out.iterations += plan.local.iterations;
    }

    for index in 0..current_groups.expiries.len() {
        let expiry = current_groups.expiries[index];
        let previous_slice = out.surface.slices.last().copied();
        let warm_start = plans[index].warm_start;
        let mut candidate = plans[index].candidate;
        let mut candidate_loss = plans[index].candidate_loss;
        let mut candidate_headroom = plans[index].candidate_headroom;
        let mut have_candidate = plans[index].have_candidate;
        let mut candidate_reused = plans[index].candidate_reused;
        let mut candidate_from_warm = plans[index].candidate_from_warm;

        let forced_full_cold = warm_start.is_some()
            && previous_degraded_age(expiry) >= DEGRADED_AGE_BEFORE_FORCED_COLD;
        if forced_full_cold {
            ensure_full_cold(
                &mut plans[index],
                &mut out,
                expiry,
                previous_slice.as_ref(),
                has_previous,
                previous_degraded_age(expiry),
            );
            let cold_converged = plans[index].full_cold_reference.converged;
            if !cold_converged {
                out.error = if plans[index].full_cold_reference.error.is_empty() {
                    "forced cold refit failed".to_owned()
                } else {
                    plans[index].full_cold_reference.error.clone()
                };
                out.path = CalibrationPath::Fallback;
                finish_result(&mut out, bound_calls);
                return out;
            }
            candidate = plans[index].full_cold_reference.surface.slices[0];
            candidate_loss = plans[index].full_cold_reference.objective;
            candidate_headroom = plans[index].full_cold_headroom;
            have_candidate = true;
            candidate_reused = false;
            candidate_from_warm = false;
            used_fallback = true;
            out.fallback_slices += 1;
        } else {
            let calendar_repair_needed = have_candidate
                && !calendar_bounds_safe(
                    &candidate,
                    &calendar_bounds(
                        previous_slice.as_ref(),
                        expiry,
                        candidate.rho,
                        candidate_headroom,
                    ),
                );
            if calendar_repair_needed {
                if let Some(warm_start_slice) = warm_start {
                    out.local_solver_slices += 1;
                    let local = {
                        let mut local_bound_calls = 0u64;
                        let _local_scope = MmBoundCallScope::new(&mut local_bound_calls);
                        let fitted = fit_slice(
                            &plans[index].prepared,
                            expiry,
                            previous_slice.as_ref(),
                            Some(&warm_start_slice),
                            None,
                            options.quote_cap,
                        );
                        out.local_mm_bound_calls += local_bound_calls;
                        bound_calls += local_bound_calls;
                        plans[index].candidate_headroom = fitted.calendar_headroom;
                        fitted
                    };
                    let local_headroom = local.calendar_headroom;
                    out.local_solver_starts += local.result.local_solver_starts;
                    out.local_objective_evaluations += local.result.objective_evaluations;
                    out.fallback_reason_counts.bound_check_failed +=
                        local.result.fallback_reason_counts.bound_check_failed;
                    out.iterations += local.result.iterations;
                    let repaired = local
                        .result
                        .converged
                        .then(|| local.result.surface.slices[0]);
                    let local_safe = repaired.is_some_and(|slice| {
                        calendar_bounds_safe(
                            &slice,
                            &calendar_bounds(
                                previous_slice.as_ref(),
                                expiry,
                                slice.rho,
                                local_headroom,
                            ),
                        )
                    });
                    if local_safe && local_headroom == CALENDAR_HEADROOM {
                        candidate = repaired.expect("local_safe implies a slice");
                        candidate_loss = local.result.objective;
                        candidate_headroom = local_headroom;
                        have_candidate = true;
                        candidate_reused = false;
                        candidate_from_warm = true;
                    } else {
                        out.fallback_reason_counts.calendar_repair_failed += 1;
                        if have_candidate
                            && calendar_bounds_safe(
                                &candidate,
                                &calendar_bounds(
                                    previous_slice.as_ref(),
                                    expiry,
                                    candidate.rho,
                                    0.0,
                                ),
                            )
                        {
                            candidate_headroom = 0.0;
                        } else if local_safe {
                            candidate = repaired.expect("local_safe implies a slice");
                            candidate_loss = local.result.objective;
                            candidate_headroom = local_headroom;
                            have_candidate = true;
                            candidate_reused = false;
                            candidate_from_warm = true;
                        } else {
                            have_candidate = false;
                        }
                    }
                }
            }

            let candidate_fails_cap = candidate_from_warm
                && have_candidate
                && !passes_quote_cap(&candidate, &plans[index].prepared, options.quote_cap);
            if candidate_fails_cap {
                out.fallback_reason_counts.loss_above_gate += 1;
            }
            let needs_restricted = warm_start.is_some() && (!have_candidate || candidate_fails_cap);
            if needs_restricted {
                out.restricted_cold_searches += 1;
                let warm_start_slice = warm_start.expect("restricted search has a warm start");
                let restricted = fit_slice(
                    &plans[index].prepared,
                    expiry,
                    previous_slice.as_ref(),
                    None,
                    Some(&warm_start_slice),
                    f64::INFINITY,
                );
                out.iterations += restricted.result.iterations;
                if restricted.result.converged {
                    let restricted_slice = restricted.result.surface.slices[0];
                    if passes_quote_cap(
                        &restricted_slice,
                        &plans[index].prepared,
                        options.quote_cap,
                    ) {
                        candidate = restricted_slice;
                        candidate_loss = restricted.result.objective;
                        candidate_headroom = restricted.calendar_headroom;
                        have_candidate = true;
                        candidate_reused = false;
                        candidate_from_warm = false;
                        used_fallback = true;
                        out.fallback_slices += 1;
                    } else {
                        out.fallback_reason_counts.loss_above_gate += 1;
                        out.quality_warm_loss +=
                            band_loss(&restricted_slice, &plans[index].prepared);
                        out.quality_loss_scale += plans[index].loss_scale;
                        ensure_full_cold(
                            &mut plans[index],
                            &mut out,
                            expiry,
                            previous_slice.as_ref(),
                            has_previous,
                            previous_degraded_age(expiry),
                        );
                        if plans[index].full_cold_budget_exhausted {
                            if !select_budget_fallback(
                                &plans[index].prepared,
                                previous_slice.as_ref(),
                                expiry,
                                warm_start,
                                Some((
                                    restricted_slice,
                                    restricted.result.objective,
                                    restricted.calendar_headroom,
                                )),
                                &mut candidate,
                                &mut candidate_loss,
                                &mut candidate_headroom,
                                &mut have_candidate,
                                &mut candidate_reused,
                                &mut candidate_from_warm,
                                &mut used_fallback,
                                &mut out,
                            ) {
                                out.error = format!(
                                    "budget exhausted without feasible candidate at t={}",
                                    cpp_dtoa(expiry)
                                );
                                out.path = CalibrationPath::Fallback;
                                finish_result(&mut out, bound_calls);
                                return out;
                            }
                        } else if !plans[index].full_cold_reference.converged {
                            out.error =
                                format!("cold quality reference failed at t={}", cpp_dtoa(expiry));
                            out.path = CalibrationPath::Fallback;
                            finish_result(&mut out, bound_calls);
                            return out;
                        } else {
                            out.quality_checks += 1;
                            let cold_slice = plans[index].full_cold_reference.surface.slices[0];
                            out.quality_reference_loss +=
                                band_loss(&cold_slice, &plans[index].prepared);
                            if passes_adaptive_quote_cap(
                                &restricted_slice,
                                &cold_slice,
                                &plans[index].prepared,
                                options.quote_cap,
                            ) {
                                candidate = restricted_slice;
                                candidate_loss = restricted.result.objective;
                                candidate_headroom = restricted.calendar_headroom;
                            } else {
                                candidate = cold_slice;
                                candidate_loss = plans[index].full_cold_reference.objective;
                                candidate_headroom = plans[index].full_cold_headroom;
                            }
                            have_candidate = true;
                            candidate_reused = false;
                            candidate_from_warm = false;
                            used_fallback = true;
                            out.fallback_slices += 1;
                        }
                    }
                } else {
                    ensure_full_cold(
                        &mut plans[index],
                        &mut out,
                        expiry,
                        previous_slice.as_ref(),
                        has_previous,
                        previous_degraded_age(expiry),
                    );
                    if plans[index].full_cold_budget_exhausted {
                        if !select_budget_fallback(
                            &plans[index].prepared,
                            previous_slice.as_ref(),
                            expiry,
                            warm_start,
                            None,
                            &mut candidate,
                            &mut candidate_loss,
                            &mut candidate_headroom,
                            &mut have_candidate,
                            &mut candidate_reused,
                            &mut candidate_from_warm,
                            &mut used_fallback,
                            &mut out,
                        ) {
                            out.error = format!(
                                "budget exhausted without feasible candidate at t={}",
                                cpp_dtoa(expiry)
                            );
                            out.path = CalibrationPath::Fallback;
                            finish_result(&mut out, bound_calls);
                            return out;
                        }
                    } else if !plans[index].full_cold_reference.converged {
                        out.error = if plans[index].full_cold_reference.error.is_empty() {
                            "slice fit failed".to_owned()
                        } else {
                            plans[index].full_cold_reference.error.clone()
                        };
                        out.path = CalibrationPath::Fallback;
                        finish_result(&mut out, bound_calls);
                        return out;
                    } else {
                        candidate = plans[index].full_cold_reference.surface.slices[0];
                        candidate_loss = plans[index].full_cold_reference.objective;
                        candidate_headroom = plans[index].full_cold_headroom;
                        have_candidate = true;
                        candidate_reused = false;
                        candidate_from_warm = false;
                        used_fallback = true;
                        out.fallback_slices += 1;
                    }
                }
            } else if !have_candidate {
                ensure_full_cold(
                    &mut plans[index],
                    &mut out,
                    expiry,
                    previous_slice.as_ref(),
                    has_previous,
                    previous_degraded_age(expiry),
                );
                if !plans[index].full_cold_reference.converged {
                    out.error = if plans[index].full_cold_budget_exhausted {
                        "full-cold budget exhausted before fitting new expiry".to_owned()
                    } else if plans[index].full_cold_reference.error.is_empty() {
                        "slice fit failed".to_owned()
                    } else {
                        plans[index].full_cold_reference.error.clone()
                    };
                    out.path = CalibrationPath::Fallback;
                    finish_result(&mut out, bound_calls);
                    return out;
                }
                candidate = plans[index].full_cold_reference.surface.slices[0];
                candidate_loss = plans[index].full_cold_reference.objective;
                candidate_headroom = plans[index].full_cold_headroom;
                have_candidate = true;
                candidate_reused = false;
                candidate_from_warm = false;
            }
        }

        if !have_candidate || !candidate_loss.is_finite() {
            out.error = format!("no finite feasible candidate at t={}", cpp_dtoa(expiry));
            out.path = if used_fallback {
                CalibrationPath::Fallback
            } else {
                CalibrationPath::Warm
            };
            finish_result(&mut out, bound_calls);
            return out;
        }

        if candidate_from_warm {
            if let Some(warm_start_slice) = warm_start {
                let before = candidate;
                let prior_mean = state_coordinates(&warm_start_slice);
                let posterior = {
                    let mut posterior_bound_calls = 0u64;
                    let _posterior_scope = MmBoundCallScope::new(&mut posterior_bound_calls);
                    let fitted = fit_slice_with_prior(
                        &plans[index].prepared,
                        expiry,
                        previous_slice.as_ref(),
                        &before,
                        options.quote_cap,
                        candidate_headroom,
                        &prior_mean,
                        &plans[index].prior_precision,
                        plans[index].loss_scale,
                    );
                    bound_calls += posterior_bound_calls;
                    out.fallback_reason_counts.bound_check_failed +=
                        fitted.fallback_reason_counts.bound_check_failed;
                    out.iterations += fitted.iterations;
                    fitted
                };
                if posterior.converged
                    && passes_quote_cap(
                        &posterior.surface.slices[0],
                        &plans[index].prepared,
                        options.quote_cap,
                    )
                {
                    candidate = posterior.surface.slices[0];
                    candidate_reused = candidate_reused
                        && candidate.theta == before.theta
                        && candidate.psi == before.psi
                        && candidate.rho == before.rho;
                    candidate_loss = band_loss(&candidate, &plans[index].prepared);
                }
            }
        }

        let result_covariance = refresh_covariance(
            &plans[index].predicted_covariance,
            &candidate,
            &plans[index].prepared,
            plans[index].loss_scale,
        );
        out.surface.slices.push(candidate);
        out.posterior_covariances.push(result_covariance);
        out.reuse_ages.push(if candidate_reused {
            previous_reuse_age(expiry) + 1
        } else {
            0
        });
        let prior_degraded_age = previous_degraded_age(expiry);
        let degraded_age =
            if passes_quote_cap(&candidate, &plans[index].prepared, options.quote_cap) {
                0
            } else {
                prior_degraded_age + 1
            };
        out.degraded_ages.push(degraded_age);
        if degraded_age > 0 {
            out.degraded_slices += 1;
        }
        let mut slice_ratio_square_sum = 0.0;
        let mut slice_quote_count = 0usize;
        let mut slice_maximum_ratio: f64 = 0.0;
        for quote in &plans[index].prepared {
            if !quote.has_band {
                continue;
            }
            let ratio = quote_excess_ratio(&candidate, quote);
            slice_ratio_square_sum += ratio * ratio;
            slice_maximum_ratio = slice_maximum_ratio.max(ratio);
            out.quality_maximum_excess_ratio = out.quality_maximum_excess_ratio.max(ratio);
            slice_quote_count += 1;
            out.quality_quote_count += 1;
        }
        quality_ratio_square_sum += slice_ratio_square_sum;
        out.worst_slice_max_excess_ratio =
            out.worst_slice_max_excess_ratio.max(slice_maximum_ratio);
        if slice_quote_count != 0 {
            let slice_rms = (slice_ratio_square_sum / slice_quote_count as f64).sqrt();
            out.worst_slice_rms_excess_ratio = out.worst_slice_rms_excess_ratio.max(slice_rms);
        }
        out.objective += candidate_loss;
        if candidate_reused {
            out.slices_reused += 1;
        } else {
            out.slices_refit += 1;
        }
        used_warm_path = used_warm_path || candidate_from_warm;
    }
    if out.quality_quote_count != 0 {
        out.quality_rms_excess_ratio =
            (quality_ratio_square_sum / out.quality_quote_count as f64).sqrt();
    }
    out.error = validate(&out.surface);
    out.converged = out.error.is_empty();
    if !out.converged {
        out.surface.slices.clear();
        out.posterior_covariances.clear();
        out.reuse_ages.clear();
        out.degraded_ages.clear();
        out.degraded_slices = 0;
    }
    out.path = if out.converged && out.fallback_reason_counts.budget_exhausted != 0 {
        CalibrationPath::Degraded
    } else if used_fallback {
        CalibrationPath::Fallback
    } else if used_warm_path {
        CalibrationPath::Warm
    } else {
        CalibrationPath::Cold
    };
    finish_result(&mut out, bound_calls);
    out
}

fn ensure_full_cold(
    plan: &mut WarmPlan,
    out: &mut CalibrationResult,
    expiry: f64,
    previous: Option<&Slice>,
    has_previous: bool,
    degraded_age: usize,
) {
    if plan.full_cold_reference_attempted {
        return;
    }
    plan.full_cold_reference_attempted = true;
    let forced = plan.force_full_cold || degraded_age >= DEGRADED_AGE_BEFORE_FORCED_COLD;
    if has_previous && !forced && out.full_cold_searches >= FULL_COLD_SEARCH_BUDGET {
        plan.full_cold_budget_exhausted = true;
        out.fallback_reason_counts.budget_exhausted += 1;
        return;
    }
    if has_previous {
        out.full_cold_searches += 1;
        if forced {
            out.forced_full_cold_searches += 1;
        }
    }
    let fitted = fit_slice(&plan.prepared, expiry, previous, None, None, f64::INFINITY);
    out.iterations += fitted.result.iterations;
    plan.full_cold_headroom = fitted.calendar_headroom;
    plan.full_cold_reference = fitted.result;
}
