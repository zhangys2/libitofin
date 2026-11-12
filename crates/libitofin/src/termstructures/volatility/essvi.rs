#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::collapsible_if)]
//! Extended SSVI total-variance surface.
//!
//! Transliteration of `ql-essvi` `cpp/essvi_surface.cpp`. The public types stay
//! standalone: they do not implement [`SmileSection`](super::SmileSection).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

const MM_GRID_SIZE: usize = 257;
const RHO_GRID_SIZE: usize = 100;
const BOUND_RELATIVE_SLACK: f64 = 5e-13;
const DEFAULT_SAFETY: f64 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundPoint {
    l: f64,
    g: f64,
    h: f64,
    g2: f64,
}

#[derive(Clone, Copy, Debug)]
struct RhoWorkspace {
    abs_rho: f64,
    s: f64,
    points: [BoundPoint; MM_GRID_SIZE],
}

#[derive(Clone, Copy, Debug)]
struct BoundEstimate {
    safe: f64,
    #[allow(dead_code)]
    grid_hint: f64,
}

thread_local! {
    static ACTIVE_MM_BOUND_COUNTER: Cell<Option<Rc<Cell<u64>>>> = const { Cell::new(None) };
    static CACHED_GRID_INDEX: Cell<i64> = const { Cell::new(-1) };
    static ARBITRARY_WORKSPACE: RefCell<Option<(f64, RhoWorkspace)>> = const { RefCell::new(None) };
    static LAST_BOUND: Cell<Option<(f64, f64, BoundEstimate)>> = const { Cell::new(None) };
}

/// Pushes `counter` as the active Martini–Mingone call counter for this thread.
pub struct MmBoundCallScope {
    previous: Option<Rc<Cell<u64>>>,
}

impl MmBoundCallScope {
    /// Count bound evaluations that occur before this scope drops into `counter`.
    pub fn new(counter: Rc<Cell<u64>>) -> Self {
        let previous = ACTIVE_MM_BOUND_COUNTER.with(|slot| slot.replace(Some(Rc::clone(&counter))));
        Self { previous }
    }
}

impl Drop for MmBoundCallScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ACTIVE_MM_BOUND_COUNTER.with(|slot| slot.set(previous));
    }
}

fn bump_mm_bound_counter() {
    ACTIVE_MM_BOUND_COUNTER.with(|slot| {
        let current = slot.take();
        if let Some(counter) = current.as_ref() {
            counter.set(counter.get() + 1);
        }
        slot.set(current);
    });
}

fn workspace_cache() -> &'static [OnceLock<RhoWorkspace>; RHO_GRID_SIZE] {
    static CACHE: [OnceLock<RhoWorkspace>; RHO_GRID_SIZE] =
        [const { OnceLock::new() }; RHO_GRID_SIZE];
    &CACHE
}

fn fill_workspace(abs_rho: f64) -> RhoWorkspace {
    let s = (1.0 - abs_rho * abs_rho).max(0.0).sqrt();
    let l2 = 1.0 / ((-abs_rho).acos() / 3.0).tan();
    let lo = l2 + 1e-8;
    let hi = 50.0_f64.max(lo + 1.0);
    let mut points = [BoundPoint {
        l: 0.0,
        g: 0.0,
        h: 0.0,
        g2: 0.0,
    }; MM_GRID_SIZE];
    for (i, point) in points.iter_mut().enumerate() {
        let u = i as f64 / 256.0;
        let l = lo * (hi / lo).powf(u);
        let root = (l * l + 1.0).sqrt();
        let n = s + abs_rho * l + root;
        let np = abs_rho + l / root;
        let npp = (l * l + 1.0).powf(-1.5);
        let g = np / 4.0;
        let h = 1.0 - (l - abs_rho / s) * np / (2.0 * n);
        let g2 = npp - np * np / (2.0 * n);
        *point = BoundPoint { l, g, h, g2 };
    }
    RhoWorkspace { abs_rho, s, points }
}

fn with_workspace<R>(abs_rho: f64, f: impl FnOnce(&RhoWorkspace) -> R) -> R {
    let grid_index = (abs_rho * 100.0).round() as i64;
    if (0..RHO_GRID_SIZE as i64).contains(&grid_index) {
        let grid_rho = grid_index as f64 / 100.0;
        if (abs_rho - grid_rho).abs() <= 1e-12 {
            let workspace =
                workspace_cache()[grid_index as usize].get_or_init(|| fill_workspace(grid_rho));
            CACHED_GRID_INDEX.with(|slot| slot.set(grid_index));
            return f(workspace);
        }
    }
    ARBITRARY_WORKSPACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let stale = slot
            .as_ref()
            .is_none_or(|(cached_rho, _)| *cached_rho != abs_rho);
        if stale {
            *slot = Some((abs_rho, fill_workspace(abs_rho)));
        }
        f(&slot.as_ref().expect("workspace was just filled").1)
    })
}

fn grid_ratio(theta: f64, workspace: &RhoWorkspace, point: &BoundPoint) -> f64 {
    let denominator = theta * workspace.s * point.g * point.g - point.g2;
    if !(denominator > 0.0) {
        return f64::INFINITY;
    }
    4.0 * theta * workspace.s * point.h * point.h / denominator
}

fn ratio_at(theta: f64, workspace: &RhoWorkspace, l: f64) -> f64 {
    let l2 = l * l + 1.0;
    let root = l2.sqrt();
    let n = workspace.s + workspace.abs_rho * l + root;
    let np = workspace.abs_rho + l / root;
    let npp = 1.0 / (l2 * root);
    let g = np / 4.0;
    let h = 1.0 - (l - workspace.abs_rho / workspace.s) * np / (2.0 * n);
    let g2 = npp - np * np / (2.0 * n);
    let denominator = theta * workspace.s * g * g - g2;
    if !(denominator > 0.0) || !(h * h >= 0.0) {
        return f64::INFINITY;
    }
    4.0 * theta * workspace.s * h * h / denominator
}

fn refine_minimum(theta: f64, workspace: &RhoWorkspace, i: usize, coarse_value: f64) -> f64 {
    let mut a = workspace.points[i - 1].l;
    let mut b = workspace.points[i + 1].l;
    const GOLDEN: f64 = 0.381966011250105;
    let mut x1 = a + GOLDEN * (b - a);
    let mut x2 = b - GOLDEN * (b - a);
    let mut f1 = ratio_at(theta, workspace, x1);
    let mut f2 = ratio_at(theta, workspace, x2);
    let mut best = coarse_value.min(f1.min(f2));
    for _ in 0..36 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + GOLDEN * (b - a);
            f1 = ratio_at(theta, workspace, x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - GOLDEN * (b - a);
            f2 = ratio_at(theta, workspace, x2);
        }
        best = best.min(f1.min(f2));
    }
    best
}

fn refined_bound(theta: f64, workspace: &RhoWorkspace) -> BoundEstimate {
    let mut previous = grid_ratio(theta, workspace, &workspace.points[0]);
    let mut current = grid_ratio(theta, workspace, &workspace.points[1]);
    let mut best = previous.min(current);
    let mut grid_best = best;
    for i in 1..MM_GRID_SIZE - 1 {
        let next = grid_ratio(theta, workspace, &workspace.points[i + 1]);
        grid_best = grid_best.min(next);
        if current.is_finite() {
            best = best.min(current);
            if current <= previous && current <= next {
                best = best.min(refine_minimum(theta, workspace, i, current));
            }
        }
        previous = current;
        current = next;
    }
    best = best.min(current);
    if !grid_best.is_finite() {
        grid_best = 0.0;
    }
    if !best.is_finite() {
        return BoundEstimate {
            safe: 0.0,
            grid_hint: grid_best,
        };
    }
    BoundEstimate {
        safe: best * (1.0 - BOUND_RELATIVE_SLACK),
        grid_hint: grid_best,
    }
}

fn bound_estimate(theta: f64, abs_rho: f64, safety: f64) -> BoundEstimate {
    bump_mm_bound_counter();
    if !(theta > safety) || !(abs_rho >= 0.0) || !(abs_rho < 1.0 - safety) {
        return BoundEstimate {
            safe: 0.0,
            grid_hint: 0.0,
        };
    }
    if let Some((last_theta, last_rho, estimate)) = LAST_BOUND.with(|slot| slot.get()) {
        if theta == last_theta && abs_rho == last_rho {
            return estimate;
        }
    }
    with_workspace(abs_rho, |workspace| {
        if !(workspace.s > safety) {
            return BoundEstimate {
                safe: 0.0,
                grid_hint: 0.0,
            };
        }
        let estimate = refined_bound(theta, workspace);
        LAST_BOUND.with(|slot| slot.set(Some((theta, abs_rho, estimate))));
        estimate
    })
}

/// One listed eSSVI expiry: forward ATM total variance `theta`, `psi = theta * phi`, and `rho`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    /// Year fraction.
    pub expiry: f64,
    /// Forward ATM total variance.
    pub theta: f64,
    /// `theta * phi`.
    pub psi: f64,
    /// Spot-vol correlation in `(-1, 1)`.
    pub rho: f64,
}

impl Default for Slice {
    fn default() -> Self {
        Self {
            expiry: 0.0,
            theta: 0.0,
            psi: 0.0,
            rho: 0.0,
        }
    }
}

/// Listed eSSVI slices, interpolated in expiry by the Corbetta rule.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Slices in increasing expiry order.
    pub slices: Vec<Slice>,
}

impl Surface {
    /// Corbetta interpolation of `(theta, psi, rho * psi)`, clamped to the listed ends.
    pub fn slice_at(&self, expiry: f64) -> Slice {
        let slices = &self.slices;
        if slices.is_empty() {
            return Slice::default();
        }
        if expiry.is_nan() {
            return Slice {
                expiry,
                theta: f64::NAN,
                psi: f64::NAN,
                rho: f64::NAN,
            };
        }
        if slices.len() == 1 || expiry <= slices[0].expiry {
            return slices[0];
        }
        if expiry >= slices[slices.len() - 1].expiry {
            return slices[slices.len() - 1];
        }
        let hi = slices.partition_point(|slice| slice.expiry < expiry);
        if hi == 0 {
            return Slice {
                expiry,
                theta: f64::NAN,
                psi: f64::NAN,
                rho: f64::NAN,
            };
        }
        let b = slices[hi];
        let a = slices[hi - 1];
        let u = (expiry - a.expiry) / (b.expiry - a.expiry);
        let psi = a.psi + u * (b.psi - a.psi);
        Slice {
            expiry,
            theta: a.theta + u * (b.theta - a.theta),
            psi,
            rho: ((1.0 - u) * a.rho * a.psi + u * b.rho * b.psi) / psi,
        }
    }

    /// Total variance at `expiry` and forward log-moneyness `k`.
    pub fn total_variance(&self, expiry: f64, k: f64) -> f64 {
        if self.slices.is_empty() {
            return f64::NAN;
        }
        slice_total_variance(&self.slice_at(expiry), k)
    }

    /// Black implied volatility at `expiry` and forward log-moneyness `k`.
    pub fn implied_volatility(&self, expiry: f64, k: f64) -> f64 {
        if !(expiry > 0.0) {
            return f64::NAN;
        }
        let w = self.total_variance(expiry, k);
        if w >= 0.0 {
            (w / expiry).sqrt()
        } else {
            f64::NAN
        }
    }
}

/// Total variance of one slice at forward log-moneyness `k`.
pub fn slice_total_variance(slice: &Slice, k: f64) -> f64 {
    let x = slice.psi * k + slice.theta * slice.rho;
    0.5 * (slice.theta
        + slice.rho * slice.psi * k
        + (x * x + slice.theta * slice.theta * (1.0 - slice.rho * slice.rho)).sqrt())
}

/// Martini–Mingone necessary-and-sufficient bound on `psi^2`.
pub fn mm_psi2_bound(theta: f64, abs_rho: f64) -> f64 {
    mm_psi2_bound_with_safety(theta, abs_rho, DEFAULT_SAFETY)
}

/// [`mm_psi2_bound`] with an explicit domain safety margin.
pub fn mm_psi2_bound_with_safety(theta: f64, abs_rho: f64, safety: f64) -> f64 {
    bound_estimate(theta, abs_rho, safety).safe
}

/// Roger-Lee wing bound plus the Martini–Mingone infimum.
pub fn butterfly_safe(slice: &Slice) -> bool {
    butterfly_safe_with_safety(slice, DEFAULT_SAFETY)
}

/// [`butterfly_safe`] with an explicit domain safety margin.
pub fn butterfly_safe_with_safety(slice: &Slice, safety: f64) -> bool {
    let abs_rho = slice.rho.abs();
    slice.theta > safety
        && slice.psi > safety
        && abs_rho < 1.0 - safety
        && slice.psi * (1.0 + abs_rho) < 4.0 - safety
        && slice.psi * slice.psi <= mm_psi2_bound_with_safety(slice.theta, abs_rho, safety) - safety
}

/// Hendriks–Martini / Corbetta adjacent-slice calendar test, plus Mingone's psi upper bound.
pub fn calendar_safe(shorter: &Slice, longer: &Slice) -> bool {
    calendar_safe_with_safety(shorter, longer, DEFAULT_SAFETY)
}

/// [`calendar_safe`] with an explicit domain safety margin.
pub fn calendar_safe_with_safety(shorter: &Slice, longer: &Slice, safety: f64) -> bool {
    if !(longer.expiry > shorter.expiry
        && longer.theta > shorter.theta + safety
        && longer.psi > shorter.psi + safety
        && shorter.psi > safety
        && longer.psi > safety)
    {
        return false;
    }
    let lower = shorter.psi
        * ((1.0 + shorter.rho) / (1.0 + longer.rho)).max((1.0 - shorter.rho) / (1.0 - longer.rho));
    let upper = shorter.psi / shorter.theta * longer.theta;
    longer.psi > lower + safety && longer.psi < upper - safety
}

/// Empty when every slice is butterfly-safe and every adjacent pair is calendar-safe.
pub fn validate(surface: &Surface) -> String {
    if surface.slices.is_empty() {
        return "surface has no slices".to_owned();
    }
    for i in 0..surface.slices.len() {
        if !butterfly_safe(&surface.slices[i]) {
            return format!("butterfly constraint failed at slice {i}");
        }
        if i > 0 && !calendar_safe(&surface.slices[i - 1], &surface.slices[i]) {
            return format!(
                "calendar constraint failed between slices {} and {i}",
                i - 1
            );
        }
    }
    String::new()
}

/// Test and calibrator helpers that mirror `essvi::detail` in the C++ tree.
pub mod detail {
    use super::{BoundPoint, DEFAULT_SAFETY, RhoWorkspace, bump_mm_bound_counter, with_workspace};

    /// Adaptive warm quote-cap test. `candidate_ratio` and `cold_ratio` are already weighted.
    pub fn passes_adaptive_weighted_quote_cap(
        candidate_ratio: f64,
        cold_ratio: f64,
        weight: f64,
        cap: f64,
    ) -> bool {
        candidate_ratio.is_finite()
            && cold_ratio.is_finite()
            && candidate_ratio <= cap.max(cold_ratio + 0.01 * weight)
    }

    /// Coarse grid envelope. Feasibility checks use [`super::mm_psi2_bound`].
    pub fn mm_psi2_bound_search_hint(theta: f64, abs_rho: f64) -> f64 {
        mm_psi2_bound_search_hint_with_safety(theta, abs_rho, DEFAULT_SAFETY)
    }

    /// [`mm_psi2_bound_search_hint`] with an explicit domain safety margin.
    pub fn mm_psi2_bound_search_hint_with_safety(theta: f64, abs_rho: f64, safety: f64) -> f64 {
        bump_mm_bound_counter();
        if !(theta > safety) || !(abs_rho >= 0.0) || !(abs_rho < 1.0 - safety) {
            return 0.0;
        }
        with_workspace(abs_rho, |workspace| {
            if !(workspace.s > safety) {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for point in &workspace.points {
                best = best.min(grid_ratio_hint(theta, workspace, point));
            }
            if best.is_finite() { best } else { 0.0 }
        })
    }

    fn grid_ratio_hint(theta: f64, workspace: &RhoWorkspace, point: &BoundPoint) -> f64 {
        let denominator = theta * workspace.s * point.g * point.g - point.g2;
        if !(denominator > 0.0) {
            return f64::INFINITY;
        }
        4.0 * theta * workspace.s * point.h * point.h / denominator
    }

    /// Independent 257-point reference used by the bound regression.
    pub fn mm_psi2_bound_reference(theta: f64, abs_rho: f64) -> f64 {
        mm_psi2_bound_reference_with_safety(theta, abs_rho, DEFAULT_SAFETY)
    }

    /// [`mm_psi2_bound_reference`] with an explicit domain safety margin.
    pub fn mm_psi2_bound_reference_with_safety(theta: f64, abs_rho: f64, safety: f64) -> f64 {
        if !(theta > safety) || !(abs_rho >= 0.0) || !(abs_rho < 1.0 - safety) {
            return 0.0;
        }
        let s = (1.0 - abs_rho * abs_rho).max(0.0).sqrt();
        if !(s > safety) {
            return 0.0;
        }
        let l2 = 1.0 / ((-abs_rho).acos() / 3.0).tan();
        let ratio = |l: f64| {
            let n = s + abs_rho * l + (l * l + 1.0).sqrt();
            let np = abs_rho + l / (l * l + 1.0).sqrt();
            let npp = (l * l + 1.0).powf(-1.5);
            let g = np / 4.0;
            let h = 1.0 - (l - abs_rho / s) * np / (2.0 * n);
            let g2 = npp - np * np / (2.0 * n);
            let den = theta * s * g * g - g2;
            if !(den > 0.0) || !(h * h >= 0.0) {
                return f64::INFINITY;
            }
            4.0 * theta * s * h * h / den
        };
        let mut best = f64::INFINITY;
        let lo = l2 + 1e-8;
        let hi = 50.0_f64.max(lo + 1.0);
        for i in 0..=256 {
            let u = i as f64 / 256.0;
            let l = lo * (hi / lo).powf(u);
            best = best.min(ratio(l));
        }
        if best.is_finite() { best } else { 0.0 }
    }
}

pub use super::essvi_calibrator::{
    CalibrationPath, CalibrationResult, FallbackReasonCounts, Quote, WarmOptions, calibrate,
    calibrate_warm,
};
