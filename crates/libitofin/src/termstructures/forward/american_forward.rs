//! De-Americanization of option quotes for the model-based American forward.
//!
//! Given a discount factor, a trial continuous yield `q` fixes a forward
//! `F = S exp((r - q) T)`. Under that yield every American quote has an implied volatility
//! from an American pricer; subtracting the early-exercise premium at that volatility gives
//! a European-equivalent price. At the right yield the call and put implied volatilities
//! agree strike by strike, so the carry is found by solving for a zero mean call/put gap.
//! The European-equivalent quotes are then handed to the parity estimator.

use super::american_model::{
    AmericanLevel, AmericanModelConfig, ForwardFlags, ForwardTier, ModelFailure,
};
use super::american_pricer::{AmericanPricer, PricerKind};
use super::implied_forward::OptionQuotePair;
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::option::OptionType::{self, Call, Put};
use crate::pricingengines::blackformula::black_formula;
use crate::types::{DiscountFactor, Rate, Real, Time, Volatility};
use std::cell::Cell;

type ModelResult<T> = Result<T, ModelFailure>;

/// Below about 0.5% volatility the Ju engine's put boundary overflows, so smaller implied
/// volatilities are treated as unsolvable.
const IV_MIN: Volatility = 0.01;
const IV_MAX: Volatility = 5.0;
const IV_ACCURACY: Real = 1e-9;
const IV_MAX_EVALUATIONS: usize = 100;
/// Strikes nearest the money that drive the carry solve.
const SOLVE_STRIKES: usize = 5;
/// Initial step when bracketing the carry from zero.
const BRACKET_STEP: Rate = 0.02;
/// A call/put implied-vol gap this small (1e-9 volatility) counts as zero.
const GAP_TOLERANCE: Real = 1e-9;
/// With two or more correction passes, a last pass that still moves the carry by more than
/// this (per year) leaves the result flagged as not converged. Each pass removes roughly
/// four fifths of the remaining carry error, so the error left is about a quarter of the
/// last move: 5e-4 per year keeps it near a basis point of the forward per year of expiry.
const CORRECTION_CONVERGENCE: Rate = 5e-4;
/// More than this fraction of the strikes entering the model being dropped flags the result.
const MAX_DROPPED_FRACTION: Real = 0.25;

/// Result of a model-based solve.
#[derive(Debug, Clone)]
pub(crate) struct ModelSolution {
    /// European-equivalent quotes for the parity estimator.
    pub pairs: Vec<OptionQuotePair>,
    /// Solved continuous yield.
    pub carry: Rate,
    /// Mean call minus put implied volatility at the solution.
    pub iv_gap: Real,
    /// Gap evaluations spent by the carry solve.
    pub iterations: usize,
    /// Accurate-pricer evaluations spent on correction.
    pub accurate_prices_used: usize,
    pub tier: ForwardTier,
    pub flags: ForwardFlags,
    /// Why the Accurate stage fell back to the Fast solution, when it did.
    pub accurate_failure: Option<ModelFailure>,
}

/// Lower ends tried for the volatility bracket: the closed forms overflow below a few percent
/// when the yield exceeds the rate, so the first volatility that prices below the target wins.
const IV_LOWER_LADDER: [Volatility; 4] = [IV_MIN, 0.03, 0.06, 0.12];

/// American implied volatility of `target` for the given pricer, or `None` when the price is
/// outside what the pricer can produce for volatilities in `[IV_MIN, IV_MAX]`.
fn american_iv(
    pricer: &AmericanPricer,
    option_type: OptionType,
    strike: Real,
    spot: Real,
    rate: Rate,
    carry: Rate,
    target: Real,
) -> Option<Volatility> {
    if !(target.is_finite() && target > 0.0) {
        return None;
    }
    let price_at = |vol: Volatility| pricer.price(option_type, strike, spot, rate, carry, vol);
    let lower = IV_LOWER_LADDER
        .iter()
        .copied()
        .find(|&vol| price_at(vol).is_ok_and(|p| p.is_finite() && p < target))?;
    let f = |vol: Volatility| price_at(vol).map_or(Real::NAN, |p| p - target);
    Brent::new()
        .with_max_evaluations(IV_MAX_EVALUATIONS)
        .solve_bracketed(f, IV_ACCURACY, 0.3_f64.max(lower * 1.01), lower, IV_MAX)
        .ok()
}

/// Market inputs shared by every pricing call of one solve.
#[derive(Clone, Copy)]
struct Market {
    spot: Real,
    rate: Rate,
    expiry: Time,
    discount: DiscountFactor,
}

/// European-equivalent price of an American quote: the Black price at the American implied vol.
fn european_equivalent(
    pricer: &AmericanPricer,
    market: Market,
    option_type: OptionType,
    strike: Real,
    carry: Rate,
    price: Real,
) -> Option<Real> {
    if price == 0.0 {
        return Some(0.0);
    }
    let vol = american_iv(
        pricer,
        option_type,
        strike,
        market.spot,
        market.rate,
        carry,
        price,
    )?;
    let forward = market.spot * ((market.rate - carry) * market.expiry).exp();
    black_formula(
        option_type,
        strike,
        forward,
        vol * market.expiry.sqrt(),
        market.discount,
        0.0,
    )
    .ok()
}

/// Call minus put implied volatility of one strike, or `None` when either cannot be inverted.
fn pair_gap(
    pricer: &AmericanPricer,
    pair: &OptionQuotePair,
    spot: Real,
    rate: Rate,
    carry: Rate,
) -> Option<Real> {
    let call = american_iv(
        pricer,
        Call,
        pair.strike,
        spot,
        rate,
        carry,
        pair.call_mid(),
    )?;
    let put = american_iv(pricer, Put, pair.strike, spot, rate, carry, pair.put_mid())?;
    Some(call - put)
}

/// Mean call minus put implied volatility over the strikes where both exist.
fn mean_iv_gap(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    carry: Rate,
) -> Option<Real> {
    let gaps: Vec<Real> = pairs
        .iter()
        .filter_map(|pair| pair_gap(pricer, pair, spot, rate, carry))
        .collect();
    (!gaps.is_empty()).then(|| gaps.iter().sum::<Real>() / gaps.len() as Real)
}

/// Mean call minus put implied volatility over a fixed set of strikes: `None` as soon as one of
/// them cannot be inverted, so the objective never changes its strike set between evaluations.
fn fixed_iv_gap(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    carry: Rate,
) -> Option<Real> {
    let mut sum = 0.0;
    for pair in pairs {
        sum += pair_gap(pricer, pair, spot, rate, carry)?;
    }
    (!pairs.is_empty()).then(|| sum / pairs.len() as Real)
}

/// The refinement may lose strikes to the bracket ends, but not most of them: at least two
/// strikes (or all of them when fewer were given) and at least half must remain.
fn enough_strikes(remaining: usize, total: usize) -> bool {
    remaining >= total.min(2) && 2 * remaining >= total
}

/// Bracket halvings tried to bring enough invertible strikes to both ends.
const MAX_BRACKET_HALVINGS: usize = 6;

/// The strikes that invert at both `x_min` and `x_max`, with the mean call/put gap over them at
/// each end.
fn common_strikes(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    x_min: Rate,
    x_max: Rate,
) -> (Vec<OptionQuotePair>, Real, Real) {
    let mut common = Vec::with_capacity(pairs.len());
    let (mut sum_min, mut sum_max) = (0.0, 0.0);
    for pair in pairs {
        if let (Some(g_min), Some(g_max)) = (
            pair_gap(pricer, pair, spot, rate, x_min),
            pair_gap(pricer, pair, spot, rate, x_max),
        ) {
            common.push(*pair);
            sum_min += g_min;
            sum_max += g_max;
        }
    }
    let count = common.len().max(1) as Real;
    (common, sum_min / count, sum_max / count)
}

/// A solved carry with the work it took and the gap left at it.
#[derive(Debug, Clone, Copy)]
struct CarrySolve {
    carry: Rate,
    evaluations: usize,
    /// Mean call minus put implied volatility over the strikes of the final objective.
    iv_gap: Real,
}

/// Solves for the carry that makes the mean call/put implied-vol gap vanish.
///
/// The gap rises with the yield (a higher yield lowers call prices and raises put prices, so
/// the call volatility must rise and the put volatility fall), so the root is bracketed by
/// stepping away from zero toward it and refined with Brent's method. The bracketing may see a
/// different set of invertible strikes at each trial carry; the bracket is halved until most
/// strikes invert at both of its ends, and the refinement runs on those strikes only. It fails
/// rather than follow a gap that loses a strike or changes sign because of the strike set.
fn solve_carry(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    config: &AmericanModelConfig,
) -> ModelResult<CarrySolve> {
    let (lo, hi) = config.carry_search_range;
    let evaluations = Cell::new(0usize);
    let gap = |carry: Rate| {
        evaluations.set(evaluations.get() + 1);
        mean_iv_gap(pricer, pairs, spot, rate, carry)
    };

    let start = 0.0_f64.clamp(lo, hi);
    let Some(g_start) = gap(start) else {
        return Err(ModelFailure::NoImpliedVolatility);
    };
    if g_start.abs() <= GAP_TOLERANCE {
        return Ok(CarrySolve {
            carry: start,
            evaluations: evaluations.get(),
            iv_gap: g_start,
        });
    }
    let direction = if g_start < 0.0 { 1.0 } else { -1.0 };
    let edge = if direction > 0.0 { hi } else { lo };
    let mut a = start;
    let mut g_a = g_start;
    let mut step = BRACKET_STEP;
    let mut shrinks = 0;
    let (b, g_b) = loop {
        let mut next = a + direction * step;
        if direction * (next - edge) >= 0.0 {
            next = edge;
        }
        match gap(next) {
            Some(g_next) if g_next.abs() <= GAP_TOLERANCE => {
                return Ok(CarrySolve {
                    carry: next,
                    evaluations: evaluations.get(),
                    iv_gap: g_next,
                });
            }
            Some(g_next) if g_next * g_start < 0.0 => break (next, g_next),
            Some(g_next) => {
                if next == edge {
                    return Err(ModelFailure::CarryNotBracketed);
                }
                a = next;
                g_a = g_next;
                step *= 2.0;
            }
            None => {
                // Some strikes cannot be inverted this far out; retry closer to the last good point.
                shrinks += 1;
                if shrinks > 4 {
                    return Err(ModelFailure::NoImpliedVolatility);
                }
                step *= 0.5;
            }
        }
    };
    // Halve the bracket (on the gap of whichever strikes invert) until enough strikes invert at
    // both of its ends to refine on.
    let (mut left, mut right) = if a < b {
        ((a, g_a), (b, g_b))
    } else {
        ((b, g_b), (a, g_a))
    };
    let mut halvings = 0;
    let (common, mean_left, mean_right) = loop {
        let (common, mean_left, mean_right) =
            common_strikes(pricer, pairs, spot, rate, left.0, right.0);
        evaluations.set(evaluations.get() + 2);
        if enough_strikes(common.len(), pairs.len()) {
            break (common, mean_left, mean_right);
        }
        halvings += 1;
        if halvings > MAX_BRACKET_HALVINGS {
            return Err(ModelFailure::CarrySolveFailed);
        }
        let mid = 0.5 * (left.0 + right.0);
        match gap(mid) {
            Some(g_mid) if g_mid.abs() <= GAP_TOLERANCE => {
                return Ok(CarrySolve {
                    carry: mid,
                    evaluations: evaluations.get(),
                    iv_gap: g_mid,
                });
            }
            Some(g_mid) if g_mid * left.1 < 0.0 => right = (mid, g_mid),
            Some(g_mid) => left = (mid, g_mid),
            None => return Err(ModelFailure::CarrySolveFailed),
        }
    };
    if mean_left * mean_right > 0.0 {
        return Err(ModelFailure::CarrySolveFailed);
    }
    let (x_min, x_max) = (left.0, right.0);
    let fixed_gap = |carry: Rate| {
        evaluations.set(evaluations.get() + 1);
        fixed_iv_gap(pricer, &common, spot, rate, carry)
    };
    let lost_a_strike = Cell::new(false);
    let carry = Brent::new()
        .with_max_evaluations(100)
        .solve_bracketed(
            |c| {
                fixed_gap(c).unwrap_or_else(|| {
                    lost_a_strike.set(true);
                    Real::NAN
                })
            },
            config.carry_tolerance,
            0.5 * (x_min + x_max),
            x_min,
            x_max,
        )
        .map_err(|_| ModelFailure::CarrySolveFailed)?;
    let iv_gap = fixed_gap(carry).unwrap_or(Real::NAN);
    if lost_a_strike.get() || !iv_gap.is_finite() {
        return Err(ModelFailure::CarrySolveFailed);
    }
    Ok(CarrySolve {
        carry,
        evaluations: evaluations.get(),
        iv_gap,
    })
}

/// At-the-money volatility implied by the quotes, or `None` when none can be inverted.
///
/// Takes the mid-price implied vols of the (up to three) strikes nearest the money forward at a
/// zero yield; the yield barely moves a window-sizing estimate.
pub(crate) fn estimate_atm_vol(
    pairs: &[OptionQuotePair],
    spot: Real,
    discount_factor: DiscountFactor,
    expiry_years: Time,
) -> Option<Volatility> {
    if !(discount_factor.is_finite() && discount_factor > 0.0 && expiry_years > 0.0) {
        return None;
    }
    let rate = -discount_factor.ln() / expiry_years;
    let pricer = AmericanPricer::new(PricerKind::JuQuadratic, expiry_years, &[]).ok()?;
    let money = spot * (rate * expiry_years).exp();
    for pair in nearest_to(pairs, money, 3) {
        let vols: Vec<Volatility> = [
            american_iv(&pricer, Call, pair.strike, spot, rate, 0.0, pair.call_mid()),
            american_iv(&pricer, Put, pair.strike, spot, rate, 0.0, pair.put_mid()),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !vols.is_empty() {
            return Some(vols.iter().sum::<Volatility>() / vols.len() as Real);
        }
    }
    None
}

/// Premium corrections of one strike: accurate minus cheap model price, for each right.
#[derive(Clone, Copy, Default)]
struct Shift {
    call: Real,
    put: Real,
}

/// The pair with the strike's premium correction removed from every bid and ask.
fn shifted(pair: &OptionQuotePair, shift: Shift) -> OptionQuotePair {
    OptionQuotePair::new(
        pair.strike,
        pair.call_bid - shift.call,
        pair.call_ask - shift.call,
        pair.put_bid - shift.put,
        pair.put_ask - shift.put,
    )
}

/// The `count` pairs nearest the money (by log distance), in ascending strike order.
fn nearest_to(pairs: &[OptionQuotePair], money: Real, count: usize) -> Vec<OptionQuotePair> {
    let mut nearest: Vec<OptionQuotePair> = pairs.to_vec();
    nearest.sort_by(|a, b| {
        let da = (a.strike / money).ln().abs();
        let db = (b.strike / money).ln().abs();
        da.total_cmp(&db)
    });
    nearest.truncate(count);
    nearest.sort_by(|a, b| a.strike.total_cmp(&b.strike));
    nearest
}

/// Converts every quote to European-equivalent bid/ask under the solved carry, with the
/// strike's premium correction removed first. Strikes that cannot be inverted are dropped.
fn deamericanize(
    pricer: &AmericanPricer,
    market: Market,
    pairs: &[OptionQuotePair],
    shifts: &[Shift],
    carry: Rate,
) -> Vec<OptionQuotePair> {
    let mut equivalent = Vec::with_capacity(pairs.len());
    for (pair, &shift) in pairs.iter().zip(shifts) {
        let pair = shifted(pair, shift);
        let convert = |ty: OptionType, price: Real| {
            european_equivalent(pricer, market, ty, pair.strike, carry, price)
        };
        if let (Some(cb), Some(ca), Some(pb), Some(pa)) = (
            convert(Call, pair.call_bid),
            convert(Call, pair.call_ask),
            convert(Put, pair.put_bid),
            convert(Put, pair.put_ask),
        ) {
            let candidate = OptionQuotePair::new(pair.strike, cb, ca, pb, pa);
            if candidate.is_valid() {
                equivalent.push(candidate);
            }
        }
    }
    equivalent
}

/// Solves one model-based American forward.
///
/// `pairs` are the (already filtered) American quotes, sorted by strike. The Fast level uses the
/// Ju closed form alone; the Accurate level adds premium-correction passes with a tree (or a
/// grid, when a dividend schedule is given) on the strikes the work budget affords.
pub(crate) fn solve(
    pairs: &[OptionQuotePair],
    spot: Real,
    discount_factor: DiscountFactor,
    expiry_years: Time,
    min_pairs: usize,
    config: &AmericanModelConfig,
) -> ModelResult<ModelSolution> {
    // The caller has checked the discount factor; a bad one would only fail every inversion.
    debug_assert!(discount_factor.is_finite() && discount_factor > 0.0);
    let rate = -discount_factor.ln() / expiry_years;
    let market = Market {
        spot,
        rate,
        expiry: expiry_years,
        discount: discount_factor,
    };
    let cheap = AmericanPricer::new(PricerKind::JuQuadratic, expiry_years, &[])
        .map_err(|_| ModelFailure::PricerUnavailable)?;
    let money = spot * (rate * expiry_years).exp();

    // Beside a schedule the flat yield q the closed form fits maps to the residual carry
    // b = q - ln(S / (S - PV)) / T, which leaves the forward unchanged. The search and flag
    // ranges are ranges of b, so the search over q is shifted by the same offset.
    let has_schedule = !config.dividends.is_empty();
    let offset = if has_schedule {
        let pv: Real = config
            .dividends
            .iter()
            .map(|&(time, amount)| amount * (-rate * time).exp())
            .sum();
        if pv >= spot {
            return Err(ModelFailure::DividendsExceedSpot);
        }
        (spot / (spot - pv)).ln() / expiry_years
    } else {
        0.0
    };
    let search = AmericanModelConfig {
        carry_search_range: (
            config.carry_search_range.0 + offset,
            config.carry_search_range.1 + offset,
        ),
        ..config.clone()
    };

    let first = solve_carry(
        &cheap,
        &nearest_to(pairs, money, SOLVE_STRIKES),
        spot,
        rate,
        &search,
    )?;
    let carry0 = first.carry;

    let level = match config.level {
        AmericanLevel::Auto if has_schedule => AmericanLevel::Accurate,
        AmericanLevel::Auto => AmericanLevel::Fast,
        level => level,
    };
    let passes = config
        .correction_passes
        .unwrap_or(if has_schedule { 2 } else { 1 });

    let mut flags = ForwardFlags::NONE;
    let mut plan = None;
    if level == AmericanLevel::Accurate && passes > 0 {
        plan = plan_correction(pairs.len(), passes, config.max_accurate_prices, min_pairs);
        if plan.is_none() {
            flags.insert(ForwardFlags::LATENCY_FALLBACK);
        }
    }

    // Fast: either requested, the budget could not pay for a correction, or the correction
    // failed after the closed-form carry was solved.
    let fast = |mut flags: ForwardFlags| -> ModelResult<ModelSolution> {
        if has_schedule {
            flags.insert(ForwardFlags::SCHEDULE_IGNORED);
        }
        let shifts = vec![Shift::default(); pairs.len()];
        let equivalent = deamericanize(&cheap, market, pairs, &shifts, carry0);
        if equivalent.len() < min_pairs {
            return Err(ModelFailure::TooFewStrikes);
        }
        flag_dropped_pairs(&mut flags, pairs.len(), equivalent.len());
        Ok(ModelSolution {
            pairs: equivalent,
            carry: carry0 - offset,
            iv_gap: first.iv_gap,
            iterations: first.evaluations,
            accurate_prices_used: 0,
            tier: ForwardTier::ModelFast,
            flags,
            accurate_failure: None,
        })
    };
    let Some(plan) = plan else {
        return fast(flags);
    };
    let accurate = accurate_stage(
        &cheap, market, pairs, first, offset, plan, min_pairs, config, &search, flags,
    );
    match accurate {
        Ok(solution) => Ok(solution),
        Err(failure) => {
            flags.insert(ForwardFlags::NON_CONVERGED);
            // If the Fast level cannot deliver either, the Accurate stage's cause is the one
            // reported: it is where the solve first stopped.
            let mut solution = fast(flags).map_err(|_| failure)?;
            solution.accurate_failure = Some(failure);
            Ok(solution)
        }
    }
}

/// The Accurate level: premium corrections from a tree (or a grid beside a schedule) on the
/// strikes the plan pays for, each followed by a new carry solve.
#[allow(clippy::too_many_arguments)]
fn accurate_stage(
    cheap: &AmericanPricer,
    market: Market,
    pairs: &[OptionQuotePair],
    first: CarrySolve,
    offset: Rate,
    plan: CorrectionPlan,
    min_pairs: usize,
    config: &AmericanModelConfig,
    search: &AmericanModelConfig,
    mut flags: ForwardFlags,
) -> ModelResult<ModelSolution> {
    let Market {
        spot,
        rate,
        expiry: expiry_years,
        ..
    } = market;
    let money = spot * (rate * expiry_years).exp();
    let has_schedule = !config.dividends.is_empty();
    let corrected = nearest_to(pairs, money, plan.strikes);
    let accurate_kind = if has_schedule {
        PricerKind::FiniteDifference {
            t_grid: 100,
            x_grid: 200,
        }
    } else {
        PricerKind::Binomial(400)
    };
    let accurate = AmericanPricer::new(accurate_kind, expiry_years, &config.dividends)
        .map_err(|_| ModelFailure::PricerUnavailable)?;
    let accurate_price = |ty: OptionType, strike: Real, carry: Rate, vol: Volatility| {
        accurate.price(ty, strike, spot, rate, carry - offset, vol)
    };

    let mut carry = first.carry;
    let mut iterations = first.evaluations;
    let mut iv_gap = first.iv_gap;
    let prices_used = Cell::new(0usize);
    let mut corrected = corrected;
    let attempted = corrected.len();
    let mut shifts = vec![Shift::default(); corrected.len()];
    let mut last_change = Real::INFINITY;
    for _ in 0..plan.passes {
        // Strikes whose mids the closed form cannot invert are dropped from the correction.
        let mut kept = Vec::with_capacity(corrected.len());
        let mut kept_shifts = Vec::with_capacity(corrected.len());
        for (pair, shift) in corrected.iter().zip(&shifts) {
            let target = shifted(pair, *shift);
            let correction = |ty: OptionType, mid: Real| -> Option<Real> {
                let vol = american_iv(cheap, ty, pair.strike, spot, rate, carry, mid)?;
                let closed = cheap.price(ty, pair.strike, spot, rate, carry, vol).ok()?;
                prices_used.set(prices_used.get() + 1);
                let exact = accurate_price(ty, pair.strike, carry, vol).ok()?;
                Some(exact - closed)
            };
            if let (Some(call), Some(put)) = (
                correction(Call, target.call_mid()),
                correction(Put, target.put_mid()),
            ) {
                // The correction is a difference of prices at the cheap pricer's implied vol, so
                // it replaces (not adds to) the previous one.
                kept.push(*pair);
                kept_shifts.push(Shift { call, put });
            }
        }
        if kept.len() < min_pairs {
            return Err(ModelFailure::TooFewStrikes);
        }
        corrected = kept;
        shifts = kept_shifts;
        let adjusted: Vec<OptionQuotePair> = corrected
            .iter()
            .zip(&shifts)
            .map(|(pair, &shift)| shifted(pair, shift))
            .collect();
        let solved = solve_carry(
            cheap,
            &nearest_to(&adjusted, money, SOLVE_STRIKES),
            spot,
            rate,
            search,
        )?;
        iterations += solved.evaluations;
        last_change = (solved.carry - carry).abs();
        carry = solved.carry;
        iv_gap = solved.iv_gap;
        if last_change <= config.carry_tolerance {
            break;
        }
    }
    let prices_used = prices_used.get();
    if plan.passes >= 2 && last_change > CORRECTION_CONVERGENCE {
        flags.insert(ForwardFlags::NON_CONVERGED);
    }
    if plan.reduced {
        flags.insert(ForwardFlags::LATENCY_FALLBACK);
    }

    let equivalent = deamericanize(cheap, market, &corrected, &shifts, carry);
    if equivalent.len() < min_pairs {
        return Err(ModelFailure::TooFewStrikes);
    }
    flag_dropped_pairs(&mut flags, attempted, equivalent.len());
    Ok(ModelSolution {
        pairs: equivalent,
        carry: carry - offset,
        iv_gap,
        iterations,
        accurate_prices_used: prices_used,
        tier: ForwardTier::ModelAccurate,
        flags,
        accurate_failure: None,
    })
}

/// Flags a result when more than a quarter of the `attempted` strikes did not survive the
/// de-Americanization.
fn flag_dropped_pairs(flags: &mut ForwardFlags, attempted: usize, kept: usize) {
    let dropped = attempted.saturating_sub(kept);
    if dropped as Real > MAX_DROPPED_FRACTION * attempted as Real {
        flags.insert(ForwardFlags::PAIRS_DROPPED);
    }
}

/// How much correction the work budget pays for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CorrectionPlan {
    /// Strikes corrected, nearest the money first.
    strikes: usize,
    passes: usize,
    /// True when the budget forced fewer strikes or passes than requested.
    reduced: bool,
}

/// Two accurate prices (call and put) are spent per strike and pass. Fewer strikes are corrected
/// before fewer passes are run, and no plan is returned when not even `min_pairs` strikes can
/// be corrected in a single pass.
fn plan_correction(
    pairs: usize,
    requested_passes: usize,
    budget: usize,
    min_pairs: usize,
) -> Option<CorrectionPlan> {
    let mut passes = requested_passes;
    loop {
        let strikes = pairs.min(budget / (2 * passes));
        if strikes >= min_pairs {
            return Some(CorrectionPlan {
                strikes,
                passes,
                reduced: strikes < pairs || passes < requested_passes,
            });
        }
        if passes == 1 {
            return None;
        }
        passes -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_cuts_strikes_before_passes() {
        let plan = |pairs, passes, budget| plan_correction(pairs, passes, budget, 2);
        // 9 strikes, 1 pass need 18 prices.
        assert_eq!(
            plan(9, 1, 18),
            Some(CorrectionPlan {
                strikes: 9,
                passes: 1,
                reduced: false
            })
        );
        // 8 prices afford 4 strikes in one pass.
        assert_eq!(
            plan(9, 1, 8),
            Some(CorrectionPlan {
                strikes: 4,
                passes: 1,
                reduced: true
            })
        );
        // Two passes over 9 strikes need 36; 8 afford 2 strikes in two passes.
        assert_eq!(
            plan(9, 2, 8),
            Some(CorrectionPlan {
                strikes: 2,
                passes: 2,
                reduced: true
            })
        );
        // 5 prices cannot pay two passes over 2 strikes (8), so one pass over 2 strikes is run.
        assert_eq!(
            plan(9, 2, 5),
            Some(CorrectionPlan {
                strikes: 2,
                passes: 1,
                reduced: true
            })
        );
        // Fewer than min_pairs strikes in one pass: no plan.
        assert_eq!(plan(9, 1, 3), None);
        assert_eq!(plan(9, 2, 0), None);
        // Fewer quotes than the budget could pay for are not "reduced".
        assert_eq!(
            plan(3, 2, 100),
            Some(CorrectionPlan {
                strikes: 3,
                passes: 2,
                reduced: false
            })
        );
    }

    fn ju_quotes(vol: Volatility, rate: Rate, carry: Rate, expiry: Time) -> Vec<OptionQuotePair> {
        let pricer = AmericanPricer::new(PricerKind::JuQuadratic, expiry, &[]).unwrap();
        [90.0, 95.0, 100.0, 105.0, 110.0]
            .iter()
            .map(|&k| {
                let c = pricer.price(Call, k, 100.0, rate, carry, vol).unwrap();
                let p = pricer.price(Put, k, 100.0, rate, carry, vol).unwrap();
                OptionQuotePair::new(k, c - 0.01, c + 0.01, p - 0.01, p + 0.01)
            })
            .collect()
    }

    #[test]
    fn an_accurate_stage_failure_keeps_the_fast_solution() {
        // A dividend after the expiry makes the grid pricer refuse to build, after the
        // closed-form carry has been solved: the Fast result is returned, flagged.
        let (rate, carry, expiry): (Rate, Rate, Time) = (0.03, 0.01, 0.5);
        let quotes = ju_quotes(0.25, rate, carry, expiry);
        let config = AmericanModelConfig {
            level: AmericanLevel::Accurate,
            dividends: vec![(expiry + 0.1, 1.0)],
            ..AmericanModelConfig::default()
        };
        let solution = solve(&quotes, 100.0, (-rate * expiry).exp(), expiry, 2, &config).unwrap();
        assert_eq!(solution.tier, ForwardTier::ModelFast);
        assert!(solution.flags.contains(ForwardFlags::NON_CONVERGED));
        assert!(solution.flags.contains(ForwardFlags::SCHEDULE_IGNORED));
        assert!(solution.pairs.len() >= 2);
        assert_eq!(
            solution.accurate_failure,
            Some(ModelFailure::PricerUnavailable)
        );
    }

    fn failure(
        quotes: &[OptionQuotePair],
        discount_factor: DiscountFactor,
        min_pairs: usize,
        config: &AmericanModelConfig,
    ) -> ModelFailure {
        solve(quotes, 100.0, discount_factor, 0.5, min_pairs, config).unwrap_err()
    }

    #[test]
    fn each_failure_reports_its_cause() {
        let (rate, expiry): (Rate, Time) = (0.03, 0.5);
        let discount = (-rate * expiry).exp();
        let quotes = ju_quotes(0.25, rate, 0.01, expiry);
        let default = AmericanModelConfig::default();

        let huge_dividend = AmericanModelConfig {
            dividends: vec![(0.25, 150.0)],
            ..default.clone()
        };
        assert_eq!(
            failure(&quotes, discount, 2, &huge_dividend),
            ModelFailure::DividendsExceedSpot
        );
        // Calls above the spot and puts above the strike cannot be priced by any volatility.
        let unpriceable: Vec<OptionQuotePair> = quotes
            .iter()
            .map(|q| OptionQuotePair::new(q.strike, 150.0, 151.0, 150.0, 151.0))
            .collect();
        assert_eq!(
            failure(&unpriceable, discount, 2, &default),
            ModelFailure::NoImpliedVolatility
        );
        // Quotes carrying about 1%, searched only above 20%.
        let far_away = AmericanModelConfig {
            carry_search_range: (0.20, 0.25),
            carry_flag_range: (0.21, 0.24),
            ..default.clone()
        };
        assert_eq!(
            failure(&quotes, discount, 2, &far_away),
            ModelFailure::CarryNotBracketed
        );
        // More strikes demanded than the chain has.
        assert_eq!(
            failure(&quotes, discount, 6, &default),
            ModelFailure::TooFewStrikes
        );
    }

    #[test]
    fn a_lone_unpriceable_strike_is_dropped_without_a_flag_but_two_of_five_are_flagged() {
        let (rate, carry, expiry): (Rate, Rate, Time) = (0.03, 0.01, 0.5);
        let discount = (-rate * expiry).exp();
        let corrupt = |quotes: &mut Vec<OptionQuotePair>, strikes: &[Real]| {
            for q in quotes.iter_mut().filter(|q| strikes.contains(&q.strike)) {
                // A call quoted above the spot has no implied volatility.
                *q = OptionQuotePair::new(q.strike, 150.0, 151.0, q.put_bid, q.put_ask);
            }
        };
        let config = AmericanModelConfig::default();

        let mut one = ju_quotes(0.25, rate, carry, expiry);
        corrupt(&mut one, &[110.0]);
        let solution = solve(&one, 100.0, discount, expiry, 2, &config).unwrap();
        assert_eq!(solution.pairs.len(), 4);
        assert!(!solution.flags.contains(ForwardFlags::PAIRS_DROPPED));
        assert!((solution.carry - carry).abs() < 0.01, "{}", solution.carry);

        let mut two = ju_quotes(0.25, rate, carry, expiry);
        corrupt(&mut two, &[90.0, 110.0]);
        let solution = solve(&two, 100.0, discount, expiry, 2, &config).unwrap();
        assert_eq!(solution.pairs.len(), 3);
        assert!(solution.flags.contains(ForwardFlags::PAIRS_DROPPED));
        assert!((solution.carry - carry).abs() < 0.01, "{}", solution.carry);
    }

    #[test]
    fn the_refinement_needs_most_of_the_strikes_it_started_with() {
        assert!(enough_strikes(1, 1));
        assert!(enough_strikes(2, 2));
        assert!(enough_strikes(3, 5));
        assert!(!enough_strikes(1, 5));
        assert!(!enough_strikes(2, 5));
        assert!(!enough_strikes(1, 2));
        assert!(!enough_strikes(0, 1));
    }

    #[test]
    fn strikes_lost_during_the_accurate_correction_are_counted_too() {
        let (rate, carry, expiry): (Rate, Rate, Time) = (0.03, 0.01, 0.5);
        let mut quotes = ju_quotes(0.25, rate, carry, expiry);
        for q in quotes
            .iter_mut()
            .filter(|q| q.strike == 90.0 || q.strike == 110.0)
        {
            // A call above the spot cannot be inverted, so the correction drops the strike.
            *q = OptionQuotePair::new(q.strike, 150.0, 151.0, q.put_bid, q.put_ask);
        }
        let config = AmericanModelConfig {
            level: AmericanLevel::Accurate,
            ..AmericanModelConfig::default()
        };
        let solution = solve(&quotes, 100.0, (-rate * expiry).exp(), expiry, 2, &config).unwrap();
        assert_eq!(solution.tier, ForwardTier::ModelAccurate);
        assert_eq!(solution.pairs.len(), 3);
        assert!(solution.flags.contains(ForwardFlags::PAIRS_DROPPED));
    }

    #[test]
    fn a_failed_fast_fallback_reports_the_accurate_stage_cause() {
        // The pricer for the dividend after the expiry cannot be built (Accurate stage), and
        // the three corrupted calls leave the Fast level fewer than the three strikes it needs.
        let (rate, carry, expiry): (Rate, Rate, Time) = (0.03, 0.01, 0.5);
        let mut quotes = ju_quotes(0.25, rate, carry, expiry);
        for q in quotes
            .iter_mut()
            .filter(|q| q.strike != 100.0 && q.strike != 105.0)
        {
            *q = OptionQuotePair::new(q.strike, 150.0, 151.0, q.put_bid, q.put_ask);
        }
        let config = AmericanModelConfig {
            level: AmericanLevel::Accurate,
            dividends: vec![(expiry + 0.1, 1.0)],
            ..AmericanModelConfig::default()
        };
        let error = solve(&quotes, 100.0, (-rate * expiry).exp(), expiry, 3, &config).unwrap_err();
        assert_eq!(error, ModelFailure::PricerUnavailable);
    }

    #[test]
    fn a_chain_that_loses_strikes_across_the_bracket_is_not_solved_on_the_remainder() {
        // Near-zero volatility and a carry at the edge of the search: the far strikes stop
        // inverting as the trial carry moves, so the refinement cannot keep most of them. The
        // answer must be a reported failure or the carry the chain was priced with.
        let (rate, carry, expiry): (Rate, Rate, Time) = (0.0, -0.09, 0.5);
        let pricer = AmericanPricer::new(PricerKind::JuQuadratic, expiry, &[]).unwrap();
        let quotes: Vec<OptionQuotePair> = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
            .iter()
            .map(|&k| {
                let c = pricer.price(Call, k, 100.0, rate, carry, 0.05).unwrap();
                let p = pricer.price(Put, k, 100.0, rate, carry, 0.05).unwrap();
                OptionQuotePair::new(k, c - 0.01, c + 0.01, (p - 0.01).max(0.0), p + 0.01)
            })
            .collect();
        let config = AmericanModelConfig::default();
        match solve(&quotes, 100.0, 1.0, expiry, 2, &config) {
            Err(failure) => assert_eq!(failure, ModelFailure::CarrySolveFailed),
            Ok(solution) => assert!(
                (solution.carry - carry).abs() < 1e-4,
                "solved {}",
                solution.carry
            ),
        }
    }

    #[test]
    fn the_solved_carry_is_right_or_an_error_never_a_strike_set_artifact() {
        // Deep in-the-money options can stop being invertible as the trial carry moves away
        // from the true one, so the strike set seen by the solver may change along the search.
        // Whatever the quotes, the answer must be the carry they were priced with or a reported
        // failure.
        let (rate, expiry): (Rate, Time) = (0.03, 0.5);
        let discount = (-rate * expiry).exp();
        let config = AmericanModelConfig::default();
        let mut solved = 0;
        for &carry in &[-0.08, -0.03, 0.0, 0.04, 0.10, 0.18, 0.22] {
            for &vol in &[0.08, 0.15, 0.25, 0.5] {
                let quotes = ju_quotes(vol, rate, carry, expiry);
                if let Ok(solution) = solve(&quotes, 100.0, discount, expiry, 2, &config) {
                    solved += 1;
                    assert!(
                        (solution.carry - carry).abs() < 1e-5,
                        "carry {carry} vol {vol}: solved {}",
                        solution.carry
                    );
                }
            }
        }
        assert!(solved >= 20, "only {solved} of 28 solved");
    }

    #[test]
    fn a_large_short_dated_special_dividend_stays_inside_the_default_carry_search() {
        // PV of the dividend is about 8, so the flat yield that matches the quotes is near
        // 0.33 a year: outside the default (-0.10, 0.25) unless the search follows the schedule.
        let (rate, expiry, dividends) = (0.03, 0.25, vec![(0.1, 8.0)]);
        let pricer = AmericanPricer::new(
            PricerKind::FiniteDifference {
                t_grid: 100,
                x_grid: 200,
            },
            expiry,
            &dividends,
        )
        .unwrap();
        let quotes: Vec<OptionQuotePair> = [90.0, 95.0, 100.0, 105.0, 110.0]
            .iter()
            .map(|&k| {
                let c = pricer.price(Call, k, 100.0, rate, 0.0, 0.3).unwrap();
                let p = pricer.price(Put, k, 100.0, rate, 0.0, 0.3).unwrap();
                OptionQuotePair::new(k, c - 0.01, c + 0.01, p - 0.01, p + 0.01)
            })
            .collect();
        let config = AmericanModelConfig {
            level: AmericanLevel::Accurate,
            dividends,
            ..AmericanModelConfig::default()
        };
        let solution = solve(&quotes, 100.0, (-rate * expiry).exp(), expiry, 2, &config).unwrap();
        assert_eq!(solution.tier, ForwardTier::ModelAccurate);
        // The residual carry is the one quoted without the schedule: about zero.
        assert!(solution.carry.abs() < 0.02, "carry {}", solution.carry);
    }

    #[test]
    fn atm_vol_is_recovered_from_the_quotes() {
        for &(vol, rate, carry, expiry) in &[
            (0.40, 0.03, 0.01, 0.5),
            (0.12, 0.05, 0.0, 1.0),
            (0.25, 0.01, 0.03, 0.25),
        ] {
            let quotes = ju_quotes(vol, rate, carry, expiry);
            let estimate =
                estimate_atm_vol(&quotes, 100.0, (-rate * expiry).exp(), expiry).unwrap();
            assert!((estimate - vol).abs() < 0.015, "{estimate} vs {vol}");
        }
    }

    #[test]
    fn atm_vol_is_none_when_nothing_can_be_inverted() {
        let quotes = vec![
            OptionQuotePair::new(100.0, 119.0, 121.0, 0.001, 0.002),
            OptionQuotePair::new(105.0, 119.0, 121.0, 0.001, 0.002),
        ];
        assert_eq!(estimate_atm_vol(&quotes, 100.0, 0.98, 0.5), None);
        assert_eq!(estimate_atm_vol(&[], 100.0, 0.98, 0.5), None);
    }

    /// Quotes priced by a 1000-step tree with a wide spread: 3% of the mid plus a cent.
    fn wide_spread_quotes(rate: Rate, carry: Rate, expiry: Time) -> Vec<OptionQuotePair> {
        let pricer = AmericanPricer::new(PricerKind::Binomial(1000), expiry, &[]).unwrap();
        [90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0]
            .iter()
            .map(|&k| {
                let c = pricer.price(Call, k, 100.0, rate, carry, 0.25).unwrap();
                let p = pricer.price(Put, k, 100.0, rate, carry, 0.25).unwrap();
                let (hc, hp) = (0.01 + 0.03 * c, 0.01 + 0.03 * p);
                OptionQuotePair::new(k, c - hc, c + hc, p - hp, p + hp)
            })
            .collect()
    }

    #[test]
    fn mid_correction_matches_a_per_side_correction_within_a_fraction_of_the_spread() {
        // The correction is computed at the mid and applied to the bid and the ask. Per-side
        // truth: invert each of the four prices with the accurate pricer at the solved carry.
        for &(rate, carry, expiry) in &[(0.05, 0.0, 2.0), (0.03, 0.06, 1.0), (0.05, 0.0, 0.5)] {
            let quotes = wide_spread_quotes(rate, carry, expiry);
            let discount = (-rate * expiry).exp();
            let config = AmericanModelConfig {
                level: AmericanLevel::Accurate,
                ..AmericanModelConfig::default()
            };
            let solution = solve(&quotes, 100.0, discount, expiry, 2, &config).unwrap();
            assert_eq!(
                solution.tier,
                ForwardTier::ModelAccurate,
                "{:?} r={rate} q={carry} T={expiry}",
                solution.accurate_failure
            );

            let accurate = AmericanPricer::new(PricerKind::Binomial(400), expiry, &[]).unwrap();
            let market = Market {
                spot: 100.0,
                rate,
                expiry,
                discount,
            };
            let mut worst: Real = 0.0;
            for model in &solution.pairs {
                let quote = quotes.iter().find(|q| q.strike == model.strike).unwrap();
                let sides = [
                    (
                        Call,
                        quote.call_bid,
                        model.call_bid,
                        0.01 + 0.03 * quote.call_mid(),
                    ),
                    (
                        Call,
                        quote.call_ask,
                        model.call_ask,
                        0.01 + 0.03 * quote.call_mid(),
                    ),
                    (
                        Put,
                        quote.put_bid,
                        model.put_bid,
                        0.01 + 0.03 * quote.put_mid(),
                    ),
                    (
                        Put,
                        quote.put_ask,
                        model.put_ask,
                        0.01 + 0.03 * quote.put_mid(),
                    ),
                ];
                for (ty, price, modeled, half_spread) in sides {
                    let reference = european_equivalent(
                        &accurate,
                        market,
                        ty,
                        quote.strike,
                        solution.carry,
                        price,
                    );
                    let Some(reference) = reference else { continue };
                    worst = worst.max((modeled - reference).abs() / half_spread);
                }
            }
            assert!(worst < 0.05, "r={rate} q={carry} T={expiry}: {worst}");
        }
    }
}
