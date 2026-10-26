//! De-Americanization of option quotes for the model-based American forward.
//!
//! Given a discount factor, a trial continuous yield `q` fixes a forward
//! `F = S exp((r - q) T)`. Under that yield every American quote has an implied volatility
//! from an American pricer; subtracting the early-exercise premium at that volatility gives
//! a European-equivalent price. At the right yield the call and put implied volatilities
//! agree strike by strike, so the carry is found by solving for a zero mean call/put gap.
//! The European-equivalent quotes are then handed to the parity estimator.

use super::american_model::{AmericanLevel, AmericanModelConfig, ForwardFlags, ForwardTier};
use super::american_pricer::{AmericanPricer, PricerKind};
use super::implied_forward::OptionQuotePair;
use crate::errors::QlResult;
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::option::OptionType::{self, Call, Put};
use crate::pricingengines::blackformula::black_formula;
use crate::types::{DiscountFactor, Rate, Real, Time, Volatility};
use crate::{fail, require};
use std::cell::Cell;

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

/// Mean call minus put implied volatility over the strikes where both exist.
fn mean_iv_gap(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    carry: Rate,
) -> Option<Real> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for pair in pairs {
        let call = american_iv(
            pricer,
            Call,
            pair.strike,
            spot,
            rate,
            carry,
            pair.call_mid(),
        );
        let put = american_iv(pricer, Put, pair.strike, spot, rate, carry, pair.put_mid());
        if let (Some(c), Some(p)) = (call, put) {
            sum += c - p;
            count += 1;
        }
    }
    (count > 0).then(|| sum / count as Real)
}

/// Solves for the carry that makes the mean call/put implied-vol gap vanish.
///
/// The gap rises with the yield (a higher yield lowers call prices and raises put prices, so
/// the call volatility must rise and the put volatility fall), so the root is bracketed by
/// stepping away from zero toward it and refined with Brent's method.
fn solve_carry(
    pricer: &AmericanPricer,
    pairs: &[OptionQuotePair],
    spot: Real,
    rate: Rate,
    config: &AmericanModelConfig,
) -> QlResult<(Rate, usize)> {
    let (lo, hi) = config.carry_search_range;
    let evaluations = Cell::new(0usize);
    let gap = |carry: Rate| {
        evaluations.set(evaluations.get() + 1);
        mean_iv_gap(pricer, pairs, spot, rate, carry)
    };

    let start = 0.0_f64.clamp(lo, hi);
    let Some(g_start) = gap(start) else {
        fail!("no implied volatility could be solved at carry {start}");
    };
    if g_start.abs() <= GAP_TOLERANCE {
        return Ok((start, evaluations.get()));
    }
    let direction = if g_start < 0.0 { 1.0 } else { -1.0 };
    let edge = if direction > 0.0 { hi } else { lo };
    let mut a = start;
    let mut step = BRACKET_STEP;
    let mut shrinks = 0;
    let b = loop {
        let mut next = a + direction * step;
        if direction * (next - edge) >= 0.0 {
            next = edge;
        }
        match gap(next) {
            Some(g_next) if g_next.abs() <= GAP_TOLERANCE => {
                return Ok((next, evaluations.get()));
            }
            Some(g_next) if g_next * g_start < 0.0 => break next,
            Some(_) => {
                require!(
                    next != edge,
                    "no carry in [{lo}, {hi}] equalizes call and put implied vols"
                );
                a = next;
                step *= 2.0;
            }
            None => {
                // Some strikes cannot be inverted this far out; retry closer to the last good point.
                shrinks += 1;
                require!(
                    shrinks <= 4,
                    "no implied volatility could be solved near carry {next}"
                );
                step *= 0.5;
            }
        }
    };
    let (x_min, x_max) = if a < b { (a, b) } else { (b, a) };
    let carry = Brent::new().with_max_evaluations(100).solve_bracketed(
        |c| gap(c).unwrap_or(Real::NAN),
        config.carry_tolerance,
        0.5 * (x_min + x_max),
        x_min,
        x_max,
    )?;
    Ok((carry, evaluations.get()))
}

/// Fast-level solve: closed-form pricer only.
///
/// `pairs` are the (already filtered) American quotes, sorted by strike.
pub(crate) fn solve_fast(
    pairs: &[OptionQuotePair],
    spot: Real,
    discount_factor: DiscountFactor,
    expiry_years: Time,
    min_pairs: usize,
    config: &AmericanModelConfig,
) -> QlResult<ModelSolution> {
    require!(
        discount_factor.is_finite() && discount_factor > 0.0,
        "discount factor must be finite and positive, got {discount_factor}"
    );
    let rate = -discount_factor.ln() / expiry_years;
    let pricer = AmericanPricer::new(PricerKind::JuQuadratic, expiry_years, &[])?;
    let market = Market {
        spot,
        rate,
        expiry: expiry_years,
        discount: discount_factor,
    };

    // The strikes nearest the money drive the carry solve.
    let money = spot * (rate * expiry_years).exp();
    let mut nearest: Vec<OptionQuotePair> = pairs.to_vec();
    nearest.sort_by(|a, b| {
        let da = (a.strike / money).ln().abs();
        let db = (b.strike / money).ln().abs();
        da.total_cmp(&db)
    });
    nearest.truncate(SOLVE_STRIKES);
    let (carry, iterations) = solve_carry(&pricer, &nearest, spot, rate, config)?;
    let iv_gap = mean_iv_gap(&pricer, &nearest, spot, rate, carry).unwrap_or(Real::NAN);

    // De-Americanize every quote at the solved carry.
    let mut equivalent = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let convert = |ty: OptionType, price: Real| {
            european_equivalent(&pricer, market, ty, pair.strike, carry, price)
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
    require!(
        equivalent.len() >= min_pairs,
        "only {} strikes could be de-Americanized, need {min_pairs}",
        equivalent.len()
    );

    let mut flags = ForwardFlags::NONE;
    if !config.dividends.is_empty()
        && matches!(config.level, AmericanLevel::Auto | AmericanLevel::Fast)
    {
        flags.insert(ForwardFlags::SCHEDULE_IGNORED);
    }
    Ok(ModelSolution {
        pairs: equivalent,
        carry,
        iv_gap,
        iterations,
        accurate_prices_used: 0,
        tier: ForwardTier::ModelFast,
        flags,
    })
}
