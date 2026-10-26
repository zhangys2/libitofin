//! Implied forward price and discount factor estimation from option quotes.
//!
//! Provides model-free estimation of forward prices $F^*$ and discount factors $D^*$
//! from two-sided option quotes with bid-ask spreads across:
//! - European cash/physical vanilla options (SPX, NDX, etc.) via Put-Call Parity:
//!   $$C(K) - P(K) = D \cdot (F - K)$$
//! - Cryptocurrency inverse options (Deribit BTC/ETH) via coin-numeraire linear parity:
//!   $$C_{\text{coin}}(K) - P_{\text{coin}}(K) = D_{\text{coin}} \left( 1 - \frac{K}{F} \right)$$
//! - American equity/ETF options (SPY, AAPL) via model-free ATM strike filtering and
//!   Merton early-exercise arbitrage bounds.

use super::american_forward;
use super::american_model::{
    AmericanModelConfig, AmericanModelDiagnostics, ForwardFlags, ForwardTier,
};
use crate::errors::QlResult;
use crate::types::{DiscountFactor, Rate, Real, Time};
use crate::{fail, require};

/// Market convention defining the quoting currency and payoff structure.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MarketConvention {
    /// European cash or physically-settled vanilla options (e.g. SPX, NDX).
    #[default]
    EuropeanVanilla,
    /// Coin-margined inverse options (e.g. Deribit BTC/ETH) where payoffs and quotes are in cryptocurrency.
    CryptoInverse,
    /// American options with early-exercise strike corridor filtering.
    AmericanEquity {
        /// Underlying spot price, used for ATM moneyness corridor filtering.
        spot: Real,
        /// Optional annualized ATM volatility to scale the corridor: $|\ln(K/S)| \le \kappa \sigma \sqrt{T}$.
        atm_vol: Option<Real>,
    },
}

/// A two-sided quote pair at a single strike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionQuotePair {
    pub strike: Real,
    pub call_bid: Real,
    pub call_ask: Real,
    pub put_bid: Real,
    pub put_ask: Real,
}

impl OptionQuotePair {
    pub fn new(strike: Real, call_bid: Real, call_ask: Real, put_bid: Real, put_ask: Real) -> Self {
        Self {
            strike,
            call_bid,
            call_ask,
            put_bid,
            put_ask,
        }
    }

    /// True if all prices and strike are non-negative, finite, ask is positive, and non-inverted ($ask \ge bid$).
    pub fn is_valid(&self) -> bool {
        self.strike > 0.0
            && self.strike.is_finite()
            && self.call_bid >= 0.0
            && self.call_bid.is_finite()
            && self.call_ask >= self.call_bid
            && self.call_ask > 0.0
            && self.call_ask.is_finite()
            && self.put_bid >= 0.0
            && self.put_bid.is_finite()
            && self.put_ask >= self.put_bid
            && self.put_ask > 0.0
            && self.put_ask.is_finite()
    }

    #[inline]
    pub fn call_mid(&self) -> Real {
        (self.call_bid + self.call_ask) * 0.5
    }

    #[inline]
    pub fn put_mid(&self) -> Real {
        (self.put_bid + self.put_ask) * 0.5
    }

    #[inline]
    pub fn call_spread(&self) -> Real {
        self.call_ask - self.call_bid
    }

    #[inline]
    pub fn put_spread(&self) -> Real {
        self.put_ask - self.put_bid
    }

    #[inline]
    pub fn combined_spread(&self) -> Real {
        self.call_spread() + self.put_spread()
    }
}

/// Qualitative status indicating health of the estimated forward and market quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForwardStatus {
    /// Clean fit, all arbitrage and consistency bounds satisfied.
    #[default]
    Valid,
    /// Strict synthetic bid-ask quotes are crossed ($F_{\text{bid}}^{\text{strict}} > F_{\text{ask}}^{\text{strict}}$).
    WarningCrossedSyntheticQuotes,
    /// Box-spread parity violated across adjacent liquid strikes.
    WarningBoxSpreadArbitrage,
    /// Implied discount factor outside plausible interval $[0.50, 1.25]$.
    WarningImplausibleDiscountFactor,
    /// Forward deviates from spot beyond threshold $|\ln(F/S)| > \delta_{\max}$.
    WarningSpotDeviationExceeded,
    /// American options early-exercise bounds widened.
    DegradedAmericanBounds,
    /// Solved carry outside the plausible range `AmericanModelConfig::carry_flag_range`.
    WarningImplausibleCarry,
}

/// Detailed diagnostic metrics accompanying the forward estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardDiagnostics {
    pub status: ForwardStatus,
    pub total_pairs_received: usize,
    pub pairs_used: usize,
    pub pairs_pruned: usize,
    pub is_crossed: bool,
    pub box_arbitrage_violations: usize,
    pub wls_rmse: Real,
    pub min_spread: Real,
    pub max_spread: Real,
}

/// Volatility assumed for the American strike window when neither the caller nor the quotes
/// provide one.
const DEFAULT_ATM_VOL: Real = 0.30;
/// Narrowest American strike window, in log moneyness.
const MIN_CORRIDOR_WINDOW: Real = 0.05;

/// Configuration options for the implied forward estimator.
#[derive(Debug, Clone)]
pub struct ImpliedForwardConfig {
    /// External discount factor $D = e^{-r T}$. If None, solved jointly from quotes.
    pub discount_factor: Option<DiscountFactor>,
    /// Underlying spot price for deviation validation and fallback.
    pub spot: Option<Real>,
    /// Minimum required valid pairs (default: 2).
    pub min_pairs: usize,
    /// Maximum pairs used in robust estimation (default: 50).
    pub max_pairs: usize,
    /// Relative spread floor regularizer $\epsilon = \text{spread\_floor} \cdot \max(K, 1.0)$ (default: 1e-4).
    pub spread_floor: Real,
    /// Maximum allowable relative deviation from spot $|F/S - 1.0|$ (default: 0.10).
    pub max_spot_deviation: Option<Real>,
    /// Multiplier for American ATM corridor: $|\ln(K/S)| \le \kappa \sigma \sqrt{T}$ (default: 0.5).
    pub american_kappa: Real,
    /// Settings of the model-based American path (ignored for other conventions).
    pub american: AmericanModelConfig,
}

impl Default for ImpliedForwardConfig {
    fn default() -> Self {
        Self {
            discount_factor: None,
            spot: None,
            min_pairs: 2,
            max_pairs: 50,
            spread_floor: 1e-4,
            max_spot_deviation: Some(0.10),
            american_kappa: 0.5,
            american: AmericanModelConfig::default(),
        }
    }
}

/// Result of the implied forward estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpliedForwardResult {
    /// Weighted point estimate $F^*$.
    pub forward: Real,
    /// Strict synthetic no-arbitrage bid bound ($\max_{i} F_{\text{bid}, i}$).
    pub forward_bid_strict: Real,
    /// Strict synthetic no-arbitrage ask bound ($\min_{i} F_{\text{ask}, i}$).
    pub forward_ask_strict: Real,
    /// Robust synthetic spread-weighted bid.
    pub forward_bid_robust: Real,
    /// Robust synthetic spread-weighted ask.
    pub forward_ask_robust: Real,
    /// Implied or anchored discount factor $D$.
    pub discount_factor: DiscountFactor,
    /// Implied annualized cost of carry $(r - q) = -\frac{\ln(D)}{T}$.
    pub implied_carry_rate: Option<Rate>,
    /// Diagnostics on quote quality, crossed quotes, and box arbitrage.
    pub diagnostics: ForwardDiagnostics,
    /// Estimator that produced the forward; `None` unless the convention is American.
    pub tier: Option<ForwardTier>,
    /// Independent signals about how an American forward was produced.
    pub flags: ForwardFlags,
    /// Numeric diagnostics of the model-based solve; `None` on the corridor tier.
    pub american: Option<AmericanModelDiagnostics>,
}

impl ImpliedForwardResult {
    /// Returns true if the estimate has `ForwardStatus::Valid`.
    pub fn is_valid(&self) -> bool {
        self.diagnostics.status == ForwardStatus::Valid
    }
}

/// High-level solver for calculating implied forward and discount factors.
pub struct ImpliedForward;

impl ImpliedForward {
    /// Calculates implied forward and discount factor from quote pairs.
    pub fn calculate(
        quotes: &[OptionQuotePair],
        convention: MarketConvention,
        config: &ImpliedForwardConfig,
        expiry_years: Time,
    ) -> QlResult<ImpliedForwardResult> {
        require!(
            expiry_years > 0.0 && expiry_years.is_finite(),
            "expiry_years must be positive and finite, got {expiry_years}"
        );
        require!(
            !quotes.is_empty(),
            "at least one option quote pair is required"
        );

        require!(
            config.min_pairs >= 2,
            "min_pairs must be at least 2, got {}",
            config.min_pairs
        );
        require!(
            config.max_pairs >= config.min_pairs,
            "max_pairs ({}) must be >= min_pairs ({})",
            config.max_pairs,
            config.min_pairs
        );

        let total_received = quotes.len();

        // 1. Initial sanity filter: drop non-positive or inverted quotes
        let valid_all: Vec<OptionQuotePair> =
            quotes.iter().copied().filter(|q| q.is_valid()).collect();

        // 2. Convention-specific strike filtering and the model-based American path. The model
        // needs a discount factor; without one, or when it cannot be solved, the corridor
        // estimate further down is returned and labeled.
        let mut model_failed = false;
        let mut corridor_window = None;
        if let MarketConvention::AmericanEquity { spot, atm_vol } = convention {
            require!(
                spot > 0.0 && spot.is_finite(),
                "AmericanEquity spot must be positive, got {spot}"
            );
            config.american.validate()?;
            config.american.validate_dividends(expiry_years)?;
            let estimated = config.discount_factor.and_then(|discount_factor| {
                american_forward::estimate_atm_vol(&valid_all, spot, discount_factor, expiry_years)
            });
            let vol = atm_vol.or(estimated).unwrap_or(DEFAULT_ATM_VOL);
            if let Some(discount_factor) = config.discount_factor {
                let window = (config.american.corridor_kappa * vol * expiry_years.sqrt())
                    .max(MIN_CORRIDOR_WINDOW);
                let attempt = Self::select_pairs(&valid_all, Some((spot, window)), config)
                    .and_then(|pairs| {
                        Self::calculate_american_model(
                            &pairs,
                            spot,
                            discount_factor,
                            config,
                            expiry_years,
                            total_received,
                        )
                    });
                match attempt {
                    Ok(result) => return Ok(result),
                    Err(_) => model_failed = true,
                }
            }
            let window =
                (config.american_kappa * vol * expiry_years.sqrt()).max(MIN_CORRIDOR_WINDOW);
            corridor_window = Some((spot, window));
        }

        let valid_pairs = Self::select_pairs(&valid_all, corridor_window, config)?;
        let pairs_used = valid_pairs.len();
        let pairs_pruned = total_received - pairs_used;

        let mut result = match convention {
            MarketConvention::CryptoInverse => Self::calculate_crypto_inverse(
                &valid_pairs,
                config,
                expiry_years,
                total_received,
                pairs_pruned,
            ),
            MarketConvention::EuropeanVanilla | MarketConvention::AmericanEquity { .. } => {
                Self::calculate_linear(
                    &valid_pairs,
                    convention,
                    config,
                    expiry_years,
                    total_received,
                    pairs_pruned,
                )
            }
        }?;
        if model_failed {
            result.flags.insert(ForwardFlags::NON_CONVERGED);
        }
        Ok(result)
    }

    /// Keeps the quotes inside the log-moneyness `window` around `spot` (when given), sorted by
    /// strike and capped to `max_pairs` nearest the center.
    fn select_pairs(
        valid: &[OptionQuotePair],
        window: Option<(Real, Real)>,
        config: &ImpliedForwardConfig,
    ) -> QlResult<Vec<OptionQuotePair>> {
        let mut pairs: Vec<OptionQuotePair> = valid.to_vec();
        if let Some((spot, window)) = window {
            pairs.retain(|q| (q.strike / spot).ln().abs() <= window);
        }

        // Sort by strike ascending
        pairs.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cap to max_pairs closest to spot (or median strike)
        if pairs.len() > config.max_pairs {
            let center = config
                .spot
                .or(window.map(|(spot, _)| spot))
                .unwrap_or_else(|| pairs[pairs.len() / 2].strike);
            pairs.sort_by(|a, b| {
                let da = (a.strike - center).abs();
                let db = (b.strike - center).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            pairs.truncate(config.max_pairs);
            pairs.sort_by(|a, b| {
                a.strike
                    .partial_cmp(&b.strike)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        require!(
            pairs.len() >= config.min_pairs,
            "insufficient valid strike pairs: expected at least {}, found {}",
            config.min_pairs,
            pairs.len()
        );
        Ok(pairs)
    }

    /// Model-based American estimate: de-Americanize the quotes under the solved carry and run
    /// the parity estimator on the European-equivalent prices.
    fn calculate_american_model(
        pairs: &[OptionQuotePair],
        spot: Real,
        discount_factor: DiscountFactor,
        config: &ImpliedForwardConfig,
        expiry_years: Time,
        total_received: usize,
    ) -> QlResult<ImpliedForwardResult> {
        let solution = american_forward::solve(
            pairs,
            spot,
            discount_factor,
            expiry_years,
            config.min_pairs,
            &config.american,
        )?;
        let backend = ImpliedForwardConfig {
            discount_factor: Some(discount_factor),
            spot: config.spot.or(Some(spot)),
            american: AmericanModelConfig::default(),
            ..config.clone()
        };
        let pairs_pruned = total_received.saturating_sub(solution.pairs.len());
        let mut result = Self::calculate_linear(
            &solution.pairs,
            MarketConvention::EuropeanVanilla,
            &backend,
            expiry_years,
            total_received,
            pairs_pruned,
        )?;

        let (flag_low, flag_high) = config.american.carry_flag_range;
        if (solution.carry < flag_low || solution.carry > flag_high)
            && result.diagnostics.status == ForwardStatus::Valid
        {
            result.diagnostics.status = ForwardStatus::WarningImplausibleCarry;
        }
        result.tier = Some(solution.tier);
        result.flags = solution.flags;
        result.american = Some(AmericanModelDiagnostics {
            solved_carry: solution.carry,
            call_put_iv_gap: solution.iv_gap,
            iterations: solution.iterations,
            accurate_prices_used: solution.accurate_prices_used,
        });
        Ok(result)
    }

    /// Parity calculation for European vanilla and American equity options.
    fn calculate_linear(
        pairs: &[OptionQuotePair],
        convention: MarketConvention,
        config: &ImpliedForwardConfig,
        expiry_years: Time,
        total_received: usize,
        _pairs_pruned: usize,
    ) -> QlResult<ImpliedForwardResult> {
        let n = pairs.len();
        let mut min_spread = Real::INFINITY;
        let mut max_spread = 0.0;

        // Compute weights
        let mut weights = Vec::with_capacity(n);
        for q in pairs {
            let cs = q.call_spread();
            let ps = q.put_spread();
            let spread_sum = cs + ps;
            if spread_sum < min_spread {
                min_spread = spread_sum;
            }
            if spread_sum > max_spread {
                max_spread = spread_sum;
            }
            let floor = config.spread_floor * q.strike.max(1.0);
            let w = 1.0 / (cs * cs + ps * ps + floor * floor);
            weights.push(w);
        }

        let total_weight: Real = weights.iter().sum();
        require!(
            total_weight > 0.0 && total_weight.is_finite(),
            "sum of regression weights is non-positive or non-finite"
        );
        for w in &mut weights {
            *w /= total_weight;
        }

        let (mut discount_factor, mut forward, mut wls_rmse, mut status) = match config
            .discount_factor
        {
            Some(df) => {
                require!(
                    df > 0.0 && df.is_finite(),
                    "discount_factor must be positive"
                );
                let mut sorted_by_spread: Vec<(Real, Real)> = pairs
                    .iter()
                    .map(|q| {
                        let spread = q.combined_spread();
                        let f_i = q.strike + (q.call_mid() - q.put_mid()) / df;
                        (spread, f_i)
                    })
                    .collect();
                sorted_by_spread
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let count = sorted_by_spread.len().min(config.max_pairs);
                let mut f_vals: Vec<Real> = sorted_by_spread[..count].iter().map(|e| e.1).collect();
                f_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let med_f = if count % 2 == 1 {
                    f_vals[count / 2]
                } else {
                    0.5 * (f_vals[count / 2 - 1] + f_vals[count / 2])
                };

                let mut sum_sq_err = 0.0;
                for (i, q) in pairs.iter().enumerate() {
                    let y = q.call_mid() - q.put_mid();
                    let model_y = df * (med_f - q.strike);
                    let err = y - model_y;
                    sum_sq_err += weights[i] * err * err;
                }
                (df, med_f, sum_sq_err.sqrt(), ForwardStatus::Valid)
            }
            None => {
                // Joint estimation via WLS
                let (df, fwd, rmse, st) = Self::fit_wls(pairs, &weights)?;
                (df, fwd, rmse, st)
            }
        };

        // Outlier pruning: refit on filtered pairs if RMSE > 0 and pairs > min_pairs + 1
        let (eval_pairs, eval_weights) = if wls_rmse > 0.0 && pairs.len() > config.min_pairs + 1 {
            let threshold = 3.5 * wls_rmse;
            let mut filtered_pairs = Vec::with_capacity(n);
            let mut filtered_weights = Vec::with_capacity(n);
            let mut pruned_any = false;

            for (i, q) in pairs.iter().enumerate() {
                let y = q.call_mid() - q.put_mid();
                let model_y = discount_factor * (forward - q.strike);
                if (y - model_y).abs() <= threshold {
                    filtered_pairs.push(*q);
                    filtered_weights.push(weights[i]);
                } else {
                    pruned_any = true;
                }
            }

            if pruned_any && filtered_pairs.len() >= config.min_pairs {
                let sum_w: Real = filtered_weights.iter().sum();
                for w in &mut filtered_weights {
                    *w /= sum_w;
                }
                if let Some(df) = config.discount_factor {
                    let mut sum_f = 0.0;
                    for (i, q) in filtered_pairs.iter().enumerate() {
                        let y = q.call_mid() - q.put_mid();
                        sum_f += filtered_weights[i] * (q.strike + y / df);
                    }
                    forward = sum_f;
                } else if let Ok((df, fwd, rmse, st)) =
                    Self::fit_wls(&filtered_pairs, &filtered_weights)
                {
                    discount_factor = df;
                    forward = fwd;
                    wls_rmse = rmse;
                    if st != ForwardStatus::Valid {
                        status = st;
                    }
                }
                (filtered_pairs, filtered_weights)
            } else {
                (pairs.to_vec(), weights)
            }
        } else {
            (pairs.to_vec(), weights)
        };

        let pairs_used = eval_pairs.len();
        let pairs_pruned = total_received.saturating_sub(pairs_used);

        // Synthetic bounds calculation using post-pruning pairs
        let mut forward_bid_strict = Real::NEG_INFINITY;
        let mut forward_ask_strict = Real::INFINITY;
        let mut forward_bid_robust = 0.0;
        let mut forward_ask_robust = 0.0;

        for (i, q) in eval_pairs.iter().enumerate() {
            let f_bid = q.strike + (q.call_bid - q.put_ask) / discount_factor;
            let f_ask = q.strike + (q.call_ask - q.put_bid) / discount_factor;

            if f_bid > forward_bid_strict {
                forward_bid_strict = f_bid;
            }
            if f_ask < forward_ask_strict {
                forward_ask_strict = f_ask;
            }

            forward_bid_robust += eval_weights[i] * f_bid;
            forward_ask_robust += eval_weights[i] * f_ask;
        }

        let is_crossed = forward_bid_strict > forward_ask_strict;
        if is_crossed && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningCrossedSyntheticQuotes;
        }

        // Box spread parity checks: (C_b1 - P_a1) - (C_a2 - P_b2) <= D(K2 - K1) <= (C_a1 - P_b1) - (C_b2 - P_a2)
        let mut box_violations = 0;
        for i in 0..eval_pairs.len().saturating_sub(1) {
            let q1 = eval_pairs[i];
            let q2 = eval_pairs[i + 1];
            let box_lower = (q1.call_bid - q1.put_ask) - (q2.call_ask - q2.put_bid);
            let box_upper = (q1.call_ask - q1.put_bid) - (q2.call_bid - q2.put_ask);
            let box_target = discount_factor * (q2.strike - q1.strike);
            if box_lower > box_target + 1e-4 || box_upper < box_target - 1e-4 {
                box_violations += 1;
            }
        }
        if box_violations > eval_pairs.len() / 2 && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningBoxSpreadArbitrage;
        }

        // Spot deviation check
        let spot_ref = config.spot.or(match convention {
            MarketConvention::AmericanEquity { spot, .. } => Some(spot),
            _ => None,
        });
        if let (Some(s), Some(max_dev)) = (spot_ref, config.max_spot_deviation) {
            let is_exceeded = (forward / s - 1.0).abs() > max_dev && status == ForwardStatus::Valid;
            if is_exceeded {
                status = ForwardStatus::WarningSpotDeviationExceeded;
            }
        }

        // American early exercise bounds check
        if let MarketConvention::AmericanEquity { .. } = convention {
            let mut f_min = Real::NEG_INFINITY;
            let mut f_max = Real::INFINITY;
            for q in &eval_pairs {
                let lb = q.strike + (q.call_bid - q.put_ask) / discount_factor;
                let ee_put = q.strike * (1.0 - discount_factor).max(0.0);
                let ub = q.strike + (q.call_ask - (q.put_bid - ee_put)) / discount_factor;
                if lb > f_min {
                    f_min = lb;
                }
                if ub < f_max {
                    f_max = ub;
                }
            }
            if (forward < f_min || forward > f_max) && status == ForwardStatus::Valid {
                status = ForwardStatus::DegradedAmericanBounds;
            }
        }

        let implied_carry_rate = if let Some(s) = spot_ref {
            if s > 0.0 && forward > 0.0 && expiry_years > 0.0 {
                Some((forward / s).ln() / expiry_years)
            } else {
                None
            }
        } else if discount_factor > 0.0 && expiry_years > 0.0 {
            Some(-discount_factor.ln() / expiry_years)
        } else {
            None
        };

        Ok(ImpliedForwardResult {
            forward,
            forward_bid_strict,
            forward_ask_strict,
            forward_bid_robust,
            forward_ask_robust,
            discount_factor,
            implied_carry_rate,
            diagnostics: ForwardDiagnostics {
                status,
                total_pairs_received: total_received,
                pairs_used,
                pairs_pruned,
                is_crossed,
                box_arbitrage_violations: box_violations,
                wls_rmse,
                min_spread,
                max_spread,
            },
            tier: matches!(convention, MarketConvention::AmericanEquity { .. })
                .then_some(ForwardTier::Corridor),
            flags: ForwardFlags::NONE,
            american: None,
        })
    }

    /// Parity calculation for coin-margined inverse cryptocurrency options (Deribit style).
    fn calculate_crypto_inverse(
        pairs: &[OptionQuotePair],
        config: &ImpliedForwardConfig,
        expiry_years: Time,
        total_received: usize,
        pairs_pruned: usize,
    ) -> QlResult<ImpliedForwardResult> {
        let n = pairs.len();
        let mut min_spread = Real::INFINITY;
        let mut max_spread = 0.0;

        let mut weights = Vec::with_capacity(n);
        for q in pairs {
            let cs = q.call_spread();
            let ps = q.put_spread();
            let spread_sum = cs + ps;
            if spread_sum < min_spread {
                min_spread = spread_sum;
            }
            if spread_sum > max_spread {
                max_spread = spread_sum;
            }
            // Coin space spread floor
            let floor = config.spread_floor;
            let w = 1.0 / (cs * cs + ps * ps + floor * floor);
            weights.push(w);
        }

        let total_weight: Real = weights.iter().sum();
        require!(
            total_weight > 0.0 && total_weight.is_finite(),
            "sum of regression weights is non-positive or non-finite"
        );
        for w in &mut weights {
            *w /= total_weight;
        }

        // Coin space linear regression:
        // y_i = C_btc - P_btc = D_coin - (D_coin / F) * K_i = alpha + beta * K_i
        let mut mean_k = 0.0;
        let mut mean_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            mean_k += weights[i] * q.strike;
            mean_y += weights[i] * y;
        }

        let mut var_k = 0.0;
        let mut cov_k_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let dk = q.strike - mean_k;
            let dy = y - mean_y;
            var_k += weights[i] * dk * dk;
            cov_k_y += weights[i] * dk * dy;
        }

        require!(
            var_k > 1e-12,
            "degenerate strike distribution in crypto inverse calculation"
        );

        let beta = cov_k_y / var_k;
        let alpha = mean_y - beta * mean_k;

        let mut status = ForwardStatus::Valid;
        let (d_coin, forward) = if beta < -1e-12 {
            let f = -alpha / beta;
            if f <= 0.0 || !f.is_finite() {
                fail!("calculated negative or infinite crypto forward: {f}");
            }
            if !(0.5..=1.5).contains(&alpha) && status == ForwardStatus::Valid {
                status = ForwardStatus::WarningImplausibleDiscountFactor;
            }
            (alpha.clamp(0.5, 1.5), f)
        } else {
            status = ForwardStatus::WarningImplausibleDiscountFactor;
            // Fallback: solve F assuming alpha = 1.0
            (1.0, mean_k / (1.0 - mean_y).max(1e-4))
        };

        // RMSE
        let mut sum_sq_err = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let pred = alpha + beta * q.strike;
            let err = y - pred;
            sum_sq_err += weights[i] * err * err;
        }
        let wls_rmse = sum_sq_err.sqrt();

        // Synthetic bounds in coin space:
        // C - P = D_coin * (1 - K / F) => F = D_coin * K / (D_coin - (C - P))
        let mut forward_bid_strict = Real::NEG_INFINITY;
        let mut forward_ask_strict = Real::INFINITY;
        let mut forward_bid_robust = 0.0;
        let mut forward_ask_robust = 0.0;

        for (i, q) in pairs.iter().enumerate() {
            let diff_ask = q.call_ask - q.put_bid;
            let diff_bid = q.call_bid - q.put_ask;

            let denom_ask = d_coin - diff_ask;
            let denom_bid = d_coin - diff_bid;

            let f_ask = if denom_ask > 1e-4 {
                d_coin * q.strike / denom_ask
            } else {
                q.strike
            };

            let f_bid = if denom_bid > 1e-4 {
                d_coin * q.strike / denom_bid
            } else {
                q.strike
            };

            if f_bid > forward_bid_strict {
                forward_bid_strict = f_bid;
            }
            if f_ask < forward_ask_strict {
                forward_ask_strict = f_ask;
            }

            forward_bid_robust += weights[i] * f_bid;
            forward_ask_robust += weights[i] * f_ask;
        }

        let is_crossed = forward_bid_strict > forward_ask_strict;
        if is_crossed && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningCrossedSyntheticQuotes;
        }

        // Spot deviation check if spot given
        if let (Some(s), Some(max_dev)) = (config.spot, config.max_spot_deviation) {
            let is_exceeded = (forward / s - 1.0).abs() > max_dev && status == ForwardStatus::Valid;
            if is_exceeded {
                status = ForwardStatus::WarningSpotDeviationExceeded;
            }
        }

        // Consistent coin discount factor
        let discount_factor = d_coin;

        let implied_carry_rate = if let Some(s) = config.spot {
            if s > 0.0 && forward > 0.0 && expiry_years > 0.0 {
                Some((forward / s).ln() / expiry_years)
            } else {
                None
            }
        } else {
            None
        };

        Ok(ImpliedForwardResult {
            forward,
            forward_bid_strict,
            forward_ask_strict,
            forward_bid_robust,
            forward_ask_robust,
            discount_factor,
            implied_carry_rate,
            diagnostics: ForwardDiagnostics {
                status,
                total_pairs_received: total_received,
                pairs_used: pairs.len(),
                pairs_pruned,
                is_crossed,
                box_arbitrage_violations: 0,
                wls_rmse,
                min_spread,
                max_spread,
            },
            tier: None,
            flags: ForwardFlags::NONE,
            american: None,
        })
    }

    /// Fits (discount_factor, forward) via weighted least squares regression:
    /// $y_i \equiv C_{\text{mid}, i} - P_{\text{mid}, i} = \alpha + \beta K_i$ with $\beta = -D, \alpha = D \cdot F$.
    fn fit_wls(
        pairs: &[OptionQuotePair],
        weights: &[Real],
    ) -> QlResult<(DiscountFactor, Real, Real, ForwardStatus)> {
        let mut mean_k = 0.0;
        let mut mean_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            mean_k += weights[i] * q.strike;
            mean_y += weights[i] * y;
        }

        let mut var_k = 0.0;
        let mut cov_k_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let dk = q.strike - mean_k;
            let dy = y - mean_y;
            var_k += weights[i] * dk * dk;
            cov_k_y += weights[i] * dk * dy;
        }

        require!(
            var_k > 1e-12,
            "degenerate strike distribution: variance of strikes is zero"
        );

        let beta = cov_k_y / var_k;
        let alpha = mean_y - beta * mean_k;

        let df_candidate = -beta;
        let mut status = ForwardStatus::Valid;

        let (discount_factor, forward) = if (0.50..=1.25).contains(&df_candidate) {
            (df_candidate, alpha / df_candidate)
        } else {
            status = ForwardStatus::WarningImplausibleDiscountFactor;
            // Fallback to D = 1.0
            (1.0, mean_k + mean_y)
        };

        // Compute RMSE
        let mut sum_sq_err = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let pred = discount_factor * (forward - q.strike);
            let err = y - pred;
            sum_sq_err += weights[i] * err * err;
        }
        let rmse = sum_sq_err.sqrt();

        Ok((discount_factor, forward, rmse, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_european_parity_unanchored() {
        let true_forward = 5050.0;
        let true_df = 0.985;
        let expiry = 0.25;

        let strikes = vec![4900.0, 4950.0, 5000.0, 5050.0, 5100.0, 5150.0, 5200.0];
        let mut quotes = Vec::new();

        for &k in &strikes {
            // Parity mid diff: C - P = df * (F - K)
            let parity_diff = true_df * (true_forward - k);
            let base_c = if parity_diff > 0.0 {
                parity_diff + 50.0
            } else {
                50.0
            };
            let base_p = base_c - parity_diff;

            // Add 1.0 bid-ask spread
            quotes.push(OptionQuotePair::new(
                k,
                base_c - 0.5,
                base_c + 0.5,
                base_p - 0.5,
                base_p + 0.5,
            ));
        }

        let config = ImpliedForwardConfig::default();
        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, expiry)
                .unwrap();

        assert!(res.is_valid());
        assert!(
            (res.forward - true_forward).abs() < 1e-4,
            "forward={}",
            res.forward
        );
        assert!(
            (res.discount_factor - true_df).abs() < 1e-4,
            "df={}",
            res.discount_factor
        );
        assert!(!res.diagnostics.is_crossed);
        assert_eq!(res.diagnostics.box_arbitrage_violations, 0);
        assert!(res.forward_bid_strict <= res.forward);
        assert!(res.forward <= res.forward_ask_strict);
    }

    #[test]
    fn test_perfect_crypto_inverse_parity() {
        let true_forward = 65000.0;
        let expiry = 0.1;
        let strikes = vec![
            60000.0, 62000.0, 64000.0, 65000.0, 66000.0, 68000.0, 70000.0,
        ];

        let mut quotes = Vec::new();
        for &k in &strikes {
            // Coin parity diff: C_btc - P_btc = 1.0 - K / F
            let diff_btc: Real = 1.0 - k / true_forward;
            let base_c = 0.05 + diff_btc.max(0.0);
            let base_p = base_c - diff_btc;

            // Spread of 0.002 BTC
            quotes.push(OptionQuotePair::new(
                k,
                base_c - 0.001,
                base_c + 0.001,
                base_p - 0.001,
                base_p + 0.001,
            ));
        }

        let config = ImpliedForwardConfig {
            spot: Some(64500.0),
            ..Default::default()
        };

        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::CryptoInverse, &config, expiry)
                .unwrap();

        assert!(res.is_valid());
        assert!(
            (res.forward - true_forward).abs() < 1.0,
            "forward={}",
            res.forward
        );
        assert!(!res.diagnostics.is_crossed);
    }

    #[test]
    fn test_crossed_synthetic_quotes_flagged() {
        let expiry = 0.1;
        // Strike with inverted call quotes
        let quotes = vec![
            OptionQuotePair::new(5000.0, 100.0, 101.0, 80.0, 81.0),
            // Bad quote with crossed synthetic bounds
            OptionQuotePair::new(5050.0, 95.0, 95.5, 50.0, 50.5),
            OptionQuotePair::new(5100.0, 50.0, 51.0, 100.0, 101.0),
        ];

        let config = ImpliedForwardConfig {
            discount_factor: Some(1.0),
            ..Default::default()
        };

        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, expiry)
                .unwrap();

        // Should return a result but flag warning
        assert!(res.diagnostics.status != ForwardStatus::Valid);
    }

    #[test]
    fn test_american_equity_atm_filtering() {
        let spot = 100.0;
        let expiry = 0.25;

        // Wide array of strikes from 50 to 150
        let strikes = vec![50.0, 70.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0, 150.0];
        let mut quotes = Vec::new();
        for &k in &strikes {
            let diff = 100.0 - k;
            quotes.push(OptionQuotePair::new(
                k,
                (diff + 10.0).max(1.0),
                (diff + 11.0).max(2.0),
                10.0,
                11.0,
            ));
        }

        let config = ImpliedForwardConfig::default();
        let res = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot,
                atm_vol: Some(0.20),
            },
            &config,
            expiry,
        )
        .unwrap();

        // Wings should have been pruned by ATM corridor
        assert!(res.diagnostics.pairs_pruned > 0);
        assert!(res.diagnostics.pairs_used < strikes.len());
    }

    #[test]
    fn test_zero_bid_quotes_accepted() {
        let pair = OptionQuotePair::new(5000.0, 0.0, 1.0, 50.0, 51.0);
        assert!(pair.is_valid());
    }

    #[test]
    fn test_crypto_inverse_parity_non_unit_df() {
        let true_forward = 70000.0;
        let d_coin = 0.95;
        let expiry = 0.25;
        let strikes = vec![65000.0, 68000.0, 70000.0, 72000.0, 75000.0];

        let mut quotes = Vec::new();
        for &k in &strikes {
            // Parity: C - P = d_coin * (1 - K / F)
            let diff_coin: Real = d_coin * (1.0 - k / true_forward);
            let base_c = 0.1 + diff_coin.max(0.0);
            let base_p = base_c - diff_coin;

            quotes.push(OptionQuotePair::new(
                k,
                base_c - 0.001,
                base_c + 0.001,
                base_p - 0.001,
                base_p + 0.001,
            ));
        }

        let config = ImpliedForwardConfig {
            spot: Some(69500.0),
            ..Default::default()
        };

        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::CryptoInverse, &config, expiry)
                .unwrap();

        assert!(res.is_valid());
        assert!(
            (res.forward - true_forward).abs() < 5.0,
            "forward={}",
            res.forward
        );
        assert!(
            (res.discount_factor - d_coin).abs() < 1e-3,
            "d_coin={}",
            res.discount_factor
        );
        assert!(res.forward_bid_strict <= res.forward && res.forward <= res.forward_ask_strict);
    }

    #[test]
    fn test_invalid_config_pair_counts() {
        let quotes = vec![
            OptionQuotePair::new(5000.0, 10.0, 11.0, 10.0, 11.0),
            OptionQuotePair::new(5100.0, 5.0, 6.0, 15.0, 16.0),
        ];
        let config = ImpliedForwardConfig {
            min_pairs: 5,
            max_pairs: 3,
            ..Default::default()
        };
        let err =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, 0.1);
        assert!(err.is_err());
    }

    fn american_parity_quotes(forward: Real, strikes: &[Real]) -> Vec<OptionQuotePair> {
        strikes
            .iter()
            .map(|&k| {
                let diff = forward - k;
                let c = diff.max(0.0) + 2.0;
                let p = c - diff;
                OptionQuotePair::new(k, c - 0.05, c + 0.05, p - 0.05, p + 0.05)
            })
            .collect()
    }

    #[test]
    fn test_american_convention_without_discount_factor_reports_corridor_tier() {
        let quotes = american_parity_quotes(101.0, &[95.0, 97.5, 100.0, 102.5, 105.0]);
        let config = ImpliedForwardConfig::default();
        let res = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot: 100.0,
                atm_vol: Some(0.25),
            },
            &config,
            0.25,
        )
        .unwrap();
        assert_eq!(res.tier, Some(ForwardTier::Corridor));
        assert!(res.flags.is_empty());
        assert!(res.american.is_none());
    }

    #[test]
    fn test_non_american_results_carry_no_tier() {
        let quotes = american_parity_quotes(5000.0, &[4900.0, 5000.0, 5100.0]);
        let config = ImpliedForwardConfig::default();
        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, 0.25)
                .unwrap();
        assert_eq!(res.tier, None);
        assert!(res.flags.is_empty());
        assert!(res.american.is_none());
    }

    #[test]
    fn test_invalid_american_config_is_rejected_only_for_american() {
        let quotes = american_parity_quotes(101.0, &[95.0, 97.5, 100.0, 102.5, 105.0]);
        let config = ImpliedForwardConfig {
            discount_factor: Some(1.0),
            american: AmericanModelConfig {
                carry_search_range: (0.25, -0.10),
                ..Default::default()
            },
            ..Default::default()
        };
        let american = MarketConvention::AmericanEquity {
            spot: 100.0,
            atm_vol: None,
        };
        assert!(ImpliedForward::calculate(&quotes, american, &config, 0.25).is_err());
        // The American settings are ignored for the other conventions.
        assert!(
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, 0.25)
                .is_ok()
        );
    }

    #[test]
    fn test_dividend_schedule_must_fall_inside_the_expiry() {
        let quotes = american_parity_quotes(101.0, &[95.0, 97.5, 100.0, 102.5, 105.0]);
        let american = MarketConvention::AmericanEquity {
            spot: 100.0,
            atm_vol: None,
        };
        let with_dividend = |time: Real, amount: Real| ImpliedForwardConfig {
            discount_factor: Some(1.0),
            american: AmericanModelConfig {
                dividends: vec![(time, amount)],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            ImpliedForward::calculate(&quotes, american, &with_dividend(0.10, 0.5), 0.25).is_ok()
        );
        assert!(
            ImpliedForward::calculate(&quotes, american, &with_dividend(0.30, 0.5), 0.25).is_err()
        );
        assert!(
            ImpliedForward::calculate(&quotes, american, &with_dividend(0.0, 0.5), 0.25).is_err()
        );
        assert!(
            ImpliedForward::calculate(&quotes, american, &with_dividend(0.10, -0.5), 0.25).is_err()
        );
    }

    use super::super::american_model::AmericanLevel;
    use super::super::american_pricer::{AmericanPricer, PricerKind};
    use crate::option::OptionType;

    /// Mid prices from `kind` at a flat vol, with a fixed absolute half spread.
    fn priced_quotes(
        kind: PricerKind,
        spot: Real,
        rate: Real,
        carry: Real,
        expiry: Real,
        vol: Real,
        strikes: &[Real],
    ) -> Vec<OptionQuotePair> {
        priced_quotes_with_dividends(kind, &[], spot, rate, carry, expiry, vol, strikes)
    }

    #[allow(clippy::too_many_arguments)]
    fn priced_quotes_with_dividends(
        kind: PricerKind,
        dividends: &[(Real, Real)],
        spot: Real,
        rate: Real,
        carry: Real,
        expiry: Real,
        vol: Real,
        strikes: &[Real],
    ) -> Vec<OptionQuotePair> {
        let pricer = AmericanPricer::new(kind, expiry, dividends).unwrap();
        let half = 0.01;
        strikes
            .iter()
            .map(|&k| {
                let c = pricer
                    .price(OptionType::Call, k, spot, rate, carry, vol)
                    .unwrap();
                let p = pricer
                    .price(OptionType::Put, k, spot, rate, carry, vol)
                    .unwrap();
                OptionQuotePair::new(k, c - half, c + half, p - half, p + half)
            })
            .collect()
    }

    fn american(spot: Real) -> MarketConvention {
        MarketConvention::AmericanEquity {
            spot,
            atm_vol: Some(0.25),
        }
    }

    fn model_config(rate: Real, expiry: Real) -> ImpliedForwardConfig {
        ImpliedForwardConfig {
            discount_factor: Some((-rate * expiry).exp()),
            // Long expiries at 5% rates put the forward more than 10% from spot.
            max_spot_deviation: None,
            ..Default::default()
        }
    }

    const STRIKES: [Real; 9] = [90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0];

    #[test]
    fn test_model_path_recovers_the_carry_of_its_own_pricer_exactly() {
        // Quotes priced by the Ju pricer the Fast level uses: only the algorithm is under test.
        for &(rate, carry, expiry) in &[(0.05, 0.0, 0.5), (0.01, 0.03, 1.0), (0.04, 0.015, 0.25)] {
            let quotes = priced_quotes(
                PricerKind::JuQuadratic,
                100.0,
                rate,
                carry,
                expiry,
                0.25,
                &STRIKES,
            );
            let config = model_config(rate, expiry);
            let res = ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
            let true_forward = 100.0 * ((rate - carry) * expiry).exp();
            assert_eq!(res.tier, Some(ForwardTier::ModelFast), "r={rate} q={carry}");
            assert!(
                res.is_valid(),
                "r={rate} q={carry}: {:?}",
                res.diagnostics.status
            );
            assert!(
                (res.forward / true_forward - 1.0).abs() < 2e-4,
                "r={rate} q={carry}: {} vs {true_forward}",
                res.forward
            );
            let diag = res.american.unwrap();
            assert!(
                (diag.solved_carry - carry).abs() < 1e-5,
                "{}",
                diag.solved_carry
            );
            assert!(
                diag.call_put_iv_gap.abs() < 1e-6,
                "{}",
                diag.call_put_iv_gap
            );
            assert_eq!(diag.accurate_prices_used, 0);
            assert!(diag.iterations >= 1);
            assert!(res.flags.is_empty());
        }
    }

    #[test]
    fn test_model_path_beats_plain_parity_on_independently_priced_quotes() {
        // Quotes from a 1000-step CRR tree, a pricer independent of the Fast level's Ju engine.
        // (r = 5%, q = 0, 2 years) is where early exercise biases plain parity the most.
        for &(rate, carry, expiry) in &[(0.05, 0.0, 2.0), (0.03, 0.06, 2.0), (0.05, 0.0, 0.5)] {
            let quotes = priced_quotes(
                PricerKind::Binomial(1000),
                100.0,
                rate,
                carry,
                expiry,
                0.25,
                &STRIKES,
            );
            let true_forward = 100.0 * ((rate - carry) * expiry).exp();
            let config = model_config(rate, expiry);
            let model =
                ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
            let plain = ImpliedForward::calculate(
                &quotes,
                MarketConvention::EuropeanVanilla,
                &config,
                expiry,
            )
            .unwrap();
            let model_error = (model.forward / true_forward - 1.0).abs();
            let plain_error = (plain.forward / true_forward - 1.0).abs();
            assert_eq!(model.tier, Some(ForwardTier::ModelFast));
            assert!(
                model_error < 8e-4,
                "r={rate} q={carry} T={expiry}: {model_error}"
            );
            assert!(
                plain_error > 2.0 * model_error,
                "r={rate} q={carry} T={expiry}: plain {plain_error} vs model {model_error}"
            );
        }
    }

    #[test]
    fn test_model_path_keeps_the_synthetic_bounds_around_the_forward() {
        let (rate, carry, expiry) = (0.05, 0.0, 1.0);
        let quotes = priced_quotes(
            PricerKind::Binomial(600),
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let config = model_config(rate, expiry);
        let res = ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
        assert!(res.forward_bid_strict <= res.forward_ask_strict);
        assert!(
            res.forward_bid_strict - 0.2 <= res.forward
                && res.forward <= res.forward_ask_strict + 0.2
        );
        assert!(res.diagnostics.pairs_used >= 5);
    }

    #[test]
    fn test_implausible_carry_is_reported_as_a_warning() {
        // A 20% yield is inside the 25% search range but outside the 15% plausibility range.
        let (rate, carry, expiry) = (0.05, 0.20, 0.5);
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let config = model_config(rate, expiry);
        let res = ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert_eq!(
            res.diagnostics.status,
            ForwardStatus::WarningImplausibleCarry
        );
        assert!((res.american.unwrap().solved_carry - carry).abs() < 1e-5);
    }

    #[test]
    fn test_unsolvable_quotes_fall_back_to_the_corridor_and_say_so() {
        // Put quotes far below intrinsic value cannot be inverted by any pricer.
        let strikes = [95.0, 97.5, 100.0, 102.5, 105.0];
        let quotes: Vec<OptionQuotePair> = strikes
            .iter()
            .map(|&k| OptionQuotePair::new(k, 12.0, 12.1, 0.001, 0.002))
            .collect();
        let config = model_config(0.05, 0.5);
        let res = ImpliedForward::calculate(&quotes, american(100.0), &config, 0.5).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::Corridor));
        assert!(res.flags.contains(ForwardFlags::NON_CONVERGED));
        assert!(res.american.is_none());
    }

    #[test]
    fn test_fast_level_flags_an_ignored_dividend_schedule() {
        let (rate, carry, expiry) = (0.05, 0.0, 0.5);
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let mut config = model_config(rate, expiry);
        config.american.dividends = vec![(0.2, 0.5)];
        config.american.level = AmericanLevel::Fast;
        let res = ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert!(res.flags.contains(ForwardFlags::SCHEDULE_IGNORED));
    }

    fn accurate_config(rate: Real, expiry: Real) -> ImpliedForwardConfig {
        let mut config = model_config(rate, expiry);
        config.american.level = AmericanLevel::Accurate;
        config
    }

    fn forward_error_bp(result: &ImpliedForwardResult, true_forward: Real) -> Real {
        (result.forward / true_forward - 1.0) * 1e4
    }

    #[test]
    fn test_accurate_level_corrects_the_closed_form_bias() {
        // Quotes from a 1000-step CRR tree. The correction pass uses a 400-step tree.
        for &(rate, carry, expiry) in &[(0.05, 0.0, 2.0), (0.03, 0.06, 2.0), (0.05, 0.0, 0.5)] {
            let quotes = priced_quotes(
                PricerKind::Binomial(1000),
                100.0,
                rate,
                carry,
                expiry,
                0.25,
                &STRIKES,
            );
            let true_forward = 100.0 * ((rate - carry) * expiry).exp();
            let fast = ImpliedForward::calculate(
                &quotes,
                american(100.0),
                &model_config(rate, expiry),
                expiry,
            )
            .unwrap();
            let accurate = ImpliedForward::calculate(
                &quotes,
                american(100.0),
                &accurate_config(rate, expiry),
                expiry,
            )
            .unwrap();
            assert_eq!(fast.tier, Some(ForwardTier::ModelFast));
            assert_eq!(accurate.tier, Some(ForwardTier::ModelAccurate));
            let fast_error = forward_error_bp(&fast, true_forward).abs();
            let accurate_error = forward_error_bp(&accurate, true_forward).abs();
            assert!(
                accurate_error < 1.5,
                "r={rate} q={carry} T={expiry}: accurate {accurate_error} bp"
            );
            assert!(
                accurate_error < fast_error,
                "r={rate} q={carry} T={expiry}: accurate {accurate_error} vs fast {fast_error}"
            );
            let diag = accurate.american.unwrap();
            // One pass, two accurate prices per strike, all nine strikes.
            assert_eq!(diag.accurate_prices_used, 2 * STRIKES.len());
            assert!(accurate.flags.is_empty());
            assert!(accurate.is_valid(), "{:?}", accurate.diagnostics.status);
        }
    }

    #[test]
    fn test_auto_level_chooses_accurate_only_with_a_schedule() {
        let (rate, carry, expiry) = (0.04, 0.0, 1.0);
        let dividends = [(0.25, 1.0), (0.75, 1.0)];
        let quotes = priced_quotes_with_dividends(
            PricerKind::FiniteDifference {
                t_grid: 100,
                x_grid: 200,
            },
            &dividends,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let auto_plain = model_config(rate, expiry);
        let res = ImpliedForward::calculate(&quotes, american(100.0), &auto_plain, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert!(res.flags.is_empty());

        let mut auto_with_schedule = model_config(rate, expiry);
        auto_with_schedule.american.dividends = dividends.to_vec();
        let res = ImpliedForward::calculate(&quotes, american(100.0), &auto_with_schedule, expiry)
            .unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelAccurate));
        assert!(res.flags.is_empty());
    }

    #[test]
    fn test_schedule_recovers_a_lumpy_dividend_that_flat_carry_cannot() {
        // A special dividend of 3.0 at day 100 of a 1-year option (r = 4%, vol 25%). Flat carry
        // misses the early exercise just before the ex-date by tens of basis points.
        let (rate, expiry) = (0.04, 1.0);
        let dividends = [(100.0 / 360.0, 3.0)];
        let quotes = priced_quotes_with_dividends(
            PricerKind::FiniteDifference {
                t_grid: 200,
                x_grid: 400,
            },
            &dividends,
            100.0,
            rate,
            0.0,
            expiry,
            0.25,
            &STRIKES,
        );
        let pv = 3.0 * (-rate * 100.0 / 360.0_f64).exp();
        let true_forward = (100.0 - pv) * (rate * expiry).exp();

        let mut accurate = accurate_config(rate, expiry);
        accurate.american.dividends = dividends.to_vec();
        let res = ImpliedForward::calculate(&quotes, american(100.0), &accurate, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelAccurate));
        assert!(
            forward_error_bp(&res, true_forward).abs() < 3.0,
            "{}",
            forward_error_bp(&res, true_forward)
        );
        // The reported carry is the residual on top of the schedule, close to zero here.
        assert!(res.american.unwrap().solved_carry.abs() < 2e-3);
        assert!(res.flags.is_empty());

        let mut fast = model_config(rate, expiry);
        fast.american.dividends = dividends.to_vec();
        fast.american.level = AmericanLevel::Fast;
        let res = ImpliedForward::calculate(&quotes, american(100.0), &fast, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert!(res.flags.contains(ForwardFlags::SCHEDULE_IGNORED));
        assert!(forward_error_bp(&res, true_forward).abs() > 15.0);
    }

    #[test]
    fn test_work_budget_limits_the_correction_deterministically() {
        let (rate, carry, expiry) = (0.05, 0.0, 1.0);
        let quotes = priced_quotes(
            PricerKind::Binomial(600),
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let run = |budget: usize| {
            let mut config = accurate_config(rate, expiry);
            config.american.max_accurate_prices = budget;
            ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap()
        };

        // Eight accurate prices pay for four strikes: still Accurate, but reduced and flagged.
        let reduced = run(8);
        assert_eq!(reduced.tier, Some(ForwardTier::ModelAccurate));
        assert!(reduced.flags.contains(ForwardFlags::LATENCY_FALLBACK));
        assert_eq!(reduced.american.unwrap().accurate_prices_used, 8);
        assert_eq!(reduced.diagnostics.pairs_used, 4);

        // One accurate price pays for no strike: Fast, flagged.
        let none = run(1);
        assert_eq!(none.tier, Some(ForwardTier::ModelFast));
        assert!(none.flags.contains(ForwardFlags::LATENCY_FALLBACK));
        assert_eq!(none.american.unwrap().accurate_prices_used, 0);

        // The same budget gives the same answer.
        assert_eq!(run(8), reduced);
    }

    #[test]
    fn test_zero_correction_passes_means_fast() {
        let (rate, carry, expiry) = (0.05, 0.0, 1.0);
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let mut config = accurate_config(rate, expiry);
        config.american.correction_passes = Some(0);
        let res = ImpliedForward::calculate(&quotes, american(100.0), &config, expiry).unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert!(res.flags.is_empty());
    }

    fn wide_strikes() -> Vec<Real> {
        (0..17).map(|i| 80.0 + 2.5 * i as Real).collect()
    }

    #[test]
    fn test_model_path_uses_a_wider_corridor_than_the_corridor_tier() {
        let (rate, carry, expiry) = (0.04, 0.01, 0.5);
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &wide_strikes(),
        );
        let with_discount = model_config(rate, expiry);
        let mut without_discount = with_discount.clone();
        without_discount.discount_factor = None;
        let model =
            ImpliedForward::calculate(&quotes, american(100.0), &with_discount, expiry).unwrap();
        let corridor =
            ImpliedForward::calculate(&quotes, american(100.0), &without_discount, expiry).unwrap();
        assert_eq!(model.tier, Some(ForwardTier::ModelFast));
        assert_eq!(corridor.tier, Some(ForwardTier::Corridor));
        assert!(
            model.diagnostics.pairs_used > corridor.diagnostics.pairs_used,
            "{} vs {}",
            model.diagnostics.pairs_used,
            corridor.diagnostics.pairs_used
        );
    }

    #[test]
    fn test_missing_atm_vol_is_estimated_from_the_quotes() {
        // Quotes at 10% volatility: the estimated vol gives a narrower window than the 30%
        // fallback, so fewer strikes are used.
        let (rate, carry, expiry) = (0.04, 0.01, 0.5);
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.10,
            &wide_strikes(),
        );
        let config = model_config(rate, expiry);
        let estimated = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot: 100.0,
                atm_vol: None,
            },
            &config,
            expiry,
        )
        .unwrap();
        let fixed = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot: 100.0,
                atm_vol: Some(0.30),
            },
            &config,
            expiry,
        )
        .unwrap();
        assert_eq!(estimated.tier, Some(ForwardTier::ModelFast));
        assert!(
            estimated.diagnostics.pairs_used < fixed.diagnostics.pairs_used,
            "{} vs {}",
            estimated.diagnostics.pairs_used,
            fixed.diagnostics.pairs_used
        );
    }

    #[test]
    fn test_corridor_tier_without_a_discount_factor_keeps_the_default_vol() {
        // Without a discount factor no vol can be estimated; the 30% default window applies.
        let quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            0.04,
            0.01,
            0.5,
            0.10,
            &wide_strikes(),
        );
        let config = ImpliedForwardConfig::default();
        let res = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot: 100.0,
                atm_vol: None,
            },
            &config,
            0.5,
        )
        .unwrap();
        assert_eq!(res.tier, Some(ForwardTier::Corridor));
        // window = 0.5 * 0.30 * sqrt(0.5) = 0.106: strikes 90 .. 110 inclusive, nine of them.
        assert_eq!(res.diagnostics.pairs_used, 9);
    }

    #[test]
    fn test_zero_rate_quotes_are_solved_instead_of_falling_back() {
        // At r = 0 the closed forms divide 0 by 0; the adapter nudges the rate off zero.
        let (rate, carry, expiry) = (0.0, 0.0, 1.0);
        let quotes = priced_quotes(
            PricerKind::Binomial(400),
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        let res = ImpliedForward::calculate(
            &quotes,
            american(100.0),
            &model_config(rate, expiry),
            expiry,
        )
        .unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelFast));
        assert!(
            forward_error_bp(&res, 100.0).abs() < 2.0,
            "{}",
            forward_error_bp(&res, 100.0)
        );
    }

    #[test]
    fn test_accurate_level_drops_strikes_it_cannot_invert() {
        let (rate, carry, expiry) = (0.05, 0.0, 1.0);
        let mut quotes = priced_quotes(
            PricerKind::JuQuadratic,
            100.0,
            rate,
            carry,
            expiry,
            0.25,
            &STRIKES,
        );
        // A put priced far below any volatility the pricer can produce.
        quotes.push(OptionQuotePair::new(120.0, 0.001, 0.002, 0.0005, 0.001));
        let res = ImpliedForward::calculate(
            &quotes,
            american(100.0),
            &accurate_config(rate, expiry),
            expiry,
        )
        .unwrap();
        assert_eq!(res.tier, Some(ForwardTier::ModelAccurate));
        assert_eq!(res.diagnostics.pairs_used, STRIKES.len());
        assert!(!res.flags.contains(ForwardFlags::NON_CONVERGED));
    }
}
